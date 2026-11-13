//! Paramagnetic metalloprotein resonance assignment (`PRE` blind-zone priors) and 5-parameter
//! traceless symmetric magnetic susceptibility anisotropy ($\Delta\boldsymbol{\chi}$) tensor
//! determination from pseudocontact shifts (`PCS`).
//!
//! Provenance: Clean-room original implementation derived strictly from peer-reviewed literature:
//! 1. Solomon, I. (1955). "Relaxation Processes in a System of Two Spins." *Phys. Rev.* 99(2),
//!    559–565 (dipolar paramagnetic relaxation enhancement $\Gamma_2 \propto r_{\text{M}}^{-6}$).
//! 2. Bertini, I., Luchinat, C. & Parigi, G. (2002). "Magnetic susceptibility in paramagnetic NMR."
//!    *Prog. Nucl. Magn. Reson. Spectrosc.* 40(3), 249–273 (pseudocontact shifts, contact shifts,
//!    gyromagnetic $\gamma_I^2$ scaling, and Solomon–Bloembergen `PRE` blind spheres around
//!    paramagnetic metal centers in metalloproteins).
//! 3. Schmitz, C., Stanton-Cook, M. J., Su, X.-C., Otting, G. & Huber, T. (2008). "NUMBAT: a
//!    graphical interface for computing $\Delta\chi$-tensors from pseudocontact shifts."
//!    *J. Biomol. NMR* 41(3), 179–189 (linear 5-component traceless symmetric $\Delta\chi$-tensor
//!    SVD fitting, principal-axis diagonalization, ZYZ Euler angles, and 3D metal-site refinement).
//! 4. Cornilescu, G., Marquardt, J. L., Ottiger, M. & Bax, A. (1998). *J. Am. Chem. Soc.* 120(27),
//!    6836–6837 (anisotropic quality factor $Q$).

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::evo_graph::{assign_evo_graph_from_peaks, EvoGraphConfig, EvoGraphReport};
use crate::noesy::{reconstruct_ca_substituents, v_dist, v_unit, PdbChain, PdbResidue, Vec3};
use crate::peaks::Peak;
use crate::priors::{one_to_three, Priors};
use crate::solver::{
    assign_from_matrices_with_locks, pair_matrix, prepare_scoring_priors,
    resolve_locked_assignments, unary_matrix, AnnealConfig, AssignResult, Matrix2D, ScoreConfig,
    FORBID,
};
use crate::spinsystems::SpinSystem;

/// Physical SI-to-NMR prefactor $\frac{10^4}{12\pi}$ converting $\Delta\chi$ in $10^{-32}\text{ m}^3$
/// and nuclear distance $r$ in Å into pseudocontact shift $\Delta\delta^{\text{PCS}}$ in ppm:
/// $$\Delta\delta^{\text{PCS}} = \frac{10^4}{12\pi r^3}\left[\Delta\chi_{\text{ax}}(3\cos^2\theta - 1) + \frac{3}{2}\Delta\chi_{\text{rh}}\sin^2\theta\cos 2\phi\right].$$
pub const PCS_SPHERICAL_PREFACTOR: f64 = 1.0e4 / (12.0 * PI);

/// Equivalent Cartesian quadratic-form prefactor $\frac{10^4}{4\pi} = 3 \times \frac{10^4}{12\pi}$
/// such that $\Delta\delta^{\text{PCS}} = \frac{10^4}{4\pi r^5}\,(\mathbf{r}^T \Delta\boldsymbol{\chi} \mathbf{r})$.
pub const PCS_CARTESIAN_PREFACTOR: f64 = 1.0e4 / (4.0 * PI);

// ============================================================================
// 1. Metalloprotein Coordination Motif Detection & `--metal-ligands` Parsing
// ============================================================================

/// Canonical metalloprotein coordination motif family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetalMotifKind {
    /// User-supplied `--metal-ligands` coordination site.
    ExplicitLigands,
    /// `CISD` / MitoNEET `[2Fe-2S]` `3Cys-1His` motif (`C-X-C-X(5..16)-C-X(1..4)-H`).
    Cisd2Fe2S,
    /// Rieske `[2Fe-2S]` `2Cys-2His` motif (`C-X-H-X(12..45)-C-X2-H`).
    Rieske2Fe2S,
    /// Bacterial/plant ferredoxin `[2Fe-2S]` or `[4Fe-4S]` `4Cys` motif (`C-X(2..4)-C-X2-C-X3-C` / `C-X(4..5)-C-X2-C-X(n)-C`).
    FerredoxinFeS,
    /// Rubredoxin or tetrahedral `Cys4` metal site (`C-X(2..4)-C-X(8..32)-C-X(2..4)-C`).
    RubredoxinOrCys4,
    /// Classical `Cys2His2` finger (`C-X(2..4)-C-X(9..15)-H-X(3..5)-H`).
    Cys2His2Finger,
    /// Retroviral / knuckle `Cys2HisCys` (`C-X2-C-X4-H-X4-C`) motif.
    Cys2HisCysFinger,
}

impl MetalMotifKind {
    /// Whether this motif represents a polynuclear iron-sulfur cluster (`[2Fe-2S]`, `[3Fe-4S]`, `[4Fe-4S]`)
    /// subject to `< 8.0 Å` Fermi-contact / spin-delocalization shifts (`Bertini et al. 2002`).
    #[inline]
    pub fn is_fes_cluster(self) -> bool {
        matches!(
            self,
            Self::Cisd2Fe2S | Self::Rieske2Fe2S | Self::FerredoxinFeS
        )
    }

    /// Default paramagnetic electronic relaxation class when no PDB ion or explicit override is given.
    #[inline]
    pub fn default_paramagnetic_class(self) -> MetalParamagneticClass {
        MetalParamagneticClass::FeSClusterOrHighSpin3d
    }
}

/// Electronic relaxation / susceptibility class of a metalloprotein cofactor (`Bertini, Luchinat & Parigi,
/// Prog. Nucl. Magn. Reson. Spectrosc. 2002, 40, 249–273`), governing the characteristic `15N`-HSQC
/// Solomon–Bloembergen / Curie-spin amide `PRE` bleaching radius $r_{\text{bleach}}$.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetalParamagneticClass {
    /// Antiferromagnetically coupled `[2Fe-2S]`/`[4Fe-4S]`/`[3Fe-4S]` clusters, mononuclear high-spin
    /// rubredoxin `Fe2+`/`Fe3+`, and fast-relaxing high-spin `3d` ions (`Co2+`, `Ni2+`): $r_{\text{bleach}} = 8.5\text{ Å}$.
    #[default]
    FeSClusterOrHighSpin3d,
    /// Slow-electron-relaxation isotropic `PRE` broadeners (`Gd3+`, `Mn2+`, type-2 `Cu2+`): $r_{\text{bleach}} = 13.5\text{ Å}$.
    IsotropicBroadener,
    /// Strongly anisotropic Curie-spin lanthanides (`Tb3+`, `Dy3+`, `Tm3+`, `Ho3+`, `Er3+`): $r_{\text{bleach}} = 9.0\text{ Å}$.
    StrongLanthanide,
    /// Moderate-susceptibility fast-relaxing lanthanides (`Yb3+`, `Ce3+`, `Pr3+`, `Nd3+`, `Sm3+`, `Eu3+`): $r_{\text{bleach}} = 5.5\text{ Å}$.
    ModerateLanthanide,
    /// Closed-shell or low-spin diamagnetic metal cofactors (`Zn2+`, `Cu+`, low-spin `Co3+`, `Mg2+`,
    /// `Ca2+`, `La3+`, `Lu3+`, low-spin `Fe2+`): $r_{\text{bleach}} = 0.0\text{ Å}$ (zero `PRE` bleaching).
    Diamagnetic,
}

impl MetalParamagneticClass {
    /// Characteristic `1H^N` `15N`-HSQC bleaching radius $r_{\text{bleach}}$ in Å (`Bertini et al. 2002`).
    #[inline]
    pub fn default_r_bleach_angstrom(self) -> f64 {
        match self {
            Self::FeSClusterOrHighSpin3d => 8.5,
            Self::IsotropicBroadener => 13.5,
            Self::StrongLanthanide => 9.0,
            Self::ModerateLanthanide => 5.5,
            Self::Diamagnetic => 0.0,
        }
    }
}

/// Detected or user-specified metal coordination cluster.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetalClusterSite {
    pub kind: MetalMotifKind,
    /// Sorted 0-based sequence indices of coordinating residues (`Cys`, `His`, `Asp`, `Glu`, `Met`/`Mse`).
    pub ligand_indices: Vec<usize>,
    /// Optional 3D Cartesian coordinate `(x, y, z)` in Å of the metal center.
    pub metal_coord: Option<Vec3>,
    /// Electronic relaxation / `PRE` bleaching class of this metal site.
    pub metal_class: MetalParamagneticClass,
}

/// Parse an explicit `--metal-ligands` specification such as `"CYS42,CYS45,HIS68,CYS71"`,
/// `"C42,C45,H68,C71"`, or 1-based sequence numbers `"42,45,68,71"` into sorted 0-based
/// indices into `sequence`.
pub fn parse_metal_ligands(
    spec: &str,
    sequence: &[String],
    seq_ids: &[i32],
) -> Result<Vec<usize>, String> {
    let mut indices = BTreeSet::new();
    for raw_tok in spec.split([',', ';', ' ', '\t', '\n']) {
        let tok = raw_tok.trim();
        if tok.is_empty() {
            continue;
        }
        let split_pos = tok
            .find(|c: char| c.is_ascii_digit() || c == '-')
            .ok_or_else(|| format!("invalid metal ligand token '{tok}': missing residue number"))?;
        let (prefix, num_str) = tok.split_at(split_pos);
        let target_sid: i32 = num_str
            .parse()
            .map_err(|_| format!("invalid residue number in metal ligand token '{tok}'"))?;

        let seq_idx = if !seq_ids.is_empty() {
            seq_ids
                .iter()
                .position(|&sid| sid == target_sid)
                .ok_or_else(|| {
                    format!("metal ligand '{tok}' (seq_id {target_sid}) not found in sequence")
                })?
        } else {
            if target_sid < 1 || (target_sid as usize) > sequence.len() {
                return Err(format!(
                    "metal ligand '{tok}' (1-based index {target_sid}) out of range 1..={}",
                    sequence.len()
                ));
            }
            (target_sid - 1) as usize
        };

        if !prefix.is_empty() {
            let expected_three = if prefix.len() == 1 {
                let c = prefix.chars().next().unwrap();
                let mapped = one_to_three(c);
                if mapped == "UNK" {
                    return Err(format!("unknown 1-letter amino acid '{prefix}' in '{tok}'"));
                }
                mapped.to_string()
            } else {
                prefix.to_ascii_uppercase()
            };
            let actual = sequence[seq_idx].trim().to_ascii_uppercase();
            if actual != expected_three {
                return Err(format!(
                    "metal ligand mismatch at seq_id {target_sid}: expected {expected_three}, sequence has {actual}"
                ));
            }
        }
        indices.insert(seq_idx);
    }

    if indices.is_empty() {
        return Err("empty --metal-ligands specification".to_string());
    }
    Ok(indices.into_iter().collect())
}

/// Scan a primary amino-acid sequence (`3`-letter codes) for canonical metalloprotein
/// coordination motifs (`[2Fe-2S]` CISD `C-X-C-X(n)-C-X(1..4)-H`, Rieske, `[2Fe-2S]`/`[4Fe-4S]`
/// ferredoxins, `Cys4`/rubredoxin, `Cys2His2`, and `CCHC` fingers).
pub fn detect_metal_coordination_motifs(sequence: &[String]) -> Vec<MetalClusterSite> {
    let n = sequence.len();
    let up: Vec<String> = sequence
        .iter()
        .map(|s| s.trim().to_ascii_uppercase())
        .collect();
    let mut used = vec![false; n];
    let mut sites = Vec::new();

    fn is_aa(idx: usize, target: &str, used: &[bool], up: &[String]) -> bool {
        idx < up.len() && !used[idx] && up[idx] == target
    }
    fn mark_used(used_arr: &mut [bool], idxs: &[usize]) {
        for &i in idxs {
            used_arr[i] = true;
        }
    }

    // Pass 1: CISD / MitoNEET [2Fe-2S] 3Cys-1His: C - X - C - X(5..16) - C - X(1..4) - H
    let mut i = 0usize;
    while i + 9 < n {
        if is_aa(i, "CYS", &used, &up) && is_aa(i + 2, "CYS", &used, &up) {
            let mut found = None;
            for c3 in (i + 8)..=(i + 19).min(n.saturating_sub(3)) {
                if !is_aa(c3, "CYS", &used, &up) {
                    continue;
                }
                // Prefer canonical C-X3-H (e.g. C-D-G-S-H / C-D-G-T-H) first, then C-X(1..4)-H
                for dh in [4usize, 3, 2, 5] {
                    let h4 = c3 + dh;
                    if is_aa(h4, "HIS", &used, &up) {
                        found = Some([i, i + 2, c3, h4]);
                        break;
                    }
                }
                if found.is_some() {
                    break;
                }
            }
            if let Some(ligs) = found {
                mark_used(&mut used, &ligs);
                sites.push(MetalClusterSite {
                    kind: MetalMotifKind::Cisd2Fe2S,
                    ligand_indices: ligs.to_vec(),
                    metal_coord: None,
                    metal_class: MetalParamagneticClass::FeSClusterOrHighSpin3d,
                });
                i = ligs[3] + 1;
                continue;
            }
        }
        i += 1;
    }

    // Pass 2: Bacterial [4Fe-4S] / [3Fe-4S] / [2Fe-2S] ferredoxin:
    //   (a) C - X(2..4) - C - X2 - C - X3 - C (canonical C-X2-C-X2-C-X3-C + Fe7S8/Fe8S8 C-X4-C-X2-C-X3-C)
    //   (b) Plant/vertebrate/ISC [2Fe-2S]: C - X(4..5) - C - X2 - C - X(18..38) - C
    i = 0;
    while i + 9 < n {
        let mut matched_2a = false;
        for d1 in [3usize, 5, 4] {
            if i + d1 + 7 < n
                && is_aa(i, "CYS", &used, &up)
                && is_aa(i + d1, "CYS", &used, &up)
                && is_aa(i + d1 + 3, "CYS", &used, &up)
                && is_aa(i + d1 + 7, "CYS", &used, &up)
            {
                let ligs = [i, i + d1, i + d1 + 3, i + d1 + 7];
                mark_used(&mut used, &ligs);
                sites.push(MetalClusterSite {
                    kind: MetalMotifKind::FerredoxinFeS,
                    ligand_indices: ligs.to_vec(),
                    metal_coord: None,
                    metal_class: MetalParamagneticClass::FeSClusterOrHighSpin3d,
                });
                i = ligs[3] + 1;
                matched_2a = true;
                break;
            }
        }
        if matched_2a {
            continue;
        }
        for gap1 in [5usize, 6] {
            let c2 = i + gap1;
            let c3 = c2 + 3;
            if is_aa(i, "CYS", &used, &up)
                && is_aa(c2, "CYS", &used, &up)
                && is_aa(c3, "CYS", &used, &up)
            {
                if let Some(c4) = ((c3 + 19)..=(c3 + 39).min(n.saturating_sub(1)))
                    .find(|&idx| is_aa(idx, "CYS", &used, &up))
                {
                    let ligs = [i, c2, c3, c4];
                    mark_used(&mut used, &ligs);
                    sites.push(MetalClusterSite {
                        kind: MetalMotifKind::FerredoxinFeS,
                        ligand_indices: ligs.to_vec(),
                        metal_coord: None,
                        metal_class: MetalParamagneticClass::FeSClusterOrHighSpin3d,
                    });
                    i = c4;
                    break;
                }
            }
        }
        i += 1;
    }

    // Pass 3: Rieske [2Fe-2S] C - X - H - X(12..45) - C - X2 - H
    i = 0;
    while i + 18 < n {
        if is_aa(i, "CYS", &used, &up) && is_aa(i + 2, "HIS", &used, &up) {
            if let Some((c3, h4)) = ((i + 15)..=(i + 48).min(n.saturating_sub(4))).find_map(|c3| {
                (is_aa(c3, "CYS", &used, &up) && is_aa(c3 + 3, "HIS", &used, &up))
                    .then_some((c3, c3 + 3))
            }) {
                let ligs = [i, i + 2, c3, h4];
                mark_used(&mut used, &ligs);
                sites.push(MetalClusterSite {
                    kind: MetalMotifKind::Rieske2Fe2S,
                    ligand_indices: ligs.to_vec(),
                    metal_coord: None,
                    metal_class: MetalParamagneticClass::FeSClusterOrHighSpin3d,
                });
                i = h4 + 1;
                continue;
            }
        }
        i += 1;
    }

    // Pass 4: Cys2His2, CCHC, and Rubredoxin / Cys4 fingers
    i = 0;
    while i + 12 < n {
        if !is_aa(i, "CYS", &used, &up) {
            i += 1;
            continue;
        }
        // CCHC: C - X2 - C - X4 - H - X4 - C
        if is_aa(i + 3, "CYS", &used, &up)
            && is_aa(i + 8, "HIS", &used, &up)
            && is_aa(i + 13, "CYS", &used, &up)
        {
            let ligs = [i, i + 3, i + 8, i + 13];
            mark_used(&mut used, &ligs);
            sites.push(MetalClusterSite {
                kind: MetalMotifKind::Cys2HisCysFinger,
                ligand_indices: ligs.to_vec(),
                metal_coord: None,
                metal_class: MetalParamagneticClass::FeSClusterOrHighSpin3d,
            });
            i += 14;
            continue;
        }

        let mut matched = false;
        for d1 in [3usize, 5, 4] {
            let c2 = i + d1;
            if !is_aa(c2, "CYS", &used, &up) {
                continue;
            }
            // Cys2His2: C - X(2..4) - C - X(9..15) - H - X(3..5) - H
            for h1 in (c2 + 10)..=(c2 + 16).min(n.saturating_sub(4)) {
                if !is_aa(h1, "HIS", &used, &up) {
                    continue;
                }
                for dh in [4usize, 5, 6] {
                    let h2 = h1 + dh;
                    if is_aa(h2, "HIS", &used, &up) {
                        let ligs = [i, c2, h1, h2];
                        mark_used(&mut used, &ligs);
                        sites.push(MetalClusterSite {
                            kind: MetalMotifKind::Cys2His2Finger,
                            ligand_indices: ligs.to_vec(),
                            metal_coord: None,
                            metal_class: MetalParamagneticClass::FeSClusterOrHighSpin3d,
                        });
                        i = h2 + 1;
                        matched = true;
                        break;
                    }
                }
                if matched {
                    break;
                }
            }
            if matched {
                break;
            }

            // Cys4 / Rubredoxin: C - X(2..4) - C - X(8..32) - C - X(2..4) - C
            for c3 in (c2 + 9)..=(c2 + 33).min(n.saturating_sub(4)) {
                if !is_aa(c3, "CYS", &used, &up) {
                    continue;
                }
                for d3 in [3usize, 5, 4] {
                    let c4 = c3 + d3;
                    if is_aa(c4, "CYS", &used, &up) {
                        let ligs = [i, c2, c3, c4];
                        mark_used(&mut used, &ligs);
                        sites.push(MetalClusterSite {
                            kind: MetalMotifKind::RubredoxinOrCys4,
                            ligand_indices: ligs.to_vec(),
                            metal_coord: None,
                            metal_class: MetalParamagneticClass::FeSClusterOrHighSpin3d,
                        });
                        i = c4 + 1;
                        matched = true;
                        break;
                    }
                }
                if matched {
                    break;
                }
            }
            if matched {
                break;
            }
        }
        if !matched {
            i += 1;
        }
    }

    sites.sort_by_key(|s| s.ligand_indices.first().copied().unwrap_or(0));
    sites
}

/// Return sorted 0-based indices of all metal-coordinating `HIS` residues across `clusters`
/// (interoperable with `aromatic::HisTautomerEvidence::metal_coordinated_hint`).
pub fn coordinating_histidine_indices(
    clusters: &[MetalClusterSite],
    sequence: &[String],
) -> Vec<usize> {
    let mut his = BTreeSet::new();
    for cl in clusters {
        for &idx in &cl.ligand_indices {
            if sequence
                .get(idx)
                .is_some_and(|r| r.trim().eq_ignore_ascii_case("HIS"))
            {
                his.insert(idx);
            }
        }
    }
    his.into_iter().collect()
}

// ============================================================================
// 2. Solomon–Bloembergen `PRE` Bleaching Probabilities
// ============================================================================

/// Compute exact 3D Solomon–Bloembergen dipolar `PRE` bleaching probability
/// $$p_{\text{bleach}}(k) = \frac{\sum_m (r_{\text{bleach},m} / r_{k,m})^6}{1 + \sum_m (r_{\text{bleach},m} / r_{k,m})^6} \in [0, 1]$$
/// allowing each metal center $m$ to carry its own characteristic electronic relaxation bleaching
/// radius $r_{\text{bleach},m}$ (`Solomon, Phys. Rev. 1955`; `Bertini, Luchinat & Parigi 2002`).
pub fn compute_bleaching_probabilities_3d_per_site(
    amide_coords: &[Option<Vec3>],
    sites: &[(Vec3, f64)],
) -> Vec<f64> {
    let n = amide_coords.len();
    if sites.is_empty() || sites.iter().all(|&(_, rb)| rb <= 0.0) {
        return vec![0.0; n];
    }
    let mut out = vec![0.0_f64; n];
    for (k, coord_opt) in amide_coords.iter().enumerate() {
        let Some(pos) = coord_opt else {
            continue;
        };
        let mut gamma_ratio = 0.0_f64;
        for &(m, r_bleach) in sites {
            if r_bleach <= 0.0 {
                continue;
            }
            let r = v_dist(*pos, m).max(0.5);
            gamma_ratio += (r_bleach / r).powi(6);
        }
        out[k] = (gamma_ratio / (1.0 + gamma_ratio)).clamp(0.0, 1.0);
    }
    out
}

/// Compute exact 3D Solomon–Bloembergen dipolar `PRE` bleaching probability
/// $$p_{\text{bleach}}(k) = \frac{\sum_m (r_{\text{bleach}} / r_{k,m})^6}{1 + \sum_m (r_{\text{bleach}} / r_{k,m})^6} \in [0, 1]$$
/// (`Solomon, Phys. Rev. 1955`; `Bertini, Luchinat & Parigi 2002`).
pub fn compute_bleaching_probabilities_3d(
    amide_coords: &[Option<Vec3>],
    metal_coords: &[Vec3],
    r_bleach_angstrom: f64,
) -> Vec<f64> {
    if metal_coords.is_empty() || r_bleach_angstrom <= 0.0 {
        return vec![0.0; amide_coords.len()];
    }
    let sites: Vec<(Vec3, f64)> = metal_coords
        .iter()
        .map(|&m| (m, r_bleach_angstrom))
        .collect();
    compute_bleaching_probabilities_3d_per_site(amide_coords, &sites)
}

#[derive(Clone, Debug)]
struct ParsedHetMetalCluster {
    centroid: Vec3,
    metal_class: MetalParamagneticClass,
    is_fes_cluster: bool,
}

#[derive(Clone, Debug)]
struct ParsedParamagneticPdb {
    chain: PdbChain,
    /// Per-residue true sidechain donor coordinates (`SG`, `ND1`/`NE2`, `OD1`/`OD2`, `OE1`/`OE2`, `SD`/`SE`).
    sidechain_donors: Vec<Vec<Vec3>>,
    /// Metal cofactor clusters associated with `chain`.
    het_clusters: Vec<ParsedHetMetalCluster>,
}

#[inline]
fn pdb_ascii_field(bytes: &[u8], start: usize, end: usize) -> Option<&str> {
    if start > end || end > bytes.len() {
        return None;
    }
    let sub = &bytes[start..end];
    if !sub.is_ascii() {
        return None;
    }
    std::str::from_utf8(sub).ok()
}

fn infer_hetatm_metal_or_chalcogen(
    res_name: &str,
    atom_name: &str,
    elem_col: &str,
    charge_col: &str,
) -> Option<(bool, MetalParamagneticClass, bool)> {
    const FES_RESIDUES: &[&str] = &["FES", "SF4", "F3S", "FS4", "FS2", "FS3"];
    const HEME_RESIDUES: &[&str] = &["HEM", "HEC", "HEA", "HEB", "SRM"];
    const KNOWN_METAL_RESIDUES: &[&str] = &[
        "FES", "SF4", "F3S", "FS4", "FS2", "FS3", "FE", "FE2", "CO", "3CO", "CU", "CU1", "CUA",
        "NI", "MN", "MN3", "GD", "TB", "DY", "TM", "HO", "ER", "YB", "CE", "PR", "ND", "SM", "EU",
        "ZN", "MG", "CA", "LA", "LU", "HEM", "HEC", "HEA", "HEB", "SRM",
    ];

    let is_fes_res = FES_RESIDUES.contains(&res_name);
    let elem = if !elem_col.is_empty() {
        elem_col.to_string()
    } else if KNOWN_METAL_RESIDUES.contains(&res_name) {
        let lead: String = atom_name
            .chars()
            .take_while(|c| c.is_ascii_alphabetic())
            .collect();
        if lead.is_empty() {
            return None;
        }
        lead
    } else {
        return None;
    };

    if is_fes_res && matches!(elem.as_str(), "S" | "SE") {
        return Some((false, MetalParamagneticClass::FeSClusterOrHighSpin3d, true));
    }

    // Guard against polypeptide/organic carbon (`CA`) when col 77-78 is blank unless residue is `CA`
    if elem == "CA" && res_name != "CA" && elem_col != "CA" {
        return None;
    }
    // In heme cofactors, only the central metal atom (`FE`, `CO`, `ZN`, `MN`) is the metal center
    if HEME_RESIDUES.contains(&res_name)
        && !matches!(elem.as_str(), "FE" | "CO" | "ZN" | "MN" | "NI" | "CU")
    {
        return None;
    }

    let class = match elem.as_str() {
        "FE" => MetalParamagneticClass::FeSClusterOrHighSpin3d,
        "NI" => MetalParamagneticClass::FeSClusterOrHighSpin3d,
        "CO" => {
            if res_name == "3CO" || charge_col.contains('3') {
                MetalParamagneticClass::Diamagnetic
            } else {
                MetalParamagneticClass::FeSClusterOrHighSpin3d
            }
        }
        "CU" => {
            if res_name == "CU1" || charge_col.contains('1') {
                MetalParamagneticClass::Diamagnetic
            } else {
                MetalParamagneticClass::IsotropicBroadener
            }
        }
        "GD" | "MN" => MetalParamagneticClass::IsotropicBroadener,
        "TB" | "DY" | "TM" | "HO" | "ER" => MetalParamagneticClass::StrongLanthanide,
        "YB" | "CE" | "PR" | "ND" | "SM" | "EU" => MetalParamagneticClass::ModerateLanthanide,
        "ZN" | "MG" | "CA" | "LA" | "LU" => MetalParamagneticClass::Diamagnetic,
        _ => return None,
    };

    Some((true, class, is_fes_res))
}

/// ASCII-safe, hostile-UTF8-proof first-model metalloprotein PDB parser:
/// - Treats full-backbone `HETATM ... MSE` selenomethionine residues as `MET` (`SE` sidechain donor);
/// - Accepts `'B'`-only alternate locations when `' '`/`'A'`/`'1'` is absent, upgrading to primary altloc if seen;
/// - Extracts true coordinating sidechain donor atoms (`SG`, `ND1`/`NE2`, `OD1`/`OD2`, `OE1`/`OE2`, `SD`/`SE`);
/// - Associates `HETATM` metal ions and `[Fe-S]` bridging chalcogens strictly with the target protein chain
///   (`<= 3.8 Å` to a target-chain donor, OR on `target_chain` and not coordinated `<= 3.8 Å` to another chain),
///   clustering polynuclear `[2Fe-2S]`/`[4Fe-4S]` sites via `<= 4.5 Å` metal/chalcogen connectivity.
fn parse_paramagnetic_pdb_first_model(
    pdb_text: &str,
    sequence: &[String],
) -> Option<ParsedParamagneticPdb> {
    #[derive(Clone, Default)]
    struct RawRes {
        chain_id: char,
        seq_id: i32,
        i_code: char,
        res_name: String,
        atoms: BTreeMap<String, (Vec3, bool)>,
    }

    #[derive(Clone, Copy)]
    struct RawHetNode {
        chain_id: char,
        seq_id: i32,
        pos: Vec3,
        is_metal: bool,
        metal_class: MetalParamagneticClass,
        is_fes_cluster: bool,
    }

    const POLYPEPTIDE_RES: &[&str] = &[
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
        "MSE", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HID", "HIE", "HIP", "CYX",
    ];

    let mut chain_order: Vec<char> = Vec::new();
    let mut chain_res_map: BTreeMap<char, Vec<RawRes>> = BTreeMap::new();
    let mut raw_hets: Vec<RawHetNode> = Vec::new();
    let mut seen_any_record = false;

    for line in pdb_text.lines() {
        let bytes = line.as_bytes();
        if bytes.starts_with(b"ENDMDL") && seen_any_record {
            break;
        }
        let is_atom = bytes.starts_with(b"ATOM  ");
        let is_het = bytes.starts_with(b"HETATM");
        if (!is_atom && !is_het) || bytes.len() < 54 {
            continue;
        }

        let Some(atom_raw) = pdb_ascii_field(bytes, 12, 16) else {
            continue;
        };
        let Some(alt_raw) = pdb_ascii_field(bytes, 16, 17) else {
            continue;
        };
        let Some(res_raw) = pdb_ascii_field(bytes, 17, 20) else {
            continue;
        };
        let Some(chain_raw) = pdb_ascii_field(bytes, 21, 22) else {
            continue;
        };
        let Some(seq_raw) = pdb_ascii_field(bytes, 22, 26) else {
            continue;
        };
        let Some(icode_raw) = pdb_ascii_field(bytes, 26, 27) else {
            continue;
        };
        let Some(x_raw) = pdb_ascii_field(bytes, 30, 38) else {
            continue;
        };
        let Some(y_raw) = pdb_ascii_field(bytes, 38, 46) else {
            continue;
        };
        let Some(z_raw) = pdb_ascii_field(bytes, 46, 54) else {
            continue;
        };

        let (Ok(x), Ok(y), Ok(z)) = (
            x_raw.trim().parse::<f64>(),
            y_raw.trim().parse::<f64>(),
            z_raw.trim().parse::<f64>(),
        ) else {
            continue;
        };
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            continue;
        }
        seen_any_record = true;
        let pos: Vec3 = [x, y, z];

        let alt_ch = alt_raw.as_bytes()[0];
        let is_primary_alt = matches!(alt_ch, b' ' | b'A' | b'1');
        let res_up = res_raw.trim().to_ascii_uppercase();
        let atom_up = atom_raw.trim().to_ascii_uppercase();
        let chain_ch = chain_raw.as_bytes()[0] as char;
        let i_code = icode_raw.as_bytes()[0] as char;

        let is_mse_poly = is_het && res_up == "MSE";
        if is_atom || (is_mse_poly && POLYPEPTIDE_RES.contains(&res_up.as_str())) {
            let Ok(seq_id) = seq_raw.trim().parse::<i32>() else {
                continue;
            };
            let norm_res = match res_up.as_str() {
                "MSE" => "MET",
                "HID" | "HIE" | "HIP" => "HIS",
                "CYX" => "CYS",
                other => {
                    if !POLYPEPTIDE_RES.contains(&other) && is_het {
                        continue;
                    }
                    other
                }
            };
            let norm_atom = match atom_up.as_str() {
                "HN" | "H1" => "H".to_string(),
                other => other.to_string(),
            };

            if !chain_order.contains(&chain_ch) {
                chain_order.push(chain_ch);
            }
            let list = chain_res_map.entry(chain_ch).or_default();
            let entry_idx = if let Some(pos_idx) = list
                .iter()
                .position(|r| r.seq_id == seq_id && r.i_code == i_code)
            {
                pos_idx
            } else {
                list.push(RawRes {
                    chain_id: chain_ch,
                    seq_id,
                    i_code,
                    res_name: norm_res.to_string(),
                    atoms: BTreeMap::new(),
                });
                list.len() - 1
            };
            let res_entry = &mut list[entry_idx];
            match res_entry.atoms.get(&norm_atom) {
                None => {
                    res_entry.atoms.insert(norm_atom, (pos, is_primary_alt));
                }
                Some(&(_, was_primary)) if !was_primary && is_primary_alt => {
                    res_entry.atoms.insert(norm_atom, (pos, true));
                }
                _ => {}
            }
            if is_atom {
                continue;
            }
        }

        if is_het && res_up != "MSE" {
            let elem_col: String = pdb_ascii_field(bytes, 76, bytes.len().min(78))
                .unwrap_or("")
                .bytes()
                .filter(|b| b.is_ascii_alphabetic())
                .map(|b| (b as char).to_ascii_uppercase())
                .collect();
            let charge_col: String = pdb_ascii_field(bytes, 78, bytes.len().min(80))
                .unwrap_or("")
                .trim()
                .to_ascii_uppercase();
            let het_seq_id = seq_raw.trim().parse::<i32>().unwrap_or(0);
            if let Some((is_metal, metal_class, is_fes_cluster)) =
                infer_hetatm_metal_or_chalcogen(&res_up, &atom_up, &elem_col, &charge_col)
            {
                raw_hets.push(RawHetNode {
                    chain_id: chain_ch,
                    seq_id: het_seq_id,
                    pos,
                    is_metal,
                    metal_class,
                    is_fes_cluster,
                });
            }
        }
    }

    // Build complete polypeptide chains (residues having N, CA, C) and select target chain
    let extract_sidechain_donors =
        |res_name: &str, atoms: &BTreeMap<String, (Vec3, bool)>| -> Vec<Vec3> {
            let names: &[&str] = match res_name {
                "CYS" => &["SG"],
                "HIS" => &["ND1", "NE2"],
                "ASP" => &["OD1", "OD2"],
                "GLU" => &["OE1", "OE2"],
                "MET" => &["SD", "SE"],
                _ => &[],
            };
            let mut out = Vec::new();
            for &nm in names {
                if let Some(&(c, _)) = atoms.get(nm) {
                    out.push(c);
                }
            }
            out
        };

    struct BuiltChainCandidate {
        chain_id: char,
        residues: Vec<PdbResidue>,
        sc_donors: Vec<Vec<Vec3>>,
        all_chain_donors: Vec<Vec3>,
    }

    let mut built_chains: Vec<BuiltChainCandidate> = Vec::new();
    for ch in chain_order {
        let Some(raw_list) = chain_res_map.get(&ch) else {
            continue;
        };
        let mut residues: Vec<PdbResidue> = Vec::new();
        let mut sc_donors = Vec::new();
        let mut all_chain_donors = Vec::new();
        for r in raw_list {
            let (Some(&(n, _)), Some(&(ca, _)), Some(&(c, _))) =
                (r.atoms.get("N"), r.atoms.get("CA"), r.atoms.get("C"))
            else {
                continue;
            };
            let sc = extract_sidechain_donors(&r.res_name, &r.atoms);
            for &dc in &sc {
                all_chain_donors.push(dc);
            }
            let o_pos = r.atoms.get("O").map(|x| x.0).unwrap_or(c);
            if r.atoms.contains_key("O") {
                all_chain_donors.push(o_pos);
            }
            let (rec_cb, rec_ha, rec_ha3, rec_qb) =
                reconstruct_ca_substituents(&r.res_name, n, ca, c);
            let h_coord = if let Some(&(h_exp, _)) = r.atoms.get("H") {
                Some(h_exp)
            } else if r.res_name != "PRO" {
                residues.last().and_then(|prev| {
                    if v_dist(prev.c, n) <= 2.0 {
                        let u_cn = v_unit([n[0] - prev.c[0], n[1] - prev.c[1], n[2] - prev.c[2]]);
                        let u_can = v_unit([n[0] - ca[0], n[1] - ca[1], n[2] - ca[2]]);
                        let dir =
                            v_unit([u_cn[0] + u_can[0], u_cn[1] + u_can[1], u_cn[2] + u_can[2]]);
                        Some([
                            n[0] + 1.01 * dir[0],
                            n[1] + 1.01 * dir[1],
                            n[2] + 1.01 * dir[2],
                        ])
                    } else {
                        None
                    }
                })
            } else {
                None
            };
            residues.push(PdbResidue {
                seq_id: r.seq_id,
                res_name: r.res_name.clone(),
                chain_id: r.chain_id,
                n,
                h: h_coord,
                ca,
                ha: r
                    .atoms
                    .get("HA")
                    .or_else(|| r.atoms.get("HA2"))
                    .map(|x| x.0)
                    .unwrap_or(rec_ha),
                ha3: r.atoms.get("HA3").map(|x| x.0).or(rec_ha3),
                c,
                o: o_pos,
                cb: r.atoms.get("CB").map(|x| x.0).or(rec_cb),
                qb: rec_qb,
            });
            sc_donors.push(sc);
        }
        if !residues.is_empty() {
            built_chains.push(BuiltChainCandidate {
                chain_id: ch,
                residues,
                sc_donors,
                all_chain_donors,
            });
        }
    }

    if built_chains.is_empty() {
        return None;
    }

    let best_chain_idx = if sequence.is_empty() {
        0
    } else {
        let seq_pairs: Vec<(i32, &str)> = sequence
            .iter()
            .enumerate()
            .map(|(i, s)| ((i + 1) as i32, s.as_str()))
            .collect();
        built_chains
            .iter()
            .enumerate()
            .max_by_key(|(_, cand)| {
                let pdb_pairs: Vec<(i32, &str)> = cand
                    .residues
                    .iter()
                    .map(|r| (r.seq_id, r.res_name.as_str()))
                    .collect();
                crate::dihedrals::align_by_res_names(&seq_pairs, &pdb_pairs).len()
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    };

    let mut other_donors = Vec::new();
    for (idx, cand) in built_chains.iter().enumerate() {
        if idx != best_chain_idx {
            other_donors.extend(cand.all_chain_donors.iter().copied());
        }
    }
    let BuiltChainCandidate {
        chain_id: target_ch,
        residues: target_residues,
        sc_donors: target_sc_donors,
        all_chain_donors: target_all_donors,
    } = built_chains.swap_remove(best_chain_idx);

    // Filter HETATM metal / bridging-S nodes strictly to those coordinated to target_ch
    // (<= 3.8 Å to a target-chain donor, OR on target_ch and not coordinated <= 3.8 Å to another chain)
    let accepted_hets: Vec<RawHetNode> = raw_hets
        .into_iter()
        .filter(|h| {
            let d_target = target_all_donors
                .iter()
                .map(|&d| v_dist(h.pos, d))
                .fold(f64::INFINITY, f64::min);
            if d_target <= 3.8 {
                return true;
            }
            let d_other = other_donors
                .iter()
                .map(|&d| v_dist(h.pos, d))
                .fold(f64::INFINITY, f64::min);
            h.chain_id == target_ch
                && d_other > 3.8
                && (h.metal_class != MetalParamagneticClass::Diamagnetic || d_target <= 4.5)
        })
        .collect();

    // Connected components over accepted metal + bridging-chalcogen nodes (<= 4.5 Å)
    let m = accepted_hets.len();
    let mut visited = vec![false; m];
    let mut het_clusters = Vec::new();
    for start in 0..m {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut queue = vec![start];
        let mut head = 0usize;
        let mut metal_pts = Vec::new();
        let mut classes = Vec::new();
        let mut comp_is_fes = false;
        while head < queue.len() {
            let cur = queue[head];
            head += 1;
            let node = accepted_hets[cur];
            comp_is_fes |= node.is_fes_cluster;
            if node.is_metal {
                metal_pts.push(node.pos);
                classes.push(node.metal_class);
            }
            for next in 0..m {
                if visited[next] {
                    continue;
                }
                let other = accepted_hets[next];
                let d = v_dist(node.pos, other.pos);
                let same_cofactor_res = node.chain_id == other.chain_id
                    && node.seq_id == other.seq_id
                    && node.seq_id != 0;
                let polynuclear_fes_link = (node.is_fes_cluster && other.is_fes_cluster)
                    || (node.metal_class == MetalParamagneticClass::FeSClusterOrHighSpin3d
                        && other.metal_class == MetalParamagneticClass::FeSClusterOrHighSpin3d
                        && d <= 3.3);
                if d <= 4.5 && (same_cofactor_res || polynuclear_fes_link) {
                    visited[next] = true;
                    queue.push(next);
                }
            }
        }
        if metal_pts.is_empty() {
            continue;
        }
        let inv = 1.0 / (metal_pts.len() as f64);
        let sx: f64 = metal_pts.iter().map(|p| p[0]).sum();
        let sy: f64 = metal_pts.iter().map(|p| p[1]).sum();
        let sz: f64 = metal_pts.iter().map(|p| p[2]).sum();
        // Pick strongest-bleaching class in the component (or FeSClusterOrHighSpin3d for Fe-S clusters)
        let cls = if comp_is_fes {
            MetalParamagneticClass::FeSClusterOrHighSpin3d
        } else {
            classes
                .into_iter()
                .max_by(|a, b| {
                    a.default_r_bleach_angstrom()
                        .total_cmp(&b.default_r_bleach_angstrom())
                })
                .unwrap_or_default()
        };
        het_clusters.push(ParsedHetMetalCluster {
            centroid: [sx * inv, sy * inv, sz * inv],
            metal_class: cls,
            is_fes_cluster: comp_is_fes,
        });
    }

    // Order paramagnetic clusters ahead of closed-shell diamagnetic cofactors while preserving PDB order
    het_clusters.sort_by_key(|hc| {
        if hc.metal_class == MetalParamagneticClass::Diamagnetic {
            1u8
        } else {
            0u8
        }
    });

    Some(ParsedParamagneticPdb {
        chain: PdbChain {
            residues: target_residues,
        },
        sidechain_donors: target_sc_donors,
        het_clusters,
    })
}

/// Metal cofactor cluster extracted from a first-model metalloprotein PDB structure.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParsedPdbMetalSite {
    pub centroid: Vec3,
    pub metal_class: MetalParamagneticClass,
    pub is_fes_cluster: bool,
}

/// Parse the target polypeptide chain (supporting `HETATM ... MSE` selenomethionine and `'B'`-only
/// alternate locations) together with target-chain-associated `HETATM` metal/cofactor clusters.
pub fn parse_paramagnetic_pdb_sites(
    pdb_text: &str,
    sequence: &[String],
) -> Option<(PdbChain, Vec<ParsedPdbMetalSite>)> {
    let parsed = parse_paramagnetic_pdb_first_model(pdb_text, sequence)?;
    let sites = parsed
        .het_clusters
        .into_iter()
        .map(|hc| ParsedPdbMetalSite {
            centroid: hc.centroid,
            metal_class: hc.metal_class,
            is_fes_cluster: hc.is_fes_cluster,
        })
        .collect();
    Some((parsed.chain, sites))
}

/// Default 3D single-linkage cutoff (`6.4 Å`) over true coordinating sidechain donor atoms
/// (`CYS SG`, `HIS ND1/NE2`, `ASP OD*`, `GLU OE*`, `MET/MSE SD/SE`):
/// - Keeps tetrahedral `[4Fe-4S]`/`[2Fe-2S]`/`Cys2His2` coordination spheres connected
///   ($d(\text{S}^\gamma, \text{S}^\gamma) \approx 5.5\text{--}6.2\text{ Å}$);
/// - Cleanly separates adjacent clusters in bacterial `2x[4Fe-4S]` ferredoxins (~11.5 Å center-to-center,
///   inter-cluster $d(\text{S}^\gamma, \text{S}^\gamma) \ge 6.7\text{ Å}$ even where backbone/$\text{C}^\beta$
///   atoms approach within `4.1 Å`).
const DONOR_CLUSTER_LINKAGE_CUTOFF_ANGSTROM: f64 = 6.4;

/// Partition explicit or detected coordinating ligand indices into 3D single-linkage donor
/// clusters (`<= cutoff_angstrom`, default `6.4 Å` on true sidechain `SG`/`ND1`/`OD*`/`OE*` donors)
/// and compute one metal centroid per cluster so closely spaced multi-cluster proteins
/// (e.g. `2x[4Fe-4S]` bacterial ferredoxins ~11.5 Å apart) never chain into a single midpoint phantom.
fn partition_ligands_3d_single_linkage(
    ligand_indices: &[usize],
    donor_coords: &[Option<Vec3>],
    cutoff_angstrom: f64,
) -> Vec<(Vec<usize>, Vec3)> {
    let mut donors: Vec<(usize, Vec3)> = Vec::new();
    for &idx in ligand_indices {
        if donors.iter().any(|&(seen, _)| seen == idx) {
            continue;
        }
        if let Some(Some(c)) = donor_coords.get(idx) {
            donors.push((idx, *c));
        }
    }
    if donors.is_empty() {
        return Vec::new();
    }

    let m = donors.len();
    let mut visited = vec![false; m];
    let mut out = Vec::new();
    for start in 0..m {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut queue = vec![start];
        let mut comp_indices = Vec::new();
        let mut sum = [0.0_f64; 3];
        let mut head = 0usize;
        while head < queue.len() {
            let cur = queue[head];
            head += 1;
            let (lig_idx, c_cur) = donors[cur];
            comp_indices.push(lig_idx);
            sum[0] += c_cur[0];
            sum[1] += c_cur[1];
            sum[2] += c_cur[2];
            for next in 0..m {
                if !visited[next] && v_dist(c_cur, donors[next].1) <= cutoff_angstrom {
                    visited[next] = true;
                    queue.push(next);
                }
            }
        }
        comp_indices.sort_unstable();
        let inv = 1.0 / (comp_indices.len() as f64);
        out.push((comp_indices, [sum[0] * inv, sum[1] * inv, sum[2] * inv]));
    }
    out
}

fn align_pdb_amides_and_donors(
    sequence: &[String],
    pdb_chain: &PdbChain,
) -> (Vec<Option<Vec3>>, Vec<Option<Vec3>>) {
    let n = sequence.len();
    let seq_pairs: Vec<(i32, &str)> = sequence
        .iter()
        .enumerate()
        .map(|(i, s)| ((i + 1) as i32, s.as_str()))
        .collect();
    let pdb_pairs: Vec<(i32, &str)> = pdb_chain
        .residues
        .iter()
        .map(|r| (r.seq_id, r.res_name.as_str()))
        .collect();
    let aligned = crate::dihedrals::align_by_res_names(&seq_pairs, &pdb_pairs);

    let mut amide_coords: Vec<Option<Vec3>> = vec![None; n];
    let mut donor_coords: Vec<Option<Vec3>> = vec![None; n];
    for (k, p_idx) in aligned {
        if k < n && p_idx < pdb_chain.residues.len() {
            let pres = &pdb_chain.residues[p_idx];
            amide_coords[k] = Some(pres.h.unwrap_or(pres.n));
            donor_coords[k] = Some(pres.cb.unwrap_or(pres.ca));
        }
    }
    (amide_coords, donor_coords)
}

fn align_parsed_paramagnetic_pdb(
    sequence: &[String],
    parsed: &ParsedParamagneticPdb,
) -> (Vec<Option<Vec3>>, Vec<Option<Vec3>>) {
    let n = sequence.len();
    let seq_pairs: Vec<(i32, &str)> = sequence
        .iter()
        .enumerate()
        .map(|(i, s)| ((i + 1) as i32, s.as_str()))
        .collect();
    let pdb_pairs: Vec<(i32, &str)> = parsed
        .chain
        .residues
        .iter()
        .map(|r| (r.seq_id, r.res_name.as_str()))
        .collect();
    let aligned = crate::dihedrals::align_by_res_names(&seq_pairs, &pdb_pairs);

    let mut amide_coords: Vec<Option<Vec3>> = vec![None; n];
    let mut donor_coords: Vec<Option<Vec3>> = vec![None; n];
    for (k, p_idx) in aligned {
        if k < n && p_idx < parsed.chain.residues.len() {
            let pres = &parsed.chain.residues[p_idx];
            amide_coords[k] = Some(pres.h.unwrap_or(pres.n));
            let sc = parsed
                .sidechain_donors
                .get(p_idx)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let dc = if !sc.is_empty() {
                let inv = 1.0 / (sc.len() as f64);
                let sx: f64 = sc.iter().map(|p| p[0]).sum();
                let sy: f64 = sc.iter().map(|p| p[1]).sum();
                let sz: f64 = sc.iter().map(|p| p[2]).sum();
                [sx * inv, sy * inv, sz * inv]
            } else {
                pres.cb.unwrap_or(pres.ca)
            };
            donor_coords[k] = Some(dc);
        }
    }
    (amide_coords, donor_coords)
}

/// Extract aligned per-residue backbone amide `1H^N` (fallback `15N` for `Pro` / N-terminus)
/// coordinates from a parsed [`PdbChain`] and estimate cluster metal coordinates from 3D
/// single-linkage (`<= 6.4 Å`) ligand donor centroids whenever `cluster.metal_coord` is `None`.
pub fn compute_bleaching_probabilities_from_pdb(
    sequence: &[String],
    pdb_chain: &PdbChain,
    clusters: &[MetalClusterSite],
    r_bleach_angstrom: f64,
) -> Vec<f64> {
    let n = sequence.len();
    if n == 0 || clusters.is_empty() || r_bleach_angstrom <= 0.0 {
        return vec![0.0; n];
    }

    let (amide_coords, donor_coords) = align_pdb_amides_and_donors(sequence, pdb_chain);

    let mut sites: Vec<(Vec3, f64)> = Vec::new();
    for cl in clusters {
        if cl.metal_class == MetalParamagneticClass::Diamagnetic {
            continue;
        }
        if let Some(mc) = cl.metal_coord {
            sites.push((mc, r_bleach_angstrom));
        } else {
            for (_, centroid) in partition_ligands_3d_single_linkage(
                &cl.ligand_indices,
                &donor_coords,
                DONOR_CLUSTER_LINKAGE_CUTOFF_ANGSTROM,
            ) {
                sites.push((centroid, r_bleach_angstrom));
            }
        }
    }

    compute_bleaching_probabilities_3d_per_site(&amide_coords, &sites)
}

/// Compute sequence-only `PRE` bleaching probability $p_{\text{bleach}}(k) \in [0, 1]$ from
/// coordinating ligand motifs (`Cys`/`His`/`Asp`/`Glu`) when no 3D structure is provided.
///
/// Closed-shell diamagnetic sites ([`MetalParamagneticClass::Diamagnetic`], e.g. `Zn2+`, `Cu+`,
/// `Co3+`, `Mg2+`, `Ca2+`, `La3+`, `Lu3+`) produce zero `PRE` bleaching ($p_{\text{bleach}} = 0$).
///
/// Reflects covalent metalloprotein coordination geometry (`Bertini, Luchinat & Parigi 2002`):
/// - Direct coordinating ligands $L_j$ and short intra-chelate linkers ($L_{j+1} - L_j \le 5$),
///   plus the covalently attached C-terminal peptide amide $b + 1$, lie inside the `< 6.5 Å`
///   blind sphere ($p_{\text{bleach}} \in [0.88, 0.96]$).
/// - C-terminal coordination-turn residues $b + 2 \dots b + 4$ donate $\text{NH}\cdots\text{S}$
///   hydrogen bonds to the cluster ($p_{\text{bleach}} \in [0.66, 0.78]$, blind-zone penumbra).
/// - N-terminal incoming scaffold residue $a - 1$ points away from the chelate ($p_{\text{bleach}} = 0.38$).
pub fn compute_bleaching_probabilities_sequence(
    n_residues: usize,
    clusters: &[MetalClusterSite],
) -> Vec<f64> {
    compute_bleaching_probabilities_sequence_with_override(n_residues, clusters, None)
}

fn compute_bleaching_probabilities_sequence_with_override(
    n_residues: usize,
    clusters: &[MetalClusterSite],
    r_bleach_override: Option<f64>,
) -> Vec<f64> {
    if r_bleach_override.is_some_and(|r| r <= 0.0) {
        return vec![0.0; n_residues];
    }
    let mut survival = vec![1.0_f64; n_residues];

    for cl in clusters {
        let eff_r = r_bleach_override.unwrap_or_else(|| cl.metal_class.default_r_bleach_angstrom());
        if eff_r <= 0.0 || cl.ligand_indices.is_empty() {
            continue;
        }
        let mut sorted = cl.ligand_indices.clone();
        sorted.sort_unstable();
        sorted.dedup();

        // Group closely spaced ligands (gap <= 5) into chelate segments [a, b]
        let mut chelates: Vec<(usize, usize)> = Vec::new();
        for &idx in &sorted {
            if idx >= n_residues {
                continue;
            }
            if let Some(last) = chelates.last_mut() {
                if idx <= last.1 + 5 {
                    last.1 = idx;
                    continue;
                }
            }
            chelates.push((idx, idx));
        }

        let mut p_cluster = vec![0.0_f64; n_residues];
        for &(a, b) in &chelates {
            for (k, pc) in p_cluster.iter_mut().enumerate().take(b + 1).skip(a) {
                let is_lig = sorted.binary_search(&k).is_ok();
                *pc = pc.max(if is_lig { 0.96 } else { 0.92 });
            }
            // C-terminal covalent amide + NH...S turn cap
            for (d_plus, p_val) in [(1usize, 0.88), (2, 0.78), (3, 0.72), (4, 0.66), (5, 0.32)] {
                let k = b + d_plus;
                if k < n_residues {
                    p_cluster[k] = p_cluster[k].max(p_val);
                }
            }
            // N-terminal incoming strand approach
            for (d_minus, p_val) in [(1usize, 0.38), (2, 0.16)] {
                if a >= d_minus {
                    let k = a - d_minus;
                    p_cluster[k] = p_cluster[k].max(p_val);
                }
            }
        }

        for (surv, &pc) in survival.iter_mut().zip(&p_cluster) {
            *surv *= 1.0 - pc;
        }
    }

    survival
        .into_iter()
        .map(|s| (1.0 - s).clamp(0.0, 1.0))
        .collect()
}

// ============================================================================
// 3. Paramagnetic Backbone Assignment Wrapper (`assign_paramagnetic_backbone`)
// ============================================================================

/// Underlying solver engine wrapped by [`assign_paramagnetic_backbone`] and
/// [`assign_paramagnetic_from_peaks`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParamagneticSolverMode {
    /// Multi-replica Metropolis–Hastings simulated annealing (`solver::assign_from_matrices_with_locks`).
    Standard,
    /// Stoichiometric evolutionary fragment-crossover + graph self-consistency (`evo_graph`).
    #[default]
    EvoGraph,
}

/// Configuration for paramagnetic metalloprotein backbone resonance assignment.
#[derive(Clone, Debug)]
pub struct ParamagneticAssignConfig {
    /// Solver backend (`Standard` or `EvoGraph`).
    pub mode: ParamagneticSolverMode,
    /// Explicit `--metal-ligands` 0-based indices (if empty and `auto_detect_motifs == true`,
    /// canonical motifs are detected from `sequence`).
    pub explicit_ligand_indices: Vec<usize>,
    /// Automatically detect `[2Fe-2S]`, `[4Fe-4S]`, `Cys2His2`, and `Cys4` coordination motifs.
    pub auto_detect_motifs: bool,
    /// Optional 3D PDB coordinate text for exact $r_{\text{M}}^{-6}$ `PRE` bleaching calculation.
    pub pdb_coordinates: Option<String>,
    /// Optional explicit 3D metal ion coordinates in Å (overrides ligand-centroid estimate when PDB is given).
    pub explicit_metal_coords: Vec<Vec3>,
    /// Optional explicit metal electronic-relaxation class override (`None` = infer per site from
    /// PDB `HETATM` element/oxidation state or motif kind).
    pub metal_class: Option<MetalParamagneticClass>,
    /// Optional explicit Solomon `PRE` amide bleaching radius $r_{\text{bleach}}$ in Å
    /// (`None` = use [`MetalParamagneticClass::default_r_bleach_angstrom`] per site).
    pub r_bleach_angstrom: Option<f64>,
    /// Blind-zone threshold (`0.65`): placements in $p_{\text{bleach}}(k) \ge 0.65$ require multi-nucleus
    /// sequential chain corroboration into the diamagnetic scaffold or abstain cleanly.
    pub blind_zone_threshold: f64,
    /// Core blind-sphere threshold (`0.85`): amides in $p_{\text{bleach}}(k) \ge 0.85$ are broadened
    /// beyond `15N`-HSQC detection and excluded from unary placement unless user-locked.
    pub hard_bleach_cutoff: f64,
    /// Log-survival unary penalty scale applied in the blind-zone penumbra.
    pub bleach_prior_weight: f64,
    /// Enable generic `CA`-less excess/artifact root filtering (independent of metal-motif coordinates).
    pub enable_generic_ca_less_filter: bool,
    /// Enable paramagnetic $u_{\text{para}}$ `PRE` blind-zone unary prior & penumbra corroboration gate.
    pub enable_u_para_bleach_prior: bool,
    /// Underlying `(U, P)` scoring parameters.
    pub score_cfg: ScoreConfig,
    /// Underlying simulated annealing parameters.
    pub anneal_cfg: AnnealConfig,
    /// Deterministic PRNG seed.
    pub seed: u64,
}

impl Default for ParamagneticAssignConfig {
    fn default() -> Self {
        Self {
            mode: ParamagneticSolverMode::EvoGraph,
            explicit_ligand_indices: Vec::new(),
            auto_detect_motifs: true,
            pdb_coordinates: None,
            explicit_metal_coords: Vec::new(),
            metal_class: None,
            r_bleach_angstrom: None,
            blind_zone_threshold: 0.65,
            hard_bleach_cutoff: 0.85,
            bleach_prior_weight: 2.2,
            enable_generic_ca_less_filter: true,
            enable_u_para_bleach_prior: true,
            score_cfg: ScoreConfig::default(),
            anneal_cfg: AnnealConfig::default(),
            seed: 0,
        }
    }
}

/// Diagnostic summary returned alongside paramagnetic backbone assignments.
#[derive(Clone, Debug)]
pub struct ParamagneticAssignReport {
    pub detected_clusters: Vec<MetalClusterSite>,
    pub coordinating_histidines: Vec<usize>,
    pub p_bleach: Vec<f64>,
    pub blind_zone_residues: Vec<usize>,
    pub n_blind_zone_abstentions: usize,
    pub evo_report: Option<EvoGraphReport>,
}

fn resolve_clusters_and_bleaching(
    sequence: &[String],
    cfg: &ParamagneticAssignConfig,
) -> (Vec<MetalClusterSite>, Vec<f64>) {
    let resolve_class = |base: MetalParamagneticClass| -> MetalParamagneticClass {
        cfg.metal_class.unwrap_or(base)
    };
    let resolve_r = |cls: MetalParamagneticClass| -> f64 {
        cfg.r_bleach_angstrom
            .unwrap_or_else(|| cls.default_r_bleach_angstrom())
    };

    let parsed_pdb = cfg
        .pdb_coordinates
        .as_deref()
        .and_then(|txt| parse_paramagnetic_pdb_first_model(txt, sequence));

    let mut clusters: Vec<MetalClusterSite> = Vec::new();

    if let Some(ref pdb) = parsed_pdb {
        let (amides, donor_coords) = align_parsed_paramagnetic_pdb(sequence, pdb);

        if !cfg.explicit_metal_coords.is_empty() {
            let candidate_ligs: Vec<usize> = if !cfg.explicit_ligand_indices.is_empty() {
                cfg.explicit_ligand_indices.clone()
            } else if cfg.auto_detect_motifs {
                detect_metal_coordination_motifs(sequence)
                    .into_iter()
                    .flat_map(|m| m.ligand_indices)
                    .collect()
            } else {
                Vec::new()
            };

            let mut per_metal_ligs: Vec<Vec<usize>> =
                vec![Vec::new(); cfg.explicit_metal_coords.len()];
            for &idx in &candidate_ligs {
                if let Some(Some(dc)) = donor_coords.get(idx) {
                    let best = cfg
                        .explicit_metal_coords
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| v_dist(*dc, **a).total_cmp(&v_dist(*dc, **b)));
                    if let Some((mi, mc)) = best {
                        if !cfg.explicit_ligand_indices.is_empty() || v_dist(*dc, *mc) <= 6.5 {
                            per_metal_ligs[mi].push(idx);
                        }
                    }
                } else if !cfg.explicit_ligand_indices.is_empty() {
                    per_metal_ligs[0].push(idx);
                }
            }

            for (mc, mut ligs) in cfg
                .explicit_metal_coords
                .iter()
                .copied()
                .zip(per_metal_ligs)
            {
                ligs.sort_unstable();
                ligs.dedup();
                let inferred_cls = pdb
                    .het_clusters
                    .iter()
                    .filter(|hc| v_dist(hc.centroid, mc) <= 3.5)
                    .min_by(|a, b| v_dist(a.centroid, mc).total_cmp(&v_dist(b.centroid, mc)))
                    .map(|hc| hc.metal_class)
                    .unwrap_or(MetalParamagneticClass::FeSClusterOrHighSpin3d);
                clusters.push(MetalClusterSite {
                    kind: MetalMotifKind::ExplicitLigands,
                    ligand_indices: ligs,
                    metal_coord: Some(mc),
                    metal_class: resolve_class(inferred_cls),
                });
            }
        } else if !pdb.het_clusters.is_empty() {
            let detected_motifs =
                if cfg.explicit_ligand_indices.is_empty() && cfg.auto_detect_motifs {
                    detect_metal_coordination_motifs(sequence)
                } else {
                    Vec::new()
                };
            let candidate_ligs: Vec<usize> = if !cfg.explicit_ligand_indices.is_empty() {
                cfg.explicit_ligand_indices.clone()
            } else {
                detected_motifs
                    .iter()
                    .flat_map(|m| m.ligand_indices.iter().copied())
                    .collect()
            };

            let mut per_het_ligs: Vec<Vec<usize>> = vec![Vec::new(); pdb.het_clusters.len()];
            let mut unmatched_ligs: Vec<usize> = Vec::new();
            for &idx in &candidate_ligs {
                if let Some(Some(dc)) = donor_coords.get(idx) {
                    let nearest = pdb
                        .het_clusters
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| {
                            v_dist(*dc, a.centroid).total_cmp(&v_dist(*dc, b.centroid))
                        });
                    if let Some((hi, hc)) = nearest {
                        if v_dist(*dc, hc.centroid) <= 5.5 {
                            per_het_ligs[hi].push(idx);
                            continue;
                        }
                    }
                }
                unmatched_ligs.push(idx);
            }

            // If explicit_ligand_indices were given and none matched a PDB HETATM cofactor,
            // honour the explicit ligands directly via 3D donor single-linkage.
            if !cfg.explicit_ligand_indices.is_empty()
                && per_het_ligs.iter().all(Vec::is_empty)
                && !unmatched_ligs.is_empty()
            {
                for (comp_ligs, centroid) in partition_ligands_3d_single_linkage(
                    &unmatched_ligs,
                    &donor_coords,
                    DONOR_CLUSTER_LINKAGE_CUTOFF_ANGSTROM,
                ) {
                    clusters.push(MetalClusterSite {
                        kind: MetalMotifKind::ExplicitLigands,
                        ligand_indices: comp_ligs,
                        metal_coord: Some(centroid),
                        metal_class: resolve_class(MetalParamagneticClass::FeSClusterOrHighSpin3d),
                    });
                }
            } else {
                for (hc, mut ligs) in pdb.het_clusters.iter().zip(per_het_ligs) {
                    ligs.sort_unstable();
                    ligs.dedup();
                    let kind = if !cfg.explicit_ligand_indices.is_empty() {
                        MetalMotifKind::ExplicitLigands
                    } else if hc.is_fes_cluster {
                        detected_motifs
                            .iter()
                            .find(|m| m.ligand_indices.iter().any(|l| ligs.contains(l)))
                            .map(|m| m.kind)
                            .unwrap_or(MetalMotifKind::FerredoxinFeS)
                    } else {
                        MetalMotifKind::ExplicitLigands
                    };
                    clusters.push(MetalClusterSite {
                        kind,
                        ligand_indices: ligs,
                        metal_coord: Some(hc.centroid),
                        metal_class: resolve_class(hc.metal_class),
                    });
                }
                if cfg.explicit_ligand_indices.is_empty() && !unmatched_ligs.is_empty() {
                    for (comp_ligs, centroid) in partition_ligands_3d_single_linkage(
                        &unmatched_ligs,
                        &donor_coords,
                        DONOR_CLUSTER_LINKAGE_CUTOFF_ANGSTROM,
                    ) {
                        if comp_ligs.len() >= 3 {
                            clusters.push(MetalClusterSite {
                                kind: MetalMotifKind::FerredoxinFeS,
                                ligand_indices: comp_ligs,
                                metal_coord: Some(centroid),
                                metal_class: resolve_class(
                                    MetalParamagneticClass::FeSClusterOrHighSpin3d,
                                ),
                            });
                        }
                    }
                }
            }
        } else if !cfg.explicit_ligand_indices.is_empty() {
            let comps = partition_ligands_3d_single_linkage(
                &cfg.explicit_ligand_indices,
                &donor_coords,
                DONOR_CLUSTER_LINKAGE_CUTOFF_ANGSTROM,
            );
            if comps.is_empty() {
                clusters.push(MetalClusterSite {
                    kind: MetalMotifKind::ExplicitLigands,
                    ligand_indices: cfg.explicit_ligand_indices.clone(),
                    metal_coord: None,
                    metal_class: resolve_class(MetalParamagneticClass::FeSClusterOrHighSpin3d),
                });
            } else {
                for (comp_ligs, centroid) in comps {
                    clusters.push(MetalClusterSite {
                        kind: MetalMotifKind::ExplicitLigands,
                        ligand_indices: comp_ligs,
                        metal_coord: Some(centroid),
                        metal_class: resolve_class(MetalParamagneticClass::FeSClusterOrHighSpin3d),
                    });
                }
            }
        } else if cfg.auto_detect_motifs {
            let raw_motifs = detect_metal_coordination_motifs(sequence);
            let mut pooled_with_coords = Vec::new();
            let mut unmodeled_motifs = Vec::new();
            for mut m in raw_motifs {
                m.metal_class = resolve_class(m.metal_class);
                if m.ligand_indices
                    .iter()
                    .any(|&idx| donor_coords.get(idx).and_then(|x| *x).is_some())
                {
                    pooled_with_coords.extend(m.ligand_indices.iter().copied());
                } else {
                    unmodeled_motifs.push(m);
                }
            }
            for (comp_ligs, centroid) in partition_ligands_3d_single_linkage(
                &pooled_with_coords,
                &donor_coords,
                DONOR_CLUSTER_LINKAGE_CUTOFF_ANGSTROM,
            ) {
                let has_his = comp_ligs.iter().any(|&idx| {
                    sequence
                        .get(idx)
                        .is_some_and(|r| r.eq_ignore_ascii_case("HIS"))
                });
                let kind = if has_his {
                    MetalMotifKind::Cisd2Fe2S
                } else {
                    MetalMotifKind::FerredoxinFeS
                };
                clusters.push(MetalClusterSite {
                    kind,
                    ligand_indices: comp_ligs,
                    metal_coord: Some(centroid),
                    metal_class: resolve_class(kind.default_paramagnetic_class()),
                });
            }
            clusters.extend(unmodeled_motifs);
        }

        let mut sites_3d: Vec<(Vec3, f64)> = Vec::new();
        let mut unmodeled_seq_clusters: Vec<MetalClusterSite> = Vec::new();
        for cl in &clusters {
            let rb = resolve_r(cl.metal_class);
            if let Some(mc) = cl.metal_coord {
                sites_3d.push((mc, rb));
            } else {
                let comps = partition_ligands_3d_single_linkage(
                    &cl.ligand_indices,
                    &donor_coords,
                    DONOR_CLUSTER_LINKAGE_CUTOFF_ANGSTROM,
                );
                if comps.is_empty() {
                    unmodeled_seq_clusters.push(cl.clone());
                } else {
                    for (_, cent) in comps {
                        sites_3d.push((cent, rb));
                    }
                }
            }
        }

        let mut p_bleach = compute_bleaching_probabilities_3d_per_site(&amides, &sites_3d);
        if !unmodeled_seq_clusters.is_empty() {
            let p_seq = compute_bleaching_probabilities_sequence_with_override(
                sequence.len(),
                &unmodeled_seq_clusters,
                cfg.r_bleach_angstrom,
            );
            for (pb, ps) in p_bleach.iter_mut().zip(p_seq) {
                *pb = 1.0 - (1.0 - *pb) * (1.0 - ps);
            }
        }
        return (clusters, p_bleach);
    }

    // Sequence-only path (no PDB structure provided or unparseable PDB)
    if !cfg.explicit_ligand_indices.is_empty() {
        if !cfg.explicit_metal_coords.is_empty() {
            for (mi, &mc) in cfg.explicit_metal_coords.iter().enumerate() {
                clusters.push(MetalClusterSite {
                    kind: MetalMotifKind::ExplicitLigands,
                    ligand_indices: if mi == 0 {
                        cfg.explicit_ligand_indices.clone()
                    } else {
                        Vec::new()
                    },
                    metal_coord: Some(mc),
                    metal_class: resolve_class(MetalParamagneticClass::FeSClusterOrHighSpin3d),
                });
            }
        } else {
            clusters.push(MetalClusterSite {
                kind: MetalMotifKind::ExplicitLigands,
                ligand_indices: cfg.explicit_ligand_indices.clone(),
                metal_coord: None,
                metal_class: resolve_class(MetalParamagneticClass::FeSClusterOrHighSpin3d),
            });
        }
    } else if cfg.auto_detect_motifs {
        clusters = detect_metal_coordination_motifs(sequence);
        for cl in &mut clusters {
            cl.metal_class = resolve_class(cl.metal_class);
        }
        for (idx, mc) in cfg.explicit_metal_coords.iter().copied().enumerate() {
            if let Some(cl) = clusters.get_mut(idx) {
                cl.metal_coord = Some(mc);
            } else {
                clusters.push(MetalClusterSite {
                    kind: MetalMotifKind::ExplicitLigands,
                    ligand_indices: Vec::new(),
                    metal_coord: Some(mc),
                    metal_class: resolve_class(MetalParamagneticClass::FeSClusterOrHighSpin3d),
                });
            }
        }
    } else {
        for &mc in &cfg.explicit_metal_coords {
            clusters.push(MetalClusterSite {
                kind: MetalMotifKind::ExplicitLigands,
                ligand_indices: Vec::new(),
                metal_coord: Some(mc),
                metal_class: resolve_class(MetalParamagneticClass::FeSClusterOrHighSpin3d),
            });
        }
    }

    let p_bleach = compute_bleaching_probabilities_sequence_with_override(
        sequence.len(),
        &clusters,
        cfg.r_bleach_angstrom,
    );
    (clusters, p_bleach)
}

fn count_positive_carbon_slots(ss: &SpinSystem) -> usize {
    ss.cands
        .values()
        .filter(|list| list.iter().any(|&(_, sup)| sup > 0))
        .count()
}

fn has_positive_ca(ss: &SpinSystem) -> bool {
    ss.cands
        .get(&("CA".to_string(), 0))
        .is_some_and(|l| l.iter().any(|&(_, sup)| sup > 0))
        || ss
            .cands
            .get(&("CA".to_string(), -1))
            .is_some_and(|l| l.iter().any(|&(_, sup)| sup > 0))
}

fn apply_paramagnetic_prior_to_unary(
    u_base: &Matrix2D,
    p_mat: &Matrix2D,
    systems: &[SpinSystem],
    sequence: &[String],
    p_bleach: &[f64],
    cfg: &ParamagneticAssignConfig,
    locks: &[(usize, usize)],
) -> Matrix2D {
    let mut u = u_base.clone();
    let locked_set: BTreeSet<(usize, usize)> = locks.iter().copied().collect();
    let n_slots: Vec<usize> = systems.iter().map(count_positive_carbon_slots).collect();
    let obs_ca: Vec<bool> = systems.iter().map(has_positive_ca).collect();

    let multi_slot_count = n_slots.iter().filter(|&&ns| ns >= 3).count();
    let total_bleached: f64 = sequence
        .iter()
        .enumerate()
        .skip(1)
        .filter(|&(_, r)| r != "PRO")
        .map(|(k, _)| p_bleach.get(k).copied().unwrap_or(0.0))
        .sum();
    // Generic CA-less artifact/excess-root filter activates on triple-resonance datasets
    // (`multi_slot_count >= 6`) whenever `enable_generic_ca_less_filter` is enabled, even when
    // no metal motif or `u_para` bleach prior is supplied (`total_bleached == 0.0`), provided
    // missing HSQC roots leave unoccupied non-Pro slots (`systems.len() + 2 <= non_pro` or `total_bleached >= 1.5`).
    let non_pro_assignable = sequence
        .iter()
        .skip(1)
        .filter(|r| r.as_str() != "PRO")
        .count();
    let has_missing_amide_slack = systems.len() > non_pro_assignable
        || systems.len() + 3 <= non_pro_assignable
        || total_bleached >= 1.5;
    let apply_ca_less_filter = cfg.enable_generic_ca_less_filter
        && cfg.score_cfg.forbid_ca_less_excess_roots
        && multi_slot_count >= 6
        && has_missing_amide_slack;

    let max_sup: Vec<usize> = systems
        .iter()
        .map(|ss| {
            ss.cands
                .values()
                .flat_map(|l| l.iter().map(|&(_, sup)| sup))
                .max()
                .unwrap_or(0)
        })
        .collect();
    let best_p: Vec<f64> = if apply_ca_less_filter {
        (0..u.rows)
            .map(|s| {
                (0..u.rows)
                    .filter(|&t| t != s)
                    .map(|t| p_mat.get(s, t).max(p_mat.get(t, s)))
                    .fold(0.0_f64, f64::max)
            })
            .collect()
    } else {
        vec![0.0; u.rows]
    };

    for (s, &ns) in n_slots.iter().enumerate().take(u.rows) {
        for k in 0..u.cols {
            if locked_set.contains(&(s, k)) {
                continue;
            }
            let mut sc = u.get(s, k);
            if sc <= FORBID * 0.5 {
                continue;
            }
            if cfg.enable_u_para_bleach_prior {
                let pb = p_bleach.get(k).copied().unwrap_or(0.0);
                if pb >= cfg.hard_bleach_cutoff {
                    u.set(s, k, FORBID);
                    continue;
                } else if pb >= cfg.blind_zone_threshold {
                    if ns < 2 {
                        u.set(s, k, FORBID);
                        continue;
                    }
                    let surv = (1.0 - pb).max(0.05);
                    sc += cfg.bleach_prior_weight * surv.ln();
                }
            }

            if apply_ca_less_filter {
                let res_k = sequence.get(k).map(String::as_str).unwrap_or("");
                let prev_pro = k > 0 && sequence.get(k - 1).is_some_and(|r| r == "PRO");
                if !obs_ca[s] && res_k != "GLY" && !prev_pro {
                    u.set(s, k, FORBID);
                    continue;
                }
                if ns <= 1 && max_sup[s] <= 1 {
                    sc -= 4.5;
                } else if best_p[s] < 2.2 {
                    sc -= 2.4;
                }
            }
            u.set(s, k, sc);
        }
    }
    u
}

/// Verify whether a tentative placement `s` at blind-zone penumbra residue `k`
/// ($0.65 \le p_{\text{bleach}}(k) < 0.85$) is genuinely corroborated by:
/// 1. Positive intrinsic typing `U_base(s, k) >= 2.5` and `>= 2` observed carbon slots;
/// 2. An unbroken sequential walk of `>= 2` links (`>= 3` residues, `P >= 2.8` at each step) reaching
///    at least one diamagnetic scaffold residue ($p_{\text{bleach}} < 0.65$);
/// 3. When the dataset includes `CB(i-1)` strips (e.g. `CBCACONH`/`HNCACB`) and `sequence[k] != "GLY"`,
///    survival of `CB(k)` as `CB(-1)` on `s_{k+1}` (diagnosing absence of `PRE` aliphatic bleaching).
#[allow(clippy::too_many_arguments)]
fn penumbra_placement_is_corroborated(
    s: usize,
    k: usize,
    at: &[i32],
    systems: &[SpinSystem],
    sequence: &[String],
    eff_priors: &Priors,
    u_base: &Matrix2D,
    p_mat: &Matrix2D,
    p_bleach: &[f64],
    cfg: &ParamagneticAssignConfig,
    dataset_has_cb_im1: bool,
) -> bool {
    if p_bleach[k] >= cfg.hard_bleach_cutoff {
        return false;
    }
    if u_base.get(s, k) < 2.5 || count_positive_carbon_slots(&systems[s]) < 2 {
        return false;
    }

    // Check CB(i-1) survival on k+1 when dataset provides CB(-1) and sequence[k] != GLY
    if dataset_has_cb_im1 && !sequence[k].eq_ignore_ascii_case("GLY") {
        if k + 1 >= sequence.len() || at[k + 1] < 0 {
            return false;
        }
        let s_next = at[k + 1] as usize;
        let has_matching_cb = systems[s_next]
            .cands
            .get(&("CB".to_string(), -1))
            .is_some_and(|list| {
                list.iter().any(|&(v, sup)| {
                    sup > 0
                        && eff_priors
                            .get(&sequence[k], "CB")
                            .is_some_and(|(mu, sd)| (v - mu).abs() <= 1.6 * sd.max(1.0))
                })
            });
        if !has_matching_cb {
            return false;
        }
    }

    // Walk C-terminally: k -> k+1 -> k+2 ... reaching p_bleach < blind_zone_threshold
    let mut c_steps = 0usize;
    let mut c_reaches_diamag = false;
    let mut cur_k = k;
    while cur_k + 1 < sequence.len() && at[cur_k] >= 0 && at[cur_k + 1] >= 0 {
        let sa = at[cur_k] as usize;
        let sb = at[cur_k + 1] as usize;
        if p_mat.get(sa, sb) < 2.8 {
            break;
        }
        c_steps += 1;
        cur_k += 1;
        if p_bleach[cur_k] < cfg.blind_zone_threshold {
            c_reaches_diamag = true;
        }
        if c_steps >= 4 && c_reaches_diamag {
            break;
        }
    }
    if c_steps >= 2 && c_reaches_diamag {
        return true;
    }

    // Or walk N-terminally: ... k-2 -> k-1 -> k reaching p_bleach < blind_zone_threshold
    let mut n_steps = 0usize;
    let mut n_reaches_diamag = false;
    cur_k = k;
    while cur_k > 1 && at[cur_k - 1] >= 0 && at[cur_k] >= 0 {
        let sa = at[cur_k - 1] as usize;
        let sb = at[cur_k] as usize;
        if p_mat.get(sa, sb) < 2.8 {
            break;
        }
        n_steps += 1;
        cur_k -= 1;
        if p_bleach[cur_k] < cfg.blind_zone_threshold {
            n_reaches_diamag = true;
        }
        if n_steps >= 4 && n_reaches_diamag {
            break;
        }
    }
    n_steps >= 2 && n_reaches_diamag
}

/// Post-filter spurious placements in `p_bleach >= blind_zone_threshold`, re-anneal any freed
/// spin systems onto open diamagnetic sequence slots, and calibrate penumbra confidences.
#[allow(clippy::too_many_arguments)]
fn filter_blind_zones_and_reoptimize(
    mut res: AssignResult,
    systems: &[SpinSystem],
    sequence: &[String],
    eff_priors: &Priors,
    u_base: &Matrix2D,
    u_para: &Matrix2D,
    p_mat: &Matrix2D,
    p_bleach: &[f64],
    cfg: &ParamagneticAssignConfig,
    user_locks: &[(usize, usize)],
    preserve_diamag_anchors: bool,
) -> (AssignResult, usize) {
    if !cfg.enable_u_para_bleach_prior && !cfg.enable_generic_ca_less_filter {
        return (res, 0);
    }

    let k_len = sequence.len();
    let s_len = systems.len();
    let locked_set: BTreeSet<(usize, usize)> = user_locks.iter().copied().collect();
    let dataset_has_cb_im1 = systems
        .iter()
        .filter(|ss| {
            ss.cands
                .get(&("CB".to_string(), -1))
                .is_some_and(|l| l.iter().any(|&(_, sup)| sup > 0))
        })
        .count()
        >= 5;

    let mut at = vec![-1i32; k_len];
    for (s, &pos) in res.position.iter().enumerate() {
        if pos >= 0 && (pos as usize) < k_len {
            at[pos as usize] = s as i32;
        }
    }

    let mut rejected_pairs: Vec<(usize, usize)> = Vec::new();
    for s in 0..s_len {
        let pos = res.position[s];
        if pos < 0 {
            continue;
        }
        let k = pos as usize;
        if locked_set.contains(&(s, k)) {
            continue;
        }
        // Evict forbidden CA-less excess/artifact placements (important in EvoGraph mode where
        // inner fragment evolution runs on `u_base`)
        if cfg.enable_generic_ca_less_filter
            && u_para.get(s, k) <= FORBID * 0.5
            && u_base.get(s, k) > FORBID * 0.5
            && (!cfg.enable_u_para_bleach_prior || p_bleach[k] < cfg.blind_zone_threshold)
        {
            rejected_pairs.push((s, k));
            continue;
        }
        if !cfg.enable_u_para_bleach_prior || p_bleach[k] < cfg.blind_zone_threshold {
            continue;
        }
        if !penumbra_placement_is_corroborated(
            s,
            k,
            &at,
            systems,
            sequence,
            eff_priors,
            u_base,
            p_mat,
            p_bleach,
            cfg,
            dataset_has_cb_im1,
        ) {
            rejected_pairs.push((s, k));
        }
    }

    let n_abstained = rejected_pairs
        .iter()
        .filter(|&&(_, k)| {
            cfg.enable_u_para_bleach_prior && p_bleach[k] >= cfg.blind_zone_threshold
        })
        .count();
    if !rejected_pairs.is_empty() {
        let _ = preserve_diamag_anchors;
        // 1. Evict all uncorroborated placements inside the PRE blind zone to the null state (-1)
        //    without crowding noise/sidechain artifact roots into settled diamagnetic chains.
        let mut evicted_backbone_roots = Vec::new();
        for &(s_bad, k_bad) in &rejected_pairs {
            res.position[s_bad] = -1;
            res.confidence[s_bad] = 0.0;
            if at[k_bad] == (s_bad as i32) {
                at[k_bad] = -1;
            }
            if count_positive_carbon_slots(&systems[s_bad]) >= 2 {
                evicted_backbone_roots.push(s_bad);
            }
        }

        // 2. Rescue any genuinely multi-slot (>= 2 carbon slots) backbone spin system onto an
        //    unoccupied diamagnetic slot when corroborated by sequential P-links or strong typing.
        for s_free in evicted_backbone_roots {
            let mut best_k: Option<(usize, f64)> = None;
            for k_open in 0..k_len {
                if at[k_open] != -1
                    || (cfg.enable_u_para_bleach_prior
                        && p_bleach[k_open] >= cfg.blind_zone_threshold)
                {
                    continue;
                }
                let u_val = u_para.get(s_free, k_open);
                if u_val <= 0.5 {
                    continue;
                }
                let mut link_gain = 0.0_f64;
                if k_open > 0 && at[k_open - 1] >= 0 {
                    link_gain += p_mat.get(at[k_open - 1] as usize, s_free).max(0.0);
                }
                if k_open + 1 < k_len && at[k_open + 1] >= 0 {
                    link_gain += p_mat.get(s_free, at[k_open + 1] as usize).max(0.0);
                }
                if link_gain >= 2.2 || u_val >= 3.0 {
                    let score_tot = u_val + link_gain;
                    if best_k.is_none_or(|(_, b_sc)| score_tot > b_sc) {
                        best_k = Some((k_open, score_tot));
                    }
                }
            }
            if let Some((k_win, sc)) = best_k {
                res.position[s_free] = k_win as i32;
                res.confidence[s_free] = (0.45 + 0.04 * sc).clamp(0.50, 0.76);
                at[k_win] = s_free as i32;
            }
        }
    }

    // Calibrate confidences in the paramagnetic penumbra so partially broadened shells reflect uncertainty
    if cfg.enable_u_para_bleach_prior {
        for s in 0..s_len {
            let pos = res.position[s];
            if pos < 0 {
                res.confidence[s] = 0.0;
                continue;
            }
            let k = pos as usize;
            if locked_set.contains(&(s, k)) {
                continue;
            }
            let pb = p_bleach[k];
            if pb >= cfg.blind_zone_threshold {
                res.confidence[s] *= (1.0 - 0.35 * pb).clamp(0.50, 0.85);
            } else if pb >= 0.35 {
                res.confidence[s] *= (1.0 - 0.15 * (pb - 0.35)).clamp(0.85, 1.0);
            }
        }
    }

    (res, n_abstained)
}

/// Assign pre-assembled [`SpinSystem`]s under a paramagnetic `PRE` blind-zone prior wrapping
/// `nmr-core`'s `(U, P)` scoring and simulated annealing pipeline.
pub fn assign_paramagnetic_backbone(
    systems: &[SpinSystem],
    sequence: &[String],
    priors: &Priors,
    cfg: &ParamagneticAssignConfig,
) -> (AssignResult, ParamagneticAssignReport) {
    let (clusters, p_bleach) = resolve_clusters_and_bleaching(sequence, cfg);
    let coord_his = coordinating_histidine_indices(&clusters, sequence);
    let blind_zone_residues: Vec<usize> = (0..sequence.len())
        .filter(|&k| p_bleach[k] >= cfg.blind_zone_threshold)
        .collect();

    let mut eff_scfg = cfg.score_cfg.clone();
    if !cfg.enable_generic_ca_less_filter {
        eff_scfg.forbid_ca_less_excess_roots = false;
    }
    let (eff_priors, _) = prepare_scoring_priors(systems, sequence, priors, &eff_scfg);
    let mut raw_scfg = eff_scfg;
    raw_scfg.auto_reference = false;
    let u_base = unary_matrix(systems, sequence, &eff_priors, &raw_scfg);
    let p_mat = pair_matrix(systems, &raw_scfg);
    let locks =
        resolve_locked_assignments(systems.len(), sequence, &cfg.score_cfg.locked_assignments);
    let u_para = apply_paramagnetic_prior_to_unary(
        &u_base, &p_mat, systems, sequence, &p_bleach, cfg, &locks,
    );

    let acfg = AnnealConfig {
        seed: cfg.seed,
        ..cfg.anneal_cfg.clone()
    };
    let initial_res = assign_from_matrices_with_locks(&u_para, &p_mat, &acfg, &locks);
    let (final_res, n_abstained) = filter_blind_zones_and_reoptimize(
        initial_res,
        systems,
        sequence,
        &eff_priors,
        &u_base,
        &u_para,
        &p_mat,
        &p_bleach,
        cfg,
        &locks,
        false,
    );

    let report = ParamagneticAssignReport {
        detected_clusters: clusters,
        coordinating_histidines: coord_his,
        p_bleach,
        blind_zone_residues,
        n_blind_zone_abstentions: n_abstained,
        evo_report: None,
    };
    (final_res, report)
}

/// End-to-end paramagnetic metalloprotein backbone assignment from raw picked [`Peak`]s,
/// supporting both [`ParamagneticSolverMode::EvoGraph`] and [`ParamagneticSolverMode::Standard`].
pub fn assign_paramagnetic_from_peaks(
    peaks: &[Peak],
    sequence: &[String],
    priors: &Priors,
    cfg: &ParamagneticAssignConfig,
) -> (Vec<SpinSystem>, AssignResult, ParamagneticAssignReport) {
    let (clusters, p_bleach) = resolve_clusters_and_bleaching(sequence, cfg);
    let coord_his = coordinating_histidine_indices(&clusters, sequence);
    let blind_zone_residues: Vec<usize> = (0..sequence.len())
        .filter(|&k| p_bleach[k] >= cfg.blind_zone_threshold)
        .collect();

    match cfg.mode {
        ParamagneticSolverMode::Standard => {
            let systems = crate::spinsystems::build_spin_systems(
                peaks,
                &crate::spinsystems::Tolerances::default(),
                3,
                false,
            )
            .unwrap_or_default();
            let (res, mut rep) = assign_paramagnetic_backbone(&systems, sequence, priors, cfg);
            rep.detected_clusters = clusters;
            rep.coordinating_histidines = coord_his;
            (systems, res, rep)
        }
        ParamagneticSolverMode::EvoGraph => {
            let mut evo_scfg = cfg.score_cfg.clone();
            if !cfg.enable_generic_ca_less_filter {
                evo_scfg.forbid_ca_less_excess_roots = false;
            }
            let evo_cfg = EvoGraphConfig {
                score_cfg: evo_scfg.clone(),
                seed: cfg.seed,
                ..Default::default()
            };

            let (systems, evo_res, evo_rep) =
                assign_evo_graph_from_peaks(peaks, sequence, priors, &evo_cfg);

            let eff_priors = evo_rep.shifted_priors(priors);
            let mut raw_scfg = evo_scfg;
            raw_scfg.auto_reference = false;
            let u_base = unary_matrix(&systems, sequence, &eff_priors, &raw_scfg);
            let p_mat = pair_matrix(&systems, &raw_scfg);
            let locks = resolve_locked_assignments(
                systems.len(),
                sequence,
                &cfg.score_cfg.locked_assignments,
            );
            let u_para = apply_paramagnetic_prior_to_unary(
                &u_base, &p_mat, &systems, sequence, &p_bleach, cfg, &locks,
            );

            let (final_res, n_abstained) = filter_blind_zones_and_reoptimize(
                evo_res,
                &systems,
                sequence,
                &eff_priors,
                &u_base,
                &u_para,
                &p_mat,
                &p_bleach,
                cfg,
                &locks,
                true,
            );

            let report = ParamagneticAssignReport {
                detected_clusters: clusters,
                coordinating_histidines: coord_his,
                p_bleach,
                blind_zone_residues,
                n_blind_zone_abstentions: n_abstained,
                evo_report: Some(evo_rep),
            };
            (systems, final_res, report)
        }
    }
}

// ============================================================================
// 4. 5-Parameter Traceless Symmetric Δχ-Tensor SVD & Metal-Site Refinement
// ============================================================================

/// Experimental pseudocontact shift (`PCS`) observation $\Delta\delta^{\text{PCS}} = \delta^{\text{para}} - \delta^{\text{dia}}$
/// paired with 3D nuclear Cartesian coordinates in Å.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PcsObservation {
    pub seq_id: i32,
    pub coord_angstrom: Vec3,
    pub pcs_ppm: f64,
    /// Measurement uncertainty $\sigma_i > 0$ in ppm (defaults to `0.05` ppm).
    pub uncertainty_ppm: f64,
    /// Optional 3D RMS nuclear coordinate uncertainty $\sigma_{\text{coord}} \ge 0$ in Å (`0.0` when exact),
    /// propagated analytically via $\nabla_{\mathbf{r}} \Delta\delta^{\text{PCS}}$ (`Schmitz et al. 2008`).
    pub coord_uncertainty_angstrom: f64,
}

impl PcsObservation {
    pub fn new(seq_id: i32, coord_angstrom: Vec3, pcs_ppm: f64) -> Self {
        Self {
            seq_id,
            coord_angstrom,
            pcs_ppm,
            uncertainty_ppm: 0.05,
            coord_uncertainty_angstrom: 0.0,
        }
    }

    pub fn with_uncertainty(mut self, sigma_ppm: f64) -> Self {
        self.uncertainty_ppm = sigma_ppm;
        self
    }

    pub fn with_coord_uncertainty(mut self, sigma_3d_angstrom: f64) -> Self {
        self.coord_uncertainty_angstrom = sigma_3d_angstrom;
        self
    }
}

/// Fitted 5-parameter traceless symmetric magnetic susceptibility anisotropy tensor $\Delta\boldsymbol{\chi}$
/// in units of $10^{-32}\text{ m}^3$ (`Schmitz et al., J. Biomol. NMR 2008, 41, 179–189`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeltaChiTensor {
    /// Metal ion position $(x_M, y_M, z_M)$ in Å in the molecular frame.
    pub metal_pos_angstrom: Vec3,
    /// Five independent molecular-frame components $[\Delta\chi_{xx}, \Delta\chi_{yy}, \Delta\chi_{xy}, \Delta\chi_{xz}, \Delta\chi_{yz}]$
    /// in $10^{-32}\text{ m}^3$ (with $\Delta\chi_{zz} = -\Delta\chi_{xx} - \Delta\chi_{yy}$).
    pub components_5: [f64; 5],
    /// Full $3 \times 3$ traceless symmetric tensor $\Delta\boldsymbol{\chi}_{\text{mol}}$ ($10^{-32}\text{ m}^3$).
    pub matrix_mol: [[f64; 3]; 3],
    /// Principal-axis eigenvalues $[\Delta\chi'_{xx}, \Delta\chi'_{yy}, \Delta\chi'_{zz}]$ ordered by the
    /// Schmitz/Otting UTS convention $|\Delta\chi'_{zz}| \ge |\Delta\chi'_{yy}| \ge |\Delta\chi'_{xx}|$
    /// (`0 <= Δχ_rh / Δχ_ax <= 2/3`).
    pub eigenvalues_paf: [f64; 3],
    /// Proper right-handed rotation matrix $R \in SO(3)$ mapping molecular vectors to the PAF ($\mathbf{r}' = R(\mathbf{r} - \mathbf{m})$).
    pub rotation_mol_to_paf: [[f64; 3]; 3],
    /// Axial anisotropy $\Delta\chi_{\text{ax}} = \Delta\chi'_{zz} - \frac{1}{2}(\Delta\chi'_{xx} + \Delta\chi'_{yy})$ ($10^{-32}\text{ m}^3$).
    pub dchi_ax: f64,
    /// Rhombic anisotropy $\Delta\chi_{\text{rh}} = \Delta\chi'_{xx} - \Delta\chi'_{yy}$ ($10^{-32}\text{ m}^3$),
    /// satisfying $0 \le \Delta\chi_{\text{rh}} / \Delta\chi_{\text{ax}} \le \frac{2}{3}$.
    pub dchi_rh: f64,
    /// Non-negative rhombicity ratio $|\Delta\chi_{\text{rh}} / \Delta\chi_{\text{ax}}| \in [0, 2/3]$.
    pub rhombicity: f64,
    /// ZYZ Euler angles $(\alpha, \beta, \gamma)$ in radians (`R = R_z(γ) * R_y(β) * R_z(α)`).
    pub euler_zyz_rad: [f64; 3],
    /// Root-mean-square PCS residual (ppm).
    pub rmsd_ppm: f64,
    /// Cornilescu-style PCS quality factor $Q = \sqrt{\sum(\Delta\delta^{\text{obs}} - \Delta\delta^{\text{calc}})^2 / \sum(\Delta\delta^{\text{obs}})^2}$.
    pub q_factor: f64,
    /// Singular value condition number $\sigma_{\max} / \sigma_{\min}$ of the $N \times 5$ design matrix.
    pub condition_number: f64,
}

/// Construct a proper right-handed rotation matrix $R = R_z(\gamma) R_y(\beta) R_z(\alpha) \in SO(3)$
/// transforming molecular-frame coordinates into the principal-axis frame ($\mathbf{r}' = R \mathbf{r}$).
pub fn rotation_matrix_from_zyz(alpha: f64, beta: f64, gamma: f64) -> [[f64; 3]; 3] {
    let (sa, ca) = alpha.sin_cos();
    let (sb, cb) = beta.sin_cos();
    let (sg, cg) = gamma.sin_cos();

    [
        [cg * cb * ca - sg * sa, cg * cb * sa + sg * ca, -cg * sb],
        [-sg * cb * ca - cg * sa, -sg * cb * sa + cg * ca, sg * sb],
        [sb * ca, sb * sa, cb],
    ]
}

/// Extract ZYZ Euler angles $(\alpha, \beta, \gamma)$ in radians from $R \in SO(3)$
/// where $R = R_z(\gamma) R_y(\beta) R_z(\alpha)$.
pub fn zyz_euler_from_rotation_matrix(r: &[[f64; 3]; 3]) -> [f64; 3] {
    let cb = r[2][2].clamp(-1.0, 1.0);
    let beta = cb.acos();
    let sb = beta.sin();
    let wrap_2pi = |ang: f64| -> f64 { ang.rem_euclid(2.0 * PI) };

    if sb > 1e-8 {
        let alpha = wrap_2pi(r[2][1].atan2(r[2][0]));
        let gamma = wrap_2pi(r[1][2].atan2(-r[0][2]));
        [alpha, beta, gamma]
    } else if cb > 0.0 {
        // beta ~ 0: R ~ R_z(alpha + gamma)
        let alpha = wrap_2pi(r[0][1].atan2(r[0][0]));
        [alpha, 0.0, 0.0]
    } else {
        // beta ~ pi: R_00 = -cos(alpha - gamma), R_01 = -sin(alpha - gamma)
        let alpha = wrap_2pi((-r[0][1]).atan2(-r[0][0]));
        [alpha, PI, 0.0]
    }
}

/// Forward spherical PAF pseudocontact shift equation (ppm) for nuclear coordinate `coord` (Å),
/// metal site `metal_pos` (Å), $(\Delta\chi_{\text{ax}}, \Delta\chi_{\text{rh}})$ in $10^{-32}\text{ m}^3$,
/// and ZYZ Euler angles $(\alpha, \beta, \gamma)$ in radians.
///
/// Follows the UTS principal-axis definition $\Delta\chi'_{zz} = \frac{2}{3}\Delta\chi_{\text{ax}}$,
/// $\Delta\chi'_{xx} - \Delta\chi'_{yy} = \Delta\chi_{\text{rh}}$:
/// $$\Delta\delta^{\text{PCS}} = \frac{10^4}{12\pi r^3}\left[\Delta\chi_{\text{ax}}(3\cos^2\theta - 1) + \frac{3}{2}\Delta\chi_{\text{rh}}\sin^2\theta\cos 2\phi\right].$$
pub fn predict_pcs_spherical_ppm(
    coord: Vec3,
    metal_pos: Vec3,
    dchi_ax: f64,
    dchi_rh: f64,
    euler_zyz_rad: [f64; 3],
) -> f64 {
    let dx = coord[0] - metal_pos[0];
    let dy = coord[1] - metal_pos[1];
    let dz = coord[2] - metal_pos[2];
    let r2 = dx * dx + dy * dy + dz * dz;
    if r2 < 1e-12 {
        return 0.0;
    }
    let r = r2.sqrt();
    let rot = rotation_matrix_from_zyz(euler_zyz_rad[0], euler_zyz_rad[1], euler_zyz_rad[2]);
    let xp = rot[0][0] * dx + rot[0][1] * dy + rot[0][2] * dz;
    let yp = rot[1][0] * dx + rot[1][1] * dy + rot[1][2] * dz;
    let zp = rot[2][0] * dx + rot[2][1] * dy + rot[2][2] * dz;

    let cos2_theta = (zp * zp) / r2;
    let sin2_cos2phi = (xp * xp - yp * yp) / r2;
    (PCS_SPHERICAL_PREFACTOR / (r * r2))
        * (dchi_ax * (3.0 * cos2_theta - 1.0) + 1.5 * dchi_rh * sin2_cos2phi)
}

/// Predict pseudocontact shift (ppm) directly from a fitted [`DeltaChiTensor`].
#[inline]
pub fn predict_pcs_from_tensor(coord: Vec3, tensor: &DeltaChiTensor) -> f64 {
    let row = pcs_design_row(coord, tensor.metal_pos_angstrom);
    row.iter()
        .zip(&tensor.components_5)
        .map(|(&r, &c)| r * c)
        .sum()
}

/// Exact 5-component traceless Cartesian design-matrix row for $\mathbf{x} = [\Delta\chi_{xx}, \Delta\chi_{yy}, \Delta\chi_{xy}, \Delta\chi_{xz}, \Delta\chi_{yz}]^T$:
/// $$\mathbf{a}(\mathbf{r}) = \frac{10^4}{4\pi r^5}\,[x^2 - z^2,\; y^2 - z^2,\; 2xy,\; 2xz,\; 2yz].$$
#[inline]
pub fn pcs_design_row(coord: Vec3, metal_pos: Vec3) -> [f64; 5] {
    let x = coord[0] - metal_pos[0];
    let y = coord[1] - metal_pos[1];
    let z = coord[2] - metal_pos[2];
    let r2 = x * x + y * y + z * z;
    if r2 < 1e-12 {
        return [0.0; 5];
    }
    let r5 = r2 * r2 * r2.sqrt();
    let pref = PCS_CARTESIAN_PREFACTOR / r5;
    [
        pref * (x * x - z * z),
        pref * (y * y - z * z),
        pref * (2.0 * x * y),
        pref * (2.0 * x * z),
        pref * (2.0 * y * z),
    ]
}

/// Maximum acceptable singular-value condition number $\kappa(A) = \sigma_{\max}/\sigma_{\min}$
/// for 5-parameter PCS linear least squares before declaring the nuclear geometry rank-deficient.
pub const MAX_PCS_DESIGN_COND: f64 = 1.0e8;

/// Compare two Cornilescu PCS $Q$-factors, treating `NaN` / non-finite $Q$ strictly as worst.
#[inline]
pub fn pcs_q_is_better(candidate_q: f64, incumbent_q: f64) -> bool {
    match (candidate_q.is_finite(), incumbent_q.is_finite()) {
        (true, true) => candidate_q < incumbent_q,
        (true, false) => true,
        (false, _) => false,
    }
}

fn validate_pcs_observation_slice(obs: &[PcsObservation], min_count: usize) -> Result<(), String> {
    if obs.len() < min_count {
        return Err(format!(
            "PCS DeltaChi tensor fit requires at least {min_count} observations (>= 1 DoF), got {}",
            obs.len()
        ));
    }
    for o in obs {
        if !o.pcs_ppm.is_finite() {
            return Err(format!(
                "non-finite PCS shift ({}) at seq_id {}",
                o.pcs_ppm, o.seq_id
            ));
        }
        if !o.coord_angstrom.iter().all(|v| v.is_finite()) {
            return Err(format!(
                "non-finite nuclear coordinate {:?} at seq_id {}",
                o.coord_angstrom, o.seq_id
            ));
        }
        if !o.uncertainty_ppm.is_finite() || o.uncertainty_ppm <= 0.0 {
            return Err(format!(
                "invalid PCS uncertainty_ppm ({}) at seq_id {}: must be finite and > 0",
                o.uncertainty_ppm, o.seq_id
            ));
        }
        if !o.coord_uncertainty_angstrom.is_finite() || o.coord_uncertainty_angstrom < 0.0 {
            return Err(format!(
                "invalid PCS coord_uncertainty_angstrom ({}) at seq_id {}: must be finite and >= 0",
                o.coord_uncertainty_angstrom, o.seq_id
            ));
        }
    }
    Ok(())
}

/// Self-contained thin Hestenes one-sided Jacobi Singular Value Decomposition of an $N \times 5$
/// matrix $A = U \Sigma V^T$ ($N \ge 6$), solving $A x \approx b$ in least squares without squaring $\kappa(A)$.
fn solve_nx5_svd(a_rows: &[[f64; 5]], b: &[f64]) -> Result<([f64; 5], f64), String> {
    let n = a_rows.len();
    if n < 6 || b.len() != n {
        return Err(format!(
            "5-parameter PCS tensor SVD requires at least 6 observations (>= 1 DoF, got {n})"
        ));
    }
    if a_rows.iter().any(|r| r.iter().any(|v| !v.is_finite())) || b.iter().any(|v| !v.is_finite()) {
        return Err("non-finite entry in PCS design matrix or observation vector".to_string());
    }

    // Store columns w_0..w_4 of length n; initialize V = I_5
    let mut cols: Vec<Vec<f64>> = (0..5)
        .map(|j| a_rows.iter().map(|r| r[j]).collect())
        .collect();
    let mut v = [[0.0_f64; 5]; 5];
    for (j, row) in v.iter_mut().enumerate() {
        row[j] = 1.0;
    }

    for _sweep in 0..40 {
        let mut max_off = 0.0_f64;
        for p in 0..5 {
            for q in (p + 1)..5 {
                let mut app = 0.0_f64;
                let mut aqq = 0.0_f64;
                let mut apq = 0.0_f64;
                for (&wp, &wq) in cols[p].iter().zip(&cols[q]) {
                    app += wp * wp;
                    aqq += wq * wq;
                    apq += wp * wq;
                }
                let denom_norm = (app * aqq).sqrt();
                if denom_norm > 0.0 {
                    max_off = max_off.max(apq.abs() / denom_norm);
                }
                if apq.abs() <= 1e-15 * denom_norm {
                    continue;
                }
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                let (left, right) = cols.split_at_mut(q);
                for (wp, wq) in left[p].iter_mut().zip(&mut right[0]) {
                    let up = *wp;
                    let uq = *wq;
                    *wp = c * up - s * uq;
                    *wq = s * up + c * uq;
                }
                for row in &mut v {
                    let vp = row[p];
                    let vq = row[q];
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
        if max_off < 1e-14 {
            break;
        }
    }

    let mut sigmas = [0.0_f64; 5];
    for j in 0..5 {
        let norm_sq: f64 = cols[j].iter().map(|&val| val * val).sum();
        sigmas[j] = norm_sq.sqrt();
    }
    let s_max = sigmas.iter().copied().fold(0.0_f64, f64::max);
    let s_min = sigmas.iter().copied().fold(f64::INFINITY, f64::min);
    if s_max <= 1e-15 || !s_max.is_finite() || !s_min.is_finite() || s_min <= 0.0 {
        return Err(
            "degenerate PCS geometry: singular or rank-deficient design matrix".to_string(),
        );
    }
    let cond = s_max / s_min;
    if !cond.is_finite() || cond > MAX_PCS_DESIGN_COND {
        return Err(format!(
            "ill-conditioned PCS geometry: design-matrix condition number {cond:.3e} exceeds {:.1e}",
            MAX_PCS_DESIGN_COND
        ));
    }

    let mut x = [0.0_f64; 5];
    for j in 0..5 {
        let mut ut_b = 0.0_f64;
        for i in 0..n {
            ut_b += (cols[j][i] / sigmas[j]) * b[i];
        }
        let coeff = ut_b / sigmas[j];
        for k in 0..5 {
            x[k] += coeff * v[k][j];
        }
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err("PCS SVD produced non-finite tensor components".to_string());
    }
    Ok((x, cond))
}

/// Exact cyclic Jacobi eigendecomposition of a $3 \times 3$ real symmetric matrix $M = Q \Lambda Q^T$.
fn jacobi_sym_3x3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut q = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..32 {
        let pairs = [(0usize, 1usize), (0, 2), (1, 2)];
        let mut off_sq = 0.0_f64;
        for &(p, r) in &pairs {
            off_sq += a[p][r] * a[p][r];
        }
        if off_sq.sqrt() < 1e-15 {
            break;
        }
        for &(p, r) in &pairs {
            let apr = a[p][r];
            if apr.abs() < 1e-16 {
                continue;
            }
            let tau = (a[r][r] - a[p][p]) / (2.0 * apr);
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;

            let app = a[p][p];
            let arr = a[r][r];
            a[p][p] = app - t * apr;
            a[r][r] = arr + t * apr;
            a[p][r] = 0.0;
            a[r][p] = 0.0;

            let m = 3 - p - r;
            let amp = a[m][p];
            let amr = a[m][r];
            a[m][p] = c * amp - s * amr;
            a[p][m] = a[m][p];
            a[m][r] = s * amp + c * amr;
            a[r][m] = a[m][r];

            for row in &mut q {
                let qp = row[p];
                let qr = row[r];
                row[p] = c * qp - s * qr;
                row[r] = s * qp + c * qr;
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2]], q)
}

/// Diagonalize the 5-component traceless symmetric tensor into the Schmitz/Otting UTS
/// principal-axis convention $|\Delta\chi'_{zz}| \ge |\Delta\chi'_{yy}| \ge |\Delta\chi'_{xx}|$
/// (`0 <= Δχ_rh / Δχ_ax <= 2/3`) and right-handed ZYZ Euler angles $(\alpha, \beta, \gamma)$.
fn build_delta_chi_tensor(
    metal_pos: Vec3,
    x: [f64; 5],
    cond: f64,
    obs: &[PcsObservation],
) -> DeltaChiTensor {
    let (xx, yy, xy, xz, yz) = (x[0], x[1], x[2], x[3], x[4]);
    let zz = -(xx + yy);
    let mat = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]];

    let (evals, q_cols) = jacobi_sym_3x3(mat);
    // Schmitz et al. (J. Biomol. NMR 2008) / Otting UTS ordering:
    // |lambda_z| >= |lambda_y| >= |lambda_x| so 0 <= dchi_rh / dchi_ax <= 2/3.
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| evals[b].abs().total_cmp(&evals[a].abs()));
    let idx_z = order[0];
    let idx_y = order[1];
    let idx_x = order[2];

    let dchi_zz_p = evals[idx_z];
    let dchi_xx_p = evals[idx_x];
    let dchi_yy_p = evals[idx_y];

    let mut ex = [q_cols[0][idx_x], q_cols[1][idx_x], q_cols[2][idx_x]];
    let mut ey = [q_cols[0][idx_y], q_cols[1][idx_y], q_cols[2][idx_y]];
    let mut ez = [q_cols[0][idx_z], q_cols[1][idx_z], q_cols[2][idx_z]];

    // Canonicalize eigenvector signs into a proper right-handed basis (det(R) = +1)
    if ez[2] < 0.0 || (ez[2].abs() < 1e-12 && ez[1] < 0.0) {
        ez = [-ez[0], -ez[1], -ez[2]];
    }
    if ex[0] < 0.0 {
        ex = [-ex[0], -ex[1], -ex[2]];
    }
    let cross_xy = [
        ex[1] * ey[2] - ex[2] * ey[1],
        ex[2] * ey[0] - ex[0] * ey[2],
        ex[0] * ey[1] - ex[1] * ey[0],
    ];
    let det = cross_xy[0] * ez[0] + cross_xy[1] * ez[1] + cross_xy[2] * ez[2];
    if det < 0.0 {
        ey = [-ey[0], -ey[1], -ey[2]];
    }

    let rot = [ex, ey, ez];
    let euler_zyz_rad = zyz_euler_from_rotation_matrix(&rot);

    let dchi_ax = dchi_zz_p - 0.5 * (dchi_xx_p + dchi_yy_p);
    let dchi_rh = dchi_xx_p - dchi_yy_p;
    let rhombicity = if dchi_ax.abs() > 1e-15 {
        (dchi_rh / dchi_ax).clamp(0.0, 2.0 / 3.0)
    } else {
        0.0
    };

    let mut sum_sq_err = 0.0_f64;
    let mut sum_sq_obs = 0.0_f64;
    for o in obs {
        let row = pcs_design_row(o.coord_angstrom, metal_pos);
        let calc: f64 = (0..5).map(|j| row[j] * x[j]).sum();
        let diff = o.pcs_ppm - calc;
        sum_sq_err += diff * diff;
        sum_sq_obs += o.pcs_ppm * o.pcs_ppm;
    }
    let rmsd_ppm = if !obs.is_empty() {
        (sum_sq_err / (obs.len() as f64)).sqrt()
    } else {
        f64::NAN
    };
    let q_factor = if sum_sq_obs > 1e-15 && sum_sq_err.is_finite() {
        (sum_sq_err / sum_sq_obs).sqrt()
    } else {
        f64::NAN
    };

    DeltaChiTensor {
        metal_pos_angstrom: metal_pos,
        components_5: x,
        matrix_mol: mat,
        eigenvalues_paf: [dchi_xx_p, dchi_yy_p, dchi_zz_p],
        rotation_mol_to_paf: rot,
        dchi_ax,
        dchi_rh,
        rhombicity,
        euler_zyz_rad,
        rmsd_ppm,
        q_factor,
        condition_number: cond,
    }
}

/// Effective PCS observation standard deviation (ppm) combining spectral shift uncertainty
/// `o.uncertainty_ppm` and isotropic 3D RMS nuclear coordinate uncertainty
/// `o.coord_uncertainty_angstrom` propagated analytically via $\nabla_{\mathbf{r}}\Delta\delta^{\text{PCS}}$
/// (`Schmitz et al. 2008`).
fn effective_pcs_sigma_ppm(o: &PcsObservation, metal_pos: Vec3, x: &[f64; 5]) -> f64 {
    let sig_ppm = o.uncertainty_ppm.max(1e-12);
    if o.coord_uncertainty_angstrom <= 1e-9 {
        return sig_ppm;
    }

    let vx = o.coord_angstrom[0] - metal_pos[0];
    let vy = o.coord_angstrom[1] - metal_pos[1];
    let vz = o.coord_angstrom[2] - metal_pos[2];
    let r2 = vx * vx + vy * vy + vz * vz;
    if r2 < 1e-12 {
        return sig_ppm;
    }
    let r = r2.sqrt();
    let r5 = r2 * r2 * r;
    let r7 = r5 * r2;

    let (xx, yy, xy, xz, yz) = (x[0], x[1], x[2], x[3], x[4]);
    let zz = -(xx + yy);
    let mat = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]];
    let mv = [
        mat[0][0] * vx + mat[0][1] * vy + mat[0][2] * vz,
        mat[1][0] * vx + mat[1][1] * vy + mat[1][2] * vz,
        mat[2][0] * vx + mat[2][1] * vy + mat[2][2] * vz,
    ];
    let vmv = vx * mv[0] + vy * mv[1] + vz * mv[2];
    let gx = PCS_CARTESIAN_PREFACTOR * (2.0 * mv[0] / r5 - 5.0 * vmv * vx / r7);
    let gy = PCS_CARTESIAN_PREFACTOR * (2.0 * mv[1] / r5 - 5.0 * vmv * vy / r7);
    let gz = PCS_CARTESIAN_PREFACTOR * (2.0 * mv[2] / r5 - 5.0 * vmv * vz / r7);
    let grad_norm_sq = gx * gx + gy * gy + gz * gz;

    // Isotropic 3D RMS sigma_coord has per-axis variance sigma_coord^2 / 3
    let var_coord =
        (o.coord_uncertainty_angstrom * o.coord_uncertainty_angstrom / 3.0) * grad_norm_sq;
    (sig_ppm * sig_ppm + var_coord).sqrt().max(1e-12)
}

/// Fit the 5-parameter traceless symmetric $\Delta\boldsymbol{\chi}$ tensor at fixed metal position
/// `metal_pos_angstrom` via weighted linear SVD, with optional analytic coordinate-gradient GLS
/// error propagation (`Schmitz et al., J. Biomol. NMR 2008`).
pub fn fit_pcs_tensor_svd(
    obs: &[PcsObservation],
    metal_pos_angstrom: Vec3,
) -> Result<DeltaChiTensor, String> {
    if !metal_pos_angstrom.iter().all(|v| v.is_finite()) {
        return Err(format!(
            "non-finite metal coordinate {metal_pos_angstrom:?} in PCS tensor SVD fit"
        ));
    }
    validate_pcs_observation_slice(obs, 6)?;

    let mut a_rows = Vec::with_capacity(obs.len());
    let mut b_vec = Vec::with_capacity(obs.len());
    let mut has_coord_unc = false;

    for o in obs {
        let r = v_dist(o.coord_angstrom, metal_pos_angstrom);
        if r < 1.2 {
            return Err(format!(
                "unphysical metal-nucleus distance {r:.3} Å (< 1.2 Å) at seq_id {}",
                o.seq_id
            ));
        }
        if o.coord_uncertainty_angstrom > 1e-9 {
            has_coord_unc = true;
        }
        let w = 1.0 / o.uncertainty_ppm.max(1e-12);
        let mut row = pcs_design_row(o.coord_angstrom, metal_pos_angstrom);
        for v in &mut row {
            *v *= w;
        }
        a_rows.push(row);
        b_vec.push(o.pcs_ppm * w);
    }

    let (mut x, mut cond) = solve_nx5_svd(&a_rows, &b_vec)?;

    // Analytic coordinate-uncertainty error propagation (Schmitz et al. 2008)
    if has_coord_unc {
        for _ in 0..2 {
            a_rows.clear();
            b_vec.clear();
            for o in obs {
                let w = 1.0 / effective_pcs_sigma_ppm(o, metal_pos_angstrom, &x);
                let mut row = pcs_design_row(o.coord_angstrom, metal_pos_angstrom);
                for v in &mut row {
                    *v *= w;
                }
                a_rows.push(row);
                b_vec.push(o.pcs_ppm * w);
            }
            let (x_gls, cond_gls) = solve_nx5_svd(&a_rows, &b_vec)?;
            x = x_gls;
            cond = cond_gls;
        }
    }

    Ok(build_delta_chi_tensor(metal_pos_angstrom, x, cond, obs))
}

/// Exclude pseudocontact shift (`PCS`) observations that fall inside the paramagnetic `PRE` core
/// blind sphere ($p_{\text{bleach}} \ge \text{hard\_bleach\_cutoff}$, default `0.85`) or inside the
/// covalent Fermi-contact / breakdown shell ($r < 8.0\text{ Å}$) of a polynuclear `[Fe-S]` cluster
/// (`Bertini, Luchinat & Parigi, Prog. NMR Spectrosc. 2002`).
pub fn filter_pcs_observations_blind_zone(
    obs: &[PcsObservation],
    metal_coords: &[Vec3],
    r_bleach_angstrom: f64,
    is_fes_cluster: bool,
    hard_bleach_cutoff: f64,
) -> Vec<PcsObservation> {
    if metal_coords.is_empty() {
        return obs.to_vec();
    }
    obs.iter()
        .copied()
        .filter(|o| {
            let r_min = metal_coords
                .iter()
                .map(|&m| v_dist(o.coord_angstrom, m))
                .fold(f64::INFINITY, f64::min);
            if is_fes_cluster && r_min < 8.0 {
                return false;
            }
            if r_bleach_angstrom > 0.0 {
                let gamma_ratio: f64 = metal_coords
                    .iter()
                    .map(|&m| (r_bleach_angstrom / v_dist(o.coord_angstrom, m).max(0.5)).powi(6))
                    .sum();
                let p_bleach = gamma_ratio / (1.0 + gamma_ratio);
                if p_bleach >= hard_bleach_cutoff {
                    return false;
                }
            }
            true
        })
        .collect()
}

/// Fit the 5-parameter $\Delta\boldsymbol{\chi}$ tensor with standardized-residual Huber Iteratively
/// Reweighted Least Squares (`IRLS`) + SVD to suppress misassigned or contact-shifted PCS outliers
/// without double-downweighting high-uncertainty observations (`Huber 1964`; `Schmitz et al. 2008`).
///
/// Each observation residual is standardized by its propagated effective standard deviation
/// $\sigma_{i,\text{eff}}$ (`uncertainty_ppm` $\oplus$ `coord_uncertainty_angstrom`):
/// $$z_i = \frac{|\Delta\delta^{\text{obs}}_i - \Delta\delta^{\text{calc}}_i|}{\sigma_{i,\text{eff}}}, \qquad w_{\text{Huber}}(z_i) = \begin{cases} 1, & z_i \le k_{\text{Huber}}, \\ \sqrt{k_{\text{Huber}} / z_i}, & z_i > k_{\text{Huber}}. \end{cases}$$
pub fn fit_pcs_tensor_robust(
    obs: &[PcsObservation],
    metal_pos_angstrom: Vec3,
    huber_k: f64,
    max_iters: usize,
) -> Result<DeltaChiTensor, String> {
    if !huber_k.is_finite() || huber_k <= 0.0 {
        return Err(format!(
            "invalid huber_k ({huber_k}): dimensionless Huber threshold must be finite and > 0"
        ));
    }
    let init = fit_pcs_tensor_svd(obs, metal_pos_angstrom)?;
    let k = huber_k.max(1e-6);
    let mut x = init.components_5;
    let mut cond = init.condition_number;

    for _ in 0..max_iters.max(1) {
        let mut a_rows = Vec::with_capacity(obs.len());
        let mut b_vec = Vec::with_capacity(obs.len());
        for o in obs {
            let base_row = pcs_design_row(o.coord_angstrom, metal_pos_angstrom);
            let pred: f64 = base_row.iter().zip(&x).map(|(&r, &c)| r * c).sum();
            let sig_eff = effective_pcs_sigma_ppm(o, metal_pos_angstrom, &x);
            let z_i = (o.pcs_ppm - pred).abs() / sig_eff;
            let base_w = 1.0 / sig_eff;
            let huber_w = if z_i <= k { 1.0 } else { (k / z_i).sqrt() };
            let w = base_w * huber_w;
            let mut row = base_row;
            for v in &mut row {
                *v *= w;
            }
            a_rows.push(row);
            b_vec.push(o.pcs_ppm * w);
        }
        let (x_new, new_cond) = solve_nx5_svd(&a_rows, &b_vec)?;
        let step_norm: f64 = x_new
            .iter()
            .zip(&x)
            .map(|(&xn, &xo)| (xn - xo).powi(2))
            .sum::<f64>()
            .sqrt();
        x = x_new;
        cond = new_cond;
        if step_norm < 1e-10 {
            break;
        }
    }

    Ok(build_delta_chi_tensor(metal_pos_angstrom, x, cond, obs))
}

/// Automatically exclude core blind-sphere ($p_{\text{bleach}} \ge \text{hard\_bleach\_cutoff}$) and
/// `[Fe-S]` Fermi-contact ($r < 8.0\text{ Å}$) observations before fitting the 5-parameter
/// $\Delta\boldsymbol{\chi}$ tensor via exact SVD (`huber_k == None`) or standardized Huber IRLS
/// (`huber_k == Some(k)`), returning `(fitted_tensor, retained_observations)`.
pub fn fit_pcs_tensor_paramagnetic(
    obs: &[PcsObservation],
    metal_pos_angstrom: Vec3,
    metal_class: MetalParamagneticClass,
    is_fes_cluster: bool,
    r_bleach_override: Option<f64>,
    hard_bleach_cutoff: f64,
    huber_k: Option<f64>,
) -> Result<(DeltaChiTensor, Vec<PcsObservation>), String> {
    let r_bleach = r_bleach_override.unwrap_or_else(|| metal_class.default_r_bleach_angstrom());
    let filtered = filter_pcs_observations_blind_zone(
        obs,
        &[metal_pos_angstrom],
        r_bleach,
        is_fes_cluster,
        hard_bleach_cutoff,
    );
    let tensor = match huber_k {
        Some(k) => fit_pcs_tensor_robust(&filtered, metal_pos_angstrom, k, 25)?,
        None => fit_pcs_tensor_svd(&filtered, metal_pos_angstrom)?,
    };
    Ok((tensor, filtered))
}

/// Variable-projection (`VarPro`) 3D metal-position and 5-parameter $\Delta\boldsymbol{\chi}$-tensor
/// refinement (`Schmitz et al. 2008`): coarse 3D grid search around `initial_metal_pos` followed by
/// 3D Nelder–Mead simplex optimization, solving exact 5-component linear SVD at every candidate site.
pub fn refine_metal_position_and_tensor(
    obs: &[PcsObservation],
    initial_metal_pos: Vec3,
    search_radius_angstrom: f64,
    grid_step_angstrom: f64,
) -> Result<DeltaChiTensor, String> {
    if !initial_metal_pos.iter().all(|v| v.is_finite())
        || !search_radius_angstrom.is_finite()
        || !grid_step_angstrom.is_finite()
    {
        return Err(
            "non-finite initial_metal_pos or grid parameter in VarPro PCS refinement".to_string(),
        );
    }
    validate_pcs_observation_slice(obs, 8)?;
    if obs.len() < 8 {
        return Err(format!(
            "joint 8-parameter metal-position + DeltaChi refinement requires >= 8 PCS observations, got {}",
            obs.len()
        ));
    }

    let eval_chi2 = |m: Vec3| -> f64 {
        for o in obs {
            if v_dist(o.coord_angstrom, m) < 2.0 {
                return 1e30;
            }
        }
        match fit_pcs_tensor_svd(obs, m) {
            Ok(t) => {
                let mut chi2 = 0.0_f64;
                for o in obs {
                    let pred = predict_pcs_from_tensor(o.coord_angstrom, &t);
                    let sig = effective_pcs_sigma_ppm(o, m, &t.components_5);
                    let z = (o.pcs_ppm - pred) / sig;
                    chi2 += z * z;
                }
                chi2
            }
            Err(_) => 1e30,
        }
    };

    // Stage 1: Coarse 3D grid search
    let mut best_m = initial_metal_pos;
    let mut best_f = eval_chi2(initial_metal_pos);
    let rad = search_radius_angstrom.max(0.0);
    let step = grid_step_angstrom.max(0.25);
    if rad > 0.0 {
        let n_steps = (rad / step).ceil() as isize;
        for ix in -n_steps..=n_steps {
            for iy in -n_steps..=n_steps {
                for iz in -n_steps..=n_steps {
                    let dm = [(ix as f64) * step, (iy as f64) * step, (iz as f64) * step];
                    if dm[0] * dm[0] + dm[1] * dm[1] + dm[2] * dm[2] > rad * rad * 1.01 {
                        continue;
                    }
                    let cand = [
                        initial_metal_pos[0] + dm[0],
                        initial_metal_pos[1] + dm[1],
                        initial_metal_pos[2] + dm[2],
                    ];
                    let f = eval_chi2(cand);
                    if f < best_f {
                        best_f = f;
                        best_m = cand;
                    }
                }
            }
        }
    }

    // Stage 2: 3D Nelder-Mead simplex refinement
    let init_delta = (step * 0.45).clamp(0.05, 0.6);
    let mut simplex: [(Vec3, f64); 4] = [
        (best_m, best_f),
        ([best_m[0] + init_delta, best_m[1], best_m[2]], 0.0),
        ([best_m[0], best_m[1] + init_delta, best_m[2]], 0.0),
        ([best_m[0], best_m[1], best_m[2] + init_delta], 0.0),
    ];
    for item in simplex.iter_mut().skip(1) {
        item.1 = eval_chi2(item.0);
    }

    for _iter in 0..200 {
        simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
        let diam = v_dist(simplex[0].0, simplex[3].0);
        if diam < 1e-6 {
            break;
        }
        let mut centroid = [0.0_f64; 3];
        for item in simplex.iter().take(3) {
            for (c_val, &p_val) in centroid.iter_mut().zip(&item.0) {
                *c_val += p_val / 3.0;
            }
        }
        let worst = simplex[3].0;
        let reflected = [
            centroid[0] + (centroid[0] - worst[0]),
            centroid[1] + (centroid[1] - worst[1]),
            centroid[2] + (centroid[2] - worst[2]),
        ];
        let f_ref = eval_chi2(reflected);

        if f_ref < simplex[0].1 {
            let expanded = [
                centroid[0] + 2.0 * (reflected[0] - centroid[0]),
                centroid[1] + 2.0 * (reflected[1] - centroid[1]),
                centroid[2] + 2.0 * (reflected[2] - centroid[2]),
            ];
            let f_exp = eval_chi2(expanded);
            simplex[3] = if f_exp < f_ref {
                (expanded, f_exp)
            } else {
                (reflected, f_ref)
            };
        } else if f_ref < simplex[2].1 {
            simplex[3] = (reflected, f_ref);
        } else {
            let contracted = if f_ref < simplex[3].1 {
                [
                    centroid[0] + 0.5 * (reflected[0] - centroid[0]),
                    centroid[1] + 0.5 * (reflected[1] - centroid[1]),
                    centroid[2] + 0.5 * (reflected[2] - centroid[2]),
                ]
            } else {
                [
                    centroid[0] + 0.5 * (worst[0] - centroid[0]),
                    centroid[1] + 0.5 * (worst[1] - centroid[1]),
                    centroid[2] + 0.5 * (worst[2] - centroid[2]),
                ]
            };
            let f_con = eval_chi2(contracted);
            if f_con < simplex[3].1 {
                simplex[3] = (contracted, f_con);
            } else {
                let best_pt = simplex[0].0;
                for item in simplex.iter_mut().skip(1) {
                    for (coord_val, &b_val) in item.0.iter_mut().zip(&best_pt) {
                        *coord_val = b_val + 0.5 * (*coord_val - b_val);
                    }
                    item.1 = eval_chi2(item.0);
                }
            }
        }
    }

    simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
    fit_pcs_tensor_svd(obs, simplex[0].0)
}
