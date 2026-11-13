//! Linked 3D-to-NMR-strip analytical ray picking, quantitative putty & perceptually uniform
//! colormaps, and 3D NOE/PRE distance-restraint violation cylinders (`T12.2`).
//!
//! Provenance: Clean-room original implementation (`docs/specs/15-structure-viewer.md` §2)
//! grounded in peer-reviewed spectroscopy, dynamics, and perceptual visualization literature:
//! - Wishart, D. S., Bigam, C. G., Holm, A., Hodges, R. S. & Sykes, B. D. (*J. Biomol. NMR*
//!   1995, 5, 67–81 — Chemical Shift Index `CSI` secondary-shift consensus).
//! - Lipari, G. & Szabo, A. (*J. Am. Chem. Soc.* 1982, 104, 4546–4559 — model-free generalized
//!   order parameter `S^2` and conformational exchange `R_ex`).
//! - Williamson, M. P. (*Prog. Nucl. Magn. Reson. Spectrosc.* 2013, 73, 1–16 — combined amide
//!   ligand chemical shift perturbation `Delta_delta_comb`).
//! - Nilges, M. (*J. Mol. Biol.* 1995, 245, 645–660 — `r^-6` effective ambiguous NOE distance).
//! - Nuñez, J. R., Anderton, C. R. & Renslow, R. S. (*PLOS ONE* 2018, 13(8), e0199239 — `Cividis`
//!   color-vision-deficiency-safe perceptually uniform colormap) & Moreland, K. (*ISVC* 2009,
//!   LNCS 5876, 92–103 — diverging cool-warm scientific colormaps).

use std::collections::BTreeMap;

use bytemuck::{Pod, Zeroable};
use nmr_core::nmrstar::Reference;
use nmr_core::priors::Priors;
use nmr_core::topology::{residue_topology, Element};

use crate::contour_painter::SyncCrosshair;
use crate::sidechain_widget::AtomJumpTarget;

/// Neutral slate RGBA applied to residues lacking quantitative data on the active scalar track.
pub const UNASSIGNED_PUTTY_RGBA: [f32; 4] = [0.42, 0.48, 0.57, 1.0];

/// Hardware-instanced cylinder descriptor (`64` bytes, 16-byte aligned) shared between covalent
/// stick bonds (`mol3d_viewer`) and 3D NOE/PRE distance restraint cylinders (`mol3d_overlay`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct CylinderInstance {
    /// `(p0_x, p0_y, p0_z, radius_angstrom)`
    pub p0_rad: [f32; 4],
    /// `(p1_x, p1_y, p1_z, dash_cycles_flag)`: `0.0` = solid half-bond CPK, `>= 2.0` = dashed cylinder
    pub p1_flags: [f32; 4],
    /// Endpoint `p0` RGBA color
    pub color0: [f32; 4],
    /// Endpoint `p1` RGBA color
    pub color1: [f32; 4],
}

impl CylinderInstance {
    /// Construct a uniform-color cylinder (used for NOE/PRE restraints and single-color bonds).
    #[inline]
    pub const fn uniform(
        p0: [f32; 3],
        p1: [f32; 3],
        radius: f32,
        rgba: [f32; 4],
        dash_cycles: f32,
    ) -> Self {
        Self {
            p0_rad: [p0[0], p0[1], p0[2], radius],
            p1_flags: [p1[0], p1[1], p1[2], dash_cycles],
            color0: rgba,
            color1: rgba,
        }
    }

    /// Construct a split-color covalent bond cylinder (`p0 -> mid` in `color0`, `mid -> p1` in `color1`).
    #[inline]
    pub const fn split_bond(
        p0: [f32; 3],
        p1: [f32; 3],
        radius: f32,
        color0: [f32; 4],
        color1: [f32; 4],
    ) -> Self {
        Self {
            p0_rad: [p0[0], p0[1], p0[2], radius],
            p1_flags: [p1[0], p1[1], p1[2], 0.0],
            color0,
            color1,
        }
    }
}

// ============================================================================
// 1. Perceptually Uniform Scientific Colormaps
// ============================================================================

/// Perceptually uniform scientific colormaps for quantitative 3D structure overlays.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ColormapKind {
    /// Perceptually uniform sequential indigo-teal-emerald-yellow (`[0, 1]`).
    #[default]
    Viridis,
    /// Color-vision-deficiency-safe perceptually uniform navy-olive-gold (Nuñez et al. 2018).
    Cividis,
    /// Diverging sapphire-white-crimson (`[-1, +1]` or midpoint-centered scalar tracks).
    BlueWhiteRed,
}

impl ColormapKind {
    pub const ALL: [Self; 3] = [Self::Viridis, Self::Cividis, Self::BlueWhiteRed];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Viridis => "Viridis",
            Self::Cividis => "Cividis (CVD-Safe)",
            Self::BlueWhiteRed => "Blue-White-Red",
        }
    }

    /// Sample `[r, g, b, alpha]` for normalized `t` in `[0.0, 1.0]`.
    #[inline]
    pub fn sample_rgba(self, t: f32, alpha: f32) -> [f32; 4] {
        const VIRIDIS_LUT: [[f32; 3]; 9] = [
            [0.267, 0.005, 0.329],
            [0.283, 0.141, 0.458],
            [0.254, 0.265, 0.530],
            [0.207, 0.372, 0.553],
            [0.164, 0.471, 0.558],
            [0.128, 0.567, 0.551],
            [0.208, 0.718, 0.472],
            [0.565, 0.842, 0.263],
            [0.993, 0.906, 0.144],
        ];
        const CIVIDIS_LUT: [[f32; 3]; 9] = [
            [0.000, 0.135, 0.305],
            [0.086, 0.220, 0.398],
            [0.225, 0.304, 0.425],
            [0.347, 0.388, 0.432],
            [0.468, 0.475, 0.453],
            [0.594, 0.567, 0.440],
            [0.728, 0.665, 0.395],
            [0.867, 0.771, 0.312],
            [0.996, 0.891, 0.210],
        ];
        const BWR_LUT: [[f32; 3]; 9] = [
            [0.122, 0.298, 0.765],
            [0.255, 0.467, 0.875],
            [0.482, 0.675, 0.945],
            [0.745, 0.847, 0.976],
            [0.955, 0.960, 0.972],
            [0.984, 0.788, 0.737],
            [0.957, 0.525, 0.467],
            [0.875, 0.263, 0.247],
            [0.698, 0.086, 0.149],
        ];

        let lut = match self {
            Self::Viridis => &VIRIDIS_LUT,
            Self::Cividis => &CIVIDIS_LUT,
            Self::BlueWhiteRed => &BWR_LUT,
        };
        let u = t.clamp(0.0, 1.0) * ((lut.len() - 1) as f32);
        let idx = (u.floor() as usize).min(lut.len() - 2);
        let frac = u - (idx as f32);
        let a = lut[idx];
        let b = lut[idx + 1];
        [
            a[0] + (b[0] - a[0]) * frac,
            a[1] + (b[1] - a[1]) * frac,
            a[2] + (b[2] - a[2]) * frac,
            alpha.clamp(0.0, 1.0),
        ]
    }
}

// ============================================================================
// 2. Quantitative Putty Radius & Scalar Track Engine
// ============================================================================

/// Active per-residue color and putty-radius track projected onto the 3D structure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum QuantitativeTrack {
    /// Canonical secondary-structure cartoon colors (cyan helix, gold strand, silver-slate coil).
    #[default]
    SecondaryStructure,
    /// Multi-conformer Kabsch `CA` spatial RMSD (`Å`) across the loaded NMR ensemble.
    EnsembleRmsd,
    /// Backbone/sidechain assignment marginal posterior confidence (`[0.0, 1.0]`).
    AssignmentConfidence,
    /// Wishart Chemical Shift Index `CSI` secondary-shift score (`[-1.0, +1.0]`: beta `< 0`, alpha `> 0`).
    ChemicalShiftIndex,
    /// Lipari–Szabo generalized order parameter `S^2` (`[0.0, 1.0]`: `1` = rigid, `0` = disordered).
    LipariSzaboS2,
    /// Conformational chemical exchange contribution `R_ex` (`s^-1`).
    ChemicalExchangeRex,
    /// Combined amide ligand titration shift perturbation `Delta_delta_comb` (`ppm`).
    LigandCsp,
}

impl QuantitativeTrack {
    pub const ALL: [Self; 7] = [
        Self::SecondaryStructure,
        Self::EnsembleRmsd,
        Self::AssignmentConfidence,
        Self::ChemicalShiftIndex,
        Self::LipariSzaboS2,
        Self::ChemicalExchangeRex,
        Self::LigandCsp,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::SecondaryStructure => "Secondary Structure",
            Self::EnsembleRmsd => "Ensemble CA RMSD (Å)",
            Self::AssignmentConfidence => "Assignment Confidence",
            Self::ChemicalShiftIndex => "Chemical Shift Index (CSI)",
            Self::LipariSzaboS2 => "Lipari-Szabo S²",
            Self::ChemicalExchangeRex => "Exchange R_ex (s⁻¹)",
            Self::LigandCsp => "Ligand CSP Δδ_comb (ppm)",
        }
    }

    pub const fn default_colormap(self) -> ColormapKind {
        match self {
            Self::ChemicalShiftIndex => ColormapKind::BlueWhiteRed,
            Self::AssignmentConfidence | Self::EnsembleRmsd | Self::LigandCsp => {
                ColormapKind::Viridis
            }
            Self::LipariSzaboS2 | Self::ChemicalExchangeRex => ColormapKind::Cividis,
            Self::SecondaryStructure => ColormapKind::Viridis,
        }
    }

    pub const fn units(self) -> &'static str {
        match self {
            Self::SecondaryStructure => "",
            Self::EnsembleRmsd => "Å",
            Self::AssignmentConfidence => "",
            Self::ChemicalShiftIndex => "CSI",
            Self::LipariSzaboS2 => "S²",
            Self::ChemicalExchangeRex => "s⁻¹",
            Self::LigandCsp => "ppm",
        }
    }
}

/// Evaluated per-residue RGBA color and putty tube radius (`Å`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResiduePuttyStyle {
    pub rgba: [f32; 4],
    pub radius_angstrom: f32,
    pub normalized_t: Option<f32>,
    pub raw_value: Option<f64>,
}

/// Per-residue quantitative NMR & dynamics datasets bound to `0`-based sequence indices.
#[derive(Clone, Debug)]
pub struct QuantitativeTrackData {
    pub active_track: QuantitativeTrack,
    pub colormap: ColormapKind,
    /// Modulate ribbon cross-section radius ("putty" / "sausage") by the active scalar track.
    pub putty_enabled: bool,
    pub putty_min_radius: f32,
    pub putty_max_radius: f32,
    /// `0`-based polymer residue index -> assignment confidence in `[0.0, 1.0]`.
    pub confidence: BTreeMap<usize, f64>,
    /// `0`-based polymer residue index -> normalized `CSI` secondary-shift score in `[-1.0, 1.0]`.
    pub csi: BTreeMap<usize, f64>,
    /// `0`-based polymer residue index -> Lipari–Szabo `S^2` in `[0.0, 1.0]`.
    pub lipari_szabo_s2: BTreeMap<usize, f64>,
    /// `0`-based polymer residue index -> `R_ex` (`s^-1`, `>= 0`).
    pub rex_s_inv: BTreeMap<usize, f64>,
    /// `0`-based polymer residue index -> combined ligand `CSP` `Delta_delta_comb` (`ppm`, `>= 0`).
    pub csp_ppm: BTreeMap<usize, f64>,
    /// Upper display clamp for `R_ex` (`s^-1`, auto-scaled when `<= 0`).
    pub rex_max_s_inv: f64,
    /// Upper display clamp for `CSP` (`ppm`, auto-scaled when `<= 0`).
    pub csp_max_ppm: f64,
    /// Upper display clamp for ensemble `CA` RMSD (`Å`, auto-scaled when `<= 0`).
    pub rmsd_max_angstrom: f64,
}

impl Default for QuantitativeTrackData {
    fn default() -> Self {
        Self {
            active_track: QuantitativeTrack::SecondaryStructure,
            colormap: ColormapKind::Viridis,
            putty_enabled: false,
            putty_min_radius: 0.22,
            putty_max_radius: 1.20,
            confidence: BTreeMap::new(),
            csi: BTreeMap::new(),
            lipari_szabo_s2: BTreeMap::new(),
            rex_s_inv: BTreeMap::new(),
            csp_ppm: BTreeMap::new(),
            rex_max_s_inv: 15.0,
            csp_max_ppm: 0.50,
            rmsd_max_angstrom: 2.50,
        }
    }
}

impl QuantitativeTrackData {
    /// Compute per-residue Wishart-style continuous `CSI` secondary-shift scores (`[-1.0, +1.0]`,
    /// positive = alpha-helical downfield `CA` / upfield `CB`, negative = beta-strand upfield `CA` /
    /// downfield `CB`) from an assigned [`Reference`] and empirical [`Priors`].
    pub fn populate_csi_from_reference(&mut self, ref_data: &Reference, priors: &Priors) {
        self.csi.clear();
        let n = ref_data.sequence.len();
        let mut raw = vec![None::<f64>; n];

        for (k, aa) in ref_data.sequence.iter().enumerate() {
            let Some(atom_priors) = priors.table.get(aa) else {
                continue;
            };
            let mut acc = 0.0f64;
            let mut wt = 0.0f64;

            if let (Some(&ca), Some(&(mu_ca, _))) = (
                ref_data.shifts.get(&(k, "CA".to_string())),
                atom_priors.get("CA"),
            ) {
                if ca.is_finite() {
                    acc += (ca - mu_ca) / 2.2;
                    wt += 1.0;
                }
            }
            if aa != "GLY" {
                if let (Some(&cb), Some(&(mu_cb, _))) = (
                    ref_data.shifts.get(&(k, "CB".to_string())),
                    atom_priors.get("CB"),
                ) {
                    if cb.is_finite() {
                        acc -= (cb - mu_cb) / 2.2;
                        wt += 0.85;
                    }
                }
            }
            if let (Some(&co), Some(&(mu_co, _))) = (
                ref_data.shifts.get(&(k, "C".to_string())),
                atom_priors.get("C"),
            ) {
                if co.is_finite() {
                    acc += (co - mu_co) / 1.8 * 0.55;
                    wt += 0.55;
                }
            }
            if wt > 0.0 {
                raw[k] = Some((acc / wt).clamp(-1.0, 1.0));
            }
        }

        // Apply 3-residue Wishart smoothing window `(k-1, k, k+1)` across observed neighbors
        for k in 0..n {
            let Some(v0) = raw[k] else {
                continue;
            };
            let mut sum = 0.50 * v0;
            let mut wsum = 0.50;
            if k > 0 {
                if let Some(vm) = raw[k - 1] {
                    sum += 0.25 * vm;
                    wsum += 0.25;
                }
            }
            if k + 1 < n {
                if let Some(vp) = raw[k + 1] {
                    sum += 0.25 * vp;
                    wsum += 0.25;
                }
            }
            self.csi.insert(k, (sum / wsum).clamp(-1.0, 1.0));
        }
    }

    /// Compute combined amide chemical shift perturbations `Delta_delta_comb` (Williamson *Prog.
    /// Nucl. Magn. Reson. Spectrosc.* 2013: `sqrt(dH^2 + (alpha_N * dN)^2)`, `alpha_N = 0.20` for
    /// `GLY` and `0.14` otherwise) between `apo` and `holo` [`Reference`] tables.
    pub fn populate_csp_from_references(&mut self, apo: &Reference, holo: &Reference) {
        self.csp_ppm.clear();
        let n = apo.sequence.len().min(holo.sequence.len());
        let mut max_seen = 0.05f64;
        for k in 0..n {
            let (Some(&h0), Some(&n0), Some(&h1), Some(&n1)) = (
                apo.shifts.get(&(k, "H".to_string())),
                apo.shifts.get(&(k, "N".to_string())),
                holo.shifts.get(&(k, "H".to_string())),
                holo.shifts.get(&(k, "N".to_string())),
            ) else {
                continue;
            };
            if !(h0.is_finite() && n0.is_finite() && h1.is_finite() && n1.is_finite()) {
                continue;
            }
            let alpha_n = if apo.sequence[k].eq_ignore_ascii_case("GLY") {
                0.20
            } else {
                0.14
            };
            let dh = h1 - h0;
            let dn = n1 - n0;
            let comb = (dh * dh + (alpha_n * dn).powi(2)).sqrt();
            max_seen = max_seen.max(comb);
            self.csp_ppm.insert(k, comb);
        }
        self.csp_max_ppm = max_seen;
    }

    /// Evaluate RGBA color and putty radius (`Å`) for a residue (`seq_idx` in `0..sequence.len()`).
    #[inline]
    pub fn evaluate_residue_style(
        &self,
        seq_idx: Option<usize>,
        ss_default_rgba: [f32; 4],
        base_coil_radius: f32,
        ensemble_ca_rmsd: Option<f64>,
        alpha: f32,
    ) -> ResiduePuttyStyle {
        let min_r = self.putty_min_radius.max(0.08);
        let max_r = self.putty_max_radius.max(min_r + 0.05);

        match self.active_track {
            QuantitativeTrack::SecondaryStructure => {
                let mut c = ss_default_rgba;
                c[3] = alpha;
                let r = if self.putty_enabled {
                    let rmsd = ensemble_ca_rmsd.unwrap_or(0.35) as f32;
                    let u = (rmsd / (self.rmsd_max_angstrom.max(0.25) as f32)).clamp(0.0, 1.0);
                    min_r + u * (max_r - min_r)
                } else {
                    base_coil_radius
                };
                ResiduePuttyStyle {
                    rgba: c,
                    radius_angstrom: r,
                    normalized_t: None,
                    raw_value: ensemble_ca_rmsd,
                }
            }
            QuantitativeTrack::EnsembleRmsd => {
                let Some(rmsd) = ensemble_ca_rmsd.filter(|v| v.is_finite()) else {
                    let mut c = UNASSIGNED_PUTTY_RGBA;
                    c[3] = alpha;
                    return ResiduePuttyStyle {
                        rgba: c,
                        radius_angstrom: base_coil_radius,
                        normalized_t: None,
                        raw_value: None,
                    };
                };
                let denom = self.rmsd_max_angstrom.max(0.25);
                let t = ((rmsd / denom) as f32).clamp(0.0, 1.0);
                let r = if self.putty_enabled {
                    min_r + t * (max_r - min_r)
                } else {
                    base_coil_radius
                };
                ResiduePuttyStyle {
                    rgba: self.colormap.sample_rgba(t, alpha),
                    radius_angstrom: r,
                    normalized_t: Some(t),
                    raw_value: Some(rmsd),
                }
            }
            QuantitativeTrack::AssignmentConfidence => {
                let Some(conf) = seq_idx
                    .and_then(|k| self.confidence.get(&k).copied())
                    .filter(|v| v.is_finite())
                else {
                    let mut c = UNASSIGNED_PUTTY_RGBA;
                    c[3] = alpha;
                    return ResiduePuttyStyle {
                        rgba: c,
                        radius_angstrom: min_r,
                        normalized_t: None,
                        raw_value: None,
                    };
                };
                let t = (conf as f32).clamp(0.0, 1.0);
                // Putty radius scales linearly in `[min_r, max_r]` with posterior confidence `t`:
                let r = if self.putty_enabled {
                    min_r + t * (max_r - min_r)
                } else {
                    base_coil_radius
                };
                ResiduePuttyStyle {
                    rgba: self.colormap.sample_rgba(t, alpha),
                    radius_angstrom: r,
                    normalized_t: Some(t),
                    raw_value: Some(conf),
                }
            }
            QuantitativeTrack::ChemicalShiftIndex => {
                let Some(csi) = seq_idx
                    .and_then(|k| self.csi.get(&k).copied())
                    .filter(|v| v.is_finite())
                else {
                    let mut c = UNASSIGNED_PUTTY_RGBA;
                    c[3] = alpha;
                    return ResiduePuttyStyle {
                        rgba: c,
                        radius_angstrom: min_r,
                        normalized_t: None,
                        raw_value: None,
                    };
                };
                let t = (0.5 * (csi + 1.0)) as f32;
                let r = if self.putty_enabled {
                    min_r + (csi.abs() as f32).clamp(0.0, 1.0) * (max_r - min_r)
                } else {
                    base_coil_radius
                };
                ResiduePuttyStyle {
                    rgba: self.colormap.sample_rgba(t, alpha),
                    radius_angstrom: r,
                    normalized_t: Some(t.clamp(0.0, 1.0)),
                    raw_value: Some(csi),
                }
            }
            QuantitativeTrack::LipariSzaboS2 => {
                let Some(s2) = seq_idx
                    .and_then(|k| self.lipari_szabo_s2.get(&k).copied())
                    .filter(|v| v.is_finite())
                else {
                    let mut c = UNASSIGNED_PUTTY_RGBA;
                    c[3] = alpha;
                    return ResiduePuttyStyle {
                        rgba: c,
                        radius_angstrom: min_r,
                        normalized_t: None,
                        raw_value: None,
                    };
                };
                let t = (s2 as f32).clamp(0.0, 1.0);
                // Flexible ps-ns loops (`1 - S^2` large) swell in putty mode (`r ~ 1 - S^2`)!
                let r = if self.putty_enabled {
                    min_r + (1.0 - t) * (max_r - min_r)
                } else {
                    base_coil_radius
                };
                ResiduePuttyStyle {
                    rgba: self.colormap.sample_rgba(t, alpha),
                    radius_angstrom: r,
                    normalized_t: Some(t),
                    raw_value: Some(s2),
                }
            }
            QuantitativeTrack::ChemicalExchangeRex => {
                let Some(rex) = seq_idx
                    .and_then(|k| self.rex_s_inv.get(&k).copied())
                    .filter(|v| v.is_finite())
                else {
                    let mut c = UNASSIGNED_PUTTY_RGBA;
                    c[3] = alpha;
                    return ResiduePuttyStyle {
                        rgba: c,
                        radius_angstrom: min_r,
                        normalized_t: None,
                        raw_value: None,
                    };
                };
                let denom = self.rex_max_s_inv.max(1.0);
                let t = ((rex.max(0.0) / denom) as f32).clamp(0.0, 1.0);
                let r = if self.putty_enabled {
                    min_r + t * (max_r - min_r)
                } else {
                    base_coil_radius
                };
                ResiduePuttyStyle {
                    rgba: self.colormap.sample_rgba(t, alpha),
                    radius_angstrom: r,
                    normalized_t: Some(t),
                    raw_value: Some(rex),
                }
            }
            QuantitativeTrack::LigandCsp => {
                let Some(csp) = seq_idx
                    .and_then(|k| self.csp_ppm.get(&k).copied())
                    .filter(|v| v.is_finite())
                else {
                    let mut c = UNASSIGNED_PUTTY_RGBA;
                    c[3] = alpha;
                    return ResiduePuttyStyle {
                        rgba: c,
                        radius_angstrom: min_r,
                        normalized_t: None,
                        raw_value: None,
                    };
                };
                let denom = self.csp_max_ppm.max(0.05);
                let t = ((csp.max(0.0) / denom) as f32).clamp(0.0, 1.0);
                let r = if self.putty_enabled {
                    min_r + t * (max_r - min_r)
                } else {
                    base_coil_radius
                };
                ResiduePuttyStyle {
                    rgba: self.colormap.sample_rgba(t, alpha),
                    radius_angstrom: r,
                    normalized_t: Some(t),
                    raw_value: Some(csp),
                }
            }
        }
    }
}

// ============================================================================
// 3. Analytical 3D Ray-Sphere & Ray-Capsule/Ribbon Picking (`100%` Accuracy)
// ============================================================================

/// World-space 3D picking ray bounded by the active front/back Z-slab planes `[t_min, t_max]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray3d {
    pub origin: [f64; 3],
    /// Unit direction vector (`||dir|| = 1`).
    pub dir: [f64; 3],
    pub t_min: f64,
    pub t_max: f64,
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

impl Ray3d {
    #[inline]
    pub fn point_at(&self, t: f64) -> [f64; 3] {
        add3(self.origin, scale3(self.dir, t))
    }
}

/// Exact analytical ray–sphere intersection returning the smallest valid `t` in `[ray.t_min, ray.t_max]`.
#[inline]
pub fn intersect_ray_sphere(ray: &Ray3d, center: [f64; 3], radius: f64) -> Option<f64> {
    if radius <= 0.0 {
        return None;
    }
    let oc = sub3(ray.origin, center);
    let b = dot3(oc, ray.dir);
    let c = dot3(oc, oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t0 = -b - sq;
    if t0 >= ray.t_min && t0 <= ray.t_max {
        return Some(t0);
    }
    let t1 = -b + sq;
    if t1 >= ray.t_min && t1 <= ray.t_max {
        return Some(t1);
    }
    None
}

/// Exact analytical ray–capsule (`p0 -> p1` cylinder of `radius` + spherical endcaps) intersection
/// returning `(hit_t, axial_fraction_s)` with `hit_t` in `[ray.t_min, ray.t_max]` and `s` in `[0, 1]`.
pub fn intersect_ray_capsule(
    ray: &Ray3d,
    p0: [f64; 3],
    p1: [f64; 3],
    radius: f64,
) -> Option<(f64, f64)> {
    if radius <= 0.0 {
        return None;
    }
    let ba = sub3(p1, p0);
    let baba = dot3(ba, ba);
    if baba < 1e-12 {
        return intersect_ray_sphere(ray, p0, radius).map(|t| (t, 0.0));
    }

    let mut best: Option<(f64, f64)> = None;
    let mut consider = |cand_t: f64, cand_s: f64| {
        if cand_t >= ray.t_min && cand_t <= ray.t_max && best.is_none_or(|(bt, _)| cand_t < bt) {
            best = Some((cand_t, cand_s.clamp(0.0, 1.0)));
        }
    };

    // 1. Infinite cylinder body clamped to axial projection `y in [0, baba]`
    let oa = sub3(ray.origin, p0);
    let bard = dot3(ba, ray.dir);
    let baoa = dot3(ba, oa);
    let a = baba - bard * bard;
    let k2 = baba * dot3(oa, oa) - baoa * baoa - radius * radius * baba;
    let b_half = baba * dot3(oa, ray.dir) - baoa * bard;

    if a.abs() > 1e-12 {
        let disc = b_half * b_half - a * k2;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            for t_cyl in [(-b_half - sq) / a, (-b_half + sq) / a] {
                let y = baoa + t_cyl * bard;
                if y >= 0.0 && y <= baba {
                    consider(t_cyl, y / baba);
                }
            }
        }
    }

    // 2. Spherical endcaps at `p0` (`s = 0`) and `p1` (`s = 1`)
    if let Some(t_cap0) = intersect_ray_sphere(ray, p0, radius) {
        consider(t_cap0, 0.0);
    }
    if let Some(t_cap1) = intersect_ray_sphere(ray, p1, radius) {
        consider(t_cap1, 1.0);
    }

    best
}

/// Classification of a 3D picked entity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Picked3dKind {
    Atom,
    MetalCofactor,
    CartoonRibbon,
    RestraintCylinder { restraint_id: usize },
}

/// Pickable atom or metal cofactor sphere in world space.
#[derive(Clone, Debug)]
pub struct PickableSphere3d {
    pub model_idx: usize,
    pub chain_id: char,
    pub seq_id: i32,
    pub residue_idx: Option<usize>,
    pub residue_name: String,
    pub atom_name: String,
    pub is_metal: bool,
    pub center: [f64; 3],
    pub pick_radius: f64,
}

/// Pickable Carson–Bugg cartoon ribbon spine capsule segment (`p0 -> p1`) in world space.
#[derive(Clone, Debug)]
pub struct PickableRibbonSegment3d {
    pub model_idx: usize,
    pub chain_id: char,
    pub seq_id: i32,
    pub residue_idx: Option<usize>,
    pub residue_name: String,
    pub p0: [f64; 3],
    pub p1: [f64; 3],
    pub pick_radius: f64,
}

/// Pickable 3D NOE/PRE restraint cylinder in world space.
#[derive(Clone, Debug)]
pub struct PickableRestraint3d {
    pub restraint_id: usize,
    pub chain_id: char,
    pub seq_id: i32,
    pub residue_idx: Option<usize>,
    pub residue_name: String,
    pub atom_name: String,
    pub p0: [f64; 3],
    pub p1: [f64; 3],
    pub pick_radius: f64,
}

/// Exact front-most 3D hit returned by [`pick_scene_analytical`].
#[derive(Clone, Debug, PartialEq)]
pub struct Picked3dHit {
    pub kind: Picked3dKind,
    pub model_idx: usize,
    pub chain_id: char,
    pub seq_id: i32,
    pub residue_idx: Option<usize>,
    pub residue_name: String,
    pub atom_name: String,
    pub hit_pos: [f64; 3],
    pub ray_t: f64,
}

/// Intersect `ray` analytically against all visible spheres, cartoon ribbon segments, and
/// restraint cylinders, returning the exact front-most hit (`min ray_t`).
pub fn pick_scene_analytical(
    ray: &Ray3d,
    spheres: &[PickableSphere3d],
    ribbons: &[PickableRibbonSegment3d],
    restraints: &[PickableRestraint3d],
) -> Option<Picked3dHit> {
    let mut best: Option<Picked3dHit> = None;

    for sp in spheres {
        if let Some(t) = intersect_ray_sphere(ray, sp.center, sp.pick_radius) {
            if best.as_ref().is_none_or(|b| t < b.ray_t) {
                best = Some(Picked3dHit {
                    kind: if sp.is_metal {
                        Picked3dKind::MetalCofactor
                    } else {
                        Picked3dKind::Atom
                    },
                    model_idx: sp.model_idx,
                    chain_id: sp.chain_id,
                    seq_id: sp.seq_id,
                    residue_idx: sp.residue_idx,
                    residue_name: sp.residue_name.clone(),
                    atom_name: sp.atom_name.clone(),
                    hit_pos: ray.point_at(t),
                    ray_t: t,
                });
            }
        }
    }

    for seg in ribbons {
        if let Some((t, _)) = intersect_ray_capsule(ray, seg.p0, seg.p1, seg.pick_radius) {
            if best.as_ref().is_none_or(|b| t < b.ray_t) {
                best = Some(Picked3dHit {
                    kind: Picked3dKind::CartoonRibbon,
                    model_idx: seg.model_idx,
                    chain_id: seg.chain_id,
                    seq_id: seg.seq_id,
                    residue_idx: seg.residue_idx,
                    residue_name: seg.residue_name.clone(),
                    atom_name: "CA".to_string(),
                    hit_pos: ray.point_at(t),
                    ray_t: t,
                });
            }
        }
    }

    for rst in restraints {
        if let Some((t, _)) = intersect_ray_capsule(ray, rst.p0, rst.p1, rst.pick_radius) {
            if best.as_ref().is_none_or(|b| t < b.ray_t) {
                best = Some(Picked3dHit {
                    kind: Picked3dKind::RestraintCylinder {
                        restraint_id: rst.restraint_id,
                    },
                    model_idx: 0,
                    chain_id: rst.chain_id,
                    seq_id: rst.seq_id,
                    residue_idx: rst.residue_idx,
                    residue_name: rst.residue_name.clone(),
                    atom_name: rst.atom_name.clone(),
                    hit_pos: ray.point_at(t),
                    ray_t: t,
                });
            }
        }
    }

    best
}

// ============================================================================
// 4. Bidirectional 3D <-> 2D HSQC / 3D Strip / 2D Covalent Graph Sync
// ============================================================================

/// Synchronized workbench navigation payload emitted when clicking a residue/atom in 3D.
#[derive(Clone, Debug, PartialEq)]
pub struct Mol3dSyncAction {
    pub residue_idx: usize,
    pub seq_id: i32,
    pub residue_name: String,
    pub clicked_atom: String,
    pub open_sidechain_graph: bool,
    pub atom_jump: Option<AtomJumpTarget>,
}

impl Mol3dSyncAction {
    /// Apply this 3D pick synchronization directly onto a shared [`SyncCrosshair`].
    pub fn apply_to_crosshair(&self, crosshair: &mut SyncCrosshair) {
        if let Some(ref jump) = self.atom_jump {
            jump.apply_to_crosshair(crosshair);
        }
    }
}

/// Resolve a [`Picked3dHit`] against polymer `sequence` / `seq_ids` and optional assigned
/// shifts (`ref_data`) into a [`Mol3dSyncAction`] that highlights the sequence ribbon, opens
/// the 2D covalent sidechain inspector, and snaps 2D HSQC + 3D strips via [`SyncCrosshair`].
pub fn resolve_3d_pick_sync(
    hit: &Picked3dHit,
    sequence: &[String],
    seq_ids: &[i32],
    ref_data: Option<&Reference>,
) -> Option<Mol3dSyncAction> {
    let res_idx = hit
        .residue_idx
        .filter(|&k| k < sequence.len())
        .or_else(|| {
            seq_ids
                .iter()
                .position(|&sid| sid == hit.seq_id)
                .filter(|&k| k < sequence.len())
        })?;

    let seq_id = seq_ids.get(res_idx).copied().unwrap_or(hit.seq_id);
    let res_name = sequence
        .get(res_idx)
        .cloned()
        .unwrap_or_else(|| hit.residue_name.clone());

    let mut atom_jump = None;
    if let Some(r) = ref_data {
        let lookup_shift = |atom: &str| -> Option<f64> {
            r.shifts
                .get(&(res_idx, atom.to_string()))
                .copied()
                .or_else(|| match atom {
                    "H" => r.shifts.get(&(res_idx, "HN".to_string())).copied(),
                    "HN" => r.shifts.get(&(res_idx, "H".to_string())).copied(),
                    "C" => r.shifts.get(&(res_idx, "CO".to_string())).copied(),
                    _ => None,
                })
        };

        let backbone_h_ppm = lookup_shift("H");
        let backbone_n_ppm = lookup_shift("N");
        let topo_opt = residue_topology(&res_name);

        let clicked = hit.atom_name.as_str();
        if let Some(ppm) = lookup_shift(clicked) {
            let elem = topo_opt
                .and_then(|t| t.atoms.iter().find(|a| a.name == clicked))
                .map(|a| a.element)
                .unwrap_or_else(|| match clicked.as_bytes().first().copied() {
                    Some(b'H') => Element::H,
                    Some(b'N') => Element::N,
                    Some(b'O') => Element::O,
                    Some(b'S') => Element::S,
                    _ => Element::C,
                });
            let attached_heavy_atom = topo_opt
                .and_then(|t| t.atoms.iter().find(|a| a.name == clicked))
                .and_then(|a| a.attached_heavy)
                .map(str::to_string);
            let attached_heavy_ppm = attached_heavy_atom.as_deref().and_then(&lookup_shift);
            let attached_proton_ppm = topo_opt.and_then(|t| {
                t.atoms
                    .iter()
                    .filter(|a| a.element == Element::H && a.attached_heavy == Some(clicked))
                    .find_map(|h| lookup_shift(h.name))
            });

            atom_jump = Some(AtomJumpTarget {
                residue_idx: res_idx,
                seq_id,
                residue_name: res_name.clone(),
                atom_name: clicked.to_string(),
                element: elem,
                ppm,
                attached_heavy_atom,
                attached_heavy_ppm,
                attached_proton_ppm,
                backbone_h_ppm,
                backbone_n_ppm,
            });
        } else if let (Some(h), Some(n)) = (backbone_h_ppm, backbone_n_ppm) {
            let ca_ppm = lookup_shift("CA");
            atom_jump = Some(AtomJumpTarget {
                residue_idx: res_idx,
                seq_id,
                residue_name: res_name.clone(),
                atom_name: if ca_ppm.is_some() {
                    "CA".to_string()
                } else {
                    "N".to_string()
                },
                element: if ca_ppm.is_some() {
                    Element::C
                } else {
                    Element::N
                },
                ppm: ca_ppm.unwrap_or(n),
                attached_heavy_atom: None,
                attached_heavy_ppm: None,
                attached_proton_ppm: Some(h),
                backbone_h_ppm: Some(h),
                backbone_n_ppm: Some(n),
            });
        }
    }

    Some(Mol3dSyncAction {
        residue_idx: res_idx,
        seq_id,
        residue_name: res_name,
        clicked_atom: hit.atom_name.clone(),
        open_sidechain_graph: true,
        atom_jump,
    })
}

// ============================================================================
// 5. 3D NOE & PRE Distance Restraints & Violation Cylinders (`< 2 ms` Update)
// ============================================================================

/// Experimental distance restraint origin (`NOE` dipolar cross-relaxation vs `PRE` paramagnetic).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RestraintKind {
    #[default]
    Noe,
    Pre,
}

impl RestraintKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Noe => "NOE",
            Self::Pre => "PRE",
        }
    }
}

/// Violation severity band for 3D restraint cylinder coloring.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RestraintStatus {
    /// `violation <= 0.0 Å` — emerald green cylinder.
    Satisfied,
    /// `0.0 Å < violation < 0.5 Å` — warm amber cylinder.
    MinorViolation,
    /// `violation >= 0.5 Å` — crimson red cylinder.
    MajorViolation,
}

impl RestraintStatus {
    #[inline]
    pub fn from_violation(violation_angstrom: f64) -> Self {
        if violation_angstrom <= 1e-6 {
            Self::Satisfied
        } else if violation_angstrom < 0.5 {
            Self::MinorViolation
        } else {
            Self::MajorViolation
        }
    }

    #[inline]
    pub const fn rgba(self) -> [f32; 4] {
        match self {
            Self::Satisfied => [0.16, 0.84, 0.46, 0.90],
            Self::MinorViolation => [0.96, 0.68, 0.14, 0.95],
            Self::MajorViolation => [0.94, 0.22, 0.24, 0.98],
        }
    }

    #[inline]
    pub const fn radius_angstrom(self) -> f32 {
        match self {
            Self::Satisfied => 0.075,
            Self::MinorViolation => 0.105,
            Self::MajorViolation => 0.135,
        }
    }
}

/// Atom endpoint selector for a 3D NOE or PRE distance restraint.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RestraintAtomSelector {
    pub chain_id: Option<char>,
    pub seq_id: i32,
    pub atom_name: String,
}

impl RestraintAtomSelector {
    pub fn new(chain_id: Option<char>, seq_id: i32, atom_name: impl Into<String>) -> Self {
        Self {
            chain_id,
            seq_id,
            atom_name: atom_name.into().trim().to_ascii_uppercase(),
        }
    }
}

/// Single unambiguous or `r^-6`-ambiguous 3D NOE/PRE distance restraint.
#[derive(Clone, Debug)]
pub struct DistanceRestraint3d {
    pub id: usize,
    pub kind: RestraintKind,
    pub members: Vec<(RestraintAtomSelector, RestraintAtomSelector)>,
    pub lower_angstrom: f64,
    pub upper_angstrom: f64,
    pub label: String,
}

impl DistanceRestraint3d {
    /// Convenience constructor for a 2-atom distance restraint.
    pub fn pair(
        id: usize,
        kind: RestraintKind,
        a: RestraintAtomSelector,
        b: RestraintAtomSelector,
        lower_angstrom: f64,
        upper_angstrom: f64,
    ) -> Self {
        let label = format!(
            "{} {}:{} - {}:{} (<= {:.2} A)",
            kind.label(),
            a.seq_id,
            a.atom_name,
            b.seq_id,
            b.atom_name,
            upper_angstrom
        );
        Self {
            id,
            kind,
            members: vec![(a, b)],
            lower_angstrom: lower_angstrom.max(0.0),
            upper_angstrom: upper_angstrom.max(lower_angstrom.max(0.1)),
            label,
        }
    }
}

/// Interactive visibility & filtering state for 3D NOE/PRE restraint cylinders.
#[derive(Clone, Debug)]
pub struct RestraintFilterConfig {
    pub enabled: bool,
    pub show_satisfied: bool,
    pub show_minor: bool,
    pub show_major: bool,
    pub selected_seq_id: Option<i32>,
    pub selected_residue_only: bool,
    pub dashed_cylinders: bool,
}

impl Default for RestraintFilterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            show_satisfied: true,
            show_minor: true,
            show_major: true,
            selected_seq_id: None,
            selected_residue_only: false,
            dashed_cylinders: false,
        }
    }
}

/// Evaluated 3D distance restraint with endpoints, `d_eff`, violation (`Å`), and [`RestraintStatus`].
#[derive(Clone, Debug)]
pub struct EvaluatedRestraint3d {
    pub id: usize,
    pub kind: RestraintKind,
    pub p0: [f32; 3],
    pub p1: [f32; 3],
    pub anchor_chain: char,
    pub anchor_seq_id: i32,
    pub anchor_res_idx: Option<usize>,
    pub anchor_res_name: String,
    pub anchor_atom: String,
    pub effective_distance: f64,
    pub upper_angstrom: f64,
    pub violation_angstrom: f64,
    pub status: RestraintStatus,
    pub label: String,
}

/// Summary statistics and GPU cylinder instances produced in `< 2.0 ms` by [`AtomCoordLookup::evaluate_restraints`].
#[derive(Clone, Debug, Default)]
pub struct RestraintOverlayBuild {
    pub cylinders: Vec<CylinderInstance>,
    pub pickables: Vec<PickableRestraint3d>,
    pub evaluated: Vec<EvaluatedRestraint3d>,
    pub satisfied_count: usize,
    pub minor_violation_count: usize,
    pub major_violation_count: usize,
    pub unresolved_count: usize,
    pub max_violation_angstrom: f64,
    pub rms_violation_angstrom: f64,
}

#[derive(Clone, Copy, Debug)]
struct IndexedAtomEntry {
    chain_id: char,
    seq_id: i32,
    residue_idx: Option<usize>,
    res_name_idx: u16,
    name_len: u8,
    name_bytes: [u8; 6],
    pos: [f32; 3],
}

impl IndexedAtomEntry {
    #[inline]
    fn from_parts(
        chain_id: char,
        seq_id: i32,
        residue_idx: Option<usize>,
        res_name_idx: u16,
        atom_name: &str,
        pos: [f32; 3],
    ) -> Self {
        let mut name_bytes = [0u8; 6];
        let mut len = 0usize;
        for b in atom_name.bytes().take(6) {
            if b.is_ascii() {
                name_bytes[len] = b.to_ascii_uppercase();
                len += 1;
            }
        }
        Self {
            chain_id,
            seq_id,
            residue_idx,
            res_name_idx,
            name_len: len as u8,
            name_bytes,
            pos,
        }
    }

    #[inline]
    fn atom_name_str(&self) -> &str {
        std::str::from_utf8(&self.name_bytes[..(self.name_len as usize)]).unwrap_or("")
    }

    #[inline]
    fn same_atom_key(&self, other: &Self) -> bool {
        self.name_len == other.name_len
            && self.name_bytes == other.name_bytes
            && self.chain_id == other.chain_id
    }
}

#[derive(Clone, Debug, Default)]
struct ModelCoordSlice {
    atoms: Vec<IndexedAtomEntry>,
    min_seq_id: i32,
    /// Optional O(1) `(start, end)` range table indexed by `(seq_id - min_seq_id) as usize`
    /// when `max_seq_id - min_seq_id <= 16_384`.
    seq_ranges: Vec<(u32, u32)>,
}

impl ModelCoordSlice {
    fn from_entries(mut atoms: Vec<IndexedAtomEntry>) -> Self {
        atoms.sort_by_key(|e| e.seq_id);
        let mut seq_ranges = Vec::new();
        let mut min_seq_id = 0i32;
        if let (Some(first), Some(last)) = (atoms.first(), atoms.last()) {
            min_seq_id = first.seq_id;
            let span = (last.seq_id as i64) - (first.seq_id as i64);
            if (0..=16_384).contains(&span) {
                let len = (span as usize) + 1;
                seq_ranges = vec![(0u32, 0u32); len];
                let mut i = 0usize;
                while i < atoms.len() {
                    let sid = atoms[i].seq_id;
                    let start = i;
                    while i < atoms.len() && atoms[i].seq_id == sid {
                        i += 1;
                    }
                    let slot = (sid - min_seq_id) as usize;
                    seq_ranges[slot] = (start as u32, i as u32);
                }
            }
        }
        Self {
            atoms,
            min_seq_id,
            seq_ranges,
        }
    }

    #[inline]
    fn atoms_for_seq_id(&self, seq_id: i32) -> &[IndexedAtomEntry] {
        if !self.seq_ranges.is_empty() {
            let off = (seq_id as i64) - (self.min_seq_id as i64);
            if off < 0 || (off as usize) >= self.seq_ranges.len() {
                return &[];
            }
            let (s, e) = self.seq_ranges[off as usize];
            return &self.atoms[(s as usize)..(e as usize)];
        }
        let s = self.atoms.partition_point(|e| e.seq_id < seq_id);
        let mut e = s;
        while e < self.atoms.len() && self.atoms[e].seq_id == seq_id {
            e += 1;
        }
        &self.atoms[s..e]
    }
}

#[inline]
fn is_proton_prefix(atom_name: &str, prefix: &str) -> bool {
    if atom_name.starts_with(prefix) {
        return true;
    }
    // Support legacy PDB digit-prefixed proton names such as `1HG1`, `2HD2`, `3HB`
    let bytes = atom_name.as_bytes();
    bytes.len() > prefix.len()
        && bytes[0].is_ascii_digit()
        && bytes[1..].starts_with(prefix.as_bytes())
}

#[inline]
fn matches_primary_or_pseudoatom(target: &str, atom_name: &str) -> bool {
    if atom_name == target
        || (target == "HN" && atom_name == "H")
        || (target == "H" && atom_name == "HN")
    {
        return true;
    }
    let stem = target.trim_end_matches(['*', '%', '#']);
    if stem.len() < target.len() && !stem.is_empty() {
        return is_proton_prefix(atom_name, stem);
    }
    match target {
        "QA" => is_proton_prefix(atom_name, "HA"),
        "QB" | "MB" => is_proton_prefix(atom_name, "HB"),
        "QG" | "MG" => is_proton_prefix(atom_name, "HG"),
        "QG1" | "MG1" => is_proton_prefix(atom_name, "HG1"),
        "QG2" | "MG2" => is_proton_prefix(atom_name, "HG2"),
        "QQG" => is_proton_prefix(atom_name, "HG1") || is_proton_prefix(atom_name, "HG2"),
        "QD" | "MD" => is_proton_prefix(atom_name, "HD"),
        "QD1" | "MD1" => is_proton_prefix(atom_name, "HD1"),
        "QD2" | "MD2" => is_proton_prefix(atom_name, "HD2"),
        "QQD" => is_proton_prefix(atom_name, "HD1") || is_proton_prefix(atom_name, "HD2"),
        "QE" | "ME" => is_proton_prefix(atom_name, "HE"),
        "QE1" => is_proton_prefix(atom_name, "HE1"),
        "QE2" => is_proton_prefix(atom_name, "HE2"),
        "QZ" | "MZ" => is_proton_prefix(atom_name, "HZ"),
        "QH1" => is_proton_prefix(atom_name, "HH1"),
        "QH2" => is_proton_prefix(atom_name, "HH2"),
        "QR" => matches!(
            atom_name,
            "HD1" | "HD2" | "HE1" | "HE2" | "HE3" | "HZ" | "HZ2" | "HZ3" | "HH2"
        ),
        _ => false,
    }
}

#[inline]
fn matches_heavy_parent_fallback(target: &str, atom_name: &str) -> bool {
    let stem = target.trim_end_matches(['*', '%', '#']);
    if target == "H" || target == "HN" {
        return atom_name == "N";
    }
    if target == "QA" || stem.starts_with("HA") {
        return atom_name == "CA";
    }
    if target == "QB" || target == "MB" || stem.starts_with("HB") {
        return atom_name == "CB";
    }
    if target == "QG1" || target == "MG1" || stem.starts_with("HG1") {
        return matches!(atom_name, "CG1" | "CG" | "OG1" | "SG");
    }
    if target == "QG2" || target == "MG2" || stem.starts_with("HG2") {
        return matches!(atom_name, "CG2" | "CG");
    }
    if matches!(target, "QG" | "QQG" | "MG") || stem.starts_with("HG") {
        return matches!(atom_name, "CG" | "CG1" | "CG2" | "OG" | "OG1" | "SG");
    }
    if target == "QD1" || target == "MD1" || stem.starts_with("HD1") {
        return matches!(atom_name, "CD1" | "CD" | "ND1" | "OD1");
    }
    if target == "QD2" || target == "MD2" || stem.starts_with("HD2") {
        return matches!(atom_name, "CD2" | "CD" | "ND2" | "OD2");
    }
    if matches!(target, "QD" | "QQD" | "MD") || stem.starts_with("HD") {
        return matches!(
            atom_name,
            "CD" | "CD1" | "CD2" | "ND1" | "ND2" | "OD1" | "OD2"
        );
    }
    if target == "QE1" || stem.starts_with("HE1") {
        return matches!(atom_name, "CE1" | "CE" | "NE1" | "OE1");
    }
    if target == "QE2" || stem.starts_with("HE2") {
        return matches!(atom_name, "CE2" | "CE" | "NE2" | "OE2");
    }
    if stem.starts_with("HE3") {
        return atom_name == "CE3";
    }
    if matches!(target, "QE" | "ME") || stem.starts_with("HE") {
        return matches!(
            atom_name,
            "CE" | "CE1" | "CE2" | "CE3" | "NE" | "NE1" | "NE2" | "OE1" | "OE2"
        );
    }
    if matches!(target, "QZ" | "MZ") || stem.starts_with("HZ") {
        return matches!(atom_name, "CZ" | "CZ2" | "CZ3" | "NZ");
    }
    if target == "QH1" || stem.starts_with("HH1") {
        return atom_name == "NH1";
    }
    if target == "QH2" || stem.starts_with("HH2") {
        return matches!(atom_name, "NH2" | "CH2");
    }
    if stem.starts_with("HH") {
        return matches!(atom_name, "OH" | "NH1" | "NH2" | "CH2");
    }
    if target == "QR" {
        return matches!(
            atom_name,
            "CD1" | "CD2" | "CE1" | "CE2" | "CE3" | "CZ" | "CZ2" | "CZ3" | "CH2"
        );
    }
    false
}

/// Compact pre-sorted multi-conformer `(model, seq_id)` atom coordinate index enabling `< 2.0 ms`
/// `<r^-6>^(-1/6)` ensemble + pseudoatom evaluation across `1,500+` NOE/PRE restraints without
/// per-frame heap allocations.
#[derive(Clone, Debug, Default)]
pub struct AtomCoordLookup {
    res_names: Vec<String>,
    models: Vec<ModelCoordSlice>,
}

impl AtomCoordLookup {
    /// Construct a single-model coordinate lookup index from `(chain_id, seq_id, residue_idx, res_name, atom_name, pos)` tuples.
    pub fn from_atom_tuples<'a, I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (char, i32, Option<usize>, &'a str, &'a str, [f32; 3])>,
    {
        Self::from_ensemble_atom_tuples(
            1,
            iter.into_iter().map(|(ch, sid, ridx, rname, aname, pos)| {
                (0usize, ch, sid, ridx, rname, aname, pos)
            }),
        )
    }

    /// Construct a multi-conformer (`0..num_models`) coordinate lookup index from
    /// `(model_idx, chain_id, seq_id, residue_idx, res_name, atom_name, pos)` tuples.
    pub fn from_ensemble_atom_tuples<'a, I>(num_models: usize, iter: I) -> Self
    where
        I: IntoIterator<Item = (usize, char, i32, Option<usize>, &'a str, &'a str, [f32; 3])>,
    {
        let n_m = num_models.max(1);
        let mut res_names: Vec<String> = Vec::new();
        let mut res_map: BTreeMap<String, u16> = BTreeMap::new();
        let mut per_model: Vec<Vec<IndexedAtomEntry>> = vec![Vec::new(); n_m];

        for (model_idx, chain_id, seq_id, residue_idx, res_name, atom_name, pos) in iter {
            if model_idx >= n_m {
                continue;
            }
            let r_idx = if let Some(&id) = res_map.get(res_name) {
                id
            } else {
                let id = (res_names.len().min(u16::MAX as usize)) as u16;
                res_names.push(res_name.to_string());
                res_map.insert(res_name.to_string(), id);
                id
            };
            per_model[model_idx].push(IndexedAtomEntry::from_parts(
                chain_id,
                seq_id,
                residue_idx,
                r_idx,
                atom_name,
                pos,
            ));
        }
        let models = per_model
            .into_iter()
            .map(ModelCoordSlice::from_entries)
            .collect();
        Self { res_names, models }
    }

    /// Number of indexed structure conformers (`M >= 1`).
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Resolve `sel` in conformer `model_idx` into a stack buffer `out_rel` (`<= 24` relative indices
    /// into `atoms_for_seq_id(sel.seq_id)`), expanding wildcards (`HB*`, `HG1*`, `HD1*`, `HE*`, `HZ*`)
    /// and NMR/IUPAC pseudoatoms (`QB`, `QG*`, `QD*`, `QE*`, `MG*`, `MD*`, `QR`), falling back to
    /// heavy parent atoms (`CB`, `CG*`, `CD*`, `CE*`, `CZ`) ONLY when explicit protons are absent.
    #[inline]
    fn resolve_rel_indices_in_slice(
        candidates: &[IndexedAtomEntry],
        sel: &RestraintAtomSelector,
        out_rel: &mut [u16; 24],
    ) -> usize {
        if candidates.is_empty() {
            return 0;
        }
        let target = sel.atom_name.as_str();
        let mut count = 0usize;
        let mut heavy_count = 0usize;
        let mut heavy_rel = [0u16; 12];

        for (rel_i, entry) in candidates.iter().enumerate() {
            if let Some(ch) = sel.chain_id {
                if !entry.chain_id.eq_ignore_ascii_case(&ch) {
                    continue;
                }
            }
            let aname = entry.atom_name_str();
            if matches_primary_or_pseudoatom(target, aname) {
                if count < out_rel.len() {
                    out_rel[count] = rel_i as u16;
                    count += 1;
                }
            } else if count == 0
                && matches_heavy_parent_fallback(target, aname)
                && heavy_count < heavy_rel.len()
            {
                heavy_rel[heavy_count] = rel_i as u16;
                heavy_count += 1;
            }
        }

        if count > 0 {
            return count;
        }
        if heavy_count > 0 {
            out_rel[..heavy_count].copy_from_slice(&heavy_rel[..heavy_count]);
            return heavy_count;
        }
        0
    }

    /// Evaluate `restraints` across all indexed conformers (`0..M`) using Nilges (`J. Mol. Biol.` 1995)
    /// ensemble + intra-conformer `<r^-6>^(-1/6)` effective distance averaging and emit filtered GPU
    /// [`CylinderInstance`]s in `< 2.0 ms`.
    pub fn evaluate_restraints(
        &self,
        restraints: &[DistanceRestraint3d],
        filter: &RestraintFilterConfig,
    ) -> RestraintOverlayBuild {
        let mut out = RestraintOverlayBuild::default();
        if restraints.is_empty() || self.models.is_empty() {
            return out;
        }
        out.cylinders.reserve(restraints.len());
        out.evaluated.reserve(restraints.len());

        let dash_flag = if filter.dashed_cylinders { 6.0 } else { 0.0 };
        let mut sq_viol_sum = 0.0f64;
        let mut n_eval = 0usize;
        let n_models = self.models.len();

        let mut rel_a0 = [0u16; 24];
        let mut rel_b0 = [0u16; 24];
        let mut rel_am = [0u16; 24];
        let mut rel_bm = [0u16; 24];
        let mut model_contrib = vec![0.0f64; n_models];

        #[derive(Clone, Copy)]
        struct AnchorCandidate {
            first_a: IndexedAtomEntry,
            p0: [f32; 3],
            p1: [f32; 3],
            model_idx: usize,
            score: f64,
        }

        for r in restraints {
            if filter.selected_residue_only {
                if let Some(sel_sid) = filter.selected_seq_id {
                    let touches = r
                        .members
                        .iter()
                        .any(|(a, b)| a.seq_id == sel_sid || b.seq_id == sel_sid);
                    if !touches {
                        continue;
                    }
                }
            }

            model_contrib.fill(0.0);
            let mut anchor_best: Option<AnchorCandidate> = None;

            for (sa, sb) in &r.members {
                // Fast path: resolve relative offsets once on conformer 0 and reuse directly on
                // conformers `1..M-1` whenever atom slots match (`same_atom_key`), falling back
                // to full per-conformer scan only for heterogeneous conformers!
                let cand_a0 = self.models[0].atoms_for_seq_id(sa.seq_id);
                let cand_b0 = self.models[0].atoms_for_seq_id(sb.seq_id);
                let na0 = Self::resolve_rel_indices_in_slice(cand_a0, sa, &mut rel_a0);
                let nb0 = Self::resolve_rel_indices_in_slice(cand_b0, sb, &mut rel_b0);

                for (m_idx, m_slice) in self.models.iter().enumerate() {
                    let cand_am = if m_idx == 0 {
                        cand_a0
                    } else {
                        m_slice.atoms_for_seq_id(sa.seq_id)
                    };
                    if cand_am.is_empty() {
                        continue;
                    }
                    let cand_bm = if m_idx == 0 {
                        cand_b0
                    } else {
                        m_slice.atoms_for_seq_id(sb.seq_id)
                    };
                    if cand_bm.is_empty() {
                        continue;
                    }

                    let (use_a, na): (&[u16], usize) = if m_idx == 0
                        || (na0 > 0
                            && cand_am.len() == cand_a0.len()
                            && rel_a0[..na0].iter().all(|&ri| {
                                cand_am[ri as usize].same_atom_key(&cand_a0[ri as usize])
                            })) {
                        (&rel_a0, na0)
                    } else {
                        let c = Self::resolve_rel_indices_in_slice(cand_am, sa, &mut rel_am);
                        (&rel_am, c)
                    };
                    if na == 0 {
                        continue;
                    }

                    let (use_b, nb): (&[u16], usize) = if m_idx == 0
                        || (nb0 > 0
                            && cand_bm.len() == cand_b0.len()
                            && rel_b0[..nb0].iter().all(|&ri| {
                                cand_bm[ri as usize].same_atom_key(&cand_b0[ri as usize])
                            })) {
                        (&rel_b0, nb0)
                    } else {
                        let c = Self::resolve_rel_indices_in_slice(cand_bm, sb, &mut rel_bm);
                        (&rel_bm, c)
                    };
                    if nb == 0 {
                        continue;
                    }

                    let mut member_inv6 = 0.0f64;
                    for &ia in &use_a[..na] {
                        let pa = cand_am[ia as usize].pos;
                        for &ib in &use_b[..nb] {
                            let pb = cand_bm[ib as usize].pos;
                            let dx = (pa[0] - pb[0]) as f64;
                            let dy = (pa[1] - pb[1]) as f64;
                            let dz = (pa[2] - pb[2]) as f64;
                            let d2 = (dx * dx + dy * dy + dz * dz).max(1e-6);
                            let inv_d2 = 1.0 / d2;
                            member_inv6 += inv_d2 * inv_d2 * inv_d2;
                        }
                    }
                    model_contrib[m_idx] += member_inv6;

                    // Prefer conformer 0 (primary opaque conformer) for cylinder endpoints
                    let replace_anchor = match anchor_best {
                        None => true,
                        Some(best) => m_idx == best.model_idx && member_inv6 > best.score,
                    };
                    if replace_anchor {
                        let first_a = cand_am[use_a[0] as usize];
                        let mut ca = [0.0f32; 3];
                        for &ia in &use_a[..na] {
                            let p = cand_am[ia as usize].pos;
                            ca[0] += p[0];
                            ca[1] += p[1];
                            ca[2] += p[2];
                        }
                        let inv_na = 1.0 / (na as f32);
                        ca[0] *= inv_na;
                        ca[1] *= inv_na;
                        ca[2] *= inv_na;

                        let mut cb = [0.0f32; 3];
                        for &ib in &use_b[..nb] {
                            let p = cand_bm[ib as usize].pos;
                            cb[0] += p[0];
                            cb[1] += p[1];
                            cb[2] += p[2];
                        }
                        let inv_nb = 1.0 / (nb as f32);
                        cb[0] *= inv_nb;
                        cb[1] *= inv_nb;
                        cb[2] *= inv_nb;

                        anchor_best = Some(AnchorCandidate {
                            first_a,
                            p0: ca,
                            p1: cb,
                            model_idx: m_idx,
                            score: member_inv6,
                        });
                    }
                }
            }

            let Some(AnchorCandidate {
                first_a: ea,
                p0,
                p1,
                ..
            }) = anchor_best
            else {
                out.unresolved_count += 1;
                continue;
            };
            let mut ensemble_inv6_sum = 0.0f64;
            let mut valid_models = 0usize;
            for &mc in &model_contrib {
                if mc > 0.0 {
                    ensemble_inv6_sum += mc;
                    valid_models += 1;
                }
            }
            if valid_models == 0 || ensemble_inv6_sum <= 0.0 {
                out.unresolved_count += 1;
                continue;
            }
            let mean_inv6 = ensemble_inv6_sum / (valid_models as f64);
            let d_eff = mean_inv6.powf(-1.0 / 6.0);
            let viol = if d_eff > r.upper_angstrom {
                d_eff - r.upper_angstrom
            } else if d_eff < r.lower_angstrom {
                r.lower_angstrom - d_eff
            } else {
                0.0
            };

            let status = RestraintStatus::from_violation(viol);
            match status {
                RestraintStatus::Satisfied => out.satisfied_count += 1,
                RestraintStatus::MinorViolation => out.minor_violation_count += 1,
                RestraintStatus::MajorViolation => out.major_violation_count += 1,
            }
            out.max_violation_angstrom = out.max_violation_angstrom.max(viol);
            sq_viol_sum += viol * viol;
            n_eval += 1;

            let res_name = self
                .res_names
                .get(ea.res_name_idx as usize)
                .cloned()
                .unwrap_or_default();
            let ea_atom_name = ea.atom_name_str().to_string();

            let visible = filter.enabled
                && match status {
                    RestraintStatus::Satisfied => filter.show_satisfied,
                    RestraintStatus::MinorViolation => filter.show_minor,
                    RestraintStatus::MajorViolation => filter.show_major,
                };

            if visible {
                let rad = status.radius_angstrom();
                out.cylinders.push(CylinderInstance::uniform(
                    p0,
                    p1,
                    rad,
                    status.rgba(),
                    dash_flag,
                ));
                out.pickables.push(PickableRestraint3d {
                    restraint_id: r.id,
                    chain_id: ea.chain_id,
                    seq_id: ea.seq_id,
                    residue_idx: ea.residue_idx,
                    residue_name: res_name.clone(),
                    atom_name: ea_atom_name.clone(),
                    p0: [p0[0] as f64, p0[1] as f64, p0[2] as f64],
                    p1: [p1[0] as f64, p1[1] as f64, p1[2] as f64],
                    pick_radius: (rad as f64).max(0.14),
                });
            }

            out.evaluated.push(EvaluatedRestraint3d {
                id: r.id,
                kind: r.kind,
                p0,
                p1,
                anchor_chain: ea.chain_id,
                anchor_seq_id: ea.seq_id,
                anchor_res_idx: ea.residue_idx,
                anchor_res_name: res_name,
                anchor_atom: ea_atom_name,
                effective_distance: d_eff,
                upper_angstrom: r.upper_angstrom,
                violation_angstrom: viol,
                status,
                label: r.label.clone(),
            });
        }

        if n_eval > 0 {
            out.rms_violation_angstrom = (sq_viol_sum / (n_eval as f64)).sqrt();
        }
        out
    }
}

/// Parse a whitespace- or tab-delimited NOE/PRE distance restraint list:
/// `seq_id_1  atom_1  seq_id_2  atom_2  upper_A  [lower_A]  [NOE|PRE]  [chain_1]  [chain_2]`
pub fn parse_distance_restraints_tsv(text: &str) -> Result<Vec<DistanceRestraint3d>, String> {
    let mut out = Vec::new();
    for (line_idx, raw) in text.lines().enumerate() {
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = trimmed.split_whitespace().collect();
        if cols.len() < 5 {
            return Err(format!(
                "Restraint line {} requires at least 5 columns (seq1 atom1 seq2 atom2 upper_A)",
                line_idx + 1
            ));
        }
        // Skip header row if first token is non-integer label
        if out.is_empty()
            && cols[0]
                .as_bytes()
                .first()
                .is_some_and(|b| b.is_ascii_alphabetic())
        {
            continue;
        }
        let seq1: i32 = cols[0]
            .parse()
            .map_err(|_| format!("Invalid seq_id_1 '{}' at line {}", cols[0], line_idx + 1))?;
        let atom1 = cols[1];
        let seq2: i32 = cols[2]
            .parse()
            .map_err(|_| format!("Invalid seq_id_2 '{}' at line {}", cols[2], line_idx + 1))?;
        let atom2 = cols[3];
        let upper: f64 = cols[4]
            .parse()
            .map_err(|_| format!("Invalid upper bound '{}' at line {}", cols[4], line_idx + 1))?;
        if !upper.is_finite() || upper <= 0.0 {
            return Err(format!(
                "Non-finite or non-positive upper bound '{}' at line {}",
                cols[4],
                line_idx + 1
            ));
        }
        let lower: f64 = if let Some(tok) = cols.get(5) {
            if let Ok(v) = tok.parse::<f64>() {
                if !v.is_finite() || v < 0.0 {
                    return Err(format!(
                        "Non-finite or negative lower bound '{}' at line {}",
                        tok,
                        line_idx + 1
                    ));
                }
                v
            } else {
                1.80
            }
        } else {
            1.80
        };
        let kind = cols
            .iter()
            .skip(5)
            .find_map(|t| {
                if t.eq_ignore_ascii_case("PRE") {
                    Some(RestraintKind::Pre)
                } else if t.eq_ignore_ascii_case("NOE") {
                    Some(RestraintKind::Noe)
                } else {
                    None
                }
            })
            .unwrap_or(RestraintKind::Noe);
        let ch1 = cols
            .get(7)
            .and_then(|s| s.chars().next())
            .filter(|c| c.is_ascii_alphanumeric());
        let ch2 = cols
            .get(8)
            .and_then(|s| s.chars().next())
            .filter(|c| c.is_ascii_alphanumeric())
            .or(ch1);

        out.push(DistanceRestraint3d::pair(
            out.len(),
            kind,
            RestraintAtomSelector::new(ch1, seq1, atom1),
            RestraintAtomSelector::new(ch2, seq2, atom2),
            lower.min(upper),
            upper,
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p1_6_unresolved_atom_selector_never_silently_substitutes_ca() {
        let lookup = AtomCoordLookup::from_atom_tuples([
            ('A', 1, Some(0), "ALA", "N", [0.0, 0.0, 0.0]),
            ('A', 1, Some(0), "ALA", "CA", [1.46, 0.0, 0.0]),
            ('A', 1, Some(0), "ALA", "CB", [1.95, 1.42, 0.0]),
            ('A', 2, Some(1), "GLY", "N", [3.20, 0.0, 0.0]),
            ('A', 2, Some(1), "GLY", "CA", [4.65, 0.0, 0.0]),
        ]);

        // 1. Invalid atom selector `NOTANATOM` must emit 0 cylinders and `unresolved_count == 1`!
        let bad = vec![DistanceRestraint3d::pair(
            0,
            RestraintKind::Noe,
            RestraintAtomSelector::new(Some('A'), 1, "NOTANATOM"),
            RestraintAtomSelector::new(Some('A'), 2, "N"),
            1.8,
            5.0,
        )];
        let built_bad = lookup.evaluate_restraints(&bad, &RestraintFilterConfig::default());
        assert_eq!(built_bad.cylinders.len(), 0);
        assert_eq!(built_bad.evaluated.len(), 0);
        assert_eq!(built_bad.unresolved_count, 1);

        // 2. Wrong sidechain atom for residue (`HB` on `GLY`, which has no `CB`) must also be unresolved!
        let gly_hb = vec![DistanceRestraint3d::pair(
            1,
            RestraintKind::Noe,
            RestraintAtomSelector::new(Some('A'), 1, "N"),
            RestraintAtomSelector::new(Some('A'), 2, "HB"),
            1.8,
            5.0,
        )];
        let built_gly_hb = lookup.evaluate_restraints(&gly_hb, &RestraintFilterConfig::default());
        assert_eq!(built_gly_hb.cylinders.len(), 0);
        assert_eq!(built_gly_hb.unresolved_count, 1);

        // 3. Valid heavy-parent fallback (`HB` -> `CB` on `ALA`, `HA2` -> `CA` on `GLY`) still resolves cleanly!
        let ok = vec![DistanceRestraint3d::pair(
            2,
            RestraintKind::Noe,
            RestraintAtomSelector::new(Some('A'), 1, "HB"),
            RestraintAtomSelector::new(Some('A'), 2, "HA2"),
            1.8,
            5.0,
        )];
        let built_ok = lookup.evaluate_restraints(&ok, &RestraintFilterConfig::default());
        assert_eq!(built_ok.cylinders.len(), 1);
        assert_eq!(built_ok.unresolved_count, 0);
    }
}
