//! High-resolution 1D Lorentzian/Voigt NMR spectrum synthesis, integration, multiplet peak picking,
//! and JCAMP-DX (`.jdx` / `.dx`) 1D NMR & MS reader/writer.
//!
//! Provenance & Licence:
//! Ported/adapted from Octet-Draw by Sam Ellis (<https://github.com/SamChemMan/Octet-Draw>, Apache-2.0)
//! and published chemical shift additivity / IUPAC nomenclature rules:
//! - McDonald, R. S.; Wilks, P. A. *Appl. Spectrosc.* **1988**, 42, 151–162 (JCAMP-DX v4.24).
//! - Davies, A. N.; Lampen, P. *Appl. Spectrosc.* **1993**, 47, 1093–1099 (JCAMP-DX NMR v5.0).
//! - Lampen, P. et al. *Pure Appl. Chem.* **1999**, 71, 1549–1556 (JCAMP-DX v5.01 / v6.0).
//! - Fulmer, G. R. et al. *Organometallics* **2010**, 29, 2176–2179 (residual NMR solvent shifts).

use crate::octet_analyt::nmr_predict::{NmrNucleus, NmrPrediction};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const MAX_JCAMP_POINTS: usize = 1 << 21; // 2,097,152 points
const MAX_ALL_POINTS: usize = 1 << 23;
const MAX_BLOCKS: usize = 64;
const MAX_DEPTH: usize = 3;
const MAX_PEAKS: usize = 100_000;
const MAX_SPECTRA: usize = 16;

/// Kind of 1D analytical spectrum stored in JCAMP-DX.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpectrumKind {
    Nmr,
    Ir,
    Uv,
    Ms,
    Other,
}

impl SpectrumKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            SpectrumKind::Nmr => "nmr",
            SpectrumKind::Ir => "ir",
            SpectrumKind::Uv => "uv",
            SpectrumKind::Ms => "ms",
            SpectrumKind::Other => "other",
        }
    }
}

/// Synthesized or parsed 1D spectrum (continuous trace or MS stick spectrum).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Spectrum1d {
    pub title: String,
    pub kind: SpectrumKind,
    pub nucleus: Option<String>,
    pub mhz: Option<f64>,
    pub solvent: Option<String>,
    pub x_unit: String,
    pub y_unit: String,
    /// X value of `y[0]` (for NMR, often high-ppm downfield end or low-ppm upfield end).
    pub first_x: f64,
    /// X value of `y[y.len() - 1]`.
    pub last_x: f64,
    /// Intensity array.
    pub y: Vec<f64>,
    /// Explicit X coordinates when unevenly spaced or stick spectrum (`sticks == true`).
    pub xs: Option<Vec<f64>>,
    /// True when `xs`/`y` represent discrete stick peaks (e.g., MS `PEAK TABLE`).
    pub sticks: bool,
    /// Optional companion peak table `(x, intensity)`.
    pub peaks: Option<Vec<(f64, f64)>>,
    /// Cumulative integral trace (normalized to total solute protons when synthesized).
    pub integral: Option<Vec<f64>>,
}

impl Spectrum1d {
    /// Return the X coordinate at sample index `idx`.
    pub fn x_at(&self, idx: usize) -> f64 {
        if let Some(xs) = &self.xs {
            xs.get(idx).copied().unwrap_or(self.first_x)
        } else if self.y.len() <= 1 {
            self.first_x
        } else {
            let frac = (idx as f64) / ((self.y.len() - 1) as f64);
            self.first_x + frac * (self.last_x - self.first_x)
        }
    }

    /// Generate the full vector of X coordinates matching `self.y`.
    pub fn x_axis(&self) -> Vec<f64> {
        if let Some(xs) = &self.xs {
            xs.clone()
        } else {
            (0..self.y.len()).map(|i| self.x_at(i)).collect()
        }
    }

    /// Integrate `[ppm_lo, ppm_hi]` using trapezoidal quadrature, scaled so `total_area` maps to `scale_norm`.
    pub fn integrate_region(&self, x_a: f64, x_b: f64) -> f64 {
        if self.y.len() < 2 {
            return 0.0;
        }
        let lo = x_a.min(x_b);
        let hi = x_a.max(x_b);
        let mut area = 0.0;
        for i in 0..(self.y.len() - 1) {
            let x0 = self.x_at(i);
            let x1 = self.x_at(i + 1);
            let seg_lo = x0.min(x1);
            let seg_hi = x0.max(x1);
            if seg_hi <= lo || seg_lo >= hi {
                continue;
            }
            let dx = (x1 - x0).abs();
            area += 0.5 * (self.y[i] + self.y[i + 1]) * dx;
        }
        area
    }
}

/// Configuration for synthesizing a 1D ¹H or ¹³C NMR Lorentzian/Voigt spectrum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Synthesize1dOptions {
    /// ¹H spectrometer frequency in MHz (e.g., `400.0`, `500.0`, `600.0`, `800.0`).
    /// For ¹³C, the Larmor observe frequency is computed as `mhz_1h * 0.25145`.
    pub mhz_1h: f64,
    /// Number of evenly spaced spectral points (default `16384`).
    pub n_points: usize,
    /// Full width at half maximum (FWHM) in Hz for sharp multiplets (default `1.2` Hz for ¹H, `2.0` Hz for ¹³C).
    pub fwhm_hz: f64,
    /// FWHM in Hz for exchangeable broad signals (`OH`, `NH`, `SH`; default `10.0` Hz).
    pub broad_fwhm_hz: f64,
    /// Pseudo-Voigt Gaussian mixing fraction `0.0..=1.0` (`0.0` = pure Lorentzian).
    pub gaussian_fraction: f64,
    /// Optional explicit `(ppm_min, ppm_max)` window; `None` auto-expands `[0, 10]` (¹H) or `[0, 200]` (¹³C).
    pub ppm_window: Option<(f64, f64)>,
    /// Include residual deuterated solvent & trace water peaks (`Fulmer 2010`).
    pub include_solvent_peaks: bool,
    /// Title written into the spectrum metadata.
    pub title: String,
}

impl Default for Synthesize1dOptions {
    fn default() -> Self {
        Self {
            mhz_1h: 400.0,
            n_points: 16384,
            fwhm_hz: 1.2,
            broad_fwhm_hz: 10.0,
            gaussian_fraction: 0.0,
            ppm_window: None,
            include_solvent_peaks: false,
            title: "Simulated 1D NMR Spectrum".to_string(),
        }
    }
}

/// Residual deuterated solvent multiplet (`solventPeaks` in `spectrumplot.ts`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResidualSolventPeak {
    pub name: String,
    pub center_ppm: f64,
    /// Individual `(ppm, relative_height)` lines after spin-1 `²H` splitting.
    pub lines: Vec<(f64, f64)>,
    pub broad: bool,
}

/// Picked 1D peak `(ppm, intensity, prominence)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Picked1dPeak {
    pub x_ppm: f64,
    pub intensity: f64,
    pub index: usize,
}

/// Picked and grouped 1D multiplet from a continuous spectrum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Picked1dMultiplet {
    pub center_ppm: f64,
    pub span_ppm: (f64, f64),
    pub peaks: Vec<Picked1dPeak>,
    pub area: f64,
    pub normalized_integral: f64,
    pub estimated_multiplicity: String,
    pub estimated_j_hz: Option<f64>,
}

/// Trinomial `1:1:1` convolution for coupling to `n_deut` spin-1 deuterium nuclei (`deuterated` in `spectrumplot.ts`).
fn deuterated_lines(
    ppm: f64,
    n_deut: usize,
    j_hz: f64,
    obs_mhz: f64,
    total_area: f64,
) -> Vec<(f64, f64)> {
    let mut row = vec![1.0f64];
    for _ in 0..n_deut {
        let mut next = vec![0.0f64; row.len() + 2];
        for (i, &val) in row.iter().enumerate() {
            next[i] += val;
            next[i + 1] += val;
            next[i + 2] += val;
        }
        row = next;
    }
    let sum: f64 = row.iter().sum();
    row.into_iter()
        .enumerate()
        .map(|(i, h)| {
            let offset_steps = (i as isize) - (n_deut as isize);
            let line_ppm = ppm + ((offset_steps as f64) * j_hz) / obs_mhz.max(1.0);
            (line_ppm, (h / sum) * total_area)
        })
        .collect()
}

/// Canonical solvent display name (`solventName` in `jcamp.ts`).
pub fn canonical_solvent_name(raw: &str) -> Option<String> {
    let cleaned = raw.replace(['<', '>'], "");
    let s = cleaned.trim();
    if s.is_empty() {
        return None;
    }
    let up = s.to_ascii_uppercase();
    let mapped = match up.as_str() {
        "CDCL3" | "CDCL₃" | "CHLOROFORM" | "CHLOROFORM-D" | "CHLOROFORMD" => "CDCl₃",
        "DMSO" | "DMSO-D6" | "DMSOD6" | "DMSO-D₆" | "D6-DMSO" | "DIMETHYLSULFOXIDE-D6" => {
            "DMSO-d₆"
        }
        "CD3OD" | "CD₃OD" | "MEOD" | "METHANOL-D4" | "METHANOLD4" => "CD₃OD",
        "D2O" | "D₂O" | "DEUTERIUM OXIDE" | "WATER-D2" => "D₂O",
        "C6D6" | "C₆D₆" | "BENZENE-D6" | "BENZENED6" => "C₆D₆",
        "ACETONE-D6" | "ACETONED6" | "ACETONE-D₆" | "(CD3)2CO" | "CD3COCD3" => "Acetone-d₆",
        "CD3CN" | "CD₃CN" | "ACETONITRILE-D3" | "ACETONITRILED3" => "CD₃CN",
        _ => return Some(s.chars().take(40).collect()),
    };
    Some(mapped.to_string())
}

/// Residual solvent and trace water peaks (`Fulmer et al., Organometallics 2010, 29, 2176`).
pub fn residual_solvent_peaks(
    solvent: &str,
    nucleus: NmrNucleus,
    mhz_1h: f64,
) -> Vec<ResidualSolventPeak> {
    let canon = canonical_solvent_name(solvent).unwrap_or_else(|| solvent.to_string());
    let obs_mhz = match nucleus {
        NmrNucleus::H1 => mhz_1h,
        NmrNucleus::C13 => mhz_1h * 0.25145,
    };

    // (h_peaks: &[(ppm, n_D, J_HD)], water_ppm, c_peaks: &[(ppm, n_D, J_CD)])
    #[allow(clippy::type_complexity)]
    let spec: (&[(f64, usize, f64)], Option<f64>, &[(f64, usize, f64)]) = match canon.as_str() {
        "CDCl₃" => (&[(7.26, 0, 0.0)], Some(1.56), &[(77.16, 1, 32.0)]),
        "DMSO-d₆" => (&[(2.50, 2, 1.9)], Some(3.33), &[(39.52, 3, 21.0)]),
        "CD₃OD" => (&[(3.31, 2, 1.7)], Some(4.87), &[(49.00, 3, 21.4)]),
        "D₂O" => (&[(4.79, 0, 0.0)], None, &[]),
        "C₆D₆" => (&[(7.16, 0, 0.0)], Some(0.40), &[(128.06, 1, 24.0)]),
        "Acetone-d₆" => (
            &[(2.05, 2, 2.2)],
            Some(2.84),
            &[(29.84, 3, 19.4), (206.26, 0, 0.0)],
        ),
        "CD₃CN" => (
            &[(1.94, 2, 2.5)],
            Some(2.13),
            &[(1.32, 3, 21.0), (118.26, 0, 0.0)],
        ),
        _ => return Vec::new(),
    };

    let mut out = Vec::new();
    match nucleus {
        NmrNucleus::H1 => {
            for &(ppm, nd, j) in spec.0 {
                out.push(ResidualSolventPeak {
                    name: canon.clone(),
                    center_ppm: ppm,
                    lines: deuterated_lines(ppm, nd, j, obs_mhz, 0.25),
                    broad: false,
                });
            }
            if let Some(w_ppm) = spec.1 {
                out.push(ResidualSolventPeak {
                    name: "H₂O".to_string(),
                    center_ppm: w_ppm,
                    lines: deuterated_lines(w_ppm, 0, 0.0, obs_mhz, 0.15),
                    broad: true,
                });
            }
        }
        NmrNucleus::C13 => {
            for &(ppm, nd, j) in spec.2 {
                out.push(ResidualSolventPeak {
                    name: canon.clone(),
                    center_ppm: ppm,
                    lines: deuterated_lines(ppm, nd, j, obs_mhz, 0.6),
                    broad: false,
                });
            }
        }
    }
    out
}

/// Default full spectral window `(ppm_min, ppm_max)` (`fullRange` in `spectrumplot.ts`).
pub fn default_ppm_window(pred: &NmrPrediction) -> (f64, f64) {
    let (mut lo, mut hi, pad) = match pred.nucleus {
        NmrNucleus::H1 => (0.0, 10.0, 0.5),
        NmrNucleus::C13 => (0.0, 200.0, 10.0),
    };
    for s in &pred.signals {
        if s.shift_ppm - pad < lo {
            lo = s.shift_ppm - pad;
        }
        if s.shift_ppm + pad > hi {
            hi = s.shift_ppm + pad;
        }
    }
    (lo, hi)
}

/// Pseudo-Voigt profile normalized to unit integral over `ppm` (`∫ V(x) dx = 1.0`).
#[inline]
fn unit_pseudo_voigt_ppm(dx_ppm: f64, fwhm_ppm: f64, eta_gauss: f64) -> f64 {
    let hwhm = (0.5 * fwhm_ppm).max(1e-9);
    let lorentz = (hwhm / std::f64::consts::PI) / (dx_ppm * dx_ppm + hwhm * hwhm);
    if eta_gauss <= 0.0 {
        return lorentz;
    }
    let sigma = fwhm_ppm / (2.0 * (2.0_f64.ln() * 2.0).sqrt()).max(1e-9);
    let gauss =
        (-0.5 * (dx_ppm / sigma).powi(2)).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt());
    let eta = eta_gauss.clamp(0.0, 1.0);
    (1.0 - eta) * lorentz + eta * gauss
}

/// Synthesize a high-resolution 1D ¹H or ¹³C Lorentzian/Voigt spectrum from predicted multiplets,
/// complete with exact proton area normalization, cumulative integration curve, and peak table.
pub fn synthesize_1d_nmr(pred: &NmrPrediction, opts: &Synthesize1dOptions) -> Spectrum1d {
    let n_pts = opts.n_points.clamp(64, MAX_JCAMP_POINTS);
    let obs_mhz = match pred.nucleus {
        NmrNucleus::H1 => opts.mhz_1h.max(10.0),
        NmrNucleus::C13 => (opts.mhz_1h * 0.25145).max(5.0),
    };
    let (lo_ppm, hi_ppm) = opts.ppm_window.unwrap_or_else(|| default_ppm_window(pred));
    // Standard NMR JCAMP convention: FIRSTX = downfield (hi_ppm), LASTX = upfield (lo_ppm).
    let first_x = hi_ppm;
    let last_x = lo_ppm;
    let step_ppm = (last_x - first_x) / ((n_pts - 1) as f64);

    let mut y_solute = vec![0.0f64; n_pts];
    let mut peak_table: Vec<(f64, f64)> = Vec::new();

    // Render solute lines
    for sig in &pred.signals {
        let fwhm_hz = if sig.broad {
            opts.broad_fwhm_hz.max(2.0)
        } else {
            opts.fwhm_hz.max(0.2)
        };
        let fwhm_ppm = fwhm_hz / obs_mhz;
        // Cutoff wing at 80 * FWHM (captures > 99.6% of Lorentzian area while staying fast)
        let wing_ppm = (80.0 * fwhm_ppm).max(0.25);

        for (line_ppm, area) in sig.lines_at_mhz(pred.nucleus, opts.mhz_1h) {
            let idx_center = ((line_ppm - first_x) / step_ppm).round() as isize;
            let half_span = ((wing_ppm / step_ppm.abs()).ceil() as isize).max(4);
            let i0 = (idx_center - half_span).clamp(0, (n_pts - 1) as isize) as usize;
            let i1 = (idx_center + half_span).clamp(0, (n_pts - 1) as isize) as usize;
            for (idx, y_val) in y_solute.iter_mut().enumerate().take(i1 + 1).skip(i0) {
                let x = first_x + (idx as f64) * step_ppm;
                *y_val +=
                    area * unit_pseudo_voigt_ppm(x - line_ppm, fwhm_ppm, opts.gaussian_fraction);
            }
            peak_table.push((
                line_ppm,
                area * unit_pseudo_voigt_ppm(0.0, fwhm_ppm, opts.gaussian_fraction),
            ));
        }
    }

    // Cumulative integration curve (from downfield first_x to upfield last_x) normalized to total solute nuclei
    let total_nuclei: f64 = pred.signals.iter().map(|s| s.nuclei as f64).sum();
    let mut integral = vec![0.0f64; n_pts];
    let dx = step_ppm.abs();
    for i in 1..n_pts {
        integral[i] = integral[i - 1] + 0.5 * (y_solute[i - 1] + y_solute[i]) * dx;
    }
    if let Some(&end_val) = integral.last() {
        if end_val > 1e-12 && total_nuclei > 0.0 {
            let scale = total_nuclei / end_val;
            for v in &mut integral {
                *v *= scale;
            }
        }
    }

    let mut y_total = y_solute;
    if opts.include_solvent_peaks {
        for sp in residual_solvent_peaks(&pred.solvent, pred.nucleus, opts.mhz_1h) {
            let fwhm_ppm = (if sp.broad {
                opts.broad_fwhm_hz
            } else {
                opts.fwhm_hz
            }) / obs_mhz;
            let wing_ppm = (60.0 * fwhm_ppm).max(0.2);
            for (line_ppm, area) in sp.lines {
                let idx_center = ((line_ppm - first_x) / step_ppm).round() as isize;
                let half_span = ((wing_ppm / step_ppm.abs()).ceil() as isize).max(4);
                let i0 = (idx_center - half_span).clamp(0, (n_pts - 1) as isize) as usize;
                let i1 = (idx_center + half_span).clamp(0, (n_pts - 1) as isize) as usize;
                for (idx, y_val) in y_total.iter_mut().enumerate().take(i1 + 1).skip(i0) {
                    let x = first_x + (idx as f64) * step_ppm;
                    *y_val += area
                        * unit_pseudo_voigt_ppm(x - line_ppm, fwhm_ppm, opts.gaussian_fraction);
                }
            }
        }
    }

    Spectrum1d {
        title: opts.title.clone(),
        kind: SpectrumKind::Nmr,
        nucleus: Some(pred.nucleus.as_str().to_string()),
        mhz: Some(obs_mhz),
        solvent: Some(pred.solvent.clone()),
        x_unit: "ppm".to_string(),
        y_unit: "intensity".to_string(),
        first_x,
        last_x,
        y: y_total,
        xs: None,
        sticks: false,
        peaks: if peak_table.is_empty() {
            None
        } else {
            Some(peak_table)
        },
        integral: Some(integral),
    }
}

/// Pick local-maximum 1D peaks above `rel_threshold * max_intensity` with 3-point parabolic refinement.
pub fn pick_1d_peaks(spec: &Spectrum1d, rel_threshold: f64) -> Vec<Picked1dPeak> {
    if spec.y.is_empty() {
        return Vec::new();
    }
    let max_y = spec
        .y
        .iter()
        .copied()
        .fold(0.0f64, |a, b| if b > a { b } else { a });
    if max_y <= 0.0 {
        return Vec::new();
    }
    let thresh = max_y * rel_threshold.clamp(1e-5, 0.99);

    if spec.sticks {
        let mut out: Vec<Picked1dPeak> = spec
            .y
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v >= thresh)
            .map(|(i, &v)| Picked1dPeak {
                x_ppm: spec.x_at(i),
                intensity: v,
                index: i,
            })
            .collect();
        out.sort_by(|a, b| {
            b.x_ppm
                .partial_cmp(&a.x_ppm)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        return out;
    }

    let mut out = Vec::new();
    for i in 1..(spec.y.len().saturating_sub(1)) {
        let y0 = spec.y[i - 1];
        let y1 = spec.y[i];
        let y2 = spec.y[i + 1];
        if y1 >= thresh && y1 >= y0 && y1 > y2 {
            // Parabolic vertex offset delta in [-0.5, 0.5]
            let denom = y0 - 2.0 * y1 + y2;
            let delta = if denom.abs() > 1e-12 {
                (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5)
            } else {
                0.0
            };
            let x_i = spec.x_at(i);
            let step = spec.x_at(i + 1) - x_i;
            let refined_x = x_i + delta * step;
            let refined_y = y1 - 0.25 * (y0 - y2) * delta;
            out.push(Picked1dPeak {
                x_ppm: refined_x,
                intensity: refined_y,
                index: i,
            });
        }
    }
    out.sort_by(|a, b| {
        b.x_ppm
            .partial_cmp(&a.x_ppm)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

/// Cluster picked 1D peaks into NMR multiplets (`max_gap_hz` typically `22.0` Hz),
/// estimating multiplet center, area, normalized proton count, multiplicity, and *J* spacing.
pub fn pick_1d_multiplets(
    spec: &Spectrum1d,
    rel_threshold: f64,
    max_gap_hz: f64,
    total_protons: Option<f64>,
) -> Vec<Picked1dMultiplet> {
    let peaks = pick_1d_peaks(spec, rel_threshold);
    if peaks.is_empty() {
        return Vec::new();
    }
    let obs_mhz = spec.mhz.unwrap_or(400.0).max(1.0);
    let max_gap_ppm = max_gap_hz.max(2.0) / obs_mhz;
    let pad_ppm = 12.0 / obs_mhz;

    let mut groups: Vec<Vec<Picked1dPeak>> = Vec::new();
    for p in peaks {
        if let Some(last_grp) = groups.last_mut() {
            let prev_x = last_grp.last().unwrap().x_ppm;
            if (prev_x - p.x_ppm).abs() <= max_gap_ppm {
                last_grp.push(p);
                continue;
            }
        }
        groups.push(vec![p]);
    }

    let mut multiplets = Vec::with_capacity(groups.len());
    let mut sum_area = 0.0;

    for grp in groups {
        let w_sum: f64 = grp.iter().map(|p| p.intensity).sum();
        let center_ppm = if w_sum > 0.0 {
            grp.iter().map(|p| p.x_ppm * p.intensity).sum::<f64>() / w_sum
        } else {
            grp[0].x_ppm
        };
        let hi_x = grp.first().unwrap().x_ppm + pad_ppm;
        let lo_x = grp.last().unwrap().x_ppm - pad_ppm;
        let area = spec.integrate_region(lo_x, hi_x);
        sum_area += area;

        let (est_mult, est_j) = classify_picked_multiplet(&grp, obs_mhz);
        multiplets.push(Picked1dMultiplet {
            center_ppm,
            span_ppm: (lo_x, hi_x),
            peaks: grp,
            area,
            normalized_integral: area,
            estimated_multiplicity: est_mult,
            estimated_j_hz: est_j,
        });
    }

    if let Some(tp) = total_protons {
        if sum_area > 1e-12 && tp > 0.0 {
            let scale = tp / sum_area;
            for m in &mut multiplets {
                m.normalized_integral = m.area * scale;
            }
        }
    }

    multiplets
}

fn classify_picked_multiplet(peaks: &[Picked1dPeak], obs_mhz: f64) -> (String, Option<f64>) {
    let n = peaks.len();
    if n <= 1 {
        return ("s".to_string(), None);
    }
    let mut spacings_hz = Vec::with_capacity(n - 1);
    for k in 0..(n - 1) {
        spacings_hz.push((peaks[k].x_ppm - peaks[k + 1].x_ppm).abs() * obs_mhz);
    }
    let mean_j = spacings_hz.iter().sum::<f64>() / (spacings_hz.len() as f64);
    let uniform = spacings_hz.iter().all(|&s| (s - mean_j).abs() < 0.8);
    if uniform {
        let label = match n {
            2 => "d",
            3 => "t",
            4 => "q",
            5 => "quint",
            6 => "sext",
            7 => "sept",
            _ => "m",
        };
        return (label.to_string(), Some(mean_j));
    }
    if n == 4 {
        let j_small = 0.5 * (spacings_hz[0] + spacings_hz[2]);
        let j_large = spacings_hz[0] + spacings_hz[1];
        if (spacings_hz[0] - spacings_hz[2]).abs() < 0.8 {
            return ("dd".to_string(), Some(j_large.max(j_small)));
        }
    }
    ("m".to_string(), Some(mean_j))
}

/// Pool-adjacent-violators isotonic non-overlapping peak label layout (`layoutLabels` in `spectrumplot.ts`).
pub fn layout_peak_labels(xs: &[f64], widths: &[f64], lo: f64, hi: f64) -> Vec<f64> {
    let n = xs.len();
    if n == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        xs[a]
            .partial_cmp(&xs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut offsets = vec![0.0f64; n];
    let mut means: Vec<f64> = Vec::new();
    let mut sizes: Vec<usize> = Vec::new();

    for k in 0..n {
        if k > 0 {
            offsets[k] = offsets[k - 1] + 0.5 * (widths[order[k - 1]] + widths[order[k]]);
        }
        let mut m = xs[order[k]] - offsets[k];
        let mut c = 1usize;
        while let Some(&prev_m) = means.last() {
            if prev_m >= m {
                means.pop();
                let pc = sizes.pop().unwrap();
                m = (prev_m * (pc as f64) + m * (c as f64)) / ((pc + c) as f64);
                c += pc;
            } else {
                break;
            }
        }
        means.push(m);
        sizes.push(c);
    }

    let first_lim = lo + 0.5 * widths[order[0]];
    let last_lim = hi - 0.5 * widths[order[n - 1]] - offsets[n - 1];
    let mut out = vec![0.0f64; n];
    let mut idx = 0usize;
    for (m, c) in means.into_iter().zip(sizes) {
        let base = if first_lim <= last_lim {
            m.clamp(first_lim, last_lim)
        } else {
            0.5 * (first_lim + last_lim)
        };
        for _ in 0..c {
            out[order[idx]] = base + offsets[idx];
            idx += 1;
        }
    }
    out
}

// ============================================================================
// JCAMP-DX (v4.24 / v5.0 / v5.01 / v6.0) Reader & Writer
// ============================================================================

#[derive(Debug, Clone)]
struct Ldr {
    label: String,
    value: String,
}

#[derive(Debug, Clone, Default)]
struct JcampBlock {
    ldrs: Vec<Ldr>,
    children: Vec<JcampBlock>,
}

fn norm_ldr_label(raw: &str) -> String {
    raw.chars()
        .filter(|c| !matches!(c, ' ' | '\t' | '-' | '/' | '_'))
        .map(|c| c.to_ascii_uppercase())
        .collect()
}

/// Return true if `text` starts with a JCAMP-DX `##TITLE=` header (ignoring UTF-8 BOM and `$$` comments).
pub fn is_jcamp(text: &str) -> bool {
    let prefix: String = text.chars().take(4096).collect();
    let trimmed = prefix.trim_start_matches('\u{feff}');
    for line in trimmed.lines() {
        let s = line.trim_start();
        if s.is_empty() || s.starts_with("$$") {
            continue;
        }
        if let Some(rest) = s.strip_prefix("##") {
            if let Some((lbl, _)) = rest.split_once('=') {
                return norm_ldr_label(lbl) == "TITLE";
            }
        }
        return false;
    }
    false
}

fn parse_jcamp_blocks(text: &str) -> Result<Vec<JcampBlock>, String> {
    if !is_jcamp(text) {
        return Err("This file is not a JCAMP-DX spectrum.".to_string());
    }
    let mut top: Vec<JcampBlock> = Vec::new();
    let mut stack: Vec<JcampBlock> = Vec::new();
    let mut count = 0usize;

    for raw_line in text.trim_start_matches('\u{feff}').lines() {
        let line = match raw_line.find("$$") {
            Some(at) => &raw_line[..at],
            None => raw_line,
        };
        let trimmed = line.trim_start();
        if let Some(after_hashes) = trimmed.strip_prefix("##") {
            if let Some((raw_lbl, val)) = after_hashes.split_once('=') {
                let label = norm_ldr_label(raw_lbl);
                if label == "TITLE" {
                    count += 1;
                    if count > MAX_BLOCKS {
                        return Err(format!(
                            "This JCAMP-DX file has more than {MAX_BLOCKS} blocks."
                        ));
                    }
                    if stack.len() >= MAX_DEPTH {
                        return Err("This JCAMP-DX file has blocks nested too deep.".to_string());
                    }
                    stack.push(JcampBlock::default());
                } else if stack.is_empty() {
                    return Err(
                        "This JCAMP-DX file is damaged: a label comes before the first TITLE."
                            .to_string(),
                    );
                }

                if let Some(cur) = stack.last_mut() {
                    cur.ldrs.push(Ldr {
                        label: label.clone(),
                        value: val.to_string(),
                    });
                }

                if label == "END" {
                    let finished = stack.pop().unwrap();
                    if let Some(parent) = stack.last_mut() {
                        parent.children.push(finished);
                    } else {
                        top.push(finished);
                    }
                }
                continue;
            }
        }

        // Continuation line of the current LDR
        if !line.trim().is_empty() {
            if let Some(cur) = stack.last_mut() {
                if let Some(last_ldr) = cur.ldrs.last_mut() {
                    last_ldr.value.push('\n');
                    last_ldr.value.push_str(line);
                }
            }
        }
    }

    // Flush unclosed top-level block if ##END= was omitted at EOF
    while let Some(finished) = stack.pop() {
        if let Some(parent) = stack.last_mut() {
            parent.children.push(finished);
        } else {
            top.push(finished);
        }
    }

    if top.is_empty() {
        return Err("This file is not a JCAMP-DX spectrum.".to_string());
    }
    Ok(top)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AsdfTokKind {
    Abs,
    Dif,
    Dup,
}

#[derive(Debug, Clone, Copy)]
struct AsdfTok {
    kind: AsdfTokKind,
    v: f64,
}

const SQZ_POS: &[u8; 10] = b"@ABCDEFGHI";
const SQZ_NEG: &[u8; 9] = b"abcdefghi";
const DIF_POS: &[u8; 10] = b"%JKLMNOPQR";
const DIF_NEG: &[u8; 9] = b"jklmnopqr";
const DUP_CHARS: &[u8; 9] = b"STUVWXYZs";

/// Tokenize one line of a JCAMP-DX ASDF `(X++(Y..Y))` or AFFN table (`tokens` in `jcamp.ts`).
fn tokenize_asdf_line(line: &str) -> Result<Vec<AsdfTok>, String> {
    let bytes = line.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;
    let mut out = Vec::new();

    let scan_digits = |mut j: usize| -> usize {
        while j < n && (bytes[j].is_ascii_digit() || bytes[j] == b'.') {
            j += 1;
        }
        j
    };
    let parse_num = |s: &str| -> Result<f64, String> {
        let v = s
            .parse::<f64>()
            .map_err(|_| format!("This JCAMP-DX file is damaged: \"{s}\" is not a number."))?;
        if !v.is_finite() || v.abs() >= 1e300 {
            return Err("This JCAMP-DX file is damaged: number out of range.".to_string());
        }
        Ok(v)
    };

    while i < n {
        let c = bytes[i];
        if matches!(c, b' ' | b'\t' | b',' | b';' | b'\r') {
            i += 1;
            continue;
        }
        if c == b'?' {
            out.push(AsdfTok {
                kind: AsdfTokKind::Abs,
                v: 0.0,
            });
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || c == b'.' || c == b'+' || c == b'-' {
            let start = i;
            let mut j = if c == b'+' || c == b'-' { i + 1 } else { i };
            j = scan_digits(j);
            if j + 2 < n
                && (bytes[j] == b'E' || bytes[j] == b'e')
                && (bytes[j + 1] == b'+' || bytes[j + 1] == b'-')
                && bytes[j + 2].is_ascii_digit()
            {
                j = scan_digits(j + 2);
            }
            if j == start + 1 && (c == b'+' || c == b'-') {
                return Err(
                    "This JCAMP-DX file is damaged: a sign stands alone in data.".to_string(),
                );
            }
            let s = std::str::from_utf8(&bytes[start..j]).unwrap();
            out.push(AsdfTok {
                kind: AsdfTokKind::Abs,
                v: parse_num(s)?,
            });
            i = j;
            continue;
        }

        let (kind, sign, lead_digit) = if let Some(k) = SQZ_POS.iter().position(|&b| b == c) {
            (AsdfTokKind::Abs, 1.0, k)
        } else if let Some(k) = SQZ_NEG.iter().position(|&b| b == c) {
            (AsdfTokKind::Abs, -1.0, k + 1)
        } else if let Some(k) = DIF_POS.iter().position(|&b| b == c) {
            (AsdfTokKind::Dif, 1.0, k)
        } else if let Some(k) = DIF_NEG.iter().position(|&b| b == c) {
            (AsdfTokKind::Dif, -1.0, k + 1)
        } else if let Some(k) = DUP_CHARS.iter().position(|&b| b == c) {
            (AsdfTokKind::Dup, 1.0, k + 1)
        } else {
            return Err(format!(
                "This JCAMP-DX file is damaged: '{}' cannot be in a data line.",
                c as char
            ));
        };

        let j = scan_digits(i + 1);
        let tail = std::str::from_utf8(&bytes[i + 1..j]).unwrap();
        let num_str = format!("{lead_digit}{tail}");
        out.push(AsdfTok {
            kind,
            v: sign * parse_num(&num_str)?,
        });
        i = j;
    }

    Ok(out)
}

/// Decode an ASDF `(X++(Y..Y))` block into uncompressed `Y` values (`asdf` in `jcamp.ts`).
fn decode_asdf_lines(lines: &[&str], max_expected: usize) -> Result<Vec<f64>, String> {
    let mut ys: Vec<f64> = Vec::new();
    let mut checked = false;

    for &line in lines {
        let toks = tokenize_asdf_line(line)?;
        if toks.is_empty() {
            continue;
        }
        if toks[0].kind != AsdfTokKind::Abs {
            return Err(
                "This JCAMP-DX file is damaged: a data line does not start with its X value."
                    .to_string(),
            );
        }
        let mut last_tok: Option<AsdfTok> = None;
        let mut y = ys.last().copied().unwrap_or(0.0);
        let mut first_in_line = true;

        for &tok in &toks[1..] {
            match tok.kind {
                AsdfTokKind::Dup => {
                    let prev = last_tok.ok_or_else(|| {
                        "This JCAMP-DX file is damaged: a repeat has nothing to repeat.".to_string()
                    })?;
                    let count_f = tok.v;
                    if count_f < 1.0 || (count_f - count_f.round()).abs() > 1e-9 {
                        return Err(
                            "This JCAMP-DX file is damaged: repeat count is not an integer."
                                .to_string(),
                        );
                    }
                    let extra = (count_f.round() as usize) - 1;
                    if ys.len() + extra > max_expected + 1 {
                        return Err("JCAMP-DX spectrum exceeds declared point limit.".to_string());
                    }
                    for _ in 0..extra {
                        if prev.kind == AsdfTokKind::Dif {
                            y += prev.v;
                        }
                        ys.push(y);
                    }
                }
                AsdfTokKind::Dif => {
                    if ys.is_empty() && first_in_line {
                        return Err(
                            "This JCAMP-DX file is damaged: data starts with a DIF difference."
                                .to_string(),
                        );
                    }
                    y += tok.v;
                    first_in_line = false;
                    if ys.len() > max_expected {
                        return Err("JCAMP-DX spectrum exceeds declared point limit.".to_string());
                    }
                    ys.push(y);
                    last_tok = Some(tok);
                }
                AsdfTokKind::Abs => {
                    y = tok.v;
                    // Y-check: if the previous line ended in DIF mode, the first Y of this line repeats the last Y of the previous line.
                    if first_in_line && checked {
                        first_in_line = false;
                        last_tok = Some(tok);
                        continue;
                    }
                    first_in_line = false;
                    if ys.len() > max_expected {
                        return Err("JCAMP-DX spectrum exceeds declared point limit.".to_string());
                    }
                    ys.push(y);
                    last_tok = Some(tok);
                }
            }
        }
        checked = last_tok.is_some_and(|t| t.kind == AsdfTokKind::Dif);
    }

    Ok(ys)
}

fn decode_free_tuples(lines: &[&str], width: usize) -> Result<Vec<Vec<f64>>, String> {
    let mut vals = Vec::new();
    for &l in lines {
        for t in tokenize_asdf_line(l)? {
            if t.kind != AsdfTokKind::Abs {
                return Err(
                    "This JCAMP-DX file is damaged: tuple table holds compressed values."
                        .to_string(),
                );
            }
            vals.push(t.v);
        }
    }
    if vals.len() % width != 0 {
        return Err("This JCAMP-DX file is damaged: tuple table has a missing value.".to_string());
    }
    Ok(vals.chunks(width).map(|c| c.to_vec()).collect())
}

fn parse_nucleus_label(raw: Option<&str>) -> Option<String> {
    let s: String = raw?
        .chars()
        .filter(|c| !matches!(c, '^' | '<' | '>' | ' ' | '\t'))
        .collect();
    if s.is_empty() {
        return None;
    }
    // Match e.g. "1H", "13C", "H1", "C13"
    let first_alpha = s.find(|c: char| c.is_ascii_alphabetic())?;
    if first_alpha > 0 {
        let (num_part, el_part) = s.split_at(first_alpha);
        if num_part.chars().all(|c| c.is_ascii_digit()) {
            return Some(format!("{num_part}{el_part}"));
        }
    } else {
        let first_digit = s.find(|c: char| c.is_ascii_digit())?;
        let (el_part, num_part) = s.split_at(first_digit);
        if num_part.chars().all(|c| c.is_ascii_digit()) {
            return Some(format!("{num_part}{el_part}"));
        }
    }
    None
}

fn detect_spectrum_kind(dtype: &str) -> SpectrumKind {
    let up = dtype.to_ascii_uppercase();
    if up.contains("NMR") {
        SpectrumKind::Nmr
    } else if up.contains("INFRARED")
        || up.contains("RAMAN")
        || up.split_whitespace().any(|w| w == "IR")
    {
        SpectrumKind::Ir
    } else if up.contains("UV") || up.contains("VIS") {
        SpectrumKind::Uv
    } else if up.contains("MASS") || up == "MS" {
        SpectrumKind::Ms
    } else {
        SpectrumKind::Other
    }
}

fn display_unit(raw_upper: &str) -> String {
    match raw_upper {
        "PPM" => "ppm".to_string(),
        "HZ" => "Hz".to_string(),
        "1/CM" => "cm⁻¹".to_string(),
        "NANOMETERS" => "nm".to_string(),
        "MICROMETERS" => "µm".to_string(),
        "M/Z" | "MZ" => "m/z".to_string(),
        "SECONDS" => "s".to_string(),
        "TRANSMITTANCE" => "%T".to_string(),
        "ABSORBANCE" => "A".to_string(),
        "RELATIVE ABUNDANCE" | "RELATIVEABUNDANCE" => "%".to_string(),
        _ => raw_upper.to_ascii_lowercase().chars().take(20).collect(),
    }
}

#[allow(clippy::type_complexity)]
fn decode_ntuples_real_page(
    b: &JcampBlock,
    all: &HashMap<String, String>,
) -> Result<Option<(f64, f64, Vec<f64>, String, String)>, String> {
    let split_csv = |k: &str| -> Vec<String> {
        all.get(k)
            .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
            .unwrap_or_default()
    };
    let symbols: Vec<String> = split_csv("SYMBOL")
        .into_iter()
        .map(|s| s.to_ascii_uppercase())
        .collect();
    let units = split_csv("UNITS");
    let firsts = split_csv("FIRST");
    let lasts = split_csv("LAST");
    let factors = split_csv("FACTOR");
    let dims = split_csv("VARDIM");

    let tables: Vec<(usize, &Ldr)> = b
        .ldrs
        .iter()
        .enumerate()
        .filter(|(_, l)| l.label == "DATATABLE")
        .collect();
    if tables.is_empty() {
        return Ok(None);
    }

    let real_entry = tables
        .iter()
        .find(|(_, t)| {
            t.value
                .lines()
                .next()
                .unwrap_or("")
                .replace(' ', "")
                .to_ascii_uppercase()
                .contains("(R..R)")
        })
        .or_else(|| {
            tables.iter().find(|(_, t)| {
                t.value
                    .lines()
                    .next()
                    .unwrap_or("")
                    .replace(' ', "")
                    .to_ascii_uppercase()
                    .contains("X++(Y..Y)")
            })
        });

    let Some(&(ldr_pos, real_ldr)) = real_entry else {
        return Ok(None);
    };

    let head_line = real_ldr.value.lines().next().unwrap_or("");
    let y_sym = if head_line.to_ascii_uppercase().contains("(R..R)") {
        "R"
    } else {
        "Y"
    };
    let xi = symbols
        .iter()
        .position(|s| s == "X")
        .ok_or_else(|| "This JCAMP-DX file is damaged: NTUPLES has no X variable.".to_string())?;
    let yi = symbols.iter().position(|s| s == y_sym);

    // Page-local NPOINTS overrides VARDIM
    let page_start = ldr_pos.saturating_sub(12);
    let page_npoints = b.ldrs[page_start..ldr_pos]
        .iter()
        .rev()
        .find(|x| x.label == "NPOINTS")
        .and_then(|x| x.value.trim().parse::<usize>().ok());

    let dim_idx = yi.unwrap_or(xi);
    let n_pts = page_npoints
        .or_else(|| dims.get(dim_idx).and_then(|s| s.parse::<usize>().ok()))
        .ok_or_else(|| {
            "This JCAMP-DX file is damaged: NTUPLES gives no number of points.".to_string()
        })?;
    if n_pts == 0 || n_pts > MAX_JCAMP_POINTS {
        return Err(format!("NTUPLES point count {n_pts} out of bounds."));
    }

    let first_x = firsts
        .get(xi)
        .and_then(|s| s.parse::<f64>().ok())
        .ok_or_else(|| "This JCAMP-DX file is damaged: NTUPLES gives no FIRST X.".to_string())?;
    let last_x = lasts
        .get(xi)
        .and_then(|s| s.parse::<f64>().ok())
        .ok_or_else(|| "This JCAMP-DX file is damaged: NTUPLES gives no LAST X.".to_string())?;
    let yf = yi
        .and_then(|idx| factors.get(idx))
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0);

    let x_u = units
        .get(xi)
        .map(|s| s.to_ascii_uppercase())
        .unwrap_or_default();
    if x_u.contains("SECONDS") || x_u.contains("TIME") {
        return Err("This file holds a raw FID. Process it into a spectrum first.".to_string());
    }
    let y_u = yi
        .and_then(|idx| units.get(idx))
        .map(|s| s.to_ascii_uppercase())
        .unwrap_or_default();

    let body_lines: Vec<&str> = real_ldr.value.lines().skip(1).collect();
    let ys: Vec<f64> = decode_asdf_lines(&body_lines, n_pts)?
        .into_iter()
        .map(|v| v * yf)
        .collect();
    if ys.len() != n_pts {
        return Err(format!(
            "This JCAMP-DX file is damaged: page declares {n_pts} points and holds {}.",
            ys.len()
        ));
    }

    Ok(Some((first_x, last_x, ys, x_u, y_u)))
}

fn block_to_spectrum(
    b: &JcampBlock,
    inherited: &HashMap<String, String>,
) -> Result<Option<Spectrum1d>, String> {
    let mut all = inherited.clone();
    for l in &b.ldrs {
        all.insert(l.label.clone(), l.value.clone());
    }
    let get_str = |k: &str| -> Option<String> {
        all.get(k)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };
    let get_num = |k: &str| -> Result<Option<f64>, String> {
        let Some(v) = get_str(k) else {
            return Ok(None);
        };
        let first_tok = v.split([' ', '\t', ',']).next().unwrap_or("");
        let n = first_tok
            .parse::<f64>()
            .map_err(|_| format!("This JCAMP-DX file is damaged: {k} is not a number."))?;
        if !n.is_finite() {
            return Err(format!("This JCAMP-DX file is damaged: {k} is not finite."));
        }
        Ok(Some(n))
    };

    let dtype = get_str("DATATYPE")
        .or_else(|| get_str("NTUPLES"))
        .unwrap_or_default();
    let kind = detect_spectrum_kind(&dtype);
    let dtype_up = dtype.to_ascii_uppercase();
    let mut x_unit_raw = get_str("XUNITS").unwrap_or_default().to_ascii_uppercase();
    let mut y_unit_raw = get_str("YUNITS").unwrap_or_default().to_ascii_uppercase();

    if dtype_up.split_whitespace().any(|w| w == "FID")
        || (kind == SpectrumKind::Nmr && x_unit_raw.contains("SECONDS"))
    {
        return Err("This file holds a raw FID. Process it into a spectrum first.".to_string());
    }
    if dtype_up.contains("2D")
        || dtype_up.contains("NMR SPECTRUM 2")
        || get_num("NUMDIM")?.unwrap_or(1.0) > 1.0
    {
        return Err("Only 1D JCAMP-DX spectra are supported here.".to_string());
    }

    let title: String = b
        .ldrs
        .iter()
        .find(|l| l.label == "TITLE")
        .map(|l| l.value.trim())
        .unwrap_or("")
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .take(200)
        .collect();

    let mhz = get_num(".OBSERVEFREQUENCY")?
        .or(get_num("$SF")?)
        .or(get_num("$BF1")?);
    let mut nucleus = if kind == SpectrumKind::Nmr {
        let nuc_raw = get_str(".OBSERVENUCLEUS")
            .or_else(|| get_str("$NUC1"))
            .or_else(|| get_str("$AXNUC"));
        parse_nucleus_label(nuc_raw.as_deref())
    } else {
        None
    };
    let solvent = canonical_solvent_name(
        &get_str(".SOLVENTNAME")
            .or_else(|| get_str("$SOLVENT"))
            .unwrap_or_default(),
    );

    let xf = get_num("XFACTOR")?.unwrap_or(1.0);
    let yf = get_num("YFACTOR")?.unwrap_or(1.0);
    let mut first_x = get_num("FIRSTX")?;
    let mut last_x = get_num("LASTX")?;
    let declared_n = match get_num("NPOINTS")? {
        Some(nf) => {
            if nf < 1.0 || (nf - nf.round()).abs() > 1e-9 {
                return Err(
                    "This JCAMP-DX file is damaged: NPOINTS is not a positive integer.".to_string(),
                );
            }
            let n = nf.round() as usize;
            if n > MAX_JCAMP_POINTS {
                return Err(format!(
                    "Spectrum has {n} points (exceeds {MAX_JCAMP_POINTS})."
                ));
            }
            Some(n)
        }
        None => None,
    };

    let find_ldr = |lbl: &str| b.ldrs.iter().find(|l| l.label == lbl);

    let mut peaks: Option<Vec<(f64, f64)>> = None;
    if let Some(table_ldr) = find_ldr("PEAKTABLE") {
        let all_lines: Vec<&str> = table_ldr.value.lines().collect();
        let head = all_lines
            .first()
            .copied()
            .unwrap_or("")
            .replace(' ', "")
            .to_ascii_uppercase();
        let width = if head.contains("XYW") || head.contains("XYM") {
            3
        } else {
            2
        };
        let data_lines: Vec<&str> = if head.starts_with('(') {
            all_lines.into_iter().skip(1).collect()
        } else {
            all_lines
        };
        let rows = decode_free_tuples(&data_lines, width)?;
        if rows.len() > MAX_PEAKS {
            return Err(format!("Spectrum has more than {MAX_PEAKS} peaks."));
        }
        peaks = Some(rows.into_iter().map(|r| (r[0] * xf, r[1] * yf)).collect());
    }

    let xy_ldr = find_ldr("XYDATA");
    let pts_ldr = find_ldr("XYPOINTS");

    let mut xs: Option<Vec<f64>> = None;
    let y: Vec<f64>;
    let mut sticks = false;

    if let Some(xy) = xy_ldr {
        let lines: Vec<&str> = xy.value.lines().collect();
        let head = lines
            .first()
            .copied()
            .unwrap_or("")
            .replace(' ', "")
            .to_ascii_uppercase();
        if head.contains("X++(Y..Y)") {
            if first_x.is_none() || last_x.is_none() {
                return Err(
                    "This JCAMP-DX file is damaged: FIRSTX or LASTX is missing.".to_string()
                );
            }
            let max_pts = declared_n.unwrap_or(MAX_JCAMP_POINTS);
            y = decode_asdf_lines(&lines[1..], max_pts)?
                .into_iter()
                .map(|v| v * yf)
                .collect();
        } else {
            let rows = decode_free_tuples(&lines[1..], 2)?;
            if rows.len() > MAX_JCAMP_POINTS {
                return Err("Too many points in XYDATA.".to_string());
            }
            xs = Some(rows.iter().map(|r| r[0] * xf).collect());
            y = rows.iter().map(|r| r[1] * yf).collect();
        }
    } else if let Some(pts) = pts_ldr {
        let lines: Vec<&str> = pts.value.lines().skip(1).collect();
        let rows = decode_free_tuples(&lines, 2)?;
        if rows.len() > MAX_JCAMP_POINTS {
            return Err("Too many points in XYPOINTS.".to_string());
        }
        xs = Some(rows.iter().map(|r| r[0] * xf).collect());
        y = rows.iter().map(|r| r[1] * yf).collect();
    } else if b.ldrs.iter().any(|l| l.label == "NTUPLES") {
        let Some((fx, lx, ny, xu, yu)) = decode_ntuples_real_page(b, &all)? else {
            return Ok(None);
        };
        first_x = Some(fx);
        last_x = Some(lx);
        y = ny;
        if !xu.is_empty() {
            x_unit_raw = xu;
        }
        if !yu.is_empty() {
            y_unit_raw = yu;
        }
    } else if let Some(pk) = peaks.take() {
        if pk.is_empty() {
            return Ok(None);
        }
        xs = Some(pk.iter().map(|p| p.0).collect());
        y = pk.iter().map(|p| p.1).collect();
        sticks = true;
    } else {
        return Ok(None);
    }

    if y.is_empty() {
        return Err("This JCAMP-DX file is damaged: the spectrum has no points.".to_string());
    }
    if let Some(dec) = declared_n {
        if xs.is_none() && y.len() != dec {
            return Err(format!(
                "This JCAMP-DX file is damaged: declares {dec} points and holds {}.",
                y.len()
            ));
        }
    }

    // Convert NMR Hz axis to ppm using observe frequency and apply .SHIFT REFERENCE if present
    let was_hz = x_unit_raw == "HZ";
    if kind == SpectrumKind::Nmr && was_hz {
        let obs = mhz.ok_or_else(|| {
            "This JCAMP-DX file is damaged: NMR spectrum in Hz has no observe frequency."
                .to_string()
        })?;
        if obs <= 0.0 {
            return Err("Observe frequency must be positive.".to_string());
        }
        if let Some(ref mut xvec) = xs {
            for v in xvec.iter_mut() {
                *v /= obs;
            }
        } else {
            first_x = first_x.map(|v| v / obs);
            last_x = last_x.map(|v| v / obs);
        }
        if let Some(ref mut pks) = peaks {
            for p in pks.iter_mut() {
                p.0 /= obs;
            }
        }
        x_unit_raw = "PPM".to_string();
    }

    if kind == SpectrumKind::Nmr && xs.is_none() {
        let fx = first_x.unwrap();
        let lx = last_x.unwrap();
        let n = y.len();
        let step = if n > 1 {
            (lx - fx) / ((n - 1) as f64)
        } else {
            0.0
        };
        let mut move_ppm = 0.0;
        if let Some(sref) = get_str(".SHIFTREFERENCE") {
            let trimmed = sref.trim().trim_start_matches('(').trim_end_matches(')');
            let parts: Vec<&str> = trimmed.split(',').map(|s| s.trim()).collect();
            if parts.len() == 4
                && matches!(
                    parts[0].to_ascii_uppercase().as_str(),
                    "INTERNAL" | "EXTERNAL"
                )
            {
                if let (Ok(pt_idx), Ok(ref_ppm)) =
                    (parts[2].parse::<usize>(), parts[3].parse::<f64>())
                {
                    if pt_idx >= 1 && pt_idx <= n {
                        move_ppm = ref_ppm - (fx + ((pt_idx - 1) as f64) * step);
                    }
                }
            }
        }
        if move_ppm.is_finite() && move_ppm.abs() > 0.0 {
            first_x = Some(fx + move_ppm);
            last_x = Some(lx + move_ppm);
            if let Some(ref mut pks) = peaks {
                for p in pks.iter_mut() {
                    p.0 += move_ppm;
                }
            }
        }
    }

    let fx_final = xs
        .as_ref()
        .map(|v| v[0])
        .or(first_x)
        .ok_or_else(|| "FIRSTX missing".to_string())?;
    let lx_final = xs
        .as_ref()
        .map(|v| *v.last().unwrap())
        .or(last_x)
        .ok_or_else(|| "LASTX missing".to_string())?;

    if kind == SpectrumKind::Nmr && nucleus.is_none() {
        nucleus = Some(if (lx_final - fx_final).abs() > 40.0 {
            "13C".to_string()
        } else {
            "1H".to_string()
        });
    }

    Ok(Some(Spectrum1d {
        title,
        kind,
        nucleus,
        mhz: mhz.filter(|&f| f > 0.0 && f < 3000.0),
        solvent,
        x_unit: display_unit(&x_unit_raw),
        y_unit: display_unit(&y_unit_raw),
        first_x: fx_final,
        last_x: lx_final,
        y,
        xs,
        sticks,
        peaks,
        integral: None,
    }))
}

/// Read all 1D spectra stored in a JCAMP-DX (`.jdx` / `.dx`) text file (`readJcamp` in `jcamp.ts`).
pub fn read_jcamp(text: &str) -> Result<Vec<Spectrum1d>, String> {
    if text.len() > 64 * 1024 * 1024 {
        return Err("JCAMP-DX file exceeds 64 MB safety limit.".to_string());
    }
    let blocks = parse_jcamp_blocks(text)?;
    let mut out = Vec::new();
    let mut total_pts = 0usize;

    fn walk(
        b: &JcampBlock,
        inherited: &HashMap<String, String>,
        out: &mut Vec<Spectrum1d>,
        total_pts: &mut usize,
    ) -> Result<(), String> {
        if !b.children.is_empty() {
            let mut own = inherited.clone();
            for l in &b.ldrs {
                if !matches!(l.label.as_str(), "TITLE" | "END" | "BLOCKS" | "BLOCKID") {
                    own.insert(l.label.clone(), l.value.clone());
                }
            }
            for c in &b.children {
                walk(c, &own, out, total_pts)?;
            }
            return Ok(());
        }
        if let Some(s) = block_to_spectrum(b, inherited)? {
            *total_pts += s.y.len();
            if *total_pts > MAX_ALL_POINTS {
                return Err("JCAMP-DX file holds too many total points.".to_string());
            }
            if out.len() < MAX_SPECTRA {
                out.push(s);
            }
        }
        Ok(())
    }

    let empty = HashMap::new();
    for b in &blocks {
        walk(b, &empty, &mut out, &mut total_pts)?;
    }
    if out.is_empty() {
        return Err("This JCAMP-DX file holds no readable 1D spectrum.".to_string());
    }
    Ok(out)
}

/// Read a Bruker 1D processed spectrum from `procs`, `1r` binary bytes, and optional `acqus` (`readBruker` in `jcamp.ts`).
pub fn read_bruker_1d(
    procs: &str,
    data_1r: &[u8],
    acqus: &str,
    name: &str,
) -> Result<Spectrum1d, String> {
    let find_param = |txt: &str, key: &str| -> Option<String> {
        for line in txt.lines() {
            if let Some(rest) = line.strip_prefix("##") {
                let rest = rest.strip_prefix('$').unwrap_or(rest);
                if let Some((k, v)) = rest.split_once('=') {
                    if norm_ldr_label(k) == key {
                        return Some(v.trim().to_string());
                    }
                }
            }
        }
        None
    };
    let req_num = |key: &str| -> Result<f64, String> {
        let s = find_param(procs, key)
            .ok_or_else(|| format!("Bruker procs is missing parameter {key}."))?;
        s.parse::<f64>()
            .map_err(|_| format!("Bruker procs parameter {key} is not a number."))
    };
    let si_f = req_num("SI")?;
    if si_f < 2.0 || (si_f - si_f.round()).abs() > 1e-9 {
        return Err("Bruker SI is not a valid integer >= 2.".to_string());
    }
    let si = si_f.round() as usize;
    if si > MAX_JCAMP_POINTS {
        return Err(format!("Bruker SI {si} exceeds {MAX_JCAMP_POINTS}."));
    }
    let offset = req_num("OFFSET")?;
    let sw_p = req_num("SWP")?;
    let sf = req_num("SF")?;
    let nc_proc = find_param(procs, "NCPROC")
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0)
        .clamp(-64, 64);
    let big_endian = find_param(procs, "BYTORDP").as_deref() == Some("1");
    let dtype = find_param(procs, "DTYPP")
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(0);
    let word_size = if dtype == 2 { 8usize } else { 4usize };
    if data_1r.len() != si * word_size {
        return Err(format!(
            "Bruker 1r holds {} points, expected {si}.",
            data_1r.len() / word_size
        ));
    }
    if sf <= 0.0 || sw_p <= 0.0 {
        return Err("Bruker SF and SW_p must be positive.".to_string());
    }
    let scale = 2.0_f64.powi(nc_proc);
    let mut y = Vec::with_capacity(si);
    for i in 0..si {
        let val = if dtype == 2 {
            let chunk: [u8; 8] = data_1r[i * 8..(i + 1) * 8].try_into().unwrap();
            if big_endian {
                f64::from_be_bytes(chunk)
            } else {
                f64::from_le_bytes(chunk)
            }
        } else {
            let chunk: [u8; 4] = data_1r[i * 4..(i + 1) * 4].try_into().unwrap();
            let raw = if big_endian {
                i32::from_be_bytes(chunk)
            } else {
                i32::from_le_bytes(chunk)
            };
            (raw as f64) * scale
        };
        if !val.is_finite() {
            return Err("Non-finite intensity in Bruker 1r.".to_string());
        }
        y.push(val);
    }

    let nuc = parse_nucleus_label(
        find_param(acqus, "NUC1")
            .or_else(|| find_param(procs, "AXNUC"))
            .as_deref(),
    )
    .unwrap_or_else(|| "1H".to_string());
    let solvent = canonical_solvent_name(&find_param(acqus, "SOLVENT").unwrap_or_default());
    let last_x = offset - (((si - 1) as f64) * sw_p) / sf / (si as f64);

    Ok(Spectrum1d {
        title: name.chars().take(200).collect(),
        kind: SpectrumKind::Nmr,
        nucleus: Some(nuc),
        mhz: Some(sf),
        solvent,
        x_unit: "ppm".to_string(),
        y_unit: String::new(),
        first_x: offset,
        last_x,
        y,
        xs: None,
        sticks: false,
        peaks: None,
        integral: None,
    })
}

// --- JCAMP-DX Writers ---

/// Encoding style for JCAMP-DX `(X++(Y..Y))` spectral arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JcampCompression {
    /// Plain ASCII Free Format Numeric (`AFFN`): maximally human-readable.
    Affn,
    /// Standard JCAMP-DX `SQZ` + `DIF` + `DUP` (`DIFDUP`) integer compression with Y-check.
    DifDup,
}

fn encode_sqz(val: i64) -> String {
    let abs_s = val.unsigned_abs().to_string();
    let bytes = abs_s.as_bytes();
    let d0 = (bytes[0] - b'0') as usize;
    let rest = &abs_s[1..];
    if val >= 0 {
        format!("{}{rest}", SQZ_POS[d0] as char)
    } else {
        // SQZ_NEG[0] is 'a' representing -1
        format!("{}{rest}", SQZ_NEG[d0 - 1] as char)
    }
}

fn encode_dif(diff: i64) -> String {
    let abs_s = diff.unsigned_abs().to_string();
    let bytes = abs_s.as_bytes();
    let d0 = (bytes[0] - b'0') as usize;
    let rest = &abs_s[1..];
    if diff >= 0 {
        format!("{}{rest}", DIF_POS[d0] as char)
    } else {
        format!("{}{rest}", DIF_NEG[d0 - 1] as char)
    }
}

fn encode_dup(count: usize) -> String {
    let s = count.to_string();
    let d0 = (s.as_bytes()[0] - b'0') as usize;
    let rest = &s[1..];
    format!("{}{rest}", DUP_CHARS[d0 - 1] as char)
}

/// Write a 1D NMR or general continuous `Spectrum1d` to a standard JCAMP-DX v5.01 `.jdx` string.
pub fn write_jcamp_nmr(spec: &Spectrum1d, compression: JcampCompression) -> String {
    let mut out = String::new();
    let clean_title = if spec.title.trim().is_empty() {
        "Simulated 1D NMR Spectrum"
    } else {
        spec.title.trim()
    };
    out.push_str(&format!("##TITLE= {clean_title}\n"));
    out.push_str("##JCAMP-DX= 5.01\n");

    let data_type = match spec.kind {
        SpectrumKind::Nmr => "NMR SPECTRUM",
        SpectrumKind::Ms => "MASS SPECTRUM",
        SpectrumKind::Ir => "INFRARED SPECTRUM",
        SpectrumKind::Uv => "UV/VIS SPECTRUM",
        SpectrumKind::Other => "SPECTRUM",
    };
    out.push_str(&format!("##DATA TYPE= {data_type}\n"));
    out.push_str("##ORIGIN= nmr-core octet_analyt\n");
    out.push_str("##OWNER= PUBLIC DOMAIN\n");

    if let Some(ref nuc) = spec.nucleus {
        out.push_str(&format!("##.OBSERVE NUCLEUS= ^{nuc}\n"));
    }
    if let Some(mhz) = spec.mhz {
        out.push_str(&format!("##.OBSERVE FREQUENCY= {mhz:.6}\n"));
    }
    if let Some(ref solv) = spec.solvent {
        out.push_str(&format!("##.SOLVENT NAME= {solv}\n"));
    }

    // Stick spectrum or irregularly spaced XYPOINTS
    if spec.sticks || spec.xs.is_some() {
        let x_u = if spec.x_unit == "m/z" {
            "M/Z"
        } else if spec.x_unit.eq_ignore_ascii_case("ppm") {
            "PPM"
        } else {
            &spec.x_unit
        };
        let y_u = if spec.y_unit == "%" {
            "RELATIVE ABUNDANCE"
        } else {
            "ARBITRARY UNITS"
        };
        out.push_str(&format!("##XUNITS= {x_u}\n"));
        out.push_str(&format!("##YUNITS= {y_u}\n"));
        out.push_str("##XFACTOR= 1.0\n");
        out.push_str("##YFACTOR= 1.0\n");
        out.push_str(&format!("##FIRSTX= {:.6}\n", spec.first_x));
        out.push_str(&format!("##LASTX= {:.6}\n", spec.last_x));
        out.push_str(&format!("##NPOINTS= {}\n", spec.y.len()));
        let blk_lbl = if spec.sticks {
            "##PEAK TABLE= (XY..XY)\n"
        } else {
            "##XYPOINTS= (XY..XY)\n"
        };
        out.push_str(blk_lbl);
        for i in 0..spec.y.len() {
            out.push_str(&format!("{:.6}, {:.6}\n", spec.x_at(i), spec.y[i]));
        }
        out.push_str("##END=\n");
        return out;
    }

    let n = spec.y.len();
    let max_abs_y = spec
        .y
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, |a, b| if b > a { b } else { a });
    let y_factor = if compression == JcampCompression::DifDup && max_abs_y > 0.0 {
        max_abs_y / 1_000_000.0
    } else {
        1.0
    };

    out.push_str("##XUNITS= PPM\n");
    out.push_str("##YUNITS= ARBITRARY UNITS\n");
    out.push_str("##XFACTOR= 1.0\n");
    out.push_str(&format!("##YFACTOR= {y_factor:.10e}\n"));
    out.push_str(&format!("##FIRSTX= {:.8}\n", spec.first_x));
    out.push_str(&format!("##LASTX= {:.8}\n", spec.last_x));
    out.push_str(&format!("##NPOINTS= {n}\n"));
    if let Some(&y0) = spec.y.first() {
        out.push_str(&format!("##FIRSTY= {:.6}\n", y0));
    }

    if let Some(ref pks) = spec.peaks {
        if !pks.is_empty() && pks.len() <= MAX_PEAKS {
            out.push_str("##PEAK TABLE= (XY..XY)\n");
            for &(px, py) in pks {
                out.push_str(&format!("{px:.6}, {:.6}\n", py / y_factor));
            }
        }
    }

    out.push_str("##XYDATA= (X++(Y..Y))\n");
    if n > 0 {
        match compression {
            JcampCompression::Affn => {
                let per_line = 8usize;
                for start in (0..n).step_by(per_line) {
                    let end = (start + per_line).min(n);
                    let x_start = spec.x_at(start);
                    let mut line = format!("{x_start:.8}");
                    for &yv in &spec.y[start..end] {
                        line.push(' ');
                        line.push_str(&format!("{:.6}", yv / y_factor));
                    }
                    line.push('\n');
                    out.push_str(&line);
                }
            }
            JcampCompression::DifDup => {
                let ints: Vec<i64> = spec
                    .y
                    .iter()
                    .map(|&v| (v / y_factor).round() as i64)
                    .collect();
                let chunk_pts = 32usize;
                let mut idx = 0usize;
                let mut prev_ended_dif = false;
                while idx < n {
                    let end = (idx + chunk_pts).min(n);
                    let x_line = spec.x_at(if prev_ended_dif { idx - 1 } else { idx });
                    let mut line = format!("{x_line:.6} ");
                    // If previous line ended in DIF mode, emit Y-check value = ints[idx - 1] first
                    let start_k = if prev_ended_dif { idx - 1 } else { idx };
                    line.push_str(&encode_sqz(ints[start_k]));

                    let mut k = start_k + 1;
                    let mut line_has_dif = false;
                    while k < end {
                        let d = ints[k] - ints[k - 1];
                        let mut run = 1usize;
                        while k + run < end && ints[k + run] - ints[k + run - 1] == d {
                            run += 1;
                        }
                        line.push_str(&encode_dif(d));
                        line_has_dif = true;
                        if run > 1 {
                            line.push_str(&encode_dup(run));
                        }
                        k += run;
                    }
                    line.push('\n');
                    out.push_str(&line);
                    prev_ended_dif = line_has_dif;
                    idx = end;
                }
            }
        }
    }
    out.push_str("##END=\n");
    out
}

/// Write a Mass Spectrum (`(m/z, rel_abundance)` peak list) to a JCAMP-DX `.jdx` string.
pub fn write_jcamp_ms(title: &str, peaks_mz_rel: &[(f64, f64)]) -> String {
    let mut sorted = peaks_mz_rel.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let first_x = sorted.first().map(|p| p.0).unwrap_or(0.0);
    let last_x = sorted.last().map(|p| p.0).unwrap_or(0.0);
    let spec = Spectrum1d {
        title: title.to_string(),
        kind: SpectrumKind::Ms,
        nucleus: None,
        mhz: None,
        solvent: None,
        x_unit: "m/z".to_string(),
        y_unit: "%".to_string(),
        first_x,
        last_x,
        xs: Some(sorted.iter().map(|p| p.0).collect()),
        y: sorted.iter().map(|p| p.1).collect(),
        sticks: true,
        peaks: None,
        integral: None,
    };
    write_jcamp_nmr(&spec, JcampCompression::Affn)
}
