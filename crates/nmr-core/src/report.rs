//! Self-contained HTML + inline vector SVG quality report generator.
//!
//! Provenance: Clean-room original implementation of publication-grade NMR assignment
//! and structure validation reporting per `docs/specs/08-service.md`.
//! Secondary chemical shifts and Chemical Shift Index (CSI) follow Wishart, Sykes & Richards,
//! *Biochemistry* 31, 1647–1651 (1992) and Wishart et al., *J. Biomol. NMR* 5, 67–81 (1995);
//! empirical backbone secondary-shift / torsion correlations follow Spera & Bax,
//! *J. Am. Chem. Soc.* 113, 5490–5492 (1991); Ramachandran $(\phi, \psi)$ basin definitions
//! follow Ramachandran, Ramakrishnan & Sasisekharan, *J. Mol. Biol.* 7, 95–99 (1963)
//! and Lovell et al., *Proteins* 50, 437–450 (2003); disordered ensemble `SSP` and 4-state `δ2D`
//! conformational population tracks follow Marsh, Singh, Jia & Forman-Kay, *Protein Sci.* 15,
//! 2795–2804 (2006) and Camilloni, De Simone, Vranken & Vendruscolo, *Biochemistry* 51,
//! 2224–2231 (2012).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::nmrstar::Reference;
use crate::priors::three_to_one;
use crate::solver::AssignResult;
use crate::spinsystems::SpinSystem;

/// Secondary structure state inferred from smoothed secondary chemical shifts (CSI).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecondaryStructureClass {
    Helix,
    Strand,
    Coil,
}

impl SecondaryStructureClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::Helix => "Alpha-Helix",
            Self::Strand => "Beta-Strand",
            Self::Coil => "Coil/Turn",
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Self::Helix => "H",
            Self::Strand => "E",
            Self::Coil => "C",
        }
    }
}

/// Per-residue $(\phi, \psi)$ backbone torsion angle point for the Ramachandran plot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DihedralPoint {
    pub seq_id: i32,
    pub residue: String,
    pub phi_deg: f64,
    pub psi_deg: f64,
    pub ss_class: SecondaryStructureClass,
}

/// Optional 3D structure ensemble and restraint validation summary for executive reporting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructureReportSummary {
    pub n_conformers: usize,
    pub backbone_rmsd_angstrom: f64,
    pub heavy_atom_rmsd_angstrom: Option<f64>,
    pub n_distance_restraints: usize,
    pub n_ambiguous_restraints: usize,
    pub n_dihedral_restraints: usize,
    pub distance_violations_gt_05a: usize,
    pub dihedral_violations_gt_5deg: usize,
    pub ramachandran_favored_pct: f64,
}

/// Complete input bundle for generating a self-contained HTML + SVG quality report.
#[derive(Clone, Debug)]
pub struct ReportInput {
    pub title: String,
    pub project_name: String,
    pub reference: Reference,
    pub confidence: BTreeMap<usize, f64>,
    pub n_peaks: Option<usize>,
    pub n_spin_systems: Option<usize>,
    pub solver_score: Option<f64>,
    pub elapsed_secs: Option<f64>,
    pub engine_label: Option<String>,
    /// Optional spin-system assembly and placement result for exact observed `i` / `i-1` strip ladders.
    pub systems_and_assignment: Option<(Vec<SpinSystem>, AssignResult)>,
    /// Optional predicted or experimental $(\phi, \psi)$ angles for Ramachandran plotting.
    pub dihedrals: Vec<DihedralPoint>,
    /// Optional multi-model or single-model PDB text; if `dihedrals` is empty and `pdb_coordinates`
    /// is provided, $(\phi, \psi)$ torsions are automatically extracted from the first model.
    pub pdb_coordinates: Option<String>,
    /// Optional 3D structure ensemble summary metrics.
    pub structure_summary: Option<StructureReportSummary>,
    /// Optional explicit IDP/ensemble `SSP` & 4-state `δ2D` conformational propensity report
    /// (rendered as a dedicated `#section-idp-propensity` SVG panel when `Some`; when `None`,
    /// an inline `<g id="svg-idp-propensity-track">` sub-panel is embedded below secondary shifts).
    pub propensity_report: Option<crate::idp_propensity::PropensityReport>,
}

impl ReportInput {
    pub fn new(
        project_name: impl Into<String>,
        reference: Reference,
        confidence: BTreeMap<usize, f64>,
    ) -> Self {
        let proj = project_name.into();
        let title = format!("{proj} — NMR Resonance Assignment & Quality Report");
        Self {
            title,
            project_name: proj,
            reference,
            confidence,
            n_peaks: None,
            n_spin_systems: None,
            solver_score: None,
            elapsed_secs: None,
            engine_label: None,
            systems_and_assignment: None,
            dihedrals: Vec::new(),
            pdb_coordinates: None,
            structure_summary: None,
            propensity_report: None,
        }
    }

    pub fn with_propensity_report(
        mut self,
        report: crate::idp_propensity::PropensityReport,
    ) -> Self {
        self.propensity_report = Some(report);
        self
    }
}

/// Wishart et al. (*J. Biomol. NMR* 1995) random-coil `13CA`, `13CB`, and `13C'` reference shifts (ppm).
pub fn random_coil_ca_cb_c(residue: &str) -> Option<(f64, Option<f64>, f64)> {
    match residue.to_ascii_uppercase().as_str() {
        "ALA" => Some((52.5, Some(19.1), 177.8)),
        "ARG" => Some((56.0, Some(30.9), 176.3)),
        "ASN" => Some((53.1, Some(38.9), 175.2)),
        "ASP" => Some((54.2, Some(41.1), 176.3)),
        "CYS" => Some((57.5, Some(28.6), 174.6)),
        "GLN" => Some((55.7, Some(29.4), 176.0)),
        "GLU" => Some((56.6, Some(29.9), 176.6)),
        "GLY" => Some((45.1, None, 173.9)),
        "HIS" => Some((55.6, Some(29.0), 174.9)),
        "ILE" => Some((61.1, Some(38.8), 176.4)),
        "LEU" => Some((55.1, Some(42.4), 177.6)),
        "LYS" => Some((56.2, Some(33.1), 176.6)),
        "MET" => Some((55.4, Some(32.9), 176.3)),
        "PHE" => Some((57.7, Some(39.6), 175.8)),
        "PRO" => Some((63.3, Some(32.1), 177.3)),
        "SER" => Some((58.3, Some(63.8), 174.6)),
        "THR" => Some((61.8, Some(69.8), 174.7)),
        "TRP" => Some((57.5, Some(29.6), 176.1)),
        "TYR" => Some((57.9, Some(38.8), 175.9)),
        "VAL" => Some((62.2, Some(32.9), 176.3)),
        _ => None,
    }
}

/// Compute per-residue $\Delta\delta_{\text{CA}} - \Delta\delta_{\text{CB}}$ secondary shifts and smoothed CSI classes.
pub fn compute_secondary_shifts(
    ref_data: &Reference,
) -> (Vec<Option<f64>>, Vec<SecondaryStructureClass>) {
    let n = ref_data.sequence.len();
    let mut raw: Vec<Option<f64>> = vec![None; n];

    for (k, res) in ref_data.sequence.iter().enumerate() {
        let Some((rc_ca, rc_cb, _)) = random_coil_ca_cb_c(res) else {
            continue;
        };
        let d_ca = ref_data
            .shifts
            .get(&(k, "CA".to_string()))
            .map(|&ca| ca - rc_ca);
        let d_cb = match rc_cb {
            Some(rcb) => ref_data
                .shifts
                .get(&(k, "CB".to_string()))
                .map(|&cb| cb - rcb),
            None => None,
        };
        raw[k] = match (d_ca, d_cb) {
            (Some(dca), Some(dcb)) => Some(dca - dcb),
            (Some(dca), None) => Some(dca),
            (None, Some(dcb)) => Some(-dcb),
            (None, None) => None,
        };
    }

    // 3-residue window smoothing for robust CSI classification (Wishart et al. 1992)
    let mut classes = vec![SecondaryStructureClass::Coil; n];
    for (k, cls) in classes.iter_mut().enumerate().take(n) {
        let mut acc = 0.0;
        let mut wt = 0.0;
        for (off, w) in [(-1isize, 0.25), (0, 0.50), (1, 0.25)] {
            let idx = (k as isize) + off;
            if idx >= 0 && (idx as usize) < n {
                if let Some(v) = raw[idx as usize] {
                    acc += w * v;
                    wt += w;
                }
            }
        }
        if wt > 0.0 {
            let smoothed = acc / wt;
            if smoothed >= 1.4 {
                *cls = SecondaryStructureClass::Helix;
            } else if smoothed <= -1.4 {
                *cls = SecondaryStructureClass::Strand;
            }
        }
    }

    (raw, classes)
}

/// Estimate continuous $(\phi, \psi)$ torsion restraints/points from secondary chemical shifts
/// (Wishart, Sykes & Richards 1992; Spera & Bax 1991).
pub fn predict_csi_dihedrals(ref_data: &Reference) -> Vec<DihedralPoint> {
    let (sec_shifts, classes) = compute_secondary_shifts(ref_data);
    let n = ref_data.sequence.len();
    let mut out = Vec::new();
    if n < 3 {
        return out;
    }

    for k in 1..(n - 1) {
        let Some(dd) = sec_shifts[k] else {
            continue;
        };
        let res = &ref_data.sequence[k];
        let seq_id = ref_data.seq_ids.get(k).copied().unwrap_or((k + 1) as i32);
        let d_co = random_coil_ca_cb_c(res)
            .and_then(|(_, _, rc_c)| {
                ref_data
                    .shifts
                    .get(&(k, "C".to_string()))
                    .map(|&c| c - rc_c)
            })
            .unwrap_or(0.0);

        let cls = classes[k];
        let (phi, psi) = if res == "PRO" {
            match cls {
                SecondaryStructureClass::Helix => (-61.0, -40.0 + (-2.0 * d_co).clamp(-8.0, 8.0)),
                SecondaryStructureClass::Strand => (-68.0, 138.0 + (-3.0 * dd).clamp(-12.0, 12.0)),
                SecondaryStructureClass::Coil => (-65.0, 142.0 + (-4.0 * d_co).clamp(-15.0, 15.0)),
            }
        } else if res == "GLY" {
            match cls {
                SecondaryStructureClass::Helix => (-63.0, -42.0),
                SecondaryStructureClass::Strand => (-135.0, 150.0),
                SecondaryStructureClass::Coil => {
                    if dd >= 0.0 {
                        (
                            58.0 + (2.5 * dd).clamp(-10.0, 10.0),
                            38.0 + (-2.0 * d_co).clamp(-10.0, 10.0),
                        )
                    } else {
                        (-78.0 + (3.0 * dd).clamp(-15.0, 15.0), 145.0)
                    }
                }
            }
        } else {
            match cls {
                SecondaryStructureClass::Helix => {
                    let p = -63.0 - (1.4 * (dd - 2.5)).clamp(-8.0, 8.0);
                    let s = -42.0 - (1.8 * d_co).clamp(-8.0, 8.0);
                    (p, s)
                }
                SecondaryStructureClass::Strand => {
                    let p = -122.0 + (3.2 * (dd + 2.5)).clamp(-18.0, 18.0);
                    let s = 136.0 - (2.8 * (dd + 2.5)).clamp(-16.0, 16.0);
                    (p, s)
                }
                SecondaryStructureClass::Coil => {
                    if dd >= 0.4 {
                        (
                            -72.0 - (4.0 * dd).clamp(-12.0, 12.0),
                            -26.0 - (5.0 * d_co).clamp(-14.0, 14.0),
                        )
                    } else {
                        (
                            -84.0 + (8.0 * dd).clamp(-25.0, 15.0),
                            132.0 - (6.0 * dd).clamp(-18.0, 18.0),
                        )
                    }
                }
            }
        };

        out.push(DihedralPoint {
            seq_id,
            residue: res.clone(),
            phi_deg: phi,
            psi_deg: psi,
            ss_class: cls,
        });
    }
    out
}

/// Extract $(\phi, \psi)$ backbone torsion angles from the first model of a PDB coordinate string.
pub fn extract_dihedrals_from_pdb(pdb_text: &str) -> Vec<DihedralPoint> {
    #[derive(Default, Clone)]
    struct ResBackbone {
        seq_id: i32,
        res_name: String,
        n: Option<[f64; 3]>,
        ca: Option<[f64; 3]>,
        c: Option<[f64; 3]>,
    }

    #[inline]
    fn pdb_ascii_col(bytes: &[u8], start: usize, end: usize) -> Option<&str> {
        let slice = bytes.get(start..end)?;
        if !slice.is_ascii() {
            return None;
        }
        std::str::from_utf8(slice).ok()
    }

    let mut map: BTreeMap<i32, ResBackbone> = BTreeMap::new();
    for line in pdb_text.lines() {
        if line.starts_with("ENDMDL") && !map.is_empty() {
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
        let atom_name = atom_col.trim();
        if !matches!(atom_name, "N" | "CA" | "C") {
            continue;
        }
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
        let entry = map.entry(seq_id).or_insert_with(|| ResBackbone {
            seq_id,
            res_name: res_name.clone(),
            ..ResBackbone::default()
        });
        match atom_name {
            "N" => entry.n = Some([x, y, z]),
            "CA" => entry.ca = Some([x, y, z]),
            "C" => entry.c = Some([x, y, z]),
            _ => {}
        }
    }

    let residues: Vec<ResBackbone> = map.into_values().collect();
    let mut out = Vec::new();
    if residues.len() < 3 {
        return out;
    }

    for i in 1..(residues.len() - 1) {
        let prev = &residues[i - 1];
        let cur = &residues[i];
        let next = &residues[i + 1];
        if cur.seq_id != prev.seq_id + 1 || next.seq_id != cur.seq_id + 1 {
            continue;
        }
        let (Some(c_prev), Some(n_i), Some(ca_i), Some(c_i), Some(n_next)) =
            (prev.c, cur.n, cur.ca, cur.c, next.n)
        else {
            continue;
        };
        let phi = torsion_deg(c_prev, n_i, ca_i, c_i);
        let psi = torsion_deg(n_i, ca_i, c_i, n_next);
        let ss_class = if (-100.0..=-35.0).contains(&phi) && (-70.0..=-10.0).contains(&psi) {
            SecondaryStructureClass::Helix
        } else if (-170.0..=-70.0).contains(&phi) && (90.0..=180.0).contains(&psi) {
            SecondaryStructureClass::Strand
        } else {
            SecondaryStructureClass::Coil
        };
        out.push(DihedralPoint {
            seq_id: cur.seq_id,
            residue: cur.res_name.clone(),
            phi_deg: phi,
            psi_deg: psi,
            ss_class,
        });
    }
    out
}

fn torsion_deg(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> f64 {
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let norm = |a: [f64; 3]| dot(a, a).sqrt();

    let b0 = sub(p0, p1);
    let b1 = sub(p2, p1);
    let b2 = sub(p3, p2);
    let b1_len = norm(b1).max(1e-12);
    let b1u = [b1[0] / b1_len, b1[1] / b1_len, b1[2] / b1_len];

    let v = [
        b0[0] - dot(b0, b1u) * b1u[0],
        b0[1] - dot(b0, b1u) * b1u[1],
        b0[2] - dot(b0, b1u) * b1u[2],
    ];
    let w = [
        b2[0] - dot(b2, b1u) * b1u[0],
        b2[1] - dot(b2, b1u) * b1u[1],
        b2[2] - dot(b2, b1u) * b1u[2],
    ];
    let x = dot(v, w);
    let y = dot(cross(b1u, v), w);
    y.atan2(x).to_degrees()
}

fn escape_xml(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Write a self-contained HTML + inline SVG quality report to disk.
pub fn write_quality_report_html(
    path: impl AsRef<Path>,
    input: &ReportInput,
) -> Result<(), String> {
    let html = generate_quality_report_html(input);
    fs::write(path.as_ref(), html).map_err(|e| {
        format!(
            "failed to write HTML report {}: {e}",
            path.as_ref().display()
        )
    })
}

/// Generate a self-contained, dependency-free single-file HTML report with inline `<svg>` plots.
pub fn generate_quality_report_html(input: &ReportInput) -> String {
    let ref_data = &input.reference;
    let n_res = ref_data.sequence.len();
    let n_assignable = ref_data
        .sequence
        .iter()
        .enumerate()
        .filter(|&(k, r)| k > 0 && r != "PRO")
        .count();

    let mut assigned_amides = 0usize;
    let mut hc_count = 0usize;
    let mut conf_sum = 0.0f64;
    let mut conf_n = 0usize;
    let mut residues_with_any_shift = 0usize;

    for k in 0..n_res {
        let has_any = ["H", "N", "CA", "CB", "C"]
            .iter()
            .any(|a| ref_data.shifts.contains_key(&(k, (*a).to_string())));
        if has_any {
            residues_with_any_shift += 1;
        }
        let has_amide = ref_data.shifts.contains_key(&(k, "H".to_string()))
            && ref_data.shifts.contains_key(&(k, "N".to_string()));
        if has_amide {
            assigned_amides += 1;
        }
        if let Some(&c) = input.confidence.get(&k) {
            if k > 0 && ref_data.sequence[k] != "PRO" && has_amide {
                conf_sum += c;
                conf_n += 1;
                if c >= 0.80 {
                    hc_count += 1;
                }
            }
        }
    }
    // Fallback if confidence covers carbons without H/N
    if conf_n == 0 && !input.confidence.is_empty() {
        for &c in input.confidence.values() {
            conf_sum += c;
            conf_n += 1;
            if c >= 0.80 {
                hc_count += 1;
            }
        }
    }
    let assigned_count = assigned_amides.max(conf_n);
    let mean_conf = if conf_n > 0 {
        conf_sum / (conf_n as f64)
    } else {
        0.0
    };

    let (sec_shifts, ss_classes) = compute_secondary_shifts(ref_data);
    let dihedrals: Vec<DihedralPoint> = if !input.dihedrals.is_empty() {
        input.dihedrals.clone()
    } else if let Some(ref pdb) = input.pdb_coordinates {
        extract_dihedrals_from_pdb(pdb)
    } else {
        Vec::new()
    };

    let auto_propensity;
    let embedded_propensity = if input.propensity_report.is_none() {
        auto_propensity = crate::idp_propensity::compute_idp_propensity(
            ref_data,
            &crate::idp_propensity::PropensityConfig::default(),
        );
        Some(&auto_propensity)
    } else {
        None
    };

    let svg_ribbon = render_confidence_ribbon_svg(ref_data, &input.confidence);
    let svg_hist = render_confidence_histogram_svg(ref_data, &input.confidence);
    let svg_csi =
        render_secondary_shifts_svg(ref_data, &sec_shifts, &ss_classes, embedded_propensity);
    let svg_strips = render_strip_connectivity_svg(
        ref_data,
        &input.confidence,
        input.systems_and_assignment.as_ref(),
    );
    let svg_rama = render_ramachandran_svg(&dihedrals);

    let mut html = String::with_capacity(65_536);
    html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
    html.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    html.push_str(&format!("<title>{}</title>\n", escape_xml(&input.title)));
    html.push_str(REPORT_CSS);
    html.push_str("</head>\n<body>\n<div class=\"container\">\n");

    // Header banner
    html.push_str("<header class=\"report-header\">\n<div>\n");
    html.push_str(&format!(
        "<div class=\"eyebrow\">NMR Resonance Assignment &amp; Validation Report</div>\n<h1>{}</h1>\n",
        escape_xml(&input.title)
    ));
    let mut sub_parts = vec![format!(
        "Project: <strong>{}</strong>",
        escape_xml(&input.project_name)
    )];
    if let Some(ref eng) = input.engine_label {
        sub_parts.push(format!("Engine: <strong>{}</strong>", escape_xml(eng)));
    }
    if let Some(np) = input.n_peaks {
        sub_parts.push(format!("Peaks: <strong>{np}</strong>"));
    }
    if let Some(ns) = input.n_spin_systems {
        sub_parts.push(format!("Spin Systems: <strong>{ns}</strong>"));
    }
    if let Some(sc) = input.solver_score {
        sub_parts.push(format!("Objective Score: <strong>{sc:.2}</strong>"));
    }
    if let Some(dt) = input.elapsed_secs {
        sub_parts.push(format!("Runtime: <strong>{dt:.3} s</strong>"));
    }
    html.push_str(&format!(
        "<p class=\"subtitle\">{}</p>\n",
        sub_parts.join(" &nbsp;&bull;&nbsp; ")
    ));
    html.push_str("</div>\n</header>\n");

    // 1. Executive Metrics Cards
    let pct_assigned = if n_assignable > 0 {
        100.0 * (assigned_count as f64) / (n_assignable as f64)
    } else {
        0.0
    };
    html.push_str("<section class=\"metrics-grid\" id=\"executive-metrics\">\n");
    html.push_str(&metric_card(
        "Protein Length",
        &format!("{n_res} aa"),
        &format!("{residues_with_any_shift}/{n_res} residues with >=1 shift"),
    ));
    html.push_str(&metric_card(
        "Assignable Amides",
        &format!("{n_assignable}"),
        "Non-proline residues (2..K)",
    ));
    html.push_str(&metric_card(
        "Assigned Amides",
        &format!("{assigned_count} ({pct_assigned:.1}%)"),
        "Placed backbone HN roots",
    ));
    html.push_str(&metric_card(
        "High Confidence (>=0.80)",
        &format!("{hc_count}"),
        "Multi-replica consensus >= 80%",
    ));
    html.push_str(&metric_card(
        "Mean Confidence",
        &format!("{:.1}%", mean_conf * 100.0),
        &format!("Across {conf_n} assigned amides"),
    ));

    // Atom completeness card
    let mut atom_counts = Vec::new();
    for atom in ["H", "N", "CA", "CB", "C"] {
        let cnt = (0..n_res)
            .filter(|&k| ref_data.shifts.contains_key(&(k, atom.to_string())))
            .count();
        atom_counts.push(format!("{atom}:{cnt}"));
    }
    html.push_str(&metric_card(
        "Atom Shift Counts",
        &format!("{} total", ref_data.shifts.len()),
        &atom_counts.join(" · "),
    ));
    html.push_str("</section>\n");

    // Optional Structure Ensemble Summary Cards
    if let Some(ref st) = input.structure_summary {
        html.push_str("<section class=\"panel\" id=\"structure-summary\">\n");
        html.push_str("<h2>3D Structure Ensemble &amp; Restraint Summary</h2>\n");
        html.push_str("<div class=\"metrics-grid\">\n");
        html.push_str(&metric_card(
            "Backbone RMSD",
            &format!("{:.2} &Aring;", st.backbone_rmsd_angstrom),
            &format!("{} conformers", st.n_conformers),
        ));
        if let Some(hrmsd) = st.heavy_atom_rmsd_angstrom {
            html.push_str(&metric_card(
                "Heavy-Atom RMSD",
                &format!("{hrmsd:.2} &Aring;"),
                "All heavy atoms",
            ));
        }
        html.push_str(&metric_card(
            "Distance Restraints",
            &format!("{}", st.n_distance_restraints),
            &format!("{} ambiguous (ADR)", st.n_ambiguous_restraints),
        ));
        html.push_str(&metric_card(
            "Dihedral Restraints",
            &format!("{}", st.n_dihedral_restraints),
            "Backbone PHI / PSI",
        ));
        html.push_str(&metric_card(
            "Restraint Violations",
            &format!(
                "{} NOE / {} dih",
                st.distance_violations_gt_05a, st.dihedral_violations_gt_5deg
            ),
            "> 0.5 A / > 5 deg",
        ));
        html.push_str(&metric_card(
            "Ramachandran Favored",
            &format!("{:.1}%", st.ramachandran_favored_pct),
            "Backbone (phi, psi) quality",
        ));
        html.push_str("</div>\n</section>\n");
    }

    // 2. Sequence Confidence Ribbon
    html.push_str("<section class=\"panel\" id=\"section-ribbon\">\n");
    html.push_str("<h2>1. Sequence Assignment Confidence Ribbon</h2>\n");
    html.push_str("<p class=\"panel-caption\">Per-residue placement consensus across simulated annealing replicas. Hover any tile to inspect exact assigned chemical shifts.</p>\n");
    html.push_str("<div class=\"svg-wrap\">");
    html.push_str(&svg_ribbon);
    html.push_str("</div>\n</section>\n");

    // Two-column row: 3. Confidence Histogram + 6. Ramachandran Plot
    html.push_str("<div class=\"two-col\">\n");
    html.push_str("<section class=\"panel\" id=\"section-histogram\">\n");
    html.push_str("<h2>2. Confidence Distribution Histogram</h2>\n");
    html.push_str("<p class=\"panel-caption\">Distribution of per-residue assignment posterior confidences (0.0–1.0) with high-confidence threshold at 0.80.</p>\n");
    html.push_str("<div class=\"svg-wrap\">");
    html.push_str(&svg_hist);
    html.push_str("</div>\n</section>\n");

    html.push_str("<section class=\"panel\" id=\"section-ramachandran\">\n");
    html.push_str("<h2>5. Ramachandran (&phi;, &psi;) Basin Plot</h2>\n");
    if dihedrals.is_empty() {
        html.push_str("<p class=\"panel-caption\">Shaded favored and allowed &alpha;<sub>R</sub>, &beta;, and &alpha;<sub>L</sub> torsion basins (no dihedral predictions or coordinates supplied).</p>\n");
    } else {
        html.push_str(&format!(
            "<p class=\"panel-caption\">{} backbone (&phi;, &psi;) torsion pairs plotted over favored and allowed &alpha;<sub>R</sub>, &beta;, and &alpha;<sub>L</sub> basins (circles: general, triangles: Gly, squares: Pro).</p>\n",
            dihedrals.len()
        ));
    }
    html.push_str("<div class=\"svg-wrap\">");
    html.push_str(&svg_rama);
    html.push_str("</div>\n</section>\n");
    html.push_str("</div>\n");

    // 4. Secondary Chemical Shift & CSI Bar Chart
    html.push_str("<section class=\"panel\" id=\"section-csi\">\n");
    html.push_str("<h2>3. Secondary Chemical Shifts (&Delta;&delta;<sub>CA</sub> &minus; &Delta;&delta;<sub>CB</sub>) &amp; Chemical Shift Index (CSI)</h2>\n");
    html.push_str("<p class=\"panel-caption\">Sequence-corrected deviations from Wishart random-coil shifts. Sustained positive values (&gt; +1.4 ppm, blue) indicate &alpha;-helices; negative values (&lt; &minus;1.4 ppm, orange) indicate &beta;-strands.</p>\n");
    html.push_str("<div class=\"svg-wrap\">");
    html.push_str(&svg_csi);
    html.push_str("</div>\n</section>\n");

    if let Some(ref prop_rep) = input.propensity_report {
        let svg_prop = render_idp_propensity_svg(prop_rep);
        html.push_str("<section class=\"panel\" id=\"section-idp-propensity\">\n");
        html.push_str("<h2>3b. Ensemble Secondary Structure Propensity (SSP) &amp; 4-State &delta;2D Populations</h2>\n");
        html.push_str("<p class=\"panel-caption\">Stacked per-residue conformational populations (&alpha;-helix, &beta;-strand, PPII, coil; Camilloni et al. 2012) and 5-residue weighted SSP score (Marsh et al. 2006).</p>\n");
        html.push_str("<div class=\"svg-wrap\">");
        html.push_str(&svg_prop);
        html.push_str("</div>\n</section>\n");
    }

    // 5. Sequential CA/CB Strip Connectivity Summary
    html.push_str("<section class=\"panel\" id=\"section-strips\">\n");
    html.push_str("<h2>4. Sequential CA / CB Strip Connectivity Ladder</h2>\n");
    html.push_str("<p class=\"panel-caption\">Sequential walk across consecutive spin systems showing intra-residue <em>i</em> (solid markers) and preceding-residue <em>i&minus;1</em> (hollow markers) <sup>13</sup>C&alpha; and <sup>13</sup>C&beta; ladder links.</p>\n");
    html.push_str("<div class=\"svg-wrap\">");
    html.push_str(&svg_strips);
    html.push_str("</div>\n</section>\n");

    // 7. Chemical Shift Summary Table
    html.push_str("<section class=\"panel\" id=\"section-table\">\n");
    html.push_str("<h2>Assigned Backbone Chemical Shift Table</h2>\n");
    html.push_str("<div class=\"table-scroll\">\n<table class=\"shift-table\">\n<thead>\n<tr>");
    for col in [
        "Seq ID",
        "Residue",
        "Confidence",
        "H (ppm)",
        "N (ppm)",
        "CA (ppm)",
        "CB (ppm)",
        "C' (ppm)",
        "&Delta;CA&minus;&Delta;CB",
        "CSI",
    ] {
        html.push_str(&format!("<th>{col}</th>"));
    }
    html.push_str("</tr>\n</thead>\n<tbody>\n");
    for k in 0..n_res {
        let sid = ref_data.seq_ids.get(k).copied().unwrap_or((k + 1) as i32);
        let res = &ref_data.sequence[k];
        let conf_str = input
            .confidence
            .get(&k)
            .map(|&c| format!("{:.2}", c))
            .unwrap_or_else(|| "&mdash;".to_string());
        let get_s = |atom: &str, prec: usize| -> String {
            ref_data
                .shifts
                .get(&(k, atom.to_string()))
                .map(|&v| format!("{v:.prec$}"))
                .unwrap_or_else(|| "&mdash;".to_string())
        };
        let dd_str = sec_shifts[k]
            .map(|v| format!("{v:+.2}"))
            .unwrap_or_else(|| "&mdash;".to_string());
        let csi = ss_classes[k].code();
        html.push_str(&format!(
            "<tr><td>{sid}</td><td><strong>{}</strong> ({})</td><td>{conf_str}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{dd_str}</td><td><span class=\"badge badge-{csi}\">{csi}</span></td></tr>\n",
            escape_xml(res),
            three_to_one(res),
            get_s("H", 3),
            get_s("N", 2),
            get_s("CA", 2),
            get_s("CB", 2),
            get_s("C", 2),
        ));
    }
    html.push_str("</tbody>\n</table>\n</div>\n</section>\n");

    html.push_str(&format!(
        "<footer class=\"report-footer\">Generated by <code>nmr-core</code> v{} &bull; Clean-room open NMR resonance assignment workbench</footer>\n",
        crate::VERSION
    ));
    html.push_str("</div>\n</body>\n</html>\n");
    html
}

fn metric_card(label: &str, value: &str, detail: &str) -> String {
    format!(
        "<div class=\"metric-card\"><div class=\"metric-label\">{label}</div><div class=\"metric-value\">{value}</div><div class=\"metric-detail\">{detail}</div></div>\n"
    )
}

// ============================================================================
// SVG 1: Sequence Confidence Ribbon
// ============================================================================

fn render_confidence_ribbon_svg(ref_data: &Reference, confidence: &BTreeMap<usize, f64>) -> String {
    let n = ref_data.sequence.len().max(1);
    let cols_per_row = 40usize;
    let n_rows = n.div_ceil(cols_per_row);
    let cell_w = 22.0f64;
    let cell_h = 34.0f64;
    let pad_l = 42.0f64;
    let pad_t = 32.0f64;
    let row_step = 62.0f64;
    let width = pad_l + (cols_per_row as f64) * cell_w + 24.0;
    let height = pad_t + (n_rows as f64) * row_step + 18.0;

    let mut svg = format!(
        "<svg id=\"svg-confidence-ribbon\" xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width:.0} {height:.0}\" width=\"100%\" role=\"img\" aria-label=\"Sequence confidence ribbon\">\n"
    );

    // Legend
    let legend = [
        ("#059669", "High (>=0.80)"),
        ("#d97706", "Medium (0.50-0.79)"),
        ("#dc2626", "Low (<0.50)"),
        ("#64748b", "Pro / N-term"),
        ("#e2e8f0", "Unassigned"),
    ];
    let mut lx = pad_l;
    for (color, text) in legend {
        svg.push_str(&format!(
            "<rect x=\"{lx:.1}\" y=\"8\" width=\"11\" height=\"11\" rx=\"2\" fill=\"{color}\"/>\n\
             <text x=\"{:.1}\" y=\"17\" font-size=\"10.5\" fill=\"#334155\">{}</text>\n",
            lx + 15.0,
            escape_xml(text)
        ));
        lx += 138.0;
    }

    for k in 0..ref_data.sequence.len() {
        let row = k / cols_per_row;
        let col = k % cols_per_row;
        let x = pad_l + (col as f64) * cell_w;
        let y = pad_t + (row as f64) * row_step + 12.0;

        let res = &ref_data.sequence[k];
        let sid = ref_data.seq_ids.get(k).copied().unwrap_or((k + 1) as i32);
        let one = three_to_one(res);
        let conf = confidence.get(&k).copied();

        if col == 0 {
            svg.push_str(&format!(
                "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"10\" fill=\"#64748b\" text-anchor=\"end\">{sid}</text>\n",
                pad_l - 6.0,
                y + 21.0
            ));
        }
        if col.is_multiple_of(5) {
            svg.push_str(&format!(
                "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"9\" fill=\"#64748b\" text-anchor=\"middle\">{sid}</text>\n",
                x + cell_w * 0.5,
                y - 3.0
            ));
        }

        let (fill, text_fill) = match conf {
            Some(c) if c >= 0.80 => ("#059669", "#ffffff"),
            Some(c) if c >= 0.50 => ("#d97706", "#ffffff"),
            Some(c) if c > 0.0 => ("#dc2626", "#ffffff"),
            _ if k == 0 || res == "PRO" => ("#64748b", "#ffffff"),
            _ => ("#e2e8f0", "#334155"),
        };

        let mut shift_parts = Vec::new();
        for atom in ["H", "N", "CA", "CB", "C"] {
            if let Some(&v) = ref_data.shifts.get(&(k, atom.to_string())) {
                shift_parts.push(format!("{atom}={v:.2}"));
            }
        }
        let shift_summary = if shift_parts.is_empty() {
            "no shifts".to_string()
        } else {
            shift_parts.join(", ")
        };
        let conf_txt = conf
            .map(|c| format!("{:.2}", c))
            .unwrap_or_else(|| "n/a".to_string());
        let tooltip = format!("{res}{sid} | conf={conf_txt} | {shift_summary}");

        svg.push_str(&format!(
            "<g class=\"res-tile\"><rect x=\"{:.1}\" y=\"{y:.1}\" width=\"{:.1}\" height=\"{cell_h:.1}\" rx=\"3\" fill=\"{fill}\" stroke=\"#ffffff\" stroke-width=\"1.2\"><title>{}</title></rect>",
            x + 0.6,
            cell_w - 1.2,
            escape_xml(&tooltip)
        ));
        svg.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" font-weight=\"600\" fill=\"{text_fill}\" text-anchor=\"middle\">{one}</text>",
            x + cell_w * 0.5,
            y + 15.0
        ));
        if let Some(c) = conf {
            let bar_w = ((cell_w - 5.0) * c.clamp(0.0, 1.0)).max(1.0);
            svg.push_str(&format!(
                "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{bar_w:.1}\" height=\"4\" rx=\"1.5\" fill=\"rgba(255,255,255,0.82)\"/>",
                x + 2.5,
                y + 24.0
            ));
        }
        svg.push_str("</g>\n");
    }

    svg.push_str("</svg>\n");
    svg
}

// ============================================================================
// SVG 2: Confidence Distribution Histogram (0.0 - 1.0)
// ============================================================================

fn render_confidence_histogram_svg(
    ref_data: &Reference,
    confidence: &BTreeMap<usize, f64>,
) -> String {
    let mut bins = [0usize; 10];
    for (k, res) in ref_data.sequence.iter().enumerate() {
        if k == 0 || res == "PRO" {
            continue;
        }
        let c = confidence.get(&k).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let b = ((c * 10.0).floor() as usize).min(9);
        bins[b] += 1;
    }
    let max_cnt = (*bins.iter().max().unwrap_or(&1)).max(1);

    let width = 440.0f64;
    let height = 270.0f64;
    let pad_l = 46.0f64;
    let pad_r = 20.0f64;
    let pad_t = 24.0f64;
    let pad_b = 44.0f64;
    let plot_w = width - pad_l - pad_r;
    let plot_h = height - pad_t - pad_b;
    let bin_w = plot_w / 10.0;

    let mut svg = format!(
        "<svg id=\"svg-confidence-histogram\" xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width:.0} {height:.0}\" width=\"100%\" role=\"img\" aria-label=\"Confidence distribution histogram\">\n"
    );

    // Horizontal grid lines
    for step in 0..=4 {
        let val = (max_cnt as f64) * (step as f64) / 4.0;
        let y = pad_t + plot_h - plot_h * (step as f64) / 4.0;
        svg.push_str(&format!(
            "<line x1=\"{pad_l:.1}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" stroke=\"#e2e8f0\" stroke-width=\"1\"/>\n\
             <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"10\" fill=\"#64748b\" text-anchor=\"end\">{val:.0}</text>\n",
            pad_l + plot_w,
            pad_l - 6.0,
            y + 3.5
        ));
    }

    for (b, &cnt) in bins.iter().enumerate() {
        let x = pad_l + (b as f64) * bin_w;
        let h = plot_h * (cnt as f64) / (max_cnt as f64);
        let y = pad_t + plot_h - h;
        let fill = if b >= 8 {
            "#059669"
        } else if b >= 5 {
            "#d97706"
        } else {
            "#94a3b8"
        };
        svg.push_str(&format!(
            "<rect x=\"{:.1}\" y=\"{y:.1}\" width=\"{:.1}\" height=\"{h:.1}\" rx=\"2\" fill=\"{fill}\"><title>Bin {:.1}-{:.1}: {cnt} residues</title></rect>\n",
            x + 3.0,
            bin_w - 6.0,
            (b as f64) * 0.1,
            ((b + 1) as f64) * 0.1
        ));
        if cnt > 0 {
            svg.push_str(&format!(
                "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"10\" font-weight=\"600\" fill=\"#1e293b\" text-anchor=\"middle\">{cnt}</text>\n",
                x + bin_w * 0.5,
                (y - 4.0).max(12.0)
            ));
        }
        svg.push_str(&format!(
            "<text x=\"{x:.1}\" y=\"{:.1}\" font-size=\"9.5\" fill=\"#475569\" text-anchor=\"middle\">{:.1}</text>\n",
            pad_t + plot_h + 15.0,
            (b as f64) * 0.1
        ));
    }
    svg.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"9.5\" fill=\"#475569\" text-anchor=\"middle\">1.0</text>\n",
        pad_l + plot_w,
        pad_t + plot_h + 15.0
    ));

    // High-confidence cutoff line at 0.80
    let hc_x = pad_l + 8.0 * bin_w;
    svg.push_str(&format!(
        "<line x1=\"{hc_x:.1}\" y1=\"{pad_t:.1}\" x2=\"{hc_x:.1}\" y2=\"{:.1}\" stroke=\"#0f766e\" stroke-width=\"1.5\" stroke-dasharray=\"4,3\"/>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"9.5\" font-weight=\"600\" fill=\"#0f766e\">HC 0.80</text>\n",
        pad_t + plot_h,
        hc_x + 4.0,
        pad_t + 10.0
    ));

    svg.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" fill=\"#334155\" text-anchor=\"middle\">Assignment Confidence</text>\n",
        pad_l + plot_w * 0.5,
        height - 8.0
    ));
    svg.push_str("</svg>\n");
    svg
}

// ============================================================================
// SVG 3: Secondary Chemical Shifts (dCA - dCB) & CSI Bar Chart
// ============================================================================

fn render_secondary_shifts_svg(
    ref_data: &Reference,
    sec_shifts: &[Option<f64>],
    ss_classes: &[SecondaryStructureClass],
    embedded_propensity: Option<&crate::idp_propensity::PropensityReport>,
) -> String {
    let n = ref_data.sequence.len().max(1);
    let bar_step = (820.0 / (n as f64)).clamp(6.5, 14.0);
    let pad_l = 52.0f64;
    let pad_r = 24.0f64;
    let pad_t = 42.0f64;
    let pad_b = 38.0f64;
    let plot_w = (n as f64) * bar_step;
    let plot_h = 170.0f64;
    let extra_track_h = if embedded_propensity.is_some() {
        172.0f64
    } else {
        0.0f64
    };
    let width = pad_l + plot_w + pad_r;
    let height = pad_t + plot_h + extra_track_h + pad_b;
    let zero_y = pad_t + plot_h * 0.5;
    let max_ppm = 7.0f64;

    let mut svg = format!(
        "<svg id=\"svg-secondary-shifts\" xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width:.0} {height:.0}\" width=\"100%\" role=\"img\" aria-label=\"Secondary chemical shifts and CSI chart\">\n"
    );

    // Legend
    svg.push_str(
        "<rect x=\"52\" y=\"8\" width=\"11\" height=\"11\" rx=\"2\" fill=\"#2563eb\"/>\
         <text x=\"67\" y=\"17\" font-size=\"10.5\" fill=\"#334155\">Alpha-Helix (&Delta;CA&minus;&Delta;CB &gt; 0)</text>\
         <rect x=\"245\" y=\"8\" width=\"11\" height=\"11\" rx=\"2\" fill=\"#ea580c\"/>\
         <text x=\"260\" y=\"17\" font-size=\"10.5\" fill=\"#334155\">Beta-Strand (&Delta;CA&minus;&Delta;CB &lt; 0)</text>\n",
    );

    // Horizontal ppm gridlines
    for ppm in [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0] {
        let y = zero_y - (ppm / max_ppm) * (plot_h * 0.5);
        let stroke = if ppm == 0.0 { "#475569" } else { "#e2e8f0" };
        let sw = if ppm == 0.0 { "1.3" } else { "1.0" };
        svg.push_str(&format!(
            "<line x1=\"{pad_l:.1}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" stroke=\"{stroke}\" stroke-width=\"{sw}\"/>\n\
             <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"9.5\" fill=\"#64748b\" text-anchor=\"end\">{ppm:+.0}</text>\n",
            pad_l + plot_w,
            pad_l - 6.0,
            y + 3.0
        ));
    }

    // Top CSI secondary structure track
    let ss_y = 26.0f64;
    svg.push_str(&format!(
        "<line x1=\"{pad_l:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#cbd5e1\" stroke-width=\"2\"/>\n",
        ss_y + 4.0,
        pad_l + plot_w,
        ss_y + 4.0
    ));
    for (k, &cls) in ss_classes.iter().enumerate() {
        let x = pad_l + (k as f64) * bar_step;
        match cls {
            SecondaryStructureClass::Helix => {
                svg.push_str(&format!(
                    "<rect x=\"{x:.1}\" y=\"{ss_y:.1}\" width=\"{bar_step:.1}\" height=\"8\" rx=\"2\" fill=\"#2563eb\"/>\n"
                ));
            }
            SecondaryStructureClass::Strand => {
                svg.push_str(&format!(
                    "<rect x=\"{x:.1}\" y=\"{ss_y:.1}\" width=\"{bar_step:.1}\" height=\"8\" rx=\"1\" fill=\"#ea580c\"/>\n"
                ));
            }
            SecondaryStructureClass::Coil => {}
        }
    }

    // Diverging bars
    let tick_y = pad_t + plot_h + extra_track_h + 15.0;
    for (k, val_opt) in sec_shifts.iter().enumerate() {
        let res = &ref_data.sequence[k];
        let sid = ref_data.seq_ids.get(k).copied().unwrap_or((k + 1) as i32);
        let x = pad_l + (k as f64) * bar_step;
        if sid % 10 == 0 || k == 0 {
            svg.push_str(&format!(
                "<text x=\"{:.1}\" y=\"{tick_y:.1}\" font-size=\"9.5\" fill=\"#475569\" text-anchor=\"middle\">{sid}</text>\n",
                x + bar_step * 0.5,
            ));
        }
        let Some(v) = val_opt else {
            continue;
        };
        let clamped = v.clamp(-max_ppm, max_ppm);
        let h = (clamped.abs() / max_ppm) * (plot_h * 0.5);
        let y = if clamped >= 0.0 { zero_y - h } else { zero_y };
        let fill = match ss_classes[k] {
            SecondaryStructureClass::Helix => "#2563eb",
            SecondaryStructureClass::Strand => "#ea580c",
            SecondaryStructureClass::Coil => {
                if *v >= 0.0 {
                    "#60a5fa"
                } else {
                    "#fb923c"
                }
            }
        };
        let res_esc = escape_xml(res);
        svg.push_str(&format!(
            "<rect x=\"{:.1}\" y=\"{y:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"{fill}\"><title>{res_esc}{sid}: dCA-dCB = {v:+.2} ppm ({})</title></rect>\n",
            x + 0.8,
            (bar_step - 1.6).max(2.0),
            h.max(1.0),
            ss_classes[k].label()
        ));
    }

    if let Some(prop_rep) = embedded_propensity {
        svg.push_str(&render_propensity_track_group(
            prop_rep,
            pad_l,
            pad_t + plot_h + 22.0,
            bar_step,
            72.0,
            64.0,
        ));
    }

    svg.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"10.5\" fill=\"#334155\" text-anchor=\"middle\">Residue Sequence Number</text>\n",
        pad_l + plot_w * 0.5,
        height - 6.0
    ));
    svg.push_str("</svg>\n");
    svg
}

/// Render an inline `<g id="svg-idp-propensity-track">` group containing stacked 4-state `δ2D`
/// conformational population bars (`p_alpha`, `p_beta`, `p_ppii`, `p_coil`) and `Marsh et al. (2006)`
/// `SSP` curve.
fn render_propensity_track_group(
    report: &crate::idp_propensity::PropensityReport,
    pad_l: f64,
    y_top: f64,
    bar_step: f64,
    pop_h: f64,
    ssp_h: f64,
) -> String {
    let n = report.residues.len().max(1);
    let plot_w = (n as f64) * bar_step;
    let mut g = String::with_capacity(n * 340 + 1024);
    g.push_str("<g id=\"svg-idp-propensity-track\">\n");

    // Mini legend above stacked 4-state bar track
    let legend_y = y_top - 6.0;
    let legend_items = [
        ("#2563eb", "p_alpha"),
        ("#ea580c", "p_beta"),
        ("#0d9488", "p_PPII"),
        ("#cbd5e1", "p_coil"),
        ("#0f172a", "SSP (-1..+1)"),
    ];
    let mut lx = pad_l;
    for (col, label) in legend_items {
        g.push_str(&format!(
            "<rect x=\"{lx:.1}\" y=\"{:.1}\" width=\"9\" height=\"9\" rx=\"1.5\" fill=\"{col}\"/>\
             <text x=\"{:.1}\" y=\"{legend_y:.1}\" font-size=\"9.2\" fill=\"#334155\">{}</text>\n",
            legend_y - 8.0,
            lx + 12.0,
            escape_xml(label)
        ));
        lx += 88.0;
    }

    // 1. Stacked 4-state conformational population bar panel (0.0 .. 1.0)
    g.push_str(&format!(
        "<rect x=\"{pad_l:.1}\" y=\"{y_top:.1}\" width=\"{plot_w:.1}\" height=\"{pop_h:.1}\" fill=\"#f8fafc\" stroke=\"#cbd5e1\" stroke-width=\"0.9\"/>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"8.8\" fill=\"#64748b\" text-anchor=\"end\">1.0</text>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"8.8\" fill=\"#64748b\" text-anchor=\"end\">0.5</text>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"8.8\" fill=\"#64748b\" text-anchor=\"end\">0.0</text>\n",
        pad_l - 5.0,
        y_top + 7.0,
        pad_l - 5.0,
        y_top + pop_h * 0.5 + 3.0,
        pad_l - 5.0,
        y_top + pop_h
    ));

    let bar_w = (bar_step - 1.0).max(1.8);
    for (k, r) in report.residues.iter().enumerate() {
        let x = pad_l + (k as f64) * bar_step + 0.5;
        let opacity = if r.has_data { "0.95" } else { "0.35" };
        let ssp_txt = r
            .ssp_score
            .map(|v| format!("{v:+.2}"))
            .unwrap_or_else(|| "n/a".to_string());
        let data_tag = if r.has_data {
            "assigned"
        } else {
            "unassigned prior"
        };
        let tip = format!(
            "{}{} ({}): SSP={} | p_alpha={:.2}, p_beta={:.2}, p_PPII={:.2}, p_coil={:.2}",
            r.residue, r.seq_id, data_tag, ssp_txt, r.p_alpha, r.p_beta, r.p_ppii, r.p_coil
        );

        let segments = [
            (r.p_alpha, "#2563eb"),
            (r.p_beta, "#ea580c"),
            (r.p_ppii, "#0d9488"),
            (r.p_coil, "#cbd5e1"),
        ];
        let mut y_cur = y_top + pop_h;
        g.push_str(&format!(
            "<g class=\"d2d-col\" fill-opacity=\"{opacity}\"><title>{}</title>",
            escape_xml(&tip)
        ));
        for (prob, color) in segments {
            let h = (prob.clamp(0.0, 1.0) * pop_h).max(0.0);
            if h > 0.05 {
                y_cur -= h;
                g.push_str(&format!(
                    "<rect x=\"{x:.1}\" y=\"{y_cur:.2}\" width=\"{bar_w:.1}\" height=\"{h:.2}\" fill=\"{color}\"/>"
                ));
            }
        }
        g.push_str("</g>\n");
    }

    // 2. Marsh et al. (2006) SSP curve sub-track (-1.0 .. +1.0)
    let ssp_top = y_top + pop_h + 10.0;
    let ssp_zero = ssp_top + ssp_h * 0.5;
    g.push_str(&format!(
        "<rect x=\"{pad_l:.1}\" y=\"{ssp_top:.1}\" width=\"{plot_w:.1}\" height=\"{ssp_h:.1}\" fill=\"#ffffff\" stroke=\"#cbd5e1\" stroke-width=\"0.9\"/>\n\
         <line x1=\"{pad_l:.1}\" y1=\"{ssp_zero:.1}\" x2=\"{:.1}\" y2=\"{ssp_zero:.1}\" stroke=\"#64748b\" stroke-width=\"1.0\" stroke-dasharray=\"3,2\"/>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"8.8\" fill=\"#64748b\" text-anchor=\"end\">+1</text>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"8.8\" fill=\"#64748b\" text-anchor=\"end\">SSP</text>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"8.8\" fill=\"#64748b\" text-anchor=\"end\">-1</text>\n",
        pad_l + plot_w,
        pad_l - 5.0,
        ssp_top + 8.0,
        pad_l - 5.0,
        ssp_zero + 3.0,
        pad_l - 5.0,
        ssp_top + ssp_h
    ));

    // Render contiguous SSP polyline segments across residues with data
    let mut poly_pts = String::new();
    for (k, r) in report.residues.iter().enumerate() {
        if let Some(ssp) = r.ssp_score {
            let cx = pad_l + ((k as f64) + 0.5) * bar_step;
            let cy = ssp_zero - ssp.clamp(-1.0, 1.0) * (ssp_h * 0.46);
            if !poly_pts.is_empty() {
                poly_pts.push(' ');
            }
            poly_pts.push_str(&format!("{cx:.1},{cy:.1}"));
        } else if !poly_pts.is_empty() {
            g.push_str(&format!(
                "<polyline fill=\"none\" stroke=\"#0f172a\" stroke-width=\"1.6\" points=\"{poly_pts}\"/>\n"
            ));
            poly_pts.clear();
        }
    }
    if !poly_pts.is_empty() {
        g.push_str(&format!(
            "<polyline fill=\"none\" stroke=\"#0f172a\" stroke-width=\"1.6\" points=\"{poly_pts}\"/>\n"
        ));
    }

    g.push_str("</g>\n");
    g
}

/// Render a standalone vector SVG panel (`<svg id="svg-idp-propensity">`) displaying stacked
/// 4-state `δ2D` conformational populations (`p_alpha`, `p_beta`, `p_ppii`, `p_coil`) and the
/// `Marsh et al. (2006)` `SSP` curve across the protein sequence.
pub fn render_idp_propensity_svg(report: &crate::idp_propensity::PropensityReport) -> String {
    let n = report.residues.len().max(1);
    let bar_step = (820.0 / (n as f64)).clamp(6.5, 14.0);
    let pad_l = 52.0f64;
    let pad_r = 24.0f64;
    let pad_t = 26.0f64;
    let pad_b = 34.0f64;
    let pop_h = 96.0f64;
    let ssp_h = 80.0f64;
    let plot_w = (n as f64) * bar_step;
    let width = pad_l + plot_w + pad_r;
    let height = pad_t + pop_h + 10.0 + ssp_h + pad_b;

    let mut svg = format!(
        "<svg id=\"svg-idp-propensity\" xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width:.0} {height:.0}\" width=\"100%\" role=\"img\">\n\
         <title>IDP SSP and d2D conformational population track</title>\n"
    );
    svg.push_str(&render_propensity_track_group(
        report, pad_l, pad_t, bar_step, pop_h, ssp_h,
    ));

    let tick_y = pad_t + pop_h + 10.0 + ssp_h + 14.0;
    for (k, r) in report.residues.iter().enumerate() {
        if r.seq_id % 10 == 0 || k == 0 {
            let cx = pad_l + ((k as f64) + 0.5) * bar_step;
            svg.push_str(&format!(
                "<text x=\"{cx:.1}\" y=\"{tick_y:.1}\" font-size=\"9.5\" fill=\"#475569\" text-anchor=\"middle\">{}</text>\n",
                r.seq_id
            ));
        }
    }
    svg.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"10.5\" fill=\"#334155\" text-anchor=\"middle\">Residue Sequence Number</text>\n",
        pad_l + plot_w * 0.5,
        height - 5.0
    ));
    svg.push_str("</svg>\n");
    svg
}

// ============================================================================
// SVG 4: Sequential CA / CB Strip Connectivity Summary
// ============================================================================

fn render_strip_connectivity_svg(
    ref_data: &Reference,
    confidence: &BTreeMap<usize, f64>,
    systems_and_assignment: Option<&(Vec<SpinSystem>, AssignResult)>,
) -> String {
    let n = ref_data.sequence.len().max(1);
    let cols_per_panel = 40usize;
    let n_panels = n.div_ceil(cols_per_panel);
    let strip_w = 21.0f64;
    let pad_l = 48.0f64;
    let pad_r = 20.0f64;
    let panel_h = 165.0f64;
    let panel_gap = 40.0f64;
    let top_legend = 28.0f64;
    let width = pad_l + (cols_per_panel as f64) * strip_w + pad_r;
    let height = top_legend + (n_panels as f64) * (panel_h + panel_gap) + 10.0;

    // Map residue k -> spin system if provided
    let mut sys_at: BTreeMap<usize, &SpinSystem> = BTreeMap::new();
    if let Some((systems, res)) = systems_and_assignment {
        for (s, &pos) in res.position.iter().enumerate() {
            if pos >= 0 {
                sys_at.insert(pos as usize, &systems[s]);
            }
        }
    }

    let c_min = 14.0f64;
    let c_max = 74.0f64;
    let map_y = |ppm: f64, y0: f64| -> f64 {
        let frac = ((ppm - c_min) / (c_max - c_min)).clamp(0.0, 1.0);
        // Conventional NMR 13C axis: upfield (small ppm) at top, downfield (large ppm) at bottom
        y0 + frac * panel_h
    };

    let mut svg = format!(
        "<svg id=\"svg-strip-connectivity\" xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width:.0} {height:.0}\" width=\"100%\" role=\"img\" aria-label=\"Sequential CA and CB strip connectivity\">\n"
    );

    // Legend
    svg.push_str(
        "<circle cx=\"56\" cy=\"13\" r=\"3.5\" fill=\"#2563eb\"/>\
         <text x=\"64\" y=\"17\" font-size=\"10\" fill=\"#334155\">CA(i)</text>\
         <circle cx=\"112\" cy=\"13\" r=\"3.2\" fill=\"#ffffff\" stroke=\"#2563eb\" stroke-width=\"1.5\"/>\
         <text x=\"120\" y=\"17\" font-size=\"10\" fill=\"#334155\">CA(i-1)</text>\
         <rect x=\"176\" y=\"9.5\" width=\"7\" height=\"7\" fill=\"#059669\"/>\
         <text x=\"188\" y=\"17\" font-size=\"10\" fill=\"#334155\">CB(i)</text>\
         <rect x=\"232\" y=\"9.5\" width=\"7\" height=\"7\" fill=\"#ffffff\" stroke=\"#059669\" stroke-width=\"1.5\"/>\
         <text x=\"244\" y=\"17\" font-size=\"10\" fill=\"#334155\">CB(i-1)</text>\
         <line x1=\"300\" y1=\"13\" x2=\"320\" y2=\"13\" stroke=\"#0d9488\" stroke-width=\"1.8\"/>\
         <text x=\"325\" y=\"17\" font-size=\"10\" fill=\"#334155\">Sequential Ladder Link</text>\n",
    );

    for p_idx in 0..n_panels {
        let k_start = p_idx * cols_per_panel;
        let k_end = ((p_idx + 1) * cols_per_panel).min(ref_data.sequence.len());
        let y0 = top_legend + (p_idx as f64) * (panel_h + panel_gap);
        let p_w = ((k_end - k_start) as f64) * strip_w;

        svg.push_str(&format!(
            "<rect x=\"{pad_l:.1}\" y=\"{y0:.1}\" width=\"{p_w:.1}\" height=\"{panel_h:.1}\" fill=\"#f8fafc\" stroke=\"#cbd5e1\" stroke-width=\"1\"/>\n"
        ));

        for ppm in [20.0, 35.0, 50.0, 65.0] {
            let py = map_y(ppm, y0);
            svg.push_str(&format!(
                "<line x1=\"{pad_l:.1}\" y1=\"{py:.1}\" x2=\"{:.1}\" y2=\"{py:.1}\" stroke=\"#e2e8f0\" stroke-width=\"1\"/>\n\
                 <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"9\" fill=\"#64748b\" text-anchor=\"end\">{ppm:.0}</text>\n",
                pad_l + p_w,
                pad_l - 5.0,
                py + 3.0
            ));
        }

        for k in k_start..k_end {
            let col = k - k_start;
            let cx = pad_l + (col as f64 + 0.5) * strip_w;
            let res = &ref_data.sequence[k];
            let sid = ref_data.seq_ids.get(k).copied().unwrap_or((k + 1) as i32);

            // Vertical strip guide
            svg.push_str(&format!(
                "<line x1=\"{cx:.1}\" y1=\"{y0:.1}\" x2=\"{cx:.1}\" y2=\"{:.1}\" stroke=\"#f1f5f9\" stroke-width=\"1\"/>\n",
                y0 + panel_h
            ));

            svg.push_str(&format!(
                "<text x=\"{cx:.1}\" y=\"{:.1}\" font-size=\"9.5\" font-weight=\"600\" fill=\"#334155\" text-anchor=\"middle\">{}</text>\n",
                y0 + panel_h + 12.0,
                three_to_one(res)
            ));
            if col.is_multiple_of(5) {
                svg.push_str(&format!(
                    "<text x=\"{cx:.1}\" y=\"{:.1}\" font-size=\"8.5\" fill=\"#64748b\" text-anchor=\"middle\">{sid}</text>\n",
                    y0 + panel_h + 23.0
                ));
            }

            let ca_i = ref_data.shifts.get(&(k, "CA".to_string())).copied();
            let cb_i = ref_data.shifts.get(&(k, "CB".to_string())).copied();

            // Preceding carbon shifts observed in strip k (either from spin system or shift table when amide k is assigned)
            let has_amide_k = ref_data.shifts.contains_key(&(k, "H".to_string()))
                && ref_data.shifts.contains_key(&(k, "N".to_string()));
            let ca_im1 = if let Some(ss) = sys_at.get(&k) {
                let target_prev = if k > 0 {
                    ref_data.shifts.get(&(k - 1, "CA".to_string())).copied()
                } else {
                    None
                };
                pick_closest_cand(ss, "CA", -1, target_prev)
            } else if k > 0 && has_amide_k {
                ref_data.shifts.get(&(k - 1, "CA".to_string())).copied()
            } else {
                None
            };

            let cb_im1 = if let Some(ss) = sys_at.get(&k) {
                let target_prev = if k > 0 {
                    ref_data.shifts.get(&(k - 1, "CB".to_string())).copied()
                } else {
                    None
                };
                pick_closest_cand(ss, "CB", -1, target_prev)
            } else if k > 0 && has_amide_k {
                ref_data.shifts.get(&(k - 1, "CB".to_string())).copied()
            } else {
                None
            };

            // Sequential walk connector from strip k-1 to strip k within the same panel
            if col > 0 {
                let prev_cx = cx - strip_w;
                if let (Some(prev_ca), Some(cur_ca_im1)) = (
                    ref_data.shifts.get(&(k - 1, "CA".to_string())).copied(),
                    ca_im1,
                ) {
                    if (prev_ca - cur_ca_im1).abs() <= 0.45 {
                        let y1 = map_y(prev_ca, y0);
                        let y2 = map_y(cur_ca_im1, y0);
                        svg.push_str(&format!(
                            "<line x1=\"{prev_cx:.1}\" y1=\"{y1:.1}\" x2=\"{:.1}\" y2=\"{y2:.1}\" stroke=\"#2563eb\" stroke-width=\"1.4\" stroke-opacity=\"0.75\"/>\n",
                            cx - 3.0
                        ));
                    }
                }
                if let (Some(prev_cb), Some(cur_cb_im1)) = (
                    ref_data.shifts.get(&(k - 1, "CB".to_string())).copied(),
                    cb_im1,
                ) {
                    if (prev_cb - cur_cb_im1).abs() <= 0.55 {
                        let y1 = map_y(prev_cb, y0);
                        let y2 = map_y(cur_cb_im1, y0);
                        svg.push_str(&format!(
                            "<line x1=\"{prev_cx:.1}\" y1=\"{y1:.1}\" x2=\"{:.1}\" y2=\"{y2:.1}\" stroke=\"#059669\" stroke-width=\"1.4\" stroke-opacity=\"0.75\"/>\n",
                            cx - 3.0
                        ));
                    }
                }
            }

            let res_esc = escape_xml(res);
            // Draw i-1 hollow markers slightly left of strip center
            if let Some(v) = ca_im1 {
                let py = map_y(v, y0);
                svg.push_str(&format!(
                    "<circle cx=\"{:.1}\" cy=\"{py:.1}\" r=\"2.8\" fill=\"#ffffff\" stroke=\"#2563eb\" stroke-width=\"1.4\"><title>{res_esc}{sid} CA(i-1) = {v:.2} ppm</title></circle>\n",
                    cx - 3.0
                ));
            }
            if let Some(v) = cb_im1 {
                let py = map_y(v, y0);
                svg.push_str(&format!(
                    "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"5.4\" height=\"5.4\" fill=\"#ffffff\" stroke=\"#059669\" stroke-width=\"1.4\"><title>{res_esc}{sid} CB(i-1) = {v:.2} ppm</title></rect>\n",
                    cx - 5.7,
                    py - 2.7
                ));
            }

            // Draw intra-residue i solid markers slightly right of strip center
            if let Some(v) = ca_i {
                let py = map_y(v, y0);
                svg.push_str(&format!(
                    "<circle cx=\"{:.1}\" cy=\"{py:.1}\" r=\"3.2\" fill=\"#2563eb\"><title>{res_esc}{sid} CA(i) = {v:.2} ppm</title></circle>\n",
                    cx + 2.5
                ));
            }
            if let Some(v) = cb_i {
                let py = map_y(v, y0);
                svg.push_str(&format!(
                    "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"6.0\" height=\"6.0\" fill=\"#059669\"><title>{res_esc}{sid} CB(i) = {v:.2} ppm</title></rect>\n",
                    cx - 0.5,
                    py - 3.0
                ));
            }

            // Bottom sequential link status bar
            if k > 0 {
                let linked = confidence.get(&k).copied().unwrap_or(0.0) >= 0.5
                    && (ca_im1.is_some() || cb_im1.is_some());
                if linked && col > 0 {
                    svg.push_str(&format!(
                        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{cx:.1}\" y2=\"{:.1}\" stroke=\"#0d9488\" stroke-width=\"3\" stroke-linecap=\"round\"/>\n",
                        cx - strip_w,
                        y0 + panel_h - 4.0,
                        y0 + panel_h - 4.0
                    ));
                }
            }
        }
    }

    svg.push_str("</svg>\n");
    svg
}

fn pick_closest_cand(ss: &SpinSystem, atom: &str, offset: i32, target: Option<f64>) -> Option<f64> {
    let list = ss.cands.get(&(atom.to_string(), offset))?;
    if list.is_empty() {
        return None;
    }
    if let Some(t) = target {
        list.iter()
            .min_by(|a, b| (a.0 - t).abs().total_cmp(&(b.0 - t).abs()))
            .map(|x| x.0)
    } else {
        Some(list[0].0)
    }
}

// ============================================================================
// SVG 5: Ramachandran (phi, psi) Scatter Plot
// ============================================================================

fn render_ramachandran_svg(dihedrals: &[DihedralPoint]) -> String {
    let size = 320.0f64;
    let pad_l = 44.0f64;
    let pad_r = 18.0f64;
    let pad_t = 18.0f64;
    let pad_b = 42.0f64;
    let plot_s = size - pad_l - pad_r; // 258
    let height = pad_t + plot_s + pad_b;

    let map_x = |phi: f64| pad_l + ((phi + 180.0) / 360.0).clamp(0.0, 1.0) * plot_s;
    let map_y = |psi: f64| pad_t + plot_s - ((psi + 180.0) / 360.0).clamp(0.0, 1.0) * plot_s;

    let mut svg = format!(
        "<svg id=\"svg-ramachandran\" xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {size:.0} {height:.0}\" width=\"100%\" role=\"img\" aria-label=\"Ramachandran phi psi plot\">\n"
    );

    svg.push_str(&format!(
        "<rect x=\"{pad_l:.1}\" y=\"{pad_t:.1}\" width=\"{plot_s:.1}\" height=\"{plot_s:.1}\" fill=\"#f8fafc\" stroke=\"#94a3b8\" stroke-width=\"1.2\"/>\n"
    ));

    // Shaded Allowed + Favored Basins (Ramachandran / Lovell contours)
    let draw_basin = |svg: &mut String,
                      phi_lo: f64,
                      phi_hi: f64,
                      psi_lo: f64,
                      psi_hi: f64,
                      fill: &str,
                      stroke: &str| {
        let x = map_x(phi_lo);
        let w = map_x(phi_hi) - x;
        let y = map_y(psi_hi);
        let h = map_y(psi_lo) - y;
        svg.push_str(&format!(
            "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{w:.1}\" height=\"{h:.1}\" rx=\"6\" fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"0.9\"/>\n"
        ));
    };

    // Allowed regions (lighter tint)
    draw_basin(
        &mut svg,
        -165.0,
        -40.0,
        -75.0,
        30.0,
        "rgba(59,130,246,0.12)",
        "rgba(59,130,246,0.30)",
    ); // alpha_R allowed
    draw_basin(
        &mut svg,
        -175.0,
        -45.0,
        75.0,
        180.0,
        "rgba(234,88,12,0.11)",
        "rgba(234,88,12,0.28)",
    ); // beta + PII allowed
    draw_basin(
        &mut svg,
        -175.0,
        -60.0,
        -180.0,
        -155.0,
        "rgba(234,88,12,0.11)",
        "rgba(234,88,12,0.28)",
    ); // beta wrap-around
    draw_basin(
        &mut svg,
        35.0,
        90.0,
        5.0,
        80.0,
        "rgba(16,185,129,0.12)",
        "rgba(16,185,129,0.30)",
    ); // alpha_L allowed

    // Favored core basins (deeper tint)
    draw_basin(
        &mut svg,
        -100.0,
        -48.0,
        -62.0,
        -18.0,
        "rgba(37,99,235,0.24)",
        "rgba(37,99,235,0.55)",
    ); // alpha_R favored
    draw_basin(
        &mut svg,
        -158.0,
        -72.0,
        98.0,
        172.0,
        "rgba(234,88,12,0.22)",
        "rgba(234,88,12,0.55)",
    ); // beta favored
    draw_basin(
        &mut svg,
        45.0,
        72.0,
        20.0,
        60.0,
        "rgba(5,150,105,0.22)",
        "rgba(5,150,105,0.55)",
    ); // alpha_L favored

    // Basin labels
    svg.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"10\" font-weight=\"600\" fill=\"#1d4ed8\">&alpha;</text>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"10\" font-weight=\"600\" fill=\"#c2410c\">&beta;</text>\n\
         <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"9.5\" font-weight=\"600\" fill=\"#047857\">L&alpha;</text>\n",
        map_x(-115.0),
        map_y(-42.0),
        map_x(-168.0),
        map_y(138.0),
        map_x(76.0),
        map_y(42.0)
    ));

    // Gridlines & ticks at -180, -90, 0, +90, +180
    for deg in [-180.0, -90.0, 0.0, 90.0, 180.0] {
        let gx = map_x(deg);
        let gy = map_y(deg);
        let dash = if deg == 0.0 {
            ""
        } else {
            " stroke-dasharray=\"3,3\""
        };
        let col = if deg == 0.0 { "#94a3b8" } else { "#e2e8f0" };
        if deg.abs() < 180.0 {
            svg.push_str(&format!(
                "<line x1=\"{gx:.1}\" y1=\"{pad_t:.1}\" x2=\"{gx:.1}\" y2=\"{:.1}\" stroke=\"{col}\" stroke-width=\"1\"{dash}/>\n\
                 <line x1=\"{pad_l:.1}\" y1=\"{gy:.1}\" x2=\"{:.1}\" y2=\"{gy:.1}\" stroke=\"{col}\" stroke-width=\"1\"{dash}/>\n",
                pad_t + plot_s,
                pad_l + plot_s
            ));
        }
        svg.push_str(&format!(
            "<text x=\"{gx:.1}\" y=\"{:.1}\" font-size=\"9.5\" fill=\"#475569\" text-anchor=\"middle\">{deg:.0}&deg;</text>\n\
             <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"9.5\" fill=\"#475569\" text-anchor=\"end\">{deg:.0}&deg;</text>\n",
            pad_t + plot_s + 14.0,
            pad_l - 5.0,
            gy + 3.0
        ));
    }

    // Scatter points
    for pt in dihedrals {
        let px = map_x(pt.phi_deg);
        let py = map_y(pt.psi_deg);
        let fill = match pt.ss_class {
            SecondaryStructureClass::Helix => "#1d4ed8",
            SecondaryStructureClass::Strand => "#ea580c",
            SecondaryStructureClass::Coil => "#0f766e",
        };
        let tip = format!(
            "{}{}: phi={:+.1} deg, psi={:+.1} deg ({})",
            pt.residue,
            pt.seq_id,
            pt.phi_deg,
            pt.psi_deg,
            pt.ss_class.label()
        );
        if pt.residue == "GLY" {
            svg.push_str(&format!(
                "<polygon points=\"{px:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\" fill=\"{fill}\" stroke=\"#ffffff\" stroke-width=\"0.8\"><title>{}</title></polygon>\n",
                py - 3.6,
                px - 3.3,
                py + 2.8,
                px + 3.3,
                py + 2.8,
                escape_xml(&tip)
            ));
        } else if pt.residue == "PRO" {
            svg.push_str(&format!(
                "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"6\" height=\"6\" fill=\"{fill}\" stroke=\"#ffffff\" stroke-width=\"0.8\"><title>{}</title></rect>\n",
                px - 3.0,
                py - 3.0,
                escape_xml(&tip)
            ));
        } else {
            svg.push_str(&format!(
                "<circle cx=\"{px:.1}\" cy=\"{py:.1}\" r=\"3.2\" fill=\"{fill}\" stroke=\"#ffffff\" stroke-width=\"0.8\"><title>{}</title></circle>\n",
                escape_xml(&tip)
            ));
        }
    }

    svg.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" fill=\"#334155\" text-anchor=\"middle\">&phi; (deg)</text>\n\
         <text x=\"13\" y=\"{:.1}\" font-size=\"11\" fill=\"#334155\" text-anchor=\"middle\" transform=\"rotate(-90 13 {:.1})\">&psi; (deg)</text>\n",
        pad_l + plot_s * 0.5,
        height - 7.0,
        pad_t + plot_s * 0.5,
        pad_t + plot_s * 0.5
    ));
    svg.push_str("</svg>\n");
    svg
}

const REPORT_CSS: &str = r#"<style>
  :root {
    --bg: #f1f5f9;
    --card: #ffffff;
    --border: #e2e8f0;
    --text: #0f172a;
    --muted: #475569;
    --accent: #0f766e;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 24px 16px 48px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.45;
  }
  .container { max-width: 1120px; margin: 0 auto; }
  .report-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #f8fafc;
    padding: 24px 28px;
    border-radius: 12px;
    margin-bottom: 18px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12);
  }
  .eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 11px;
    color: #5eead4;
    font-weight: 600;
    margin-bottom: 4px;
  }
  .report-header h1 { margin: 0 0 8px; font-size: 22px; font-weight: 700; }
  .subtitle { margin: 0; font-size: 13px; color: #cbd5e1; }
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(168px, 1fr));
    gap: 12px;
    margin-bottom: 18px;
  }
  .metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
  }
  .metric-label { font-size: 11.5px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; }
  .metric-value { font-size: 21px; font-weight: 700; color: #0f172a; margin: 4px 0 2px; }
  .metric-detail { font-size: 11.5px; color: #64748b; }
  .panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 18px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
  }
  .panel h2 { margin: 0 0 4px; font-size: 16px; color: #0f172a; }
  .panel-caption { margin: 0 0 12px; font-size: 12.5px; color: var(--muted); }
  .two-col {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    gap: 18px;
  }
  .svg-wrap { width: 100%; overflow-x: auto; }
  .table-scroll { max-height: 420px; overflow: auto; border: 1px solid var(--border); border-radius: 8px; }
  .shift-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  .shift-table th {
    position: sticky; top: 0;
    background: #f8fafc;
    color: #334155;
    text-align: left;
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    font-weight: 600;
  }
  .shift-table td { padding: 6px 10px; border-bottom: 1px solid #f1f5f9; font-variant-numeric: tabular-nums; }
  .shift-table tr:hover td { background: #f8fafc; }
  .badge { display: inline-block; padding: 1px 7px; border-radius: 999px; font-size: 11px; font-weight: 600; }
  .badge-H { background: #dbeafe; color: #1d4ed8; }
  .badge-E { background: #ffedd5; color: #c2410c; }
  .badge-C { background: #f1f5f9; color: #475569; }
  .report-footer { text-align: center; font-size: 12px; color: #64748b; margin-top: 20px; }
  @media print {
    body { background: #ffffff; padding: 0; }
    .table-scroll { max-height: none; overflow: visible; }
    .panel { break-inside: avoid; box-shadow: none; }
  }
</style>
"#;
