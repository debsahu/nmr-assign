//! Peak data model, triple-resonance / sidechain / NOESY experiment rules, and TSV I/O.
//!
//! Provenance: Clean-room original implementation of the TSV peak format and canonical
//! triple-resonance correlation rules (Kay, Ikura, Tschudin & Bax, *J. Magn. Reson.* 1990,
//! 89, 496–514 for `HNCA`/`HNCO`; Grzesiek & Bax, *J. Magn. Reson.* 1992, 99, 201–207 for
//! `CBCA(CO)NH`; Wittekind & Mueller, *J. Magn. Reson. B* 1993, 101, 201–205 for `HNCACB`),
//! per `docs/specs/01-data-io.md`, `03-assignment.md`, `04-structure.md`, and `07-sidechain-assignment.md`.

use std::fs;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct Peak {
    pub exp: String,
    pub h: f64,
    pub n: f64,
    pub c: Option<f64>,
    pub sign: i32,
    /// Optional absolute peak height (amplitude) for intensity-aware `i` / `i-1` & NOE scoring.
    pub height: Option<f64>,
    /// Optional integrated peak volume.
    pub volume: Option<f64>,
    /// Optional second proton dimension (`1H_indirect`) for 3D NOESY / HCCH-TOCSY / H(CCCO)NH.
    pub h2: Option<f64>,
}

impl Peak {
    pub fn hsqc(h: f64, n: f64) -> Self {
        Self {
            exp: "HSQC".to_string(),
            h,
            n,
            c: None,
            sign: 1,
            height: None,
            volume: None,
            h2: None,
        }
    }

    pub fn triple(exp: impl Into<String>, h: f64, n: f64, c: f64, sign: i32) -> Self {
        Self {
            exp: exp.into(),
            h,
            n,
            c: Some(c),
            sign,
            height: None,
            volume: None,
            h2: None,
        }
    }

    pub fn with_intensity(mut self, height: f64, volume: Option<f64>) -> Self {
        self.height = Some(height);
        self.volume = volume;
        self
    }

    pub fn c13_hsqc(h: f64, c: f64, sign: i32) -> Self {
        Self {
            exp: "C13HSQC".to_string(),
            h,
            n: 0.0,
            c: Some(c),
            sign,
            height: None,
            volume: None,
            h2: None,
        }
    }

    pub fn tocsy_3d(
        exp: impl Into<String>,
        h: f64,
        hetero: f64,
        c_or_h2: f64,
        h2: Option<f64>,
    ) -> Self {
        Self {
            exp: exp.into(),
            h,
            n: hetero,
            c: Some(c_or_h2),
            sign: 1,
            height: None,
            volume: None,
            h2,
        }
    }

    pub fn noesy_3d(
        exp: impl Into<String>,
        h_direct: f64,
        heavy: f64,
        h_indirect: f64,
        volume: f64,
    ) -> Self {
        let e = exp.into();
        let (n, c) = if e == "NOESY13C" {
            (0.0, Some(heavy))
        } else {
            (heavy, None)
        };
        Self {
            exp: e,
            h: h_direct,
            n,
            c,
            sign: 1,
            height: Some(volume),
            volume: Some(volume),
            h2: Some(h_indirect),
        }
    }
}

pub const EXPERIMENT_NAMES: [&str; 7] = [
    "HSQC", "HNCA", "HNCOCA", "HNCACB", "CBCACONH", "HNCO", "HNCACO",
];

pub const ALL_EXPERIMENT_NAMES: [&str; 13] = [
    "HSQC",
    "HNCA",
    "HNCOCA",
    "HNCACB",
    "CBCACONH",
    "HNCO",
    "HNCACO",
    "C13HSQC",
    "CCCONH",
    "HCCCONH",
    "HCCHTOCSY",
    "NOESY15N",
    "NOESY13C",
];

pub fn is_known_experiment(exp: &str) -> bool {
    ALL_EXPERIMENT_NAMES.contains(&exp)
}

/// Expected `(atom, offset, sign)` peaks per residue root for a given backbone experiment.
pub fn experiment_rules(exp: &str) -> Option<&'static [(&'static str, i32, i32)]> {
    match exp {
        "HSQC" => Some(&[]),
        "HNCA" => Some(&[("CA", 0, 1), ("CA", -1, 1)]),
        "HNCOCA" => Some(&[("CA", -1, 1)]),
        "HNCACB" => Some(&[("CA", 0, 1), ("CB", 0, -1), ("CA", -1, 1), ("CB", -1, -1)]),
        "CBCACONH" => Some(&[("CA", -1, 1), ("CB", -1, 1)]),
        "HNCO" => Some(&[("C", -1, 1)]),
        "HNCACO" => Some(&[("C", 0, 1), ("C", -1, 1)]),
        _ => None,
    }
}

pub fn is_i_minus_1_only(exp: &str) -> bool {
    matches!(exp, "HNCOCA" | "HNCO" | "CBCACONH")
}

pub fn write_peaks(path: impl AsRef<Path>, peaks: &[Peak]) -> Result<(), String> {
    let has_h2 = peaks.iter().any(|p| p.h2.is_some());
    let has_intensity = peaks
        .iter()
        .any(|p| p.height.is_some() || p.volume.is_some());

    let mut out = if has_h2 {
        String::from("experiment\tH\tN\tC\tsign\theight\tvolume\tH2\n")
    } else if has_intensity {
        String::from("experiment\tH\tN\tC\tsign\theight\tvolume\n")
    } else {
        String::from("experiment\tH\tN\tC\tsign\n")
    };

    for p in peaks {
        let c_str = p.c.map(|v| format!("{v:.3}")).unwrap_or_default();
        if has_h2 {
            let h_str = p.height.map(|v| format!("{v:.2}")).unwrap_or_default();
            let v_str = p.volume.map(|v| format!("{v:.2}")).unwrap_or_default();
            let h2_str = p.h2.map(|v| format!("{v:.4}")).unwrap_or_default();
            out.push_str(&format!(
                "{}\t{:.4}\t{:.3}\t{}\t{}\t{}\t{}\t{}\n",
                p.exp, p.h, p.n, c_str, p.sign, h_str, v_str, h2_str
            ));
        } else if has_intensity {
            let h_str = p.height.map(|v| format!("{v:.2}")).unwrap_or_default();
            let v_str = p.volume.map(|v| format!("{v:.2}")).unwrap_or_default();
            out.push_str(&format!(
                "{}\t{:.4}\t{:.3}\t{}\t{}\t{}\t{}\n",
                p.exp, p.h, p.n, c_str, p.sign, h_str, v_str
            ));
        } else {
            out.push_str(&format!(
                "{}\t{:.4}\t{:.3}\t{}\t{}\n",
                p.exp, p.h, p.n, c_str, p.sign
            ));
        }
    }
    fs::write(path.as_ref(), out)
        .map_err(|e| format!("failed to write peaks {}: {e}", path.as_ref().display()))
}

pub fn read_peaks(path: impl AsRef<Path>) -> Result<Vec<Peak>, String> {
    let text = fs::read_to_string(path.as_ref())
        .map_err(|e| format!("failed to read peaks {}: {e}", path.as_ref().display()))?;
    read_peaks_str(&text)
}

fn parse_opt_f64(
    fields: &[&str],
    idx: usize,
    line_no: usize,
    label: &str,
) -> Result<Option<f64>, String> {
    if fields.len() > idx && !fields[idx].trim().is_empty() && fields[idx].trim() != "." {
        fields[idx]
            .trim()
            .parse::<f64>()
            .map(Some)
            .map_err(|e| format!("line {}: invalid {label}: {e}", line_no + 1))
    } else {
        Ok(None)
    }
}

pub fn read_peaks_str(text: &str) -> Result<Vec<Peak>, String> {
    let mut out = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("experiment") || trimmed.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 {
            return Err(format!(
                "line {}: expected at least 3 tab-separated columns",
                line_no + 1
            ));
        }
        let exp = fields[0]
            .trim()
            .to_ascii_uppercase()
            .replace(['(', ')', '-'], "");
        if !is_known_experiment(&exp) {
            return Err(format!(
                "line {}: unknown experiment {:?}; expected one of {:?}",
                line_no + 1,
                fields[0],
                ALL_EXPERIMENT_NAMES
            ));
        }
        let h = fields[1]
            .trim()
            .parse::<f64>()
            .map_err(|e| format!("line {}: invalid H shift: {e}", line_no + 1))?;
        let n = if fields[2].trim().is_empty() || fields[2].trim() == "." {
            0.0
        } else {
            fields[2]
                .trim()
                .parse::<f64>()
                .map_err(|e| format!("line {}: invalid N shift: {e}", line_no + 1))?
        };
        let c = parse_opt_f64(&fields, 3, line_no, "C shift")?;
        let sign = if fields.len() > 4 && !fields[4].trim().is_empty() && fields[4].trim() != "." {
            fields[4]
                .trim()
                .parse::<i32>()
                .map_err(|e| format!("line {}: invalid sign: {e}", line_no + 1))?
        } else {
            1
        };
        let height = parse_opt_f64(&fields, 5, line_no, "height")?;
        let volume = parse_opt_f64(&fields, 6, line_no, "volume")?;
        let h2 = parse_opt_f64(&fields, 7, line_no, "H2 shift")?;

        out.push(Peak {
            exp,
            h,
            n,
            c,
            sign,
            height,
            volume,
            h2,
        });
    }
    Ok(out)
}
