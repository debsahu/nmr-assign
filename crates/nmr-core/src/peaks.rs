//! Peak data model, triple-resonance experiment rules, and TSV I/O.
//!
//! Provenance: Original implementation of the TSV peak format and experiment rules
//! in `docs/specs/03-assignment.md` sections 1 and 4.

use std::fs;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct Peak {
    pub exp: String,
    pub h: f64,
    pub n: f64,
    pub c: Option<f64>,
    pub sign: i32,
}

impl Peak {
    pub fn hsqc(h: f64, n: f64) -> Self {
        Self {
            exp: "HSQC".to_string(),
            h,
            n,
            c: None,
            sign: 1,
        }
    }

    pub fn triple(exp: impl Into<String>, h: f64, n: f64, c: f64, sign: i32) -> Self {
        Self {
            exp: exp.into(),
            h,
            n,
            c: Some(c),
            sign,
        }
    }
}

pub const EXPERIMENT_NAMES: [&str; 7] = [
    "HSQC", "HNCA", "HNCOCA", "HNCACB", "CBCACONH", "HNCO", "HNCACO",
];

/// Expected `(atom, offset, sign)` peaks per residue root for a given experiment.
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
    let mut out = String::from("experiment\tH\tN\tC\tsign\n");
    for p in peaks {
        let c_str = p.c.map(|v| format!("{v:.3}")).unwrap_or_default();
        out.push_str(&format!("{}\t{:.4}\t{:.3}\t{}\t{}\n", p.exp, p.h, p.n, c_str, p.sign));
    }
    fs::write(path.as_ref(), out)
        .map_err(|e| format!("failed to write peaks {}: {e}", path.as_ref().display()))
}

pub fn read_peaks(path: impl AsRef<Path>) -> Result<Vec<Peak>, String> {
    let text = fs::read_to_string(path.as_ref())
        .map_err(|e| format!("failed to read peaks {}: {e}", path.as_ref().display()))?;
    read_peaks_str(&text)
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
            return Err(format!("line {}: expected at least 3 tab-separated columns", line_no + 1));
        }
        let exp = fields[0]
            .trim()
            .to_ascii_uppercase()
            .replace(['(', ')'], "");
        if experiment_rules(&exp).is_none() {
            return Err(format!(
                "line {}: unknown experiment {:?}; expected one of {:?}",
                line_no + 1,
                fields[0],
                EXPERIMENT_NAMES
            ));
        }
        let h = fields[1]
            .trim()
            .parse::<f64>()
            .map_err(|e| format!("line {}: invalid H shift: {e}", line_no + 1))?;
        let n = fields[2]
            .trim()
            .parse::<f64>()
            .map_err(|e| format!("line {}: invalid N shift: {e}", line_no + 1))?;
        let c = if fields.len() > 3 && !fields[3].trim().is_empty() {
            Some(
                fields[3]
                    .trim()
                    .parse::<f64>()
                    .map_err(|e| format!("line {}: invalid C shift: {e}", line_no + 1))?,
            )
        } else {
            None
        };
        let sign = if fields.len() > 4 && !fields[4].trim().is_empty() {
            fields[4]
                .trim()
                .parse::<i32>()
                .map_err(|e| format!("line {}: invalid sign: {e}", line_no + 1))?
        } else {
            1
        };
        out.push(Peak { exp, h, n, c, sign });
    }
    Ok(out)
}
