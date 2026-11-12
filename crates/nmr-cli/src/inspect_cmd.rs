//! `nmrassign inspect <path> [--json]` command implementation (`docs/specs/12-workbench-ux.md` §1.5).
//!
//! Provenance: Clean-room original CLI wrapper around `nmr_core::classify::classify_nmr_workspace`.

use std::path::PathBuf;

use nmr_core::classify::{classify_nmr_workspace, WorkspaceInventory};

/// Format a human-readable tabular summary of `WorkspaceInventory`.
pub fn format_inventory_table(inv: &WorkspaceInventory) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Workspace: {}  (regime: {}, scanned in {:.2} ms)\n",
        inv.root_name, inv.problem_regime, inv.elapsed_ms
    ));
    if !inv.secondary_regimes.is_empty() {
        let sec: Vec<String> = inv
            .secondary_regimes
            .iter()
            .map(ToString::to_string)
            .collect();
        out.push_str(&format!("Secondary regimes: {}\n", sec.join(", ")));
    }
    out.push_str(&format!(
        "Experiments: {}  |  Auxiliary sequence/STAR/NEF files: {}\n\n",
        inv.experiments.len(),
        inv.auxiliary_files.len()
    ));

    if !inv.experiments.is_empty() {
        out.push_str(&format!(
            "{:<3} {:<30} {:<10} {:<8} {:<9} {:>7} {:<18} {:<14} {:<10} {:<14}\n",
            "Sel", "Path", "Vendor", "Dim", "State", "MHz", "Axes", "PulseClass", "NUS", "Quad"
        ));
        out.push_str(&"-".repeat(128));
        out.push('\n');

        for exp in &inv.experiments {
            let sel = if exp.selected_by_default {
                "[x]"
            } else {
                "[ ]"
            };
            let mhz = exp
                .spectrometer_mhz
                .map(|m| format!("{m:.1}"))
                .unwrap_or_else(|| "-".to_string());
            let axes_str = exp
                .axes
                .iter()
                .map(|a| {
                    if a.nucleus.is_empty() {
                        a.label.clone()
                    } else {
                        a.nucleus.clone()
                    }
                })
                .collect::<Vec<_>>()
                .join(",");
            let short_path = truncate_middle(&exp.relative_path, 30);
            out.push_str(&format!(
                "{:<3} {:<30} {:<10} {:<8} {:<9} {:>7} {:<18} {:<14} {:<10} {:<14}\n",
                sel,
                short_path,
                exp.vendor.as_str(),
                exp.dimensionality.as_str(),
                exp.data_state.as_str(),
                mhz,
                truncate_middle(&axes_str, 18),
                exp.pulse_class.as_str(),
                truncate_middle(&exp.nus.badge, 10),
                truncate_middle(&exp.quadrature_badge, 14),
            ));
        }
        out.push('\n');
    }

    if !inv.auxiliary_files.is_empty() {
        out.push_str("Auxiliary Files:\n");
        for aux in &inv.auxiliary_files {
            out.push_str(&format!(
                "  - {} ({}, {} B): {}\n",
                aux.relative_path, aux.vendor, aux.total_bytes, aux.summary
            ));
        }
        out.push('\n');
    }

    if !inv.missing_input_hints.is_empty() {
        out.push_str("Hints:\n");
        for hint in &inv.missing_input_hints {
            out.push_str(&format!("  ! {hint}\n"));
        }
    }

    out
}

fn truncate_middle(s: &str, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars || max_chars < 5 {
        return s.to_string();
    }
    let keep_left = (max_chars - 3) / 2;
    let keep_right = max_chars - 3 - keep_left;
    let left: String = chars[..keep_left].iter().collect();
    let right: String = chars[chars.len() - keep_right..].iter().collect();
    format!("{left}...{right}")
}

/// Execute `nmrassign inspect <path> [--json]`.
pub fn cmd_inspect(args: &[String]) -> Result<(), String> {
    let mut target_path: Option<PathBuf> = None;
    let mut json_output = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!(
                    "Usage: nmrassign inspect <path> [--json]\n\n\
                     Inspect a spectrometer folder, single spectrum, or .tar/.tar.gz/.tgz archive\n\
                     without reading bulk binary FID or cube payloads."
                );
                return Ok(());
            }
            "--json" | "-j" => {
                json_output = true;
            }
            "--in" | "--path" | "-i" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "missing value after --in/--path".to_string())?;
                target_path = Some(PathBuf::from(val));
            }
            other if other.starts_with('-') => {
                return Err(format!(
                    "unknown inspect option {other:?}; expected <path> [--json]"
                ));
            }
            positional => {
                if target_path.is_some() {
                    return Err(format!(
                        "unexpected extra positional argument {positional:?}; expected single <path>"
                    ));
                }
                target_path = Some(PathBuf::from(positional));
            }
        }
        i += 1;
    }

    let path = target_path.ok_or_else(|| {
        "missing workspace or archive path; usage: nmrassign inspect <path> [--json]".to_string()
    })?;

    let inv = classify_nmr_workspace(&path)?;
    if json_output {
        let json = inv.to_json_pretty()?;
        println!("{json}");
    } else {
        print!("{}", format_inventory_table(&inv));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    use std::process;

    #[test]
    fn test_inspect_cli_table_and_json_modes() {
        let dir = env::temp_dir().join(format!("nmr_cli_inspect_{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        let exp = dir.join("10");
        fs::create_dir_all(&exp).unwrap();
        fs::write(
            exp.join("acqus"),
            "##TITLE= test\n##$PULPROG= <hsqcetfpf3gpsi>\n##$NUC1= <1H>\n##$SFO1= 600.13\n##$BF1= 600.13\n##$SW= 14.0\n##$TD= 2048\n##$FnMODE= 0\n##$FnTYPE= 0\n##END=\n",
        )
        .unwrap();
        fs::write(
            exp.join("acqu2s"),
            "##TITLE= test\n##$NUC1= <15N>\n##$SFO1= 60.81\n##$BF1= 60.81\n##$SW= 32.0\n##$TD= 256\n##$FnMODE= 6\n##END=\n",
        )
        .unwrap();
        fs::write(exp.join("ser"), vec![0u8; 64]).unwrap();

        cmd_inspect(&[dir.to_string_lossy().into_owned()]).unwrap();
        cmd_inspect(&[dir.to_string_lossy().into_owned(), "--json".to_string()]).unwrap();

        let inv = classify_nmr_workspace(&dir).unwrap();
        let tbl = format_inventory_table(&inv);
        assert!(tbl.contains("N15Hsqc"), "got:\n{tbl}");
        assert!(tbl.contains("Echo-Antiecho"), "got:\n{tbl}");

        let _ = fs::remove_dir_all(&dir);
    }
}
