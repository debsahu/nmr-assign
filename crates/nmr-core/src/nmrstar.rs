//! Minimal NMR-STAR reader and writer for backbone and sidechain chemical shifts.
//!
//! Provenance: Clean-room original implementation of the public STAR / NMR-STAR v3 and v2.1
//! syntax (`loop_`/`stop_` blocks, quoted and semicolon-delimited values; Hall, Allen & Brown,
//! *Acta Crystallogr. A* 1991, 47, 655–685; Ulrich et al., *Nucleic Acids Res.* 2008, 36,
//! D402–D408, "BioMagResBank"), per `docs/specs/01-data-io.md`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::priors::{three_to_one, ATOMS};

#[derive(Clone, Debug, Default)]
pub struct Reference {
    /// Three-letter residue codes, indexed `0..K-1`.
    pub sequence: Vec<String>,
    /// Original sequence numbering (`Seq_ID`).
    pub seq_ids: Vec<i32>,
    /// Assigned backbone/sidechain chemical shifts keyed by `(residue_index, atom_name)`.
    pub shifts: BTreeMap<(usize, String), f64>,
    /// Optional NMR-STAR v3 `_Atom_chem_shift.Ambiguity_code` (`1..=9`) keyed by `(residue_index, atom_name)`.
    pub ambiguity_codes: BTreeMap<(usize, String), u8>,
}

impl Reference {
    pub fn new(sequence: Vec<String>, seq_ids: Vec<i32>) -> Self {
        Self {
            sequence,
            seq_ids,
            shifts: BTreeMap::new(),
            ambiguity_codes: BTreeMap::new(),
        }
    }

    pub fn one_letter(&self) -> String {
        self.sequence.iter().map(|r| three_to_one(r)).collect()
    }
}

fn atom_alias(atom: &str) -> &str {
    match atom {
        "HN" => "H",
        "CO" => "C",
        other => other,
    }
}

fn tokenize_line(line: &str, out: &mut Vec<(String, bool)>) {
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '#' {
            break;
        }
        if c == '\'' || c == '"' {
            let quote = c;
            i += 1;
            let mut buf = String::new();
            while i < n {
                if chars[i] == quote && (i + 1 == n || chars[i + 1].is_whitespace()) {
                    i += 1;
                    break;
                }
                buf.push(chars[i]);
                i += 1;
            }
            out.push((buf, true));
        } else {
            let mut buf = String::new();
            while i < n && !chars[i].is_whitespace() {
                buf.push(chars[i]);
                i += 1;
            }
            out.push((buf, false));
        }
    }
}

fn tokenize(text: &str) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if let Some(rest) = line.strip_prefix(';') {
            let mut buf = vec![rest];
            i += 1;
            while i < lines.len() && !lines[i].starts_with(';') {
                buf.push(lines[i]);
                i += 1;
            }
            out.push((buf.join("\n").trim().to_string(), true));
            i += 1;
            continue;
        }
        tokenize_line(line, &mut out);
        i += 1;
    }
    out
}

fn parse_loops(tokens: &[(String, bool)]) -> Vec<(Vec<String>, Vec<Vec<String>>)> {
    let mut loops = Vec::new();
    let mut i = 0;
    let n = tokens.len();
    while i < n {
        if !tokens[i].1 && tokens[i].0.eq_ignore_ascii_case("loop_") {
            i += 1;
            let mut tags = Vec::new();
            while i < n && !tokens[i].1 && tokens[i].0.starts_with('_') {
                tags.push(tokens[i].0.clone());
                i += 1;
            }
            let k_cols = tags.len().max(1);
            let mut vals = Vec::new();
            while i < n {
                let (ref tok, is_quoted) = tokens[i];
                if !is_quoted
                    && (tok.eq_ignore_ascii_case("stop_")
                        || tok.starts_with("save_")
                        || tok.eq_ignore_ascii_case("loop_")
                        || (vals.len() % k_cols == 0 && tok.starts_with('_')))
                {
                    break;
                }
                vals.push(tok.clone());
                i += 1;
            }
            if !tags.is_empty() && vals.len() >= tags.len() {
                let k = tags.len();
                let n_rows = vals.len() / k;
                let mut rows = Vec::with_capacity(n_rows);
                for r in 0..n_rows {
                    rows.push(vals[r * k..(r + 1) * k].to_vec());
                }
                loops.push((tags, rows));
            }
        } else {
            i += 1;
        }
    }
    loops
}

fn find_col(tags: &[String], names: &[&str]) -> Option<usize> {
    let low: Vec<String> = tags.iter().map(|t| t.to_ascii_lowercase()).collect();
    for name in names {
        let target = name.to_ascii_lowercase();
        let suffix = format!(".{}", target.trim_start_matches('_'));
        for (j, t) in low.iter().enumerate() {
            if t == &target || t.ends_with(&suffix) {
                return Some(j);
            }
        }
    }
    None
}

pub fn read_star(path: impl AsRef<Path>, entity: Option<&str>) -> Result<Reference, String> {
    let bytes = fs::read(path.as_ref())
        .map_err(|e| format!("failed to read {}: {e}", path.as_ref().display()))?;
    let text = String::from_utf8_lossy(&bytes);
    read_star_str(&text, entity).map_err(|e| format!("{} in {}", e, path.as_ref().display()))
}

pub fn read_star_all_atoms(
    path: impl AsRef<Path>,
    entity: Option<&str>,
) -> Result<Reference, String> {
    let bytes = fs::read(path.as_ref())
        .map_err(|e| format!("failed to read {}: {e}", path.as_ref().display()))?;
    let text = String::from_utf8_lossy(&bytes);
    read_star_str_all_atoms(&text, entity)
        .map_err(|e| format!("{} in {}", e, path.as_ref().display()))
}

pub fn read_star_str(text: &str, entity: Option<&str>) -> Result<Reference, String> {
    read_star_str_inner(text, entity, true)
}

pub fn read_star_str_all_atoms(text: &str, entity: Option<&str>) -> Result<Reference, String> {
    read_star_str_inner(text, entity, false)
}

fn read_star_str_inner(
    text: &str,
    entity: Option<&str>,
    backbone_only: bool,
) -> Result<Reference, String> {
    let tokens = tokenize(text);
    let loops = parse_loops(&tokens);

    let mut shift_rows: Vec<(String, i32, String, String, f64, Option<u8>)> = Vec::new();
    let mut seq_rows: Vec<(String, i32, String)> = Vec::new();
    let mut loaded_shift_loop = false;

    for (tags, rows) in &loops {
        if let Some(val_col) = find_col(tags, &["_Atom_chem_shift.Val", "_Chem_shift_value"]) {
            // Use only the primary assigned chemical shift loop so multi-condition BMRB entries do not collide.
            if loaded_shift_loop {
                continue;
            }
            let sid_col = find_col(
                tags,
                &[
                    "_Atom_chem_shift.Seq_ID",
                    "_Atom_chem_shift.Comp_index_ID",
                    "_Residue_seq_code",
                ],
            );
            let comp_col = find_col(tags, &["_Atom_chem_shift.Comp_ID", "_Residue_label"]);
            let atom_col = find_col(tags, &["_Atom_chem_shift.Atom_ID", "_Atom_name"]);
            let ent_col = find_col(tags, &["_Atom_chem_shift.Entity_ID"]);
            let amb_col = find_col(
                tags,
                &[
                    "_Atom_chem_shift.Ambiguity_code",
                    "_Chem_shift_ambiguity_code",
                ],
            );
            let (Some(sid_c), Some(comp_c), Some(atom_c)) = (sid_col, comp_col, atom_col) else {
                continue;
            };
            for r in rows {
                let Ok(sid) = r[sid_c].parse::<i32>() else {
                    continue;
                };
                let Ok(val) = r[val_col].parse::<f64>() else {
                    continue;
                };
                if !val.is_finite() {
                    return Err(format!(
                        "non-finite chemical shift value {:?} at Seq_ID {sid} in NMR-STAR",
                        r[val_col]
                    ));
                }
                let ent = ent_col
                    .map(|c| r[c].clone())
                    .unwrap_or_else(|| "1".to_string());
                let amb = amb_col
                    .and_then(|c| r[c].parse::<u8>().ok())
                    .filter(|&a| (1..=9).contains(&a));
                shift_rows.push((
                    ent,
                    sid,
                    r[comp_c].to_ascii_uppercase(),
                    r[atom_c].to_ascii_uppercase(),
                    val,
                    amb,
                ));
            }
            if !shift_rows.is_empty() {
                loaded_shift_loop = true;
            }
            continue;
        }

        let num_col = find_col(tags, &["_Entity_poly_seq.Num", "_Residue_seq_code"]);
        let mon_col = find_col(tags, &["_Entity_poly_seq.Mon_ID", "_Residue_label"]);
        if let (Some(num_c), Some(mon_c)) = (num_col, mon_col) {
            if find_col(tags, &["_Atom_name"]).is_none() {
                let ent_col = find_col(tags, &["_Entity_poly_seq.Entity_ID"]);
                for r in rows {
                    let Ok(num) = r[num_c].parse::<i32>() else {
                        continue;
                    };
                    let ent = ent_col
                        .map(|c| r[c].clone())
                        .unwrap_or_else(|| "1".to_string());
                    seq_rows.push((ent, num, r[mon_c].to_ascii_uppercase()));
                }
            }
        }
    }

    if shift_rows.is_empty() {
        return Err("no chemical shift loop found".to_string());
    }

    let chosen_entity: String = if let Some(e) = entity {
        e.to_string()
    } else {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for (e, ..) in &shift_rows {
            *counts.entry(e.clone()).or_insert(0) += 1;
        }
        counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(e, _)| e)
            .unwrap_or_else(|| "1".to_string())
    };

    let mut comps: BTreeMap<i32, String> = BTreeMap::new();
    for (e, s, c, _, _, _) in &shift_rows {
        if e == &chosen_entity {
            comps.insert(*s, c.clone());
        }
    }
    let any_matching_seq_ent = seq_rows.iter().any(|(e, _, _)| e == &chosen_entity);
    for (e, s, c) in &seq_rows {
        if e == &chosen_entity || !any_matching_seq_ent {
            comps.entry(*s).or_insert_with(|| c.clone());
        }
    }

    const MAX_STAR_SEQ_SPAN: i64 = 100_000;
    let (&lo, &hi) = (
        comps.keys().next().ok_or("empty sequence in NMR-STAR")?,
        comps
            .keys()
            .next_back()
            .ok_or("empty sequence in NMR-STAR")?,
    );
    let span = i64::from(hi) - i64::from(lo);
    if !(0..=MAX_STAR_SEQ_SPAN).contains(&span) {
        return Err(format!(
            "NMR-STAR Seq_ID span {lo}..={hi} ({} residues) exceeds safety limit ({MAX_STAR_SEQ_SPAN})",
            span.saturating_add(1)
        ));
    }
    let seq_ids: Vec<i32> = (lo..=hi).collect();
    let sequence: Vec<String> = seq_ids
        .iter()
        .map(|s| comps.get(s).cloned().unwrap_or_else(|| "UNK".to_string()))
        .collect();

    let mut ref_data = Reference::new(sequence, seq_ids);
    for (e, s, _, atom_raw, v, amb) in shift_rows {
        let atom = atom_alias(&atom_raw);
        if e == chosen_entity
            && !atom.is_empty()
            && atom != "."
            && atom != "?"
            && (!backbone_only || ATOMS.contains(&atom))
            && s >= lo
            && s <= hi
        {
            let key = ((s - lo) as usize, atom.to_string());
            ref_data.shifts.insert(key.clone(), v);
            if let Some(code) = amb {
                ref_data.ambiguity_codes.insert(key, code);
            }
        }
    }
    Ok(ref_data)
}

pub fn write_star(
    path: impl AsRef<Path>,
    ref_data: &Reference,
    confidence: Option<&BTreeMap<usize, f64>>,
) -> Result<(), String> {
    write_star_with_header(path, ref_data, confidence, None)
}

pub fn write_star_with_header(
    path: impl AsRef<Path>,
    ref_data: &Reference,
    confidence: Option<&BTreeMap<usize, f64>>,
    header_comment: Option<&str>,
) -> Result<(), String> {
    let mut lines = Vec::new();
    if let Some(hdr) = header_comment {
        for l in hdr.lines() {
            lines.push(format!(
                "# {}",
                l.trim_start_matches("# ").trim_start_matches('#')
            ));
        }
        lines.push(String::new());
    }
    lines.extend([
        "data_nmr_assign_output".to_string(),
        String::new(),
        "save_entity_1".to_string(),
        "   _Entity.Sf_category  entity".to_string(),
        "   _Entity.ID           1".to_string(),
        String::new(),
        "   loop_".to_string(),
        "      _Entity_poly_seq.Entity_ID".to_string(),
        "      _Entity_poly_seq.Num".to_string(),
        "      _Entity_poly_seq.Mon_ID".to_string(),
        String::new(),
    ]);
    for (seq_id, mon_id) in ref_data.seq_ids.iter().zip(ref_data.sequence.iter()) {
        lines.push(format!("      1 {:5} {:3}", seq_id, mon_id));
    }
    lines.extend([
        "   stop_".to_string(),
        "save_".to_string(),
        String::new(),
        "save_assigned_chem_shift_list_1".to_string(),
        "   _Assigned_chem_shift_list.Sf_category  assigned_chemical_shifts".to_string(),
        String::new(),
        "   loop_".to_string(),
    ]);
    for t in [
        "ID",
        "Entity_ID",
        "Seq_ID",
        "Comp_ID",
        "Atom_ID",
        "Atom_type",
        "Val",
        "Ambiguity_code",
        "Details",
    ] {
        lines.push(format!("      _Atom_chem_shift.{t}"));
    }
    lines.push(String::new());

    let mut entries: Vec<(usize, &str, f64)> = ref_data
        .shifts
        .iter()
        .map(|(&(idx, ref atom), &v)| (idx, atom.as_str(), v))
        .collect();
    entries.sort_by(|&(ia, aa, _), &(ib, ab, _)| {
        let oa = ATOMS.iter().position(|&a| a == aa).unwrap_or(99);
        let ob = ATOMS.iter().position(|&a| a == ab).unwrap_or(99);
        (ia, oa, aa).cmp(&(ib, ob, ab))
    });

    for (i, (idx, atom, v)) in entries.into_iter().enumerate() {
        let det = confidence
            .and_then(|c| c.get(&idx))
            .map(|&c| format!("conf={c:.2}"))
            .unwrap_or_else(|| ".".to_string());
        let atom_type = atom
            .chars()
            .find(|c| c.is_ascii_alphabetic())
            .map(|c| c.to_string())
            .unwrap_or_else(|| "X".to_string());
        let amb = ref_data
            .ambiguity_codes
            .get(&(idx, atom.to_string()))
            .copied()
            .unwrap_or_else(|| {
                crate::stereospec::default_atom_ambiguity_code(&ref_data.sequence[idx], atom)
            });
        lines.push(format!(
            "      {:5} 1 {:5} {:3} {:4} {} {:9.3} {} {}",
            i + 1,
            ref_data.seq_ids[idx],
            ref_data.sequence[idx],
            atom,
            atom_type,
            v,
            amb,
            det
        ));
    }
    lines.push("   stop_".to_string());
    lines.push("save_".to_string());
    lines.push(String::new());

    fs::write(path.as_ref(), lines.join("\n"))
        .map_err(|e| format!("failed to write {}: {e}", path.as_ref().display()))
}
