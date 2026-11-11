//! Biopolymer Sequence Builders: Peptide (`H-Ala-Gly-Pro-Phe-OH` / 1-letter `MQIFVK...`)
//! and DNA/RNA Oligonucleotide (`5'-pATGC-3'` / `AACGUU`) -> All-Atom 2D Chemical Structures
//! with L-Amino Acid and β-D-Ribofuranose / 2'-Deoxy-β-D-Ribofuranose Wedge/Hash Stereocenters.
//!
//! Ported/adapted from Octet-Draw by Sam Ellis (https://github.com/SamChemMan/Octet-Draw, Apache-2.0:
//! `peptide.ts`, `nucleic.ts`, `sequence.ts`).

use super::graph::{ChemError, Molecule};
use super::layout::clean_2d;
use super::smiles_inchi::parse_smiles;

/// Standard 20 canonical L-amino acids: `(one_letter, three_letter, side_chain_smiles_from_ca)`.
pub const AMINO_ACIDS: &[(char, &str, &str)] = &[
    ('A', "Ala", "C"),
    ('R', "Arg", "CCCNC(=N)N"),
    ('N', "Asn", "CC(=O)N"),
    ('D', "Asp", "CC(=O)O"),
    ('C', "Cys", "CS"),
    ('Q', "Gln", "CCC(=O)N"),
    ('E', "Glu", "CCC(=O)O"),
    ('G', "Gly", ""),
    ('H', "His", "CC1=CNC=N1"),
    ('I', "Ile", "[C@@H](C)CC"),
    ('L', "Leu", "CC(C)C"),
    ('K', "Lys", "CCCCN"),
    ('M', "Met", "CCSC"),
    ('F', "Phe", "CC1=CC=CC=C1"),
    ('P', "Pro", ""), // special cyclic imino acid
    ('S', "Ser", "CO"),
    ('T', "Thr", "[C@@H](C)O"),
    ('W', "Trp", "CC1=CNC2=CC=CC=C12"),
    ('Y', "Tyr", "CC1=CC=C(O)C=C1"),
    ('V', "Val", "C(C)C"),
];

const MAX_PEPTIDE_RESIDUES: usize = 120;
const MAX_NUCLEOTIDES: usize = 40;

fn residue_smiles(one_letter: char) -> Result<String, ChemError> {
    if one_letter == 'G' {
        return Ok("NCC(=O)".to_string());
    }
    if one_letter == 'P' {
        return Ok("N1CCC[C@H]1C(=O)".to_string());
    }
    for &(code, _, side) in AMINO_ACIDS {
        if code == one_letter {
            return Ok(format!("N[C@@H]({})C(=O)", side));
        }
    }
    Err(ChemError::new(format!(
        "Unknown amino acid residue code '{}'",
        one_letter
    )))
}

fn three_to_one(tok: &str) -> Option<char> {
    for &(c, three, _) in AMINO_ACIDS {
        if three.eq_ignore_ascii_case(tok) {
            return Some(c);
        }
    }
    None
}

/// Convert a peptide sequence (either 3-letter such as `"H-Ala-Gly-Pro-Phe-OH"`, `"Ac-Ala-Cys-NH2"`,
/// `"Ala-Gly-Val"`, or 1-letter such as `"MQIFVK"`, `"peptide: ACDEFG"`) into an all-atom L-amino acid SMILES string.
pub fn peptide_to_smiles(seq: &str) -> Result<String, ChemError> {
    let mut s = seq.trim();
    if let Some(idx) = s.find(':') {
        let prefix = s[..idx].trim();
        if prefix.eq_ignore_ascii_case("peptide") {
            s = s[idx + 1..].trim();
        }
    }
    let compact: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if compact.is_empty() {
        return Err(ChemError::new("Empty peptide sequence"));
    }

    // Parse optional N-terminus (`H-`, `H2N-`, `Ac-`, `Boc-`, `Fmoc-`, `Cbz-`) and C-terminus (`-OH`, `-NH2`, `-OMe`, `-OEt`)
    let mut body = compact.as_str();
    let mut n_cap = "";
    for (prefix, smi) in [
        ("Ac-", "CC(=O)"),
        ("ac-", "CC(=O)"),
        ("Boc-", "CC(C)(C)OC(=O)"),
        ("Fmoc-", "C1=CC=C2C(=C1)C3=CC=CC=C3C2COC(=O)"),
        ("Cbz-", "C1=CC=CC=C1COC(=O)"),
        ("H2N-", ""),
        ("H-", ""),
        ("h-", ""),
    ] {
        if let Some(rest) = body.strip_prefix(prefix) {
            n_cap = smi;
            body = rest;
            break;
        }
    }

    let mut c_cap = "O"; // free carboxylic acid -OH by default
    for (suffix, smi) in [
        ("-NH2", "N"),
        ("-nh2", "N"),
        ("-OMe", "OC"),
        ("-OEt", "OCC"),
        ("-OH", "O"),
        ("-oh", "O"),
    ] {
        if let Some(rest) = body.strip_suffix(suffix) {
            c_cap = smi;
            body = rest;
            break;
        }
    }

    if body.is_empty() {
        return Err(ChemError::new("Peptide sequence contains no residues"));
    }

    // Determine whether `body` is hyphenated 3-letter codes (`Ala-Gly-Pro-Phe`) or 1-letter codes (`MQIFVK`)
    let letters: Vec<char> =
        if body.contains('-') || (body.len() == 3 && three_to_one(body).is_some()) {
            let mut codes = Vec::new();
            for part in body.split('-') {
                if part.is_empty() {
                    continue;
                }
                if let Some(c) = three_to_one(part) {
                    codes.push(c);
                } else if part.len() == 1 {
                    let ch = part.chars().next().unwrap().to_ascii_uppercase();
                    if AMINO_ACIDS.iter().any(|&(k, _, _)| k == ch) {
                        codes.push(ch);
                    } else {
                        return Err(ChemError::new(format!("Unknown amino acid '{}'", part)));
                    }
                } else {
                    return Err(ChemError::new(format!("Unknown amino acid '{}'", part)));
                }
            }
            codes
        } else {
            let mut codes = Vec::new();
            for ch in body.chars() {
                if !ch.is_ascii_uppercase() {
                    return Err(ChemError::new(format!(
                        "1-letter peptide sequence must use uppercase standard codes (got '{}')",
                        ch
                    )));
                }
                if !AMINO_ACIDS.iter().any(|&(k, _, _)| k == ch) {
                    return Err(ChemError::new(format!(
                        "Unknown 1-letter amino acid code '{}'",
                        ch
                    )));
                }
                codes.push(ch);
            }
            codes
        };

    if letters.is_empty() {
        return Err(ChemError::new("Empty peptide residue list"));
    }
    if letters.len() > MAX_PEPTIDE_RESIDUES {
        return Err(ChemError::new(format!(
            "Peptide sequence length {} exceeds maximum {}",
            letters.len(),
            MAX_PEPTIDE_RESIDUES
        )));
    }

    let mut out = String::from(n_cap);
    for ch in letters {
        out.push_str(&residue_smiles(ch)?);
    }
    out.push_str(c_cap);
    Ok(out)
}

/// Build a full all-atom 2D `Molecule` from a 3-letter (`"H-Ala-Gly-Pro-Phe-OH"`) or 1-letter (`"MQIFVK"`)
/// peptide sequence, complete with ACS zig-zag backbone coordinates and L-amino acid wedge/hash stereocenters!
pub fn build_peptide_2d(seq: &str) -> Result<Molecule, ChemError> {
    let smi = peptide_to_smiles(seq)?;
    let mut mol = parse_smiles(&smi)?;
    mol.name = seq.trim().to_string();
    clean_2d(&mut mol);
    Ok(mol)
}

/// Nucleobase SMILES attached from glycosidic N9 (purines A, G) or N1 (pyrimidines C, T, U), using ring numbers 2 & 3.
fn nucleobase_smiles(base: char) -> Option<&'static str> {
    match base {
        'A' => Some("N2C=NC3=C2N=CN=C3N"),
        'G' => Some("N2C=NC3=C2N=C(N)NC3=O"),
        'C' => Some("N2C=CC(=NC2=O)N"),
        'U' => Some("N2C=CC(=O)NC2=O"),
        'T' => Some("N2C=C(C)C(=O)NC2=O"),
        _ => None,
    }
}

fn nucleoside_unit_smiles(base: char, is_dna: bool) -> Result<String, ChemError> {
    let b_smi = nucleobase_smiles(base)
        .ok_or_else(|| ChemError::new(format!("Unknown nucleobase '{}'", base)))?;
    let c2_prime = if is_dna { "C" } else { "[C@H](O)" };
    Ok(format!("OC[C@H]1O[C@@H]({}){}[C@@H]1O", b_smi, c2_prime))
}

/// Convert a DNA or RNA sequence (e.g. `"ATGC"`, `"5'-pATGC-3'"`, `"dna: ATGC"`, `"rna: AACGUU"`)
/// into a full β-D-furanose phosphodiester SMILES string.
pub fn oligonucleotide_to_smiles(seq: &str, is_dna: bool) -> Result<String, ChemError> {
    let mut s = seq.trim();
    let mut dna_mode = is_dna;
    if let Some(idx) = s.find(':') {
        let prefix = s[..idx].trim();
        if prefix.eq_ignore_ascii_case("dna") {
            dna_mode = true;
            s = s[idx + 1..].trim();
        } else if prefix.eq_ignore_ascii_case("rna") {
            dna_mode = false;
            s = s[idx + 1..].trim();
        }
    }

    let mut bare: String = s
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| if c == '’' || c == '′' { '\'' } else { c })
        .collect();

    if let Some(rest) = bare.strip_prefix("5'-") {
        bare = rest.to_string();
    } else if let Some(rest) = bare.strip_prefix("5'") {
        bare = rest.to_string();
    }
    if let Some(rest) = bare.strip_suffix("-3'") {
        bare = rest.to_string();
    } else if let Some(rest) = bare.strip_suffix("3'") {
        bare = rest.to_string();
    }

    let mut five_phosphate = false;
    let mut three_phosphate = false;
    if let Some(rest) = bare.strip_prefix("p-").or_else(|| bare.strip_prefix('p')) {
        five_phosphate = true;
        bare = rest.to_string();
    }
    if let Some(rest) = bare.strip_suffix("-p").or_else(|| bare.strip_suffix('p')) {
        if !rest.is_empty() {
            three_phosphate = true;
            bare = rest.to_string();
        }
    }

    // Also allow hyphen-separated `dA-dT-dG-dC` or `rA-rU-rG-rC`
    let cleaned_bases: String = if bare.contains('-') {
        let mut tmp = String::new();
        for tok in bare.split('-') {
            let t = tok
                .strip_prefix('d')
                .or_else(|| tok.strip_prefix('r'))
                .unwrap_or(tok);
            tmp.push_str(t);
        }
        tmp
    } else {
        bare
    };

    if cleaned_bases.is_empty() {
        return Err(ChemError::new("Empty oligonucleotide sequence"));
    }
    if cleaned_bases.len() > MAX_NUCLEOTIDES {
        return Err(ChemError::new(format!(
            "Oligonucleotide sequence length {} exceeds maximum {}",
            cleaned_bases.len(),
            MAX_NUCLEOTIDES
        )));
    }

    let mut units = Vec::with_capacity(cleaned_bases.len());
    for raw_ch in cleaned_bases.chars() {
        let mut ch = raw_ch.to_ascii_uppercase();
        if dna_mode && ch == 'U' {
            ch = 'T';
        } else if !dna_mode && ch == 'T' {
            ch = 'U';
        }
        units.push(nucleoside_unit_smiles(ch, dna_mode)?);
    }

    let phosphate = "P(=O)(O)";
    let mut out = String::new();
    if five_phosphate {
        out.push('O');
        out.push_str(phosphate);
    }
    out.push_str(&units.join(phosphate));
    if three_phosphate {
        out.push_str(phosphate);
        out.push('O');
    }
    Ok(out)
}

/// Build a full all-atom 2D `Molecule` from a DNA or RNA sequence (`"5'-pATGC-3'"`, `"AACGUU"`),
/// complete with 2D coordinates and β-D-sugar wedge/hash stereocenters!
pub fn build_oligonucleotide_2d(seq: &str, is_dna: bool) -> Result<Molecule, ChemError> {
    let smi = oligonucleotide_to_smiles(seq, is_dna)?;
    let mut mol = parse_smiles(&smi)?;
    mol.name = seq.trim().to_string();
    clean_2d(&mut mol);
    Ok(mol)
}

/// Unified biopolymer sequence parser (`"peptide: ..."`, `"dna: ..."`, `"rna: ..."`, `"H-Ala-Gly-OH"`, `"5'-ATGC-3'"`).
pub fn parse_biopolymer_sequence(text: &str) -> Result<Molecule, ChemError> {
    let trimmed = text.trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("dna:") || lower.starts_with("5'") {
        build_oligonucleotide_2d(trimmed, true)
    } else if lower.starts_with("rna:") {
        build_oligonucleotide_2d(trimmed, false)
    } else {
        build_peptide_2d(trimmed)
    }
}
