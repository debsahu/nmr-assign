//! ChemDraw XML (`.cdxml`), ChemDraw Binary (`.cdx` — magic `VjCD0100` tag tree parser & writer),
//! and Lossless Octet Container (`.oct`) Reader and Writer.
//!
//! Ported/adapted from Octet-Draw by Sam Ellis (https://github.com/SamChemMan/Octet-Draw, Apache-2.0:
//! `cdxml.ts`, `cdx.ts`, `cdxtags.ts`, `oct.ts`) and the public Cambridgesoft/Revvity CDX & CDXML
//! file format specifications.

use super::graph::{
    symbol_from_atomic_number, Atom, BondKind, Bracket, BracketStyle, BracketUsage, CanvasShape,
    ChemDocument, ChemError, CurvedArrow, DoubleBondGeometry, MechanismAnchor, Molecule, Orbital,
    OrbitalKind, OrbitalPhase, Radical, ReactionArrow, ReactionArrowKind, RichTextBox, ShapeKind,
    TextSpan, TlcPlate, TlcSpot, TlcSpotStyle,
};
use super::layout::ANGSTROM_TO_PT;
use super::stereo::{kekulize, perceive_cip, CipAtomDescriptor, CipBondDescriptor};
use std::collections::BTreeMap;

/// ChemDraw binary `.cdx` 8-byte ASCII magic signature.
pub const CDX_MAGIC: &[u8; 8] = b"VjCD0100";

/// Base64 prefix of `VjCD0100` for detecting base64-wrapped CDX payloads in `read_oct`.
pub const CDX_BASE64_MAGIC: &str = "VmpDRDAxMDA";

const OCT_COMMENT_MARK: &str = "<!--Octet 1 ";
const PAGE_ORIGIN_X_PT: f64 = 144.0;
const PAGE_ORIGIN_Y_PT: f64 = 250.0;

#[inline]
fn ang_to_cdx_xy(x_ang: f64, y_ang: f64) -> (f64, f64) {
    (
        PAGE_ORIGIN_X_PT + x_ang * ANGSTROM_TO_PT,
        PAGE_ORIGIN_Y_PT - y_ang * ANGSTROM_TO_PT,
    )
}

#[inline]
fn cdx_to_ang_xy(cx_pt: f64, cy_pt: f64) -> (f64, f64) {
    (
        (cx_pt - PAGE_ORIGIN_X_PT) / ANGSTROM_TO_PT,
        (PAGE_ORIGIN_Y_PT - cy_pt) / ANGSTROM_TO_PT,
    )
}

// ============================================================================
// 1. Minimal Strict XML AST (`XmlNode`)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct XmlNode {
    pub name: String,
    pub attrs: BTreeMap<String, String>,
    pub kids: Vec<XmlNode>,
    pub text: String,
}

impl XmlNode {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            attrs: BTreeMap::new(),
            kids: Vec::new(),
            text: String::new(),
        }
    }

    pub fn with_attr(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.attrs.insert(k.into(), v.into());
        self
    }
}

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(ch),
        }
    }
    out
}

fn xml_unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut i = 0usize;
    while i < s.len() {
        if s[i..].starts_with('&') {
            if let Some(semi) = s[i..].find(';') {
                let ent = &s[i + 1..i + semi];
                let decoded = match ent {
                    "amp" => Some('&'),
                    "lt" => Some('<'),
                    "gt" => Some('>'),
                    "quot" => Some('"'),
                    "apos" => Some('\''),
                    _ if ent.starts_with("#x") || ent.starts_with("#X") => {
                        u32::from_str_radix(&ent[2..], 16)
                            .ok()
                            .and_then(char::from_u32)
                    }
                    _ if ent.starts_with('#') => {
                        ent[1..].parse::<u32>().ok().and_then(char::from_u32)
                    }
                    _ => None,
                };
                if let Some(c) = decoded {
                    out.push(c);
                    i += semi + 1;
                    continue;
                }
            }
        }
        let ch = s[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Serialize an `XmlNode` tree into formatted CDXML text with XML & DOCTYPE headers.
pub fn write_xml_document(root: &XmlNode) -> String {
    let mut out = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n<!DOCTYPE CDXML SYSTEM \"http://www.cambridgesoft.com/xml/cdxml.dtd\" >\n",
    );
    write_xml_element(root, 0, &mut out);
    out
}

fn write_xml_element(node: &XmlNode, indent: usize, out: &mut String) {
    let pad = "  ".repeat(indent);
    out.push_str(&pad);
    out.push('<');
    out.push_str(&node.name);
    for (k, v) in &node.attrs {
        out.push(' ');
        out.push_str(k);
        out.push_str("=\"");
        out.push_str(&xml_escape(v));
        out.push('"');
    }
    if node.kids.is_empty() && node.text.is_empty() {
        out.push_str("/>\n");
        return;
    }
    out.push('>');
    if node.kids.is_empty() {
        out.push_str(&xml_escape(&node.text));
        out.push_str("</");
        out.push_str(&node.name);
        out.push_str(">\n");
    } else if node.name == "t" {
        // Keep `<t><s ...>...</s></t>` compact without introducing extra whitespace into text runs!
        for kid in &node.kids {
            out.push('<');
            out.push_str(&kid.name);
            for (k, v) in &kid.attrs {
                out.push(' ');
                out.push_str(k);
                out.push_str("=\"");
                out.push_str(&xml_escape(v));
                out.push('"');
            }
            out.push('>');
            out.push_str(&xml_escape(&kid.text));
            out.push_str("</");
            out.push_str(&kid.name);
            out.push('>');
        }
        out.push_str("</t>\n");
    } else {
        out.push('\n');
        for kid in &node.kids {
            write_xml_element(kid, indent + 1, out);
        }
        if !node.text.is_empty() {
            out.push_str(&pad);
            out.push_str("  ");
            out.push_str(&xml_escape(&node.text));
            out.push('\n');
        }
        out.push_str(&pad);
        out.push_str("</");
        out.push_str(&node.name);
        out.push_str(">\n");
    }
}

/// Parse an XML string into its root `XmlNode` (skipping XML declarations, DOCTYPE, and comments).
pub fn parse_xml(xml_text: &str) -> Result<XmlNode, ChemError> {
    let bytes = xml_text.as_bytes();
    let mut pos = 0usize;
    // Skip UTF-8 BOM if present
    if xml_text.starts_with('\u{feff}') {
        pos += 3;
    }

    let mut stack: Vec<XmlNode> = Vec::new();
    let mut root: Option<XmlNode> = None;

    while pos < bytes.len() {
        if let Some(lt_rel) = xml_text[pos..].find('<') {
            let text_chunk = &xml_text[pos..pos + lt_rel];
            if let Some(top) = stack.last_mut() {
                if !text_chunk.is_empty() && (top.name == "s" || !text_chunk.trim().is_empty()) {
                    top.text.push_str(&xml_unescape(text_chunk));
                }
            }
            pos += lt_rel;
        } else {
            break;
        }

        // Now `bytes[pos] == b'<'`
        if xml_text[pos..].starts_with("<!--") {
            let end_c = xml_text[pos + 4..]
                .find("-->")
                .ok_or_else(|| ChemError::new("Unclosed XML comment"))?;
            pos += 4 + end_c + 3;
            continue;
        }
        if xml_text[pos..].starts_with("<![CDATA[") {
            let end_cd = xml_text[pos + 9..]
                .find("]]>")
                .ok_or_else(|| ChemError::new("Unclosed CDATA section"))?;
            if let Some(top) = stack.last_mut() {
                top.text.push_str(&xml_text[pos + 9..pos + 9 + end_cd]);
            }
            pos += 9 + end_cd + 3;
            continue;
        }
        if xml_text[pos..].starts_with("<?") {
            let end_pi = xml_text[pos + 2..]
                .find("?>")
                .ok_or_else(|| ChemError::new("Unclosed XML processing instruction"))?;
            pos += 2 + end_pi + 2;
            continue;
        }
        if xml_text[pos..].starts_with("<!") {
            // Skip `<!DOCTYPE ...>` (handling internal `[...]` subset if present)
            let mut depth = 0i32;
            let mut j = pos + 2;
            while j < bytes.len() {
                if bytes[j] == b'[' {
                    depth += 1;
                } else if bytes[j] == b']' {
                    depth -= 1;
                } else if bytes[j] == b'>' && depth <= 0 {
                    break;
                }
                j += 1;
            }
            pos = (j + 1).min(bytes.len());
            continue;
        }

        // Closing tag `</name>`
        if xml_text[pos..].starts_with("</") {
            let gt_rel = xml_text[pos + 2..]
                .find('>')
                .ok_or_else(|| ChemError::new("Unclosed '</...>' tag"))?;
            let close_name = xml_text[pos + 2..pos + 2 + gt_rel].trim();
            pos += 2 + gt_rel + 1;
            let finished = stack
                .pop()
                .ok_or_else(|| ChemError::new("Unexpected closing XML tag"))?;
            if finished.name != close_name {
                return Err(ChemError::new(format!(
                    "Mismatched XML closing tag: expected '</{}>', found '</{}>'",
                    finished.name, close_name
                )));
            }
            if let Some(parent) = stack.last_mut() {
                parent.kids.push(finished);
            } else {
                root = Some(finished);
                break;
            }
            continue;
        }

        // Opening or self-closing tag `<name attr="val" ...>`
        let (node, self_closing, next_pos) = parse_xml_open_tag(xml_text, pos)?;
        pos = next_pos;
        if self_closing {
            if let Some(parent) = stack.last_mut() {
                parent.kids.push(node);
            } else {
                root = Some(node);
                break;
            }
        } else {
            if stack.len() > 64 {
                return Err(ChemError::new("XML nesting depth exceeds limit (64)"));
            }
            stack.push(node);
        }
    }

    root.ok_or_else(|| ChemError::new("No root XML element found"))
}

fn parse_xml_open_tag(s: &str, start_lt: usize) -> Result<(XmlNode, bool, usize), ChemError> {
    let bytes = s.as_bytes();
    let mut i = start_lt + 1;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let name_start = i;
    while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'>' && bytes[i] != b'/'
    {
        i += 1;
    }
    if i == name_start {
        return Err(ChemError::new("Empty XML tag name"));
    }
    let mut node = XmlNode::new(&s[name_start..i]);

    loop {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            return Err(ChemError::new("Unexpected EOF inside XML tag"));
        }
        if bytes[i] == b'>' {
            return Ok((node, false, i + 1));
        }
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'>' {
            return Ok((node, true, i + 2));
        }

        let k_start = i;
        while i < bytes.len()
            && !bytes[i].is_ascii_whitespace()
            && bytes[i] != b'='
            && bytes[i] != b'>'
            && bytes[i] != b'/'
        {
            i += 1;
        }
        let key = s[k_start..i].to_string();
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'=' {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= bytes.len() || (bytes[i] != b'"' && bytes[i] != b'\'') {
                return Err(ChemError::new("Expected quoted XML attribute value"));
            }
            let quote = bytes[i];
            i += 1;
            let v_start = i;
            while i < bytes.len() && bytes[i] != quote {
                i += 1;
            }
            if i >= bytes.len() {
                return Err(ChemError::new("Unclosed attribute quote in XML"));
            }
            let val = xml_unescape(&s[v_start..i]);
            i += 1;
            node.attrs.insert(key, val);
        } else if !key.is_empty() {
            node.attrs.insert(key, "yes".to_string());
        }
    }
}

// ============================================================================
// 2. ChemDocument <-> CDXML AST (`doc_to_cdxml_ast` & `cdxml_ast_to_doc`)
// ============================================================================

/// Serialize a `ChemDocument` to a standard ChemDraw `.cdxml` string.
pub fn write_cdxml(doc: &ChemDocument) -> String {
    let ast = doc_to_cdxml_ast(doc);
    write_xml_document(&ast)
}

/// Read a ChemDraw `.cdxml` string into a `ChemDocument`.
pub fn read_cdxml(cdxml_text: &str) -> Result<ChemDocument, ChemError> {
    let ast = parse_xml(cdxml_text)?;
    cdxml_ast_to_doc(&ast)
}

pub fn doc_to_cdxml_ast(doc: &ChemDocument) -> XmlNode {
    let mut next_id: i32 = 1;
    let mut alloc_id = || -> i32 {
        let v = next_id;
        next_id += 1;
        v
    };

    let mut root = XmlNode::new("CDXML")
        .with_attr("CreationProgram", "OctetMol 0.2")
        .with_attr("BondLength", "14.4")
        .with_attr("LineWidth", "0.6")
        .with_attr("BoldWidth", "2")
        .with_attr("ChainAngle", "120")
        .with_attr("BondSpacing", "18")
        .with_attr("LabelFont", "3")
        .with_attr("LabelSize", "10")
        .with_attr("CaptionFont", "3")
        .with_attr("CaptionSize", "10");
    if !doc.title.is_empty() {
        root.attrs.insert("Name".to_string(), doc.title.clone());
    }

    // Standard fonttable & colortable so both ChemDraw and `cdx_to_xml` know font 3 (Arial UTF-8 / ISO-8859-1) & colors!
    let mut fonttable = XmlNode::new("fonttable");
    fonttable.kids.push(
        XmlNode::new("font")
            .with_attr("id", "3")
            .with_attr("charset", "utf-8")
            .with_attr("name", "Arial"),
    );
    root.kids.push(fonttable);

    let mut colortable = XmlNode::new("colortable");
    colortable.kids.push(
        XmlNode::new("color")
            .with_attr("r", "1")
            .with_attr("g", "1")
            .with_attr("b", "1"),
    );
    colortable.kids.push(
        XmlNode::new("color")
            .with_attr("r", "0")
            .with_attr("g", "0")
            .with_attr("b", "0"),
    );
    root.kids.push(colortable);

    let mut page = XmlNode::new("page")
        .with_attr("id", alloc_id().to_string())
        .with_attr("BoundingBox", "0 0 540 720")
        .with_attr("WidthPages", "1")
        .with_attr("HeightPages", "1");

    // Track `(mol_idx, atom_idx) -> cdxml_id` and `(mol_idx, bond_idx) -> cdxml_id` for anchors/brackets!
    let mut atom_ids: BTreeMap<(usize, usize), i32> = BTreeMap::new();
    let mut bond_ids: BTreeMap<(usize, usize), i32> = BTreeMap::new();

    // 1. Molecules -> `<fragment>`
    for (mi, mol) in doc.molecules.iter().enumerate() {
        let frag = molecule_to_cdxml_fragment(mol, mi, &mut alloc_id, &mut atom_ids, &mut bond_ids);
        page.kids.push(frag);
    }

    // 2. Reaction Arrows -> `<arrow>`
    for ra in &doc.reaction_arrows {
        let id = alloc_id();
        let (tx, ty) = ang_to_cdx_xy(ra.tail.0, ra.tail.1);
        let (hx, hy) = ang_to_cdx_xy(ra.head.0, ra.head.1);
        let min_x = tx.min(hx);
        let min_y = ty.min(hy);
        let max_x = tx.max(hx);
        let max_y = ty.max(hy);
        let mut arr = XmlNode::new("arrow")
            .with_attr("id", id.to_string())
            .with_attr(
                "BoundingBox",
                format!("{:.2} {:.2} {:.2} {:.2}", min_x, min_y, max_x, max_y),
            )
            .with_attr("Tail3D", format!("{:.2} {:.2} 0", tx, ty))
            .with_attr("Head3D", format!("{:.2} {:.2} 0", hx, hy));

        match ra.kind {
            ReactionArrowKind::Solid => {
                arr.attrs.insert("ArrowheadType".into(), "Solid".into());
                arr.attrs.insert("ArrowheadHead".into(), "Full".into());
            }
            ReactionArrowKind::Equilibrium => {
                arr.attrs.insert("ArrowheadType".into(), "Solid".into());
                arr.attrs.insert("ArrowheadHead".into(), "HalfLeft".into());
                arr.attrs.insert("ArrowheadTail".into(), "HalfLeft".into());
                arr.attrs.insert("ArrowShaftSpacing".into(), "200".into());
            }
            ReactionArrowKind::UnbalancedEquilibrium => {
                arr.attrs.insert("ArrowheadType".into(), "Solid".into());
                arr.attrs.insert("ArrowheadHead".into(), "HalfLeft".into());
                arr.attrs.insert("ArrowheadTail".into(), "HalfLeft".into());
                arr.attrs
                    .insert("ArrowEquilibriumRatio".into(), "150".into());
            }
            ReactionArrowKind::Resonance => {
                arr.attrs.insert("ArrowheadType".into(), "Solid".into());
                arr.attrs.insert("ArrowheadHead".into(), "Full".into());
                arr.attrs.insert("ArrowheadTail".into(), "Full".into());
            }
            ReactionArrowKind::Retrosynthetic => {
                arr.attrs.insert("ArrowheadType".into(), "Hollow".into());
                arr.attrs.insert("ArrowheadHead".into(), "Full".into());
                arr.attrs.insert("ArrowShaftSpacing".into(), "300".into());
            }
            ReactionArrowKind::NoGoCross => {
                arr.attrs.insert("ArrowheadType".into(), "Solid".into());
                arr.attrs.insert("ArrowheadHead".into(), "Full".into());
                arr.attrs.insert("NoGo".into(), "Cross".into());
            }
            ReactionArrowKind::NoGoHash => {
                arr.attrs.insert("ArrowheadType".into(), "Solid".into());
                arr.attrs.insert("ArrowheadHead".into(), "Full".into());
                arr.attrs.insert("NoGo".into(), "Hash".into());
            }
            ReactionArrowKind::Dashed => {
                arr.attrs.insert("ArrowheadType".into(), "Solid".into());
                arr.attrs.insert("ArrowheadHead".into(), "Full".into());
                arr.attrs.insert("LineType".into(), "Dashed".into());
            }
        }

        // Encode `above_text` & `below_text` inside `Name` and `Comment` (both valid CDX string tags 0x0008 and 0x0009!)
        if let Some(ab) = &ra.above_text {
            arr.attrs.insert("Name".into(), ab.clone());
        }
        if let Some(bl) = &ra.below_text {
            arr.attrs.insert("Comment".into(), bl.clone());
        }
        page.kids.push(arr);
    }

    // 3. Curved Electron-Pushing Arrows -> `<curve>`
    for ca in &doc.curved_arrows {
        let id = alloc_id();
        let (x0, y0) = ang_to_cdx_xy(ca.start.0, ca.start.1);
        let (cx, cy) = ang_to_cdx_xy(ca.control.0, ca.control.1);
        let (x1, y1) = ang_to_cdx_xy(ca.end.0, ca.end.1);
        // ChemDraw Bézier CurvePoints: 4 cubic control points `start, ctrl1, ctrl2, end`
        let c1x = x0 + (2.0 / 3.0) * (cx - x0);
        let c1y = y0 + (2.0 / 3.0) * (cy - y0);
        let c2x = x1 + (2.0 / 3.0) * (cx - x1);
        let c2y = y1 + (2.0 / 3.0) * (cy - y1);
        let mut crv = XmlNode::new("curve")
            .with_attr("id", id.to_string())
            .with_attr(
                "CurvePoints",
                format!(
                    "{:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} {:.2}",
                    x0, y0, c1x, c1y, c2x, c2y, x1, y1
                ),
            )
            .with_attr("ArrowheadType", "Solid")
            .with_attr(
                "ArrowheadHead",
                if ca.electrons == 1 {
                    "HalfLeft"
                } else {
                    "Full"
                },
            );

        if let Some(src) = ca.source_anchor {
            let sid = match src {
                MechanismAnchor::Atom { mol, atom } => atom_ids.get(&(mol, atom)).copied(),
                MechanismAnchor::Bond { mol, bond } => bond_ids.get(&(mol, bond)).copied(),
            };
            if let Some(v) = sid {
                crv.attrs.insert("ArrowSource".into(), v.to_string());
            }
        }
        if let Some(dst) = ca.target_anchor {
            let tid = match dst {
                MechanismAnchor::Atom { mol, atom } => atom_ids.get(&(mol, atom)).copied(),
                MechanismAnchor::Bond { mol, bond } => bond_ids.get(&(mol, bond)).copied(),
            };
            if let Some(v) = tid {
                crv.attrs.insert("ArrowTarget".into(), v.to_string());
            }
        }
        page.kids.push(crv);
    }

    // 4. Brackets -> `<bracketedgroup>`
    for br in &doc.brackets {
        let id = alloc_id();
        let (x0, y0) = ang_to_cdx_xy(br.bounds.0, br.bounds.3); // min_x, max_y_ang -> top-left in Y-down
        let (x1, y1) = ang_to_cdx_xy(br.bounds.2, br.bounds.1); // max_x, min_y_ang -> bottom-right in Y-down
        let btype = match br.style {
            BracketStyle::Square => "SquarePair",
            BracketStyle::Round => "RoundPair",
            BracketStyle::Brace => "CurlyPair",
        };
        let busage = match br.usage {
            BracketUsage::Sru => "SRU",
            BracketUsage::MultipleGroup => "MultipleGroup",
            BracketUsage::Copolymer => "Copolymer",
            BracketUsage::CopolymerAlternating => "CopolymerAlternating",
            BracketUsage::CopolymerBlock => "CopolymerBlock",
            BracketUsage::CopolymerRandom => "CopolymerRandom",
            BracketUsage::Monomer => "Monomer",
            BracketUsage::Mixture => "MixtureUnordered",
            BracketUsage::Generic => "Generic",
        };
        let mut bg = XmlNode::new("bracketedgroup")
            .with_attr("id", id.to_string())
            .with_attr(
                "BoundingBox",
                format!("{:.2} {:.2} {:.2} {:.2}", x0, y0, x1, y1),
            )
            .with_attr("BracketType", btype)
            .with_attr("BracketUsage", busage)
            .with_attr("SRULabel", br.subscript.clone());
        if let Some(sup) = &br.superscript {
            bg.attrs.insert("Comment".into(), sup.clone());
        }
        let enclosed_ids: Vec<String> = br
            .enclosed_atoms
            .iter()
            .filter_map(|&(m, a)| atom_ids.get(&(m, a)).map(|v| v.to_string()))
            .collect();
        if !enclosed_ids.is_empty() {
            bg.attrs
                .insert("BracketedObjectIDs".into(), enclosed_ids.join(" "));
        }
        page.kids.push(bg);
    }

    // 5. Orbitals -> `<graphic GraphicType="Orbital" .../>`
    for orb in &doc.orbitals {
        let id = alloc_id();
        let (cx, cy) = ang_to_cdx_xy(orb.center.0, orb.center.1);
        let (ax, ay) = ang_to_cdx_xy(orb.axis_end.0, orb.axis_end.1);
        let otype = match (orb.kind, orb.phase) {
            (OrbitalKind::S, OrbitalPhase::FilledPlus) => "sFilled",
            (OrbitalKind::S, OrbitalPhase::ShadedMinus) => "sShaded",
            (OrbitalKind::S, OrbitalPhase::Hollow) => "s",
            (OrbitalKind::P, OrbitalPhase::FilledPlus) => "pFilled",
            (OrbitalKind::P, OrbitalPhase::ShadedMinus) => "pShaded",
            (OrbitalKind::P, OrbitalPhase::Hollow) => "p",
            (OrbitalKind::Sp3, OrbitalPhase::FilledPlus) => "hybridPlusFilled",
            (OrbitalKind::Sp3, OrbitalPhase::ShadedMinus) => "hybridMinusFilled",
            (OrbitalKind::Sp3, OrbitalPhase::Hollow) => "hybridPlus",
            (OrbitalKind::Dz2, OrbitalPhase::FilledPlus) => "dz2PlusFilled",
            (OrbitalKind::Dz2, OrbitalPhase::ShadedMinus) => "dz2MinusFilled",
            (OrbitalKind::Dz2, OrbitalPhase::Hollow) => "dz2Plus",
            (OrbitalKind::Dxy, OrbitalPhase::FilledPlus) => "dxyFilled",
            (OrbitalKind::Dxy, _) => "dxy",
        };
        let mut gr = XmlNode::new("graphic")
            .with_attr("id", id.to_string())
            .with_attr(
                "BoundingBox",
                format!("{:.2} {:.2} {:.2} {:.2}", cx, cy, ax, ay),
            )
            .with_attr("GraphicType", "Orbital")
            .with_attr("OrbitalType", otype);
        if let Some((m, a)) = orb.atom_anchor {
            if let Some(&aid) = atom_ids.get(&(m, a)) {
                gr.attrs.insert("SupersededBy".into(), aid.to_string());
            }
        }
        page.kids.push(gr);
    }

    // 6. TLC Plates -> `<tlcplate>` -> `<tlclane>` -> `<tlcspot>`
    for tlc in &doc.tlc_plates {
        let id = alloc_id();
        let (x0, y0) = ang_to_cdx_xy(tlc.bounds.0, tlc.bounds.3);
        let (x1, y1) = ang_to_cdx_xy(tlc.bounds.2, tlc.bounds.1);
        let mut plate = XmlNode::new("tlcplate")
            .with_attr("id", id.to_string())
            .with_attr(
                "BoundingBox",
                format!("{:.2} {:.2} {:.2} {:.2}", x0, y0, x1, y1),
            )
            .with_attr("OriginFraction", format!("{}", tlc.origin_fraction))
            .with_attr(
                "SolventFrontFraction",
                format!("{}", tlc.solvent_front_fraction),
            )
            .with_attr("ShowOrigin", "yes")
            .with_attr("ShowSolventFront", "yes")
            .with_attr("ShowBorders", "yes");

        let n_lanes = tlc
            .lane_labels
            .len()
            .max(tlc.spots.iter().map(|s| s.lane + 1).max().unwrap_or(1));
        for lane_idx in 0..n_lanes {
            let mut lane_el = XmlNode::new("tlclane").with_attr("id", alloc_id().to_string());
            if let Some(lbl) = tlc.lane_labels.get(lane_idx) {
                lane_el.attrs.insert("Name".into(), lbl.clone());
            }
            for sp in tlc.spots.iter().filter(|s| s.lane == lane_idx) {
                let mut spot_el = XmlNode::new("tlcspot")
                    .with_attr("id", alloc_id().to_string())
                    .with_attr("Rf", format!("{}", sp.rf))
                    .with_attr("ShowRf", "yes");
                let ctype = match sp.style {
                    TlcSpotStyle::Solid => 2,
                    TlcSpotStyle::Hollow => 0,
                    TlcSpotStyle::Shaded => 4,
                    TlcSpotStyle::UvActive => 8,
                };
                spot_el.attrs.insert("CurveType".into(), ctype.to_string());
                if let Some(lbl) = &sp.label {
                    spot_el.attrs.insert("Comment".into(), lbl.clone());
                }
                lane_el.kids.push(spot_el);
            }
            plate.kids.push(lane_el);
        }
        page.kids.push(plate);
    }

    // 7. RichTextBoxes -> `<t>` -> `<s>`
    for tb in &doc.text_boxes {
        let id = alloc_id();
        let (px, py) = ang_to_cdx_xy(tb.position.0, tb.position.1);
        let mut t_el = XmlNode::new("t")
            .with_attr("id", id.to_string())
            .with_attr("p", format!("{:.2} {:.2}", px, py));
        for sp in &tb.spans {
            let mut face: u16 = 0;
            if sp.bold {
                face |= 1;
            }
            if sp.italic {
                face |= 2;
            }
            if sp.subscript {
                face |= 32;
            }
            if sp.superscript {
                face |= 64;
            }
            let mut s_el = XmlNode::new("s")
                .with_attr("font", "3")
                .with_attr("size", format!("{}", sp.font_size_pt.unwrap_or(10.0)))
                .with_attr("color", "1");
            if face != 0 {
                s_el.attrs.insert("face".into(), face.to_string());
            }
            s_el.text = sp.text.clone();
            t_el.kids.push(s_el);
        }
        page.kids.push(t_el);
    }

    // 8. CanvasShapes -> `<graphic GraphicType="Rectangle|Oval|Line" .../>`
    for sh in &doc.shapes {
        let id = alloc_id();
        let (x0, y0) = ang_to_cdx_xy(sh.bounds.0, sh.bounds.3);
        let (x1, y1) = ang_to_cdx_xy(sh.bounds.2, sh.bounds.1);
        let mut gr = XmlNode::new("graphic")
            .with_attr("id", id.to_string())
            .with_attr(
                "BoundingBox",
                format!("{:.2} {:.2} {:.2} {:.2}", x0, y0, x1, y1),
            );
        match sh.kind {
            ShapeKind::Rectangle | ShapeKind::RoundedRectangle => {
                gr.attrs.insert("GraphicType".into(), "Rectangle".into());
                let mut flags = Vec::new();
                if sh.kind == ShapeKind::RoundedRectangle {
                    flags.push("RoundEdge");
                }
                if sh.filled {
                    flags.push("Filled");
                }
                if sh.dashed {
                    flags.push("Dashed");
                }
                if flags.is_empty() {
                    flags.push("Plain");
                }
                gr.attrs.insert("RectangleType".into(), flags.join(" "));
            }
            ShapeKind::Oval | ShapeKind::Circle => {
                gr.attrs.insert("GraphicType".into(), "Oval".into());
                let mut flags = Vec::new();
                if sh.kind == ShapeKind::Circle {
                    flags.push("Circle");
                }
                if sh.filled {
                    flags.push("Filled");
                }
                if sh.dashed {
                    flags.push("Dashed");
                }
                if !flags.is_empty() {
                    gr.attrs.insert("OvalType".into(), flags.join(" "));
                }
            }
            ShapeKind::Line => {
                gr.attrs.insert("GraphicType".into(), "Line".into());
                if sh.dashed {
                    gr.attrs.insert("LineType".into(), "Dashed".into());
                }
            }
        }
        page.kids.push(gr);
    }

    root.kids.push(page);
    root
}

fn molecule_to_cdxml_fragment(
    mol: &Molecule,
    mol_idx: usize,
    alloc_id: &mut dyn FnMut() -> i32,
    atom_ids: &mut BTreeMap<(usize, usize), i32>,
    bond_ids: &mut BTreeMap<(usize, usize), i32>,
) -> XmlNode {
    let mut frag = XmlNode::new("fragment").with_attr("id", alloc_id().to_string());
    if !mol.name.is_empty() {
        frag.attrs.insert("Name".into(), mol.name.clone());
    }
    if !mol.comment.is_empty() {
        frag.attrs.insert("Comment".into(), mol.comment.clone());
    }

    let cip = perceive_cip(mol);
    let mut local_atom_ids = Vec::with_capacity(mol.atoms.len());

    for (ai, a) in mol.atoms.iter().enumerate() {
        let nid = alloc_id();
        local_atom_ids.push(nid);
        atom_ids.insert((mol_idx, ai), nid);

        let (px, py) = ang_to_cdx_xy(a.x, a.y);
        let mut n_el = XmlNode::new("n")
            .with_attr("id", nid.to_string())
            .with_attr("p", format!("{:.2} {:.2}", px, py));

        if let Some(zc) = a.z_coord {
            n_el.attrs.insert(
                "xyz".into(),
                format!("{:.2} {:.2} {:.2}", px, py, zc * ANGSTROM_TO_PT),
            );
        }

        if let Some(lbl) = &a.superatom_label {
            n_el.attrs.insert("NodeType".into(), "Nickname".into());
            let mut t_el = XmlNode::new("t")
                .with_attr("id", alloc_id().to_string())
                .with_attr("p", format!("{:.2} {:.2}", px, py));
            let mut s_el = XmlNode::new("s")
                .with_attr("font", "3")
                .with_attr("size", "10");
            s_el.text = lbl.clone();
            t_el.kids.push(s_el);
            n_el.kids.push(t_el);
        } else if !a.multi_center_atoms.is_empty() {
            n_el.attrs
                .insert("NodeType".into(), "MultiAttachment".into());
        } else if a.z != 6 {
            // In CDXML, Element="6" (Carbon) is the default when omitted; write explicit Element for all non-C!
            n_el.attrs.insert("Element".into(), a.z.to_string());
        }

        if let Some(iso) = a.isotope {
            n_el.attrs.insert("Isotope".into(), iso.to_string());
        }
        if a.formal_charge != 0 {
            n_el.attrs
                .insert("Charge".into(), a.formal_charge.to_string());
        }
        match a.radical {
            Radical::None => {}
            Radical::Singlet => {
                n_el.attrs.insert("Radical".into(), "Singlet".into());
            }
            Radical::Doublet => {
                n_el.attrs.insert("Radical".into(), "Doublet".into());
            }
            Radical::Triplet => {
                n_el.attrs.insert("Radical".into(), "Triplet".into());
            }
        }
        if let Some(eh) = a.explicit_hydrogens {
            n_el.attrs.insert("NumHydrogens".into(), eh.to_string());
        }
        if let Some(anum) = &a.atom_number {
            n_el.attrs.insert("AtomNumber".into(), anum.clone());
        }
        if let Some(&desc) = cip.atoms.get(&ai) {
            let as_code = match desc {
                CipAtomDescriptor::R => "R",
                CipAtomDescriptor::S => "S",
                CipAtomDescriptor::PseudoR => "r",
                CipAtomDescriptor::PseudoS => "s",
            };
            n_el.attrs.insert("AS".into(), as_code.into());
        }

        frag.kids.push(n_el);
    }

    // Fill in MultiAttachment `Attachments` IDs now that all `local_atom_ids` exist
    for (node_cursor, a) in mol.atoms.iter().enumerate() {
        if !a.multi_center_atoms.is_empty() {
            let att_ids: Vec<String> = a
                .multi_center_atoms
                .iter()
                .filter_map(|&m| local_atom_ids.get(m).map(|v| v.to_string()))
                .collect();
            frag.kids[node_cursor]
                .attrs
                .insert("Attachments".into(), att_ids.join(" "));
        }
    }

    for (bi, b) in mol.bonds.iter().enumerate() {
        let bid = alloc_id();
        bond_ids.insert((mol_idx, bi), bid);

        let mut b_el = XmlNode::new("b")
            .with_attr("id", bid.to_string())
            .with_attr("B", local_atom_ids[b.a].to_string())
            .with_attr("E", local_atom_ids[b.b].to_string());

        match b.kind {
            BondKind::Single => {} // Order="1" Solid is CDXML default
            BondKind::Double => {
                b_el.attrs.insert("Order".into(), "2".into());
                if b.double_geometry == DoubleBondGeometry::Either {
                    b_el.attrs.insert("Display".into(), "Wavy".into());
                }
            }
            BondKind::Triple => {
                b_el.attrs.insert("Order".into(), "3".into());
            }
            BondKind::Aromatic => {
                b_el.attrs.insert("Order".into(), "1.5".into());
            }
            BondKind::WedgeUp => {
                b_el.attrs.insert("Display".into(), "WedgeBegin".into());
            }
            BondKind::HashDown => {
                b_el.attrs
                    .insert("Display".into(), "WedgedHashBegin".into());
            }
            BondKind::HollowWedge => {
                b_el.attrs
                    .insert("Display".into(), "HollowWedgeBegin".into());
            }
            BondKind::Wavy => {
                b_el.attrs.insert("Display".into(), "Wavy".into());
            }
            BondKind::Bold => {
                b_el.attrs.insert("Display".into(), "Bold".into());
            }
            BondKind::Dashed => {
                b_el.attrs.insert("Display".into(), "Dash".into());
            }
            BondKind::Dative => {
                b_el.attrs.insert("Order".into(), "dative".into());
            }
            BondKind::Hydrogen => {
                b_el.attrs.insert("Order".into(), "hydrogen".into());
            }
            BondKind::Ionic => {
                b_el.attrs.insert("Order".into(), "ionic".into());
            }
        }

        if let Some(&bdesc) = cip.bonds.get(&bi) {
            b_el.attrs.insert(
                "BS".into(),
                match bdesc {
                    CipBondDescriptor::E => "E",
                    CipBondDescriptor::Z => "Z",
                }
                .into(),
            );
        }

        frag.kids.push(b_el);
    }

    frag
}

/// Convert a parsed `<CDXML>` `XmlNode` AST into a `ChemDocument`.
pub fn cdxml_ast_to_doc(root: &XmlNode) -> Result<ChemDocument, ChemError> {
    if root.name != "CDXML" {
        return Err(ChemError::new(format!(
            "Expected <CDXML> root element, found <{}>",
            root.name
        )));
    }

    let mut doc = ChemDocument::new();
    if let Some(nm) = root.attrs.get("Name") {
        doc.title = nm.clone();
    }

    // Collect top-level children inside `<page>` (or directly inside `<CDXML>` if pageless)
    let containers: Vec<&XmlNode> = {
        let pages: Vec<&XmlNode> = root.kids.iter().filter(|k| k.name == "page").collect();
        if pages.is_empty() {
            vec![root]
        } else {
            pages
        }
    };

    let mut id_to_atom: BTreeMap<i32, (usize, usize)> = BTreeMap::new();
    let mut id_to_bond: BTreeMap<i32, (usize, usize)> = BTreeMap::new();

    // Pass 1: Parse all `<fragment>` elements first so `id_to_atom` and `id_to_bond` are populated for anchors/brackets!
    for page in &containers {
        collect_fragments_recursive(page, &mut doc, &mut id_to_atom, &mut id_to_bond)?;
    }

    // Pass 2: Parse page objects (`<arrow>`, `<curve>`, `<bracketedgroup>`, `<tlcplate>`, `<t>`, `<graphic>`)
    for page in &containers {
        for el in &page.kids {
            match el.name.as_str() {
                "fragment" | "fonttable" | "colortable" => {}
                "arrow" => {
                    if let Some(ra) = parse_cdxml_arrow(el) {
                        doc.reaction_arrows.push(ra);
                    }
                }
                "curve" => {
                    if let Some(ca) = parse_cdxml_curve(el, &id_to_atom, &id_to_bond) {
                        doc.curved_arrows.push(ca);
                    }
                }
                "bracketedgroup" => {
                    if let Some(br) = parse_cdxml_bracket(el, &id_to_atom) {
                        doc.brackets.push(br);
                    }
                }
                "tlcplate" => {
                    if let Some(tlc) = parse_cdxml_tlc(el) {
                        doc.tlc_plates.push(tlc);
                    }
                }
                "t" => {
                    if let Some(tb) = parse_cdxml_textbox(el) {
                        doc.text_boxes.push(tb);
                    }
                }
                "graphic" => {
                    parse_cdxml_graphic(el, &mut doc, &id_to_atom);
                }
                _ => {}
            }
        }
    }

    Ok(doc)
}

fn collect_fragments_recursive(
    container: &XmlNode,
    doc: &mut ChemDocument,
    id_to_atom: &mut BTreeMap<i32, (usize, usize)>,
    id_to_bond: &mut BTreeMap<i32, (usize, usize)>,
) -> Result<(), ChemError> {
    for el in &container.kids {
        if el.name == "fragment" {
            let mol_idx = doc.molecules.len();
            let mol = parse_cdxml_fragment(el, mol_idx, id_to_atom, id_to_bond)?;
            doc.molecules.push(mol);
        } else if el.name == "group" || el.name == "scheme" {
            collect_fragments_recursive(el, doc, id_to_atom, id_to_bond)?;
        }
    }
    Ok(())
}

fn parse_cdxml_fragment(
    frag: &XmlNode,
    mol_idx: usize,
    id_to_atom: &mut BTreeMap<i32, (usize, usize)>,
    id_to_bond: &mut BTreeMap<i32, (usize, usize)>,
) -> Result<Molecule, ChemError> {
    let mut mol = Molecule::new();
    if let Some(nm) = frag.attrs.get("Name") {
        mol.name = nm.clone();
    }
    if let Some(cm) = frag.attrs.get("Comment") {
        mol.comment = cm.clone();
    }

    let mut local_id_map: BTreeMap<i32, usize> = BTreeMap::new();
    let mut pending_attachments: Vec<(usize, String)> = Vec::new();
    let mut any_3d = false;
    let mut raw_z_coords: Vec<f64> = Vec::new();

    for kid in &frag.kids {
        if kid.name != "n" {
            continue;
        }
        let nid = kid
            .attrs
            .get("id")
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0);
        let node_type = kid
            .attrs
            .get("NodeType")
            .map(|s| s.as_str())
            .unwrap_or("Element");

        let (px, py) = kid
            .attrs
            .get("p")
            .and_then(|s| parse_two_f64(s))
            .unwrap_or((PAGE_ORIGIN_X_PT, PAGE_ORIGIN_Y_PT));
        let (x_ang, y_ang) = cdx_to_ang_xy(px, py);

        let mut z_ang = 0.0;
        if let Some(xyz_str) = kid.attrs.get("xyz") {
            let nums = parse_f64_list(xyz_str);
            if nums.len() >= 3 {
                z_ang = nums[2] / ANGSTROM_TO_PT;
                any_3d = true;
            }
        }
        raw_z_coords.push(z_ang);

        let mut atom = if node_type == "Nickname" || node_type == "Fragment" {
            let lbl = kid
                .kids
                .iter()
                .find(|k| k.name == "t")
                .map(extract_plain_text)
                .unwrap_or_else(|| "R".to_string());
            let mut a = Atom::new(&lbl);
            a.z = 0;
            a.symbol = lbl.clone();
            a.superatom_label = Some(lbl);
            a
        } else if node_type == "MultiAttachment" {
            Atom::new("*")
        } else {
            let z = kid
                .attrs
                .get("Element")
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(6);
            Atom::new(symbol_from_atomic_number(z))
        };

        atom.x = x_ang;
        atom.y = y_ang;
        if let Some(iso) = kid.attrs.get("Isotope").and_then(|s| s.parse::<u16>().ok()) {
            atom.isotope = Some(iso);
        }
        if let Some(chg) = kid.attrs.get("Charge").and_then(|s| s.parse::<f64>().ok()) {
            atom.formal_charge = chg.round() as i8;
        }
        if let Some(rad) = kid.attrs.get("Radical") {
            atom.radical = match rad.as_str() {
                "Singlet" | "1" => Radical::Singlet,
                "Doublet" | "2" => Radical::Doublet,
                "Triplet" | "3" => Radical::Triplet,
                _ => Radical::None,
            };
        }
        if let Some(nh) = kid
            .attrs
            .get("NumHydrogens")
            .and_then(|s| s.parse::<u8>().ok())
        {
            atom.explicit_hydrogens = Some(nh);
        }
        if let Some(anum) = kid.attrs.get("AtomNumber") {
            atom.atom_number = Some(anum.clone());
        }
        if let Some(att) = kid.attrs.get("Attachments") {
            pending_attachments.push((mol.atoms.len(), att.clone()));
        }

        let ai = mol.add_atom(atom);
        if nid != 0 {
            local_id_map.insert(nid, ai);
            id_to_atom.insert(nid, (mol_idx, ai));
        }
    }

    if any_3d {
        for (i, zc) in raw_z_coords.into_iter().enumerate() {
            mol.atoms[i].z_coord = Some(zc);
        }
    }

    for (ai, att_str) in pending_attachments {
        mol.atoms[ai].multi_center_atoms = att_str
            .split_whitespace()
            .filter_map(|s| {
                s.parse::<i32>()
                    .ok()
                    .and_then(|id| local_id_map.get(&id).copied())
            })
            .collect();
    }

    let mut has_arom = false;
    for kid in &frag.kids {
        if kid.name != "b" {
            continue;
        }
        let bid = kid
            .attrs
            .get("id")
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0);
        let Some(b_id) = kid.attrs.get("B").and_then(|s| s.parse::<i32>().ok()) else {
            continue;
        };
        let Some(e_id) = kid.attrs.get("E").and_then(|s| s.parse::<i32>().ok()) else {
            continue;
        };
        let (Some(&a_idx), Some(&b_idx)) = (local_id_map.get(&b_id), local_id_map.get(&e_id))
        else {
            continue;
        };

        let order_str = kid.attrs.get("Order").map(|s| s.as_str()).unwrap_or("1");
        let disp_str = kid
            .attrs
            .get("Display")
            .map(|s| s.as_str())
            .unwrap_or("Solid");

        let (mut start_a, mut end_b, kind, crossed_dbl) = match (order_str, disp_str) {
            ("2", "Wavy") => (a_idx, b_idx, BondKind::Double, true),
            ("2", _) => (a_idx, b_idx, BondKind::Double, false),
            ("3", _) => (a_idx, b_idx, BondKind::Triple, false),
            ("1.5", _) => {
                has_arom = true;
                (a_idx, b_idx, BondKind::Aromatic, false)
            }
            ("dative", _) => (a_idx, b_idx, BondKind::Dative, false),
            ("hydrogen", _) => (a_idx, b_idx, BondKind::Hydrogen, false),
            ("ionic", _) => (a_idx, b_idx, BondKind::Ionic, false),
            (_, "WedgeBegin") => (a_idx, b_idx, BondKind::WedgeUp, false),
            (_, "WedgeEnd") => (b_idx, a_idx, BondKind::WedgeUp, false),
            (_, "WedgedHashBegin" | "Hash") => (a_idx, b_idx, BondKind::HashDown, false),
            (_, "WedgedHashEnd") => (b_idx, a_idx, BondKind::HashDown, false),
            (_, "HollowWedgeBegin") => (a_idx, b_idx, BondKind::HollowWedge, false),
            (_, "HollowWedgeEnd") => (b_idx, a_idx, BondKind::HollowWedge, false),
            (_, "Wavy") => (a_idx, b_idx, BondKind::Wavy, false),
            (_, "Bold") => (a_idx, b_idx, BondKind::Bold, false),
            (_, "Dash" | "Dot") => (a_idx, b_idx, BondKind::Dashed, false),
            _ => (a_idx, b_idx, BondKind::Single, false),
        };
        let _ = (&mut start_a, &mut end_b);
        let bi = mol.add_bond(start_a, end_b, kind);
        if crossed_dbl {
            mol.bonds[bi].double_geometry = DoubleBondGeometry::Either;
        }
        if bid != 0 {
            id_to_bond.insert(bid, (mol_idx, bi));
        }
    }

    if has_arom {
        kekulize(&mut mol)?;
    }

    Ok(mol)
}

fn parse_cdxml_arrow(el: &XmlNode) -> Option<ReactionArrow> {
    let tail_pt = el
        .attrs
        .get("Tail3D")
        .and_then(|s| parse_two_f64(s))
        .or_else(|| {
            el.attrs.get("BoundingBox").and_then(|s| {
                let v = parse_f64_list(s);
                (v.len() >= 4).then(|| (v[0], v[1]))
            })
        })?;
    let head_pt = el
        .attrs
        .get("Head3D")
        .and_then(|s| parse_two_f64(s))
        .or_else(|| {
            el.attrs.get("BoundingBox").and_then(|s| {
                let v = parse_f64_list(s);
                (v.len() >= 4).then(|| (v[2], v[3]))
            })
        })?;

    let tail = cdx_to_ang_xy(tail_pt.0, tail_pt.1);
    let head = cdx_to_ang_xy(head_pt.0, head_pt.1);

    let ah_type = el
        .attrs
        .get("ArrowheadType")
        .map(|s| s.as_str())
        .unwrap_or("Solid");
    let ah_head = el
        .attrs
        .get("ArrowheadHead")
        .map(|s| s.as_str())
        .unwrap_or("Full");
    let ah_tail = el
        .attrs
        .get("ArrowheadTail")
        .map(|s| s.as_str())
        .unwrap_or("None");
    let nogo = el.attrs.get("NoGo").map(|s| s.as_str()).unwrap_or("");
    let ltype = el.attrs.get("LineType").map(|s| s.as_str()).unwrap_or("");

    let kind = if nogo == "Cross" {
        ReactionArrowKind::NoGoCross
    } else if nogo == "Hash" {
        ReactionArrowKind::NoGoHash
    } else if ltype.contains("Dashed") {
        ReactionArrowKind::Dashed
    } else if ah_type == "Hollow" {
        ReactionArrowKind::Retrosynthetic
    } else if ah_head.starts_with("Half") && ah_tail.starts_with("Half") {
        if el.attrs.contains_key("ArrowEquilibriumRatio") {
            ReactionArrowKind::UnbalancedEquilibrium
        } else {
            ReactionArrowKind::Equilibrium
        }
    } else if ah_head == "Full" && ah_tail == "Full" {
        ReactionArrowKind::Resonance
    } else {
        ReactionArrowKind::Solid
    };

    Some(ReactionArrow {
        kind,
        tail,
        head,
        above_text: el.attrs.get("Name").cloned().filter(|s| !s.is_empty()),
        below_text: el.attrs.get("Comment").cloned().filter(|s| !s.is_empty()),
    })
}

fn parse_cdxml_curve(
    el: &XmlNode,
    id_to_atom: &BTreeMap<i32, (usize, usize)>,
    id_to_bond: &BTreeMap<i32, (usize, usize)>,
) -> Option<CurvedArrow> {
    let pts = parse_f64_list(el.attrs.get("CurvePoints")?);
    if pts.len() < 6 {
        return None;
    }
    let start_pt = (pts[0], pts[1]);
    let end_pt = (pts[pts.len() - 2], pts[pts.len() - 1]);
    // Reconstruct quadratic apex control point from cubic control points `c1 = start + 2/3*(ctrl - start)`
    let ctrl_pt = if pts.len() >= 8 {
        let c1x = pts[2];
        let c1y = pts[3];
        (
            start_pt.0 + 1.5 * (c1x - start_pt.0),
            start_pt.1 + 1.5 * (c1y - start_pt.1),
        )
    } else {
        (pts[2], pts[3])
    };

    let ah_head = el
        .attrs
        .get("ArrowheadHead")
        .map(|s| s.as_str())
        .unwrap_or("Full");
    let electrons = if ah_head.starts_with("Half") { 1 } else { 2 };

    let resolve_anchor = |attr_name: &str| -> Option<MechanismAnchor> {
        let id = el.attrs.get(attr_name)?.parse::<i32>().ok()?;
        if let Some(&(mol, atom)) = id_to_atom.get(&id) {
            Some(MechanismAnchor::Atom { mol, atom })
        } else {
            id_to_bond
                .get(&id)
                .map(|&(mol, bond)| MechanismAnchor::Bond { mol, bond })
        }
    };

    Some(CurvedArrow {
        electrons,
        start: cdx_to_ang_xy(start_pt.0, start_pt.1),
        control: cdx_to_ang_xy(ctrl_pt.0, ctrl_pt.1),
        end: cdx_to_ang_xy(end_pt.0, end_pt.1),
        source_anchor: resolve_anchor("ArrowSource"),
        target_anchor: resolve_anchor("ArrowTarget"),
    })
}

fn parse_cdxml_bracket(
    el: &XmlNode,
    id_to_atom: &BTreeMap<i32, (usize, usize)>,
) -> Option<Bracket> {
    let bb = parse_f64_list(el.attrs.get("BoundingBox")?);
    if bb.len() < 4 {
        return None;
    }
    let (min_x, max_y) = cdx_to_ang_xy(bb[0], bb[1]);
    let (max_x, min_y) = cdx_to_ang_xy(bb[2], bb[3]);
    let bounds = (
        min_x.min(max_x),
        min_y.min(max_y),
        min_x.max(max_x),
        min_y.max(max_y),
    );

    let style = match el
        .attrs
        .get("BracketType")
        .map(|s| s.as_str())
        .unwrap_or("SquarePair")
    {
        "RoundPair" | "Round" => BracketStyle::Round,
        "CurlyPair" | "Curly" => BracketStyle::Brace,
        _ => BracketStyle::Square,
    };
    let usage = match el
        .attrs
        .get("BracketUsage")
        .map(|s| s.as_str())
        .unwrap_or("SRU")
    {
        "MultipleGroup" => BracketUsage::MultipleGroup,
        "Copolymer" => BracketUsage::Copolymer,
        "CopolymerAlternating" => BracketUsage::CopolymerAlternating,
        "CopolymerBlock" => BracketUsage::CopolymerBlock,
        "CopolymerRandom" => BracketUsage::CopolymerRandom,
        "Monomer" => BracketUsage::Monomer,
        "MixtureUnordered" | "MixtureOrdered" => BracketUsage::Mixture,
        "Generic" | "Unspecified" => BracketUsage::Generic,
        _ => BracketUsage::Sru,
    };

    let subscript = el.attrs.get("SRULabel").cloned().unwrap_or_default();
    let superscript = el.attrs.get("Comment").cloned().filter(|s| !s.is_empty());
    let enclosed_atoms = el
        .attrs
        .get("BracketedObjectIDs")
        .map(|s| {
            s.split_whitespace()
                .filter_map(|tok| {
                    tok.parse::<i32>()
                        .ok()
                        .and_then(|id| id_to_atom.get(&id).copied())
                })
                .collect()
        })
        .unwrap_or_default();

    Some(Bracket {
        style,
        usage,
        bounds,
        subscript,
        superscript,
        enclosed_atoms,
    })
}

fn parse_cdxml_tlc(el: &XmlNode) -> Option<TlcPlate> {
    let bb = parse_f64_list(el.attrs.get("BoundingBox")?);
    if bb.len() < 4 {
        return None;
    }
    let (min_x, max_y) = cdx_to_ang_xy(bb[0], bb[1]);
    let (max_x, min_y) = cdx_to_ang_xy(bb[2], bb[3]);
    let bounds = (
        min_x.min(max_x),
        min_y.min(max_y),
        min_x.max(max_x),
        min_y.max(max_y),
    );
    let origin_fraction = el
        .attrs
        .get("OriginFraction")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.10);
    let solvent_front_fraction = el
        .attrs
        .get("SolventFrontFraction")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.10);

    let mut lane_labels = Vec::new();
    let mut spots = Vec::new();
    for (lane_idx, lane_el) in el.kids.iter().filter(|k| k.name == "tlclane").enumerate() {
        lane_labels.push(lane_el.attrs.get("Name").cloned().unwrap_or_default());
        for spot_el in lane_el.kids.iter().filter(|k| k.name == "tlcspot") {
            let rf = spot_el
                .attrs
                .get("Rf")
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.5);
            let style = match spot_el
                .attrs
                .get("CurveType")
                .and_then(|s| s.parse::<i32>().ok())
                .unwrap_or(2)
            {
                0 => TlcSpotStyle::Hollow,
                4 => TlcSpotStyle::Shaded,
                8 => TlcSpotStyle::UvActive,
                _ => TlcSpotStyle::Solid,
            };
            spots.push(TlcSpot {
                lane: lane_idx,
                rf,
                style,
                label: spot_el
                    .attrs
                    .get("Comment")
                    .cloned()
                    .filter(|s| !s.is_empty()),
            });
        }
    }

    Some(TlcPlate {
        bounds,
        origin_fraction,
        solvent_front_fraction,
        lane_labels,
        spots,
    })
}

fn parse_cdxml_textbox(el: &XmlNode) -> Option<RichTextBox> {
    let (px, py) = el
        .attrs
        .get("p")
        .and_then(|s| parse_two_f64(s))
        .unwrap_or((PAGE_ORIGIN_X_PT, PAGE_ORIGIN_Y_PT));
    let position = cdx_to_ang_xy(px, py);
    let mut spans = Vec::new();
    for s_el in el.kids.iter().filter(|k| k.name == "s") {
        let face = s_el
            .attrs
            .get("face")
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(0);
        let size = s_el.attrs.get("size").and_then(|s| s.parse::<f64>().ok());
        spans.push(TextSpan {
            text: s_el.text.clone(),
            bold: (face & 1) != 0,
            italic: (face & 2) != 0,
            subscript: (face & 32) != 0,
            superscript: (face & 64) != 0,
            font_size_pt: size,
            color_rgb: None,
        });
    }
    if spans.is_empty() && !el.text.is_empty() {
        spans.push(TextSpan::plain(el.text.clone()));
    }
    if spans.is_empty() {
        None
    } else {
        Some(RichTextBox { position, spans })
    }
}

fn parse_cdxml_graphic(
    el: &XmlNode,
    doc: &mut ChemDocument,
    id_to_atom: &BTreeMap<i32, (usize, usize)>,
) {
    let gtype = el
        .attrs
        .get("GraphicType")
        .map(|s| s.as_str())
        .unwrap_or("");
    let Some(bb_str) = el.attrs.get("BoundingBox") else {
        return;
    };
    let bb = parse_f64_list(bb_str);
    if bb.len() < 4 {
        return;
    }

    if gtype == "Orbital" {
        let center = cdx_to_ang_xy(bb[0], bb[1]);
        let axis_end = cdx_to_ang_xy(bb[2], bb[3]);
        let otype = el
            .attrs
            .get("OrbitalType")
            .map(|s| s.as_str())
            .unwrap_or("pFilled");
        let (kind, phase) = match otype {
            "sFilled" => (OrbitalKind::S, OrbitalPhase::FilledPlus),
            "sShaded" => (OrbitalKind::S, OrbitalPhase::ShadedMinus),
            "s" => (OrbitalKind::S, OrbitalPhase::Hollow),
            "pFilled" => (OrbitalKind::P, OrbitalPhase::FilledPlus),
            "pShaded" => (OrbitalKind::P, OrbitalPhase::ShadedMinus),
            "p" => (OrbitalKind::P, OrbitalPhase::Hollow),
            "hybridPlusFilled" => (OrbitalKind::Sp3, OrbitalPhase::FilledPlus),
            "hybridMinusFilled" | "hybridMinus" => (OrbitalKind::Sp3, OrbitalPhase::ShadedMinus),
            "hybridPlus" => (OrbitalKind::Sp3, OrbitalPhase::Hollow),
            "dz2PlusFilled" => (OrbitalKind::Dz2, OrbitalPhase::FilledPlus),
            "dz2MinusFilled" | "dz2Minus" => (OrbitalKind::Dz2, OrbitalPhase::ShadedMinus),
            "dz2Plus" => (OrbitalKind::Dz2, OrbitalPhase::Hollow),
            "dxyFilled" => (OrbitalKind::Dxy, OrbitalPhase::FilledPlus),
            "dxy" => (OrbitalKind::Dxy, OrbitalPhase::Hollow),
            _ => (OrbitalKind::P, OrbitalPhase::FilledPlus),
        };
        let atom_anchor = el
            .attrs
            .get("SupersededBy")
            .and_then(|s| s.parse::<i32>().ok())
            .and_then(|id| id_to_atom.get(&id).copied());
        doc.orbitals.push(Orbital {
            kind,
            phase,
            center,
            axis_end,
            atom_anchor,
        });
    } else if matches!(gtype, "Rectangle" | "Oval" | "Line") {
        let (min_x, max_y) = cdx_to_ang_xy(bb[0], bb[1]);
        let (max_x, min_y) = cdx_to_ang_xy(bb[2], bb[3]);
        let bounds = (
            min_x.min(max_x),
            min_y.min(max_y),
            min_x.max(max_x),
            min_y.max(max_y),
        );
        let rtype = el
            .attrs
            .get("RectangleType")
            .map(|s| s.as_str())
            .unwrap_or("");
        let otype = el.attrs.get("OvalType").map(|s| s.as_str()).unwrap_or("");
        let ltype = el.attrs.get("LineType").map(|s| s.as_str()).unwrap_or("");
        let kind = match gtype {
            "Rectangle" => {
                if rtype.contains("RoundEdge") {
                    ShapeKind::RoundedRectangle
                } else {
                    ShapeKind::Rectangle
                }
            }
            "Oval" => {
                if otype.contains("Circle") {
                    ShapeKind::Circle
                } else {
                    ShapeKind::Oval
                }
            }
            _ => ShapeKind::Line,
        };
        let dashed =
            rtype.contains("Dashed") || otype.contains("Dashed") || ltype.contains("Dashed");
        let filled = rtype.contains("Filled") || otype.contains("Filled");
        doc.shapes.push(CanvasShape {
            kind,
            bounds,
            dashed,
            filled,
        });
    }
}

fn extract_plain_text(t_el: &XmlNode) -> String {
    let mut s = String::new();
    for k in &t_el.kids {
        if k.name == "s" {
            s.push_str(&k.text);
        }
    }
    if s.is_empty() {
        s.push_str(&t_el.text);
    }
    s
}

fn parse_two_f64(s: &str) -> Option<(f64, f64)> {
    let mut it = s.split_whitespace();
    let a = it.next()?.parse::<f64>().ok()?;
    let b = it.next()?.parse::<f64>().ok()?;
    Some((a, b))
}

fn parse_f64_list(s: &str) -> Vec<f64> {
    s.split_whitespace()
        .filter_map(|tok| tok.parse::<f64>().ok())
        .collect()
}

// ============================================================================
// 3. ChemDraw Binary (`.cdx` — Magic `VjCD0100`) Tag Tree Reader & Writer
// ============================================================================

const CDX_OBJECTS: &[(u16, &str)] = &[
    (0x8000, "CDXML"),
    (0x8001, "page"),
    (0x8002, "group"),
    (0x8003, "fragment"),
    (0x8004, "n"),
    (0x8005, "b"),
    (0x8006, "t"),
    (0x8007, "graphic"),
    (0x8008, "curve"),
    (0x800D, "scheme"),
    (0x800E, "step"),
    (0x8010, "spectrum"),
    (0x8011, "objecttag"),
    (0x8013, "sequence"),
    (0x8017, "bracketedgroup"),
    (0x8018, "bracketattachment"),
    (0x8019, "crossingbond"),
    (0x801D, "tlcplate"),
    (0x801E, "tlclane"),
    (0x801F, "tlcspot"),
    (0x8021, "arrow"),
];

#[derive(Clone, Copy)]
enum PropKind {
    Flag,
    Bool,
    I8,
    I16,
    U16,
    I32,
    E8,
    E16,
    B16,
    Order,
    Charge,
    Coord,
    P2,
    P3,
    Rect,
    Angle,
    Spacing,
    F64,
    Str,
    Plain,
    Ids,
    Attachments,
    Curve,
}

struct CdxPropSpec {
    tag: u16,
    name: &'static str,
    kind: PropKind,
    enums: &'static [(i32, &'static str)],
}

const NODE_TYPE_ENUMS: &[(i32, &str)] = &[
    (0, "Unspecified"),
    (1, "Element"),
    (4, "Nickname"),
    (5, "Fragment"),
    (7, "GenericNickname"),
    (10, "MultiAttachment"),
    (12, "ExternalConnectionPoint"),
];

const RADICAL_ENUMS: &[(i32, &str)] =
    &[(0, "None"), (1, "Singlet"), (2, "Doublet"), (3, "Triplet")];

const AS_ENUMS: &[(i32, &str)] = &[
    (0, "U"),
    (1, "N"),
    (2, "R"),
    (3, "S"),
    (4, "r"),
    (5, "s"),
    (6, "u"),
];

const ORDER_ENUMS: &[(i32, &str)] = &[
    (1, "1"),
    (2, "2"),
    (4, "3"),
    (8, "4"),
    (128, "1.5"),
    (4096, "dative"),
    (8192, "ionic"),
    (16384, "hydrogen"),
];

const DISPLAY_ENUMS: &[(i32, &str)] = &[
    (0, "Solid"),
    (1, "Dash"),
    (2, "Hash"),
    (3, "WedgedHashBegin"),
    (4, "WedgedHashEnd"),
    (5, "Bold"),
    (6, "WedgeBegin"),
    (7, "WedgeEnd"),
    (8, "Wavy"),
    (9, "HollowWedgeBegin"),
    (10, "HollowWedgeEnd"),
];

const BS_ENUMS: &[(i32, &str)] = &[(0, "U"), (1, "N"), (2, "E"), (3, "Z")];

const GRAPHIC_TYPE_ENUMS: &[(i32, &str)] = &[
    (0, "Undefined"),
    (1, "Line"),
    (2, "Arc"),
    (3, "Rectangle"),
    (4, "Oval"),
    (5, "Orbital"),
    (6, "Bracket"),
    (7, "Symbol"),
];

const LINE_TYPE_BITS: &[(i32, &str)] = &[(0, "Solid"), (1, "Dashed"), (2, "Bold"), (4, "Wavy")];

const RECT_TYPE_BITS: &[(i32, &str)] = &[
    (0, "Plain"),
    (1, "RoundEdge"),
    (2, "Shadow"),
    (4, "Shaded"),
    (8, "Filled"),
    (16, "Dashed"),
    (32, "Bold"),
];

const OVAL_TYPE_BITS: &[(i32, &str)] = &[
    (1, "Circle"),
    (2, "Shaded"),
    (4, "Filled"),
    (8, "Dashed"),
    (16, "Bold"),
];

const ORBITAL_TYPE_ENUMS: &[(i32, &str)] = &[
    (0, "s"),
    (3, "p"),
    (4, "hybridPlus"),
    (5, "hybridMinus"),
    (6, "dz2Plus"),
    (7, "dz2Minus"),
    (8, "dxy"),
    (256, "sShaded"),
    (259, "pShaded"),
    (512, "sFilled"),
    (515, "pFilled"),
    (516, "hybridPlusFilled"),
    (517, "hybridMinusFilled"),
    (518, "dz2PlusFilled"),
    (519, "dz2MinusFilled"),
    (520, "dxyFilled"),
];

const BRACKET_TYPE_ENUMS: &[(i32, &str)] = &[
    (0, "RoundPair"),
    (1, "SquarePair"),
    (2, "CurlyPair"),
    (3, "Square"),
    (4, "Curly"),
    (5, "Round"),
];

const BRACKET_USAGE_ENUMS: &[(i32, &str)] = &[
    (0, "Unspecified"),
    (3, "SRU"),
    (4, "Monomer"),
    (6, "Copolymer"),
    (7, "CopolymerAlternating"),
    (8, "CopolymerRandom"),
    (9, "CopolymerBlock"),
    (14, "MixtureUnordered"),
    (15, "MixtureOrdered"),
    (16, "MultipleGroup"),
    (17, "Generic"),
];

const ARROWHEAD_TYPE_ENUMS: &[(i32, &str)] = &[
    (0, "Unspecified"),
    (1, "Solid"),
    (2, "Hollow"),
    (3, "Angle"),
];

const ARROWHEAD_END_ENUMS: &[(i32, &str)] = &[
    (0, "Unspecified"),
    (1, "None"),
    (2, "Full"),
    (3, "HalfLeft"),
    (4, "HalfRight"),
];

const NOGO_ENUMS: &[(i32, &str)] = &[(0, "Unspecified"), (1, "None"), (2, "Cross"), (3, "Hash")];

const CDX_PROPS: &[CdxPropSpec] = &[
    CdxPropSpec {
        tag: 0x0003,
        name: "CreationProgram",
        kind: PropKind::Str,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0008,
        name: "Name",
        kind: PropKind::Str,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0009,
        name: "Comment",
        kind: PropKind::Str,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x000A,
        name: "Z",
        kind: PropKind::I16,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0011,
        name: "Visible",
        kind: PropKind::Bool,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0013,
        name: "SupersededBy",
        kind: PropKind::I32,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0200,
        name: "p",
        kind: PropKind::P2,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0201,
        name: "xyz",
        kind: PropKind::P3,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0204,
        name: "BoundingBox",
        kind: PropKind::Rect,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0207,
        name: "Head3D",
        kind: PropKind::P3,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0208,
        name: "Tail3D",
        kind: PropKind::P3,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0301,
        name: "color",
        kind: PropKind::U16,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0302,
        name: "bgcolor",
        kind: PropKind::U16,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0400,
        name: "NodeType",
        kind: PropKind::E16,
        enums: NODE_TYPE_ENUMS,
    },
    CdxPropSpec {
        tag: 0x0402,
        name: "Element",
        kind: PropKind::U16,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0420,
        name: "Isotope",
        kind: PropKind::I16,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0421,
        name: "Charge",
        kind: PropKind::Charge,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0422,
        name: "Radical",
        kind: PropKind::E8,
        enums: RADICAL_ENUMS,
    },
    CdxPropSpec {
        tag: 0x042B,
        name: "NumHydrogens",
        kind: PropKind::U16,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0432,
        name: "Attachments",
        kind: PropKind::Attachments,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0437,
        name: "AS",
        kind: PropKind::E8,
        enums: AS_ENUMS,
    },
    CdxPropSpec {
        tag: 0x0439,
        name: "AtomNumber",
        kind: PropKind::Plain,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0600,
        name: "Order",
        kind: PropKind::Order,
        enums: ORDER_ENUMS,
    },
    CdxPropSpec {
        tag: 0x0601,
        name: "Display",
        kind: PropKind::E16,
        enums: DISPLAY_ENUMS,
    },
    CdxPropSpec {
        tag: 0x0604,
        name: "B",
        kind: PropKind::I32,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0605,
        name: "E",
        kind: PropKind::I32,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x060A,
        name: "BS",
        kind: PropKind::E8,
        enums: BS_ENUMS,
    },
    CdxPropSpec {
        tag: 0x0803,
        name: "ChainAngle",
        kind: PropKind::Angle,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0804,
        name: "BondSpacing",
        kind: PropKind::Spacing,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0805,
        name: "BondLength",
        kind: PropKind::Coord,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0806,
        name: "BoldWidth",
        kind: PropKind::Coord,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0807,
        name: "LineWidth",
        kind: PropKind::Coord,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x080F,
        name: "WidthPages",
        kind: PropKind::U16,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0810,
        name: "HeightPages",
        kind: PropKind::U16,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0A00,
        name: "GraphicType",
        kind: PropKind::E16,
        enums: GRAPHIC_TYPE_ENUMS,
    },
    CdxPropSpec {
        tag: 0x0A01,
        name: "LineType",
        kind: PropKind::B16,
        enums: LINE_TYPE_BITS,
    },
    CdxPropSpec {
        tag: 0x0A03,
        name: "RectangleType",
        kind: PropKind::B16,
        enums: RECT_TYPE_BITS,
    },
    CdxPropSpec {
        tag: 0x0A04,
        name: "OvalType",
        kind: PropKind::B16,
        enums: OVAL_TYPE_BITS,
    },
    CdxPropSpec {
        tag: 0x0A05,
        name: "OrbitalType",
        kind: PropKind::E16,
        enums: ORBITAL_TYPE_ENUMS,
    },
    CdxPropSpec {
        tag: 0x0A06,
        name: "BracketType",
        kind: PropKind::E16,
        enums: BRACKET_TYPE_ENUMS,
    },
    CdxPropSpec {
        tag: 0x0A08,
        name: "CurveType",
        kind: PropKind::I16,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0A23,
        name: "CurvePoints",
        kind: PropKind::Curve,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0A24,
        name: "BracketUsage",
        kind: PropKind::E16,
        enums: BRACKET_USAGE_ENUMS,
    },
    CdxPropSpec {
        tag: 0x0A27,
        name: "BracketedObjectIDs",
        kind: PropKind::Ids,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0A2A,
        name: "SRULabel",
        kind: PropKind::Str,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0A2F,
        name: "ArrowheadType",
        kind: PropKind::E16,
        enums: ARROWHEAD_TYPE_ENUMS,
    },
    CdxPropSpec {
        tag: 0x0A33,
        name: "ArrowShaftSpacing",
        kind: PropKind::I16,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0A34,
        name: "ArrowEquilibriumRatio",
        kind: PropKind::I16,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0A35,
        name: "ArrowheadHead",
        kind: PropKind::E16,
        enums: ARROWHEAD_END_ENUMS,
    },
    CdxPropSpec {
        tag: 0x0A36,
        name: "ArrowheadTail",
        kind: PropKind::E16,
        enums: ARROWHEAD_END_ENUMS,
    },
    CdxPropSpec {
        tag: 0x0A3B,
        name: "NoGo",
        kind: PropKind::E8,
        enums: NOGO_ENUMS,
    },
    CdxPropSpec {
        tag: 0x0A3E,
        name: "ArrowSource",
        kind: PropKind::I16,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0A3F,
        name: "ArrowTarget",
        kind: PropKind::I16,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0AA0,
        name: "OriginFraction",
        kind: PropKind::F64,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0AA1,
        name: "SolventFrontFraction",
        kind: PropKind::F64,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0AA2,
        name: "ShowOrigin",
        kind: PropKind::Flag,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0AA3,
        name: "ShowSolventFront",
        kind: PropKind::Flag,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0AA4,
        name: "ShowBorders",
        kind: PropKind::Flag,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0AB0,
        name: "Rf",
        kind: PropKind::F64,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0AB2,
        name: "ShowRf",
        kind: PropKind::Flag,
        enums: &[],
    },
    CdxPropSpec {
        tag: 0x0401,
        name: "LabelDisplay",
        kind: PropKind::I8,
        enums: &[],
    },
];

/// Write a `ChemDocument` as a binary ChemDraw `.cdx` file (`VjCD0100` header + binary tag tree).
pub fn write_cdx(doc: &ChemDocument) -> Vec<u8> {
    let ast = doc_to_cdxml_ast(doc);
    xml_to_cdx(&ast)
}

/// Read a binary ChemDraw `.cdx` file (`VjCD0100` header + binary tag tree) into a `ChemDocument`.
pub fn read_cdx(bytes: &[u8]) -> Result<ChemDocument, ChemError> {
    let ast = cdx_to_xml(bytes)?;
    cdxml_ast_to_doc(&ast)
}

/// Encode a `<CDXML>` `XmlNode` tree into standard ChemDraw binary `.cdx` (`VjCD0100`) bytes.
pub fn xml_to_cdx(root: &XmlNode) -> Vec<u8> {
    let mut out = Vec::with_capacity(1024);
    // 22-byte CDX header: "VjCD0100" + 04 03 02 01 + 10 zero reserved bytes
    out.extend_from_slice(CDX_MAGIC);
    out.extend_from_slice(&[4, 3, 2, 1]);
    out.extend_from_slice(&[0u8; 10]);

    write_cdx_object(root, 0, &mut out);
    out
}

fn write_cdx_object(el: &XmlNode, depth: usize, out: &mut Vec<u8>) {
    if depth > 64 {
        return;
    }
    let Some(&(obj_tag, _)) = CDX_OBJECTS.iter().find(|&&(_, nm)| nm == el.name) else {
        return;
    };
    let id = el
        .attrs
        .get("id")
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0);
    out.extend_from_slice(&obj_tag.to_le_bytes());
    out.extend_from_slice(&(id as u32).to_le_bytes());

    let put_prop = |tag: u16, data: &[u8], buf: &mut Vec<u8>| {
        buf.extend_from_slice(&tag.to_le_bytes());
        if data.len() >= 0xffff {
            buf.extend_from_slice(&0xffffu16.to_le_bytes());
            buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
        } else {
            buf.extend_from_slice(&(data.len() as u16).to_le_bytes());
        }
        buf.extend_from_slice(data);
    };

    // Encode attributes via `CDX_PROPS`
    for (k, v) in &el.attrs {
        if k == "id" {
            continue;
        }
        let Some(spec) = CDX_PROPS.iter().find(|p| p.name == k) else {
            continue;
        };
        if let Some(encoded) = encode_cdx_prop(spec, v) {
            put_prop(spec.tag, &encoded, out);
        }
    }

    // Encode `<fonttable>`, `<colortable>`, and `<s>` text runs as CDX properties!
    for kid in &el.kids {
        if kid.name == "colortable" {
            let mut buf = Vec::new();
            buf.extend_from_slice(&(kid.kids.len() as u16).to_le_bytes());
            for c in &kid.kids {
                for ch in ["r", "g", "b"] {
                    let fv = c
                        .attrs
                        .get(ch)
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0)
                        .clamp(0.0, 1.0);
                    buf.extend_from_slice(&((fv * 65535.0).round() as u16).to_le_bytes());
                }
            }
            put_prop(0x0300, &buf, out);
        } else if kid.name == "fonttable" {
            let mut buf = Vec::new();
            buf.extend_from_slice(&1u16.to_le_bytes()); // platform
            buf.extend_from_slice(&(kid.kids.len() as u16).to_le_bytes());
            for f in &kid.kids {
                let fid = f
                    .attrs
                    .get("id")
                    .and_then(|s| s.parse::<u16>().ok())
                    .unwrap_or(3);
                let cs: u16 = 65001; // UTF-8
                let nm_bytes = f
                    .attrs
                    .get("name")
                    .map(|s| s.as_bytes())
                    .unwrap_or(b"Arial");
                buf.extend_from_slice(&fid.to_le_bytes());
                buf.extend_from_slice(&cs.to_le_bytes());
                buf.extend_from_slice(&(nm_bytes.len() as u16).to_le_bytes());
                buf.extend_from_slice(nm_bytes);
            }
            put_prop(0x0100, &buf, out);
        }
    }

    // Text runs (`<s>`) -> property `0x0700` (legacy ASCII/Latin) and `0x0709` (UTF-8 styled runs)!
    let s_runs: Vec<&XmlNode> = el.kids.iter().filter(|k| k.name == "s").collect();
    if !s_runs.is_empty() {
        let mut run_headers: Vec<[u16; 5]> = Vec::new();
        let mut utf8_bytes: Vec<u8> = Vec::new();
        for s in s_runs {
            if s.text.is_empty() {
                continue;
            }
            let font = s
                .attrs
                .get("font")
                .and_then(|v| v.parse::<u16>().ok())
                .unwrap_or(3);
            let face = s
                .attrs
                .get("face")
                .and_then(|v| v.parse::<u16>().ok())
                .unwrap_or(0);
            let size = s
                .attrs
                .get("size")
                .and_then(|v| v.parse::<f64>().ok())
                .map(|sz| (sz * 20.0).round() as u16)
                .unwrap_or(200);
            let color = s
                .attrs
                .get("color")
                .and_then(|v| v.parse::<u16>().ok())
                .unwrap_or(1);
            run_headers.push([utf8_bytes.len() as u16, font, face, size, color]);
            utf8_bytes.extend_from_slice(s.text.as_bytes());
        }
        let mut packed = Vec::with_capacity(2 + 10 * run_headers.len() + utf8_bytes.len());
        packed.extend_from_slice(&(run_headers.len() as u16).to_le_bytes());
        for rh in &run_headers {
            for &word in rh {
                packed.extend_from_slice(&word.to_le_bytes());
            }
        }
        packed.extend_from_slice(&utf8_bytes);
        put_prop(0x0700, &packed, out);
        put_prop(0x0709, &packed, out);
    }

    // Write child objects
    for kid in &el.kids {
        if matches!(kid.name.as_str(), "colortable" | "fonttable" | "s") {
            continue;
        }
        write_cdx_object(kid, depth + 1, out);
    }

    // End-of-object tag `0x0000`
    out.extend_from_slice(&0u16.to_le_bytes());
}

fn fixed_32(v: f64) -> i32 {
    (v * 65536.0).round() as i32
}

fn encode_cdx_prop(spec: &CdxPropSpec, val: &str) -> Option<Vec<u8>> {
    let lookup_enum = |s: &str| -> Option<i32> {
        spec.enums
            .iter()
            .find(|&&(_, nm)| nm.eq_ignore_ascii_case(s))
            .map(|&(code, _)| code)
            .or_else(|| s.parse::<i32>().ok())
    };

    match spec.kind {
        PropKind::Flag => (val == "yes" || val == "true").then(Vec::new),
        PropKind::Bool => Some(vec![if val == "yes" || val == "true" { 1 } else { 0 }]),
        PropKind::I8 => Some(vec![val.parse::<i8>().ok()? as u8]),
        PropKind::I16 => Some(val.parse::<i16>().ok()?.to_le_bytes().to_vec()),
        PropKind::U16 => Some(val.parse::<u16>().ok()?.to_le_bytes().to_vec()),
        PropKind::I32 => Some(val.parse::<i32>().ok()?.to_le_bytes().to_vec()),
        PropKind::E8 => Some(vec![lookup_enum(val.trim())? as i8 as u8]),
        PropKind::E16 => Some((lookup_enum(val.trim())? as i16).to_le_bytes().to_vec()),
        PropKind::B16 => {
            let mut mask: i32 = 0;
            for part in val.split_whitespace() {
                mask |= lookup_enum(part)?;
            }
            Some((mask as u16).to_le_bytes().to_vec())
        }
        PropKind::Order => {
            let mut mask: i32 = 0;
            for part in val.split_whitespace() {
                mask |= lookup_enum(part)?;
            }
            Some((mask as u16).to_le_bytes().to_vec())
        }
        PropKind::Charge => {
            let q = val.parse::<i8>().ok()?;
            Some(vec![q as u8])
        }
        PropKind::Coord | PropKind::Angle => {
            let r = val.parse::<f64>().ok()?;
            Some(fixed_32(r).to_le_bytes().to_vec())
        }
        PropKind::Spacing => {
            let r = val.parse::<f64>().ok()?;
            Some(((r * 10.0).round() as u16).to_le_bytes().to_vec())
        }
        PropKind::P2 => {
            // Note ChemDraw CDX stores p2 as `(y, x)` in two 32-bit 16.16 fixed-point integers!
            let (x, y) = parse_two_f64(val)?;
            let mut b = Vec::with_capacity(8);
            b.extend_from_slice(&fixed_32(y).to_le_bytes());
            b.extend_from_slice(&fixed_32(x).to_le_bytes());
            Some(b)
        }
        PropKind::P3 => {
            let v = parse_f64_list(val);
            if v.len() < 2 {
                return None;
            }
            let z = v.get(2).copied().unwrap_or(0.0);
            let mut b = Vec::with_capacity(12);
            b.extend_from_slice(&fixed_32(v[0]).to_le_bytes());
            b.extend_from_slice(&fixed_32(v[1]).to_le_bytes());
            b.extend_from_slice(&fixed_32(z).to_le_bytes());
            Some(b)
        }
        PropKind::Rect => {
            // CDXML `BoundingBox="left top right bottom"` <-> CDX `(top, left, bottom, right)`!
            let v = parse_f64_list(val);
            if v.len() != 4 {
                return None;
            }
            let mut b = Vec::with_capacity(16);
            b.extend_from_slice(&fixed_32(v[1]).to_le_bytes());
            b.extend_from_slice(&fixed_32(v[0]).to_le_bytes());
            b.extend_from_slice(&fixed_32(v[3]).to_le_bytes());
            b.extend_from_slice(&fixed_32(v[2]).to_le_bytes());
            Some(b)
        }
        PropKind::F64 => {
            let r = val.parse::<f64>().ok()?;
            Some(r.to_le_bytes().to_vec())
        }
        PropKind::Str => {
            // CDX styled string with 0 runs: `u16(0)` + UTF-8 bytes
            let mut b = Vec::with_capacity(2 + val.len());
            b.extend_from_slice(&0u16.to_le_bytes());
            b.extend_from_slice(val.as_bytes());
            Some(b)
        }
        PropKind::Plain => Some(val.as_bytes().to_vec()),
        PropKind::Ids => {
            let mut b = Vec::new();
            for tok in val.split_whitespace() {
                let id = tok.parse::<i32>().ok()?;
                b.extend_from_slice(&id.to_le_bytes());
            }
            Some(b)
        }
        PropKind::Attachments => {
            let ids: Option<Vec<i32>> = val
                .split_whitespace()
                .map(|t| t.parse::<i32>().ok())
                .collect();
            let ids = ids?;
            let mut b = Vec::with_capacity(2 + 4 * ids.len());
            b.extend_from_slice(&(ids.len() as u16).to_le_bytes());
            for id in ids {
                b.extend_from_slice(&id.to_le_bytes());
            }
            Some(b)
        }
        PropKind::Curve => {
            let v = parse_f64_list(val);
            if !v.len().is_multiple_of(2) {
                return None;
            }
            let cnt = v.len() / 2;
            let mut b = Vec::with_capacity(2 + 8 * cnt);
            b.extend_from_slice(&(cnt as u16).to_le_bytes());
            for k in 0..cnt {
                // `(y, x)` per point in CDX `CurvePoints`!
                b.extend_from_slice(&fixed_32(v[2 * k + 1]).to_le_bytes());
                b.extend_from_slice(&fixed_32(v[2 * k]).to_le_bytes());
            }
            Some(b)
        }
    }
}

/// Decode a binary ChemDraw `.cdx` (`VjCD0100`) byte buffer into a `<CDXML>` `XmlNode` AST.
pub fn cdx_to_xml(bytes: &[u8]) -> Result<XmlNode, ChemError> {
    if bytes.len() < 28 || &bytes[0..8] != CDX_MAGIC {
        return Err(ChemError::new(
            "Invalid CDX file: missing 'VjCD0100' magic signature",
        ));
    }
    let little = if bytes[8..12] == [4, 3, 2, 1] {
        true
    } else if bytes[8..12] == [1, 2, 3, 4] {
        false
    } else {
        return Err(ChemError::new("Invalid CDX byte-order marker"));
    };

    let mut pos = 22usize;
    let mut obj_count = 0usize;

    let read_u16 = |b: &[u8], p: &mut usize| -> Result<u16, ChemError> {
        if *p + 2 > b.len() {
            return Err(ChemError::new("Truncated CDX stream reading u16"));
        }
        let arr = [b[*p], b[*p + 1]];
        *p += 2;
        Ok(if little {
            u16::from_le_bytes(arr)
        } else {
            u16::from_be_bytes(arr)
        })
    };
    let read_u32 = |b: &[u8], p: &mut usize| -> Result<u32, ChemError> {
        if *p + 4 > b.len() {
            return Err(ChemError::new("Truncated CDX stream reading u32"));
        }
        let arr = [b[*p], b[*p + 1], b[*p + 2], b[*p + 3]];
        *p += 4;
        Ok(if little {
            u32::from_le_bytes(arr)
        } else {
            u32::from_be_bytes(arr)
        })
    };

    let doc_tag = read_u16(bytes, &mut pos)?;
    let doc_id = read_u32(bytes, &mut pos)? as i32;
    if doc_tag != 0x8000 && (doc_tag != 0 || doc_id != 0) {
        return Err(ChemError::new("Invalid root CDX document tag"));
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_object(
        bytes: &[u8],
        pos: &mut usize,
        little: bool,
        tag: u16,
        id: i32,
        depth: usize,
        obj_count: &mut usize,
    ) -> Result<XmlNode, ChemError> {
        *obj_count += 1;
        if *obj_count > 500_000 || depth > 64 {
            return Err(ChemError::new("CDX object count or depth limit exceeded"));
        }
        let el_name = if depth == 0 {
            "CDXML".to_string()
        } else {
            CDX_OBJECTS
                .iter()
                .find(|&&(t, _)| t == tag)
                .map(|&(_, n)| n.to_string())
                .unwrap_or_else(|| format!("object{:04x}", tag))
        };
        let mut el = XmlNode::new(el_name);
        if id != 0 {
            el.attrs.insert("id".into(), id.to_string());
        }

        let mut legacy_text: Option<&[u8]> = None;
        let mut unicode_text: Option<&[u8]> = None;
        let mut child_objects: Vec<XmlNode> = Vec::new();

        loop {
            if depth == 0 && *pos >= bytes.len() {
                break;
            }
            if *pos + 2 > bytes.len() {
                return Err(ChemError::new("Unexpected EOF inside CDX object"));
            }
            let t = if little {
                u16::from_le_bytes([bytes[*pos], bytes[*pos + 1]])
            } else {
                u16::from_be_bytes([bytes[*pos], bytes[*pos + 1]])
            };
            *pos += 2;
            if t == 0 {
                break;
            }
            if (t & 0x8000) != 0 {
                if *pos + 4 > bytes.len() {
                    return Err(ChemError::new("Truncated CDX child object ID"));
                }
                let cid = if little {
                    i32::from_le_bytes([
                        bytes[*pos],
                        bytes[*pos + 1],
                        bytes[*pos + 2],
                        bytes[*pos + 3],
                    ])
                } else {
                    i32::from_be_bytes([
                        bytes[*pos],
                        bytes[*pos + 1],
                        bytes[*pos + 2],
                        bytes[*pos + 3],
                    ])
                };
                *pos += 4;
                child_objects.push(decode_object(
                    bytes,
                    pos,
                    little,
                    t,
                    cid,
                    depth + 1,
                    obj_count,
                )?);
                continue;
            }

            // Property `(tag, len, payload)`
            if *pos + 2 > bytes.len() {
                return Err(ChemError::new("Truncated CDX property length"));
            }
            let short_len = if little {
                u16::from_le_bytes([bytes[*pos], bytes[*pos + 1]])
            } else {
                u16::from_be_bytes([bytes[*pos], bytes[*pos + 1]])
            };
            *pos += 2;
            let len = if short_len == 0xffff {
                if *pos + 4 > bytes.len() {
                    return Err(ChemError::new("Truncated CDX 32-bit property length"));
                }
                let l32 = if little {
                    u32::from_le_bytes([
                        bytes[*pos],
                        bytes[*pos + 1],
                        bytes[*pos + 2],
                        bytes[*pos + 3],
                    ])
                } else {
                    u32::from_be_bytes([
                        bytes[*pos],
                        bytes[*pos + 1],
                        bytes[*pos + 2],
                        bytes[*pos + 3],
                    ])
                } as usize;
                *pos += 4;
                l32
            } else {
                short_len as usize
            };

            if *pos + len > bytes.len() {
                return Err(ChemError::new("Truncated CDX property payload"));
            }
            let payload = &bytes[*pos..*pos + len];
            *pos += len;

            if t == 0x0700 {
                legacy_text = Some(payload);
            } else if t == 0x0709 {
                unicode_text = Some(payload);
            } else if t == 0x0100 {
                if let Some(ft) = decode_cdx_fonttable(payload, little) {
                    el.kids.push(ft);
                }
            } else if t == 0x0300 {
                if let Some(ct) = decode_cdx_colortable(payload, little) {
                    el.kids.push(ct);
                }
            } else if let Some(spec) = CDX_PROPS.iter().find(|p| p.tag == t) {
                if let Some(val_str) = decode_cdx_prop(spec, payload, little) {
                    el.attrs.insert(spec.name.to_string(), val_str);
                }
            }
        }

        if let Some(tb) = unicode_text.or(legacy_text) {
            el.kids.extend(decode_cdx_text_runs(tb, little));
        }
        el.kids.extend(child_objects);

        Ok(el)
    }

    decode_object(bytes, &mut pos, little, 0x8000, doc_id, 0, &mut obj_count)
}

fn read_int_var(b: &[u8], little: bool, signed: bool) -> Option<i32> {
    match b.len() {
        1 => Some(if signed {
            (b[0] as i8) as i32
        } else {
            b[0] as i32
        }),
        2 => {
            let u = if little {
                u16::from_le_bytes([b[0], b[1]])
            } else {
                u16::from_be_bytes([b[0], b[1]])
            };
            Some(if signed { (u as i16) as i32 } else { u as i32 })
        }
        4 => {
            let u = if little {
                u32::from_le_bytes([b[0], b[1], b[2], b[3]])
            } else {
                u32::from_be_bytes([b[0], b[1], b[2], b[3]])
            };
            Some(u as i32)
        }
        _ => None,
    }
}

fn read_coord_at(b: &[u8], offset: usize, little: bool) -> Option<f64> {
    if offset + 4 > b.len() {
        return None;
    }
    let arr = [b[offset], b[offset + 1], b[offset + 2], b[offset + 3]];
    let raw = if little {
        i32::from_le_bytes(arr)
    } else {
        i32::from_be_bytes(arr)
    };
    Some((raw as f64) / 65536.0)
}

fn decode_bytes_lossless_utf8(b: &[u8]) -> String {
    match std::str::from_utf8(b) {
        Ok(s) => s.replace("\r\n", "\n").replace('\r', "\n"),
        Err(_) => b.iter().map(|&c| c as char).collect(),
    }
}

fn decode_cdx_prop(spec: &CdxPropSpec, b: &[u8], little: bool) -> Option<String> {
    let lookup_name = |code: i32| -> String {
        spec.enums
            .iter()
            .find(|&&(c, _)| c == code)
            .or_else(|| spec.enums.iter().find(|&&(c, _)| c == (code & 0xffff)))
            .map(|&(_, nm)| nm.to_string())
            .unwrap_or_else(|| code.to_string())
    };

    match spec.kind {
        PropKind::Flag => Some("yes".to_string()),
        PropKind::Bool => Some(if b.iter().any(|&x| x != 0) {
            "yes".to_string()
        } else {
            "no".to_string()
        }),
        PropKind::I8 | PropKind::I16 | PropKind::I32 => {
            Some(read_int_var(b, little, true)?.to_string())
        }
        PropKind::U16 => Some(read_int_var(b, little, false)?.to_string()),
        PropKind::E8 | PropKind::E16 => Some(lookup_name(read_int_var(b, little, true)?)),
        PropKind::B16 => {
            let v = read_int_var(b, little, false)?;
            let names: Vec<&str> = spec
                .enums
                .iter()
                .filter(|&&(bit, _)| bit != 0 && (v & bit) == bit)
                .map(|&(_, nm)| nm)
                .collect();
            if names.is_empty() {
                Some(
                    spec.enums
                        .iter()
                        .find(|&&(bit, _)| bit == 0)
                        .map(|&(_, nm)| nm.to_string())
                        .unwrap_or_else(|| "0".to_string()),
                )
            } else {
                Some(names.join(" "))
            }
        }
        PropKind::Order => {
            let v = read_int_var(b, little, false)?;
            let names: Vec<&str> = spec
                .enums
                .iter()
                .filter(|&&(bit, _)| (v & bit) != 0)
                .map(|&(_, nm)| nm)
                .collect();
            Some(if names.is_empty() {
                "1".to_string()
            } else {
                names.join(" ")
            })
        }
        PropKind::Charge => {
            if b.len() == 4 {
                let raw = read_int_var(b, little, true)? as f64;
                Some(format!("{}", (raw / 16777216.0).round() as i32))
            } else {
                Some(read_int_var(b, little, true)?.to_string())
            }
        }
        PropKind::Coord | PropKind::Angle => Some(format!("{}", read_coord_at(b, 0, little)?)),
        PropKind::Spacing => Some(format!(
            "{}",
            (read_int_var(b, little, false)? as f64) / 10.0
        )),
        PropKind::P2 => {
            let y = read_coord_at(b, 0, little)?;
            let x = read_coord_at(b, 4, little)?;
            Some(format!("{} {}", x, y))
        }
        PropKind::P3 => {
            let x = read_coord_at(b, 0, little)?;
            let y = read_coord_at(b, 4, little)?;
            let z = read_coord_at(b, 8, little)?;
            Some(format!("{} {} {}", x, y, z))
        }
        PropKind::Rect => {
            let top = read_coord_at(b, 0, little)?;
            let left = read_coord_at(b, 4, little)?;
            let bottom = read_coord_at(b, 8, little)?;
            let right = read_coord_at(b, 12, little)?;
            Some(format!("{} {} {} {}", left, top, right, bottom))
        }
        PropKind::F64 => {
            if b.len() != 8 {
                return None;
            }
            let arr: [u8; 8] = b.try_into().ok()?;
            let v = if little {
                f64::from_le_bytes(arr)
            } else {
                f64::from_be_bytes(arr)
            };
            Some(format!("{}", v))
        }
        PropKind::Str => {
            // Styled string: `u16 n_runs` + `10 * n_runs` bytes + text
            if b.len() >= 2 {
                let n_runs = if little {
                    u16::from_le_bytes([b[0], b[1]])
                } else {
                    u16::from_be_bytes([b[0], b[1]])
                } as usize;
                if 2 + 10 * n_runs <= b.len() {
                    return Some(decode_bytes_lossless_utf8(&b[2 + 10 * n_runs..]));
                }
            }
            Some(decode_bytes_lossless_utf8(b))
        }
        PropKind::Plain => Some(decode_bytes_lossless_utf8(b)),
        PropKind::Ids => {
            if !b.len().is_multiple_of(4) {
                return None;
            }
            let cnt = b.len() / 4;
            let mut ids = Vec::with_capacity(cnt);
            for k in 0..cnt {
                ids.push(read_int_var(&b[4 * k..4 * k + 4], little, true)?.to_string());
            }
            Some(ids.join(" "))
        }
        PropKind::Attachments => {
            let has_count = b.len() >= 2 && {
                let c = if little {
                    u16::from_le_bytes([b[0], b[1]])
                } else {
                    u16::from_be_bytes([b[0], b[1]])
                } as usize;
                b.len() == 2 + 4 * c
            };
            let start = if has_count { 2 } else { 0 };
            if !(b.len() - start).is_multiple_of(4) {
                return None;
            }
            let cnt = (b.len() - start) / 4;
            let mut ids = Vec::with_capacity(cnt);
            for k in 0..cnt {
                let off = start + 4 * k;
                ids.push(read_int_var(&b[off..off + 4], little, true)?.to_string());
            }
            Some(ids.join(" "))
        }
        PropKind::Curve => {
            if b.len() < 2 {
                return None;
            }
            let cnt = if little {
                u16::from_le_bytes([b[0], b[1]])
            } else {
                u16::from_be_bytes([b[0], b[1]])
            } as usize;
            if b.len() != 2 + 8 * cnt {
                return None;
            }
            let mut coords = Vec::with_capacity(2 * cnt);
            for k in 0..cnt {
                let y = read_coord_at(b, 2 + 8 * k, little)?;
                let x = read_coord_at(b, 6 + 8 * k, little)?;
                coords.push(format!("{} {}", x, y));
            }
            Some(coords.join(" "))
        }
    }
}

fn decode_cdx_text_runs(b: &[u8], little: bool) -> Vec<XmlNode> {
    if b.len() < 2 {
        return Vec::new();
    }
    let n_runs = if little {
        u16::from_le_bytes([b[0], b[1]])
    } else {
        u16::from_be_bytes([b[0], b[1]])
    } as usize;
    if 2 + 10 * n_runs > b.len() {
        let mut s = XmlNode::new("s");
        s.text = decode_bytes_lossless_utf8(b);
        return vec![s];
    }

    let mut runs = Vec::with_capacity(n_runs);
    let u16_at = |off: usize| -> u16 {
        if little {
            u16::from_le_bytes([b[off], b[off + 1]])
        } else {
            u16::from_be_bytes([b[off], b[off + 1]])
        }
    };
    for k in 0..n_runs {
        let base = 2 + 10 * k;
        runs.push([
            u16_at(base) as usize,
            u16_at(base + 2) as usize,
            u16_at(base + 4) as usize,
            u16_at(base + 6) as usize,
            u16_at(base + 8) as usize,
        ]);
    }
    let text_bytes = &b[2 + 10 * n_runs..];
    if runs.is_empty() {
        if text_bytes.is_empty() {
            return Vec::new();
        }
        let mut s = XmlNode::new("s");
        s.text = decode_bytes_lossless_utf8(text_bytes);
        return vec![s];
    }

    let mut out = Vec::new();
    for (k, r) in runs.iter().enumerate() {
        let start = r[0].min(text_bytes.len());
        let end = if k + 1 < runs.len() {
            runs[k + 1][0].max(start).min(text_bytes.len())
        } else {
            text_bytes.len()
        };
        if start >= end {
            continue;
        }
        let mut s = XmlNode::new("s");
        if r[1] != 0xffff {
            s.attrs.insert("font".into(), r[1].to_string());
        }
        if r[2] != 0 {
            s.attrs.insert("face".into(), r[2].to_string());
        }
        if r[3] != 0 {
            s.attrs
                .insert("size".into(), format!("{}", (r[3] as f64) / 20.0));
        }
        s.attrs.insert("color".into(), r[4].to_string());
        s.text = decode_bytes_lossless_utf8(&text_bytes[start..end]);
        out.push(s);
    }
    out
}

fn decode_cdx_fonttable(b: &[u8], little: bool) -> Option<XmlNode> {
    if b.len() < 4 {
        return None;
    }
    let count = if little {
        u16::from_le_bytes([b[2], b[3]])
    } else {
        u16::from_be_bytes([b[2], b[3]])
    } as usize;
    let mut ft = XmlNode::new("fonttable");
    let mut at = 4usize;
    for _ in 0..count {
        if at + 6 > b.len() {
            return None;
        }
        let id = if little {
            u16::from_le_bytes([b[at], b[at + 1]])
        } else {
            u16::from_be_bytes([b[at], b[at + 1]])
        };
        let len = if little {
            u16::from_le_bytes([b[at + 4], b[at + 5]])
        } else {
            u16::from_be_bytes([b[at + 4], b[at + 5]])
        } as usize;
        if at + 6 + len > b.len() {
            return None;
        }
        let nm = decode_bytes_lossless_utf8(&b[at + 6..at + 6 + len]);
        ft.kids.push(
            XmlNode::new("font")
                .with_attr("id", id.to_string())
                .with_attr("charset", "utf-8")
                .with_attr("name", nm),
        );
        at += 6 + len;
    }
    Some(ft)
}

fn decode_cdx_colortable(b: &[u8], little: bool) -> Option<XmlNode> {
    if b.len() < 2 {
        return None;
    }
    let count = if little {
        u16::from_le_bytes([b[0], b[1]])
    } else {
        u16::from_be_bytes([b[0], b[1]])
    } as usize;
    if b.len() != 2 + 6 * count {
        return None;
    }
    let mut ct = XmlNode::new("colortable");
    for k in 0..count {
        let off = 2 + 6 * k;
        let ch = |o: usize| -> String {
            let v = if little {
                u16::from_le_bytes([b[o], b[o + 1]])
            } else {
                u16::from_be_bytes([b[o], b[o + 1]])
            };
            format!("{:.4}", (v as f64) / 65535.0)
        };
        ct.kids.push(
            XmlNode::new("color")
                .with_attr("r", ch(off))
                .with_attr("g", ch(off + 2))
                .with_attr("b", ch(off + 4)),
        );
    }
    Some(ct)
}

// ============================================================================
// 4. Lossless Octet `.oct` Format (`write_oct` & `read_oct`, matching `oct.ts`)
// ============================================================================

/// Dual-seed FNV-1a change-detection hash over UTF-16 code units (matching `oct.ts` `hash(text)`!).
pub fn octet_fnv_hash(text: &str) -> String {
    let mut a: u32 = 0x811c9dc5;
    let mut b: u32 = 0x01000193 ^ 0x5bd1e995;
    for unit in text.encode_utf16() {
        let c = unit as u32;
        a = (a ^ c).wrapping_mul(0x01000193);
        b = (b ^ c).wrapping_mul(0x01000193) ^ (b >> 13);
    }
    format!("{:08x}{:08x}", a, b)
}

/// Write a `ChemDocument` in Octet `.oct` format: valid ChemDraw `.cdxml` followed by
/// `<!--Octet 1 <16-hex-fnv-hash> <base64-json>-->\n` so ChemDraw opens the CDXML directly while
/// Octet restores 100% lossless `ChemDocument` metadata (and detects external edits via hash mismatch!).
pub fn write_oct(doc: &ChemDocument) -> Result<String, ChemError> {
    let mut cdxml = write_cdxml(doc);
    if !cdxml.ends_with('\n') {
        cdxml.push('\n');
    }
    let json = serde_json::to_string(doc)
        .map_err(|e| ChemError::new(format!("Failed to serialize ChemDocument JSON: {}", e)))?;
    let h = octet_fnv_hash(&cdxml);
    let b64 = base64_encode(json.as_bytes());
    Ok(format!("{}{}{h} {b64}-->\n", cdxml, OCT_COMMENT_MARK))
}

/// Read an Octet `.oct` (or `.cdxml`, raw `.octet` JSON, or base64 `.cdx`) document into a `ChemDocument`.
/// Verifies the embedded FNV-1a hash of the preceding CDXML; if modified externally, falls back to `read_cdxml`!
pub fn read_oct(text: &str) -> Result<ChemDocument, ChemError> {
    let trimmed_start = text.trim_start();
    if trimmed_start.starts_with(CDX_BASE64_MAGIC) {
        let raw = base64_decode(trimmed_start.trim())?;
        return read_cdx(&raw);
    }
    if trimmed_start.starts_with('{') {
        return serde_json::from_str::<ChemDocument>(trimmed_start)
            .map_err(|e| ChemError::new(format!("Invalid .octet JSON: {}", e)));
    }

    if let Some(mark_pos) = text.rfind(OCT_COMMENT_MARK) {
        let body = &text[..mark_pos];
        let after = text[mark_pos + OCT_COMMENT_MARK.len()..].trim_end();
        if let Some(comment_inner) = after.strip_suffix("-->") {
            if let Some((hash_part, b64_part)) = comment_inner.split_once(' ') {
                if hash_part.len() == 16 && octet_fnv_hash(body) == hash_part {
                    if let Ok(json_bytes) = base64_decode(b64_part.trim()) {
                        if let Ok(doc) = serde_json::from_slice::<ChemDocument>(&json_bytes) {
                            return Ok(doc);
                        }
                    }
                }
            }
        }
    }

    // Fallback when CDXML was modified by another program or no Octet comment was present
    read_cdxml(text)
}

// Pure-Rust RFC 4648 Base64 encoder & decoder (zero external crates)
const B64_ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let n = (b0 << 16) | (b1 << 8) | b2;

        out.push(B64_ALPHABET[((n >> 18) & 0x3f) as usize] as char);
        out.push(B64_ALPHABET[((n >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            out.push(B64_ALPHABET[((n >> 6) & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64_ALPHABET[(n & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

pub fn base64_decode(b64: &str) -> Result<Vec<u8>, ChemError> {
    let mut vals = Vec::with_capacity(b64.len());
    for b in b64.bytes() {
        if b.is_ascii_whitespace() || b == b'=' {
            continue;
        }
        let v = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return Err(ChemError::new("Invalid character in Base64 payload")),
        };
        vals.push(v);
    }

    let mut out = Vec::with_capacity((vals.len() * 3) / 4);
    for chunk in vals.chunks(4) {
        if chunk.len() < 2 {
            break;
        }
        let n0 = chunk[0] as u32;
        let n1 = chunk[1] as u32;
        out.push(((n0 << 2) | (n1 >> 4)) as u8);
        if chunk.len() > 2 {
            let n2 = chunk[2] as u32;
            out.push((((n1 & 0x0f) << 4) | (n2 >> 2)) as u8);
            if chunk.len() > 3 {
                let n3 = chunk[3] as u32;
                out.push((((n2 & 0x03) << 6) | n3) as u8);
            }
        }
    }
    Ok(out)
}
