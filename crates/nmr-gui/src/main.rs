//! Interactive WebGPU + `eframe`/`egui` NMR spectrum and backbone resonance assignment workbench.
//!
//! Provenance: Original scientific GUI built on `eframe`, `wgpu`, `nmr-core`, and `nmr-gpu`
//! (native Metal/Vulkan/DX12 and browser `wasm32-unknown-unknown` + WebGPU).

mod contour_painter;
#[allow(dead_code)]
mod octet_canvas;
#[allow(dead_code)]
mod session_io;
#[allow(dead_code)]
mod strip_tiler;

use std::collections::BTreeMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use contour_painter::{ContourCallback, ContourPipelineResource, ContourUniforms};
use eframe::{egui, egui_wgpu};
use nmr_core::evaluate::{score, to_reference, Metrics};
use nmr_core::evo_graph::{assign_evo_graph_from_peaks, EvoGraphConfig};
use nmr_core::nef::read_nef_str;
use nmr_core::neural_picker::{pick_hsqc_2d_neural, NeuralPickerConfig};
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::nmrstar::{read_star, write_star};
use nmr_core::nmrstar::{read_star_str, Reference};
use nmr_core::octet_analyt::iupac_naming::{iupac_name_to_mol, mol_to_iupac_name};
use nmr_core::octet_analyt::jcamp_spec1d::{
    pick_1d_multiplets, read_jcamp, synthesize_1d_nmr, write_jcamp_nmr, JcampCompression,
    Synthesize1dOptions,
};
use nmr_core::octet_analyt::mass_props::compute_phys_chem_props;
use nmr_core::octet_analyt::nmr_predict::{predict_nmr, NmrNucleus};
use nmr_core::octet_analyt::AnalytMol;
use nmr_core::octet_mol::{
    build_peptide_2d, clean_2d, expand_abbreviations, kekulize, parse_biopolymer_sequence,
    parse_condensed_formula, parse_inchi, parse_reaction_smiles, parse_smiles, perceive_sssr,
    read_cdx, read_cdxml, read_mol, read_oct, read_sdf, write_canonical_smiles, write_cdx,
    write_cdxml, write_inchi, write_inchikey, write_mol_v2000, write_oct, write_reaction_smiles,
    write_sdf, ChemDocument, Molecule, SdfRecord,
};
use nmr_core::octet_render::{
    render_scene_svg, DocumentStyle, RenderScene, SceneSpectrum1D, SpectrumSignal1D,
};
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::peaks::read_peaks;
use nmr_core::peaks::{read_peaks_str, Peak};
use nmr_core::picker::{estimate_noise_mad, pick_hsqc_2d, PickerConfig};
use nmr_core::priors::{one_to_three, three_to_one, Priors, ATOMS};
use nmr_core::rng::Rng;
use nmr_core::simulate::{
    render_hsqc_2d, render_triple_3d, simulate, SimConfig, SpectrumRenderConfig,
};
use nmr_core::solver::{
    assign_from_matrices, pair_matrix, unary_matrix, AnnealConfig, AssignResult, Matrix2D,
    ScoreConfig, FORBID,
};
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::spectrum::{read_bruker_pdata_2d, read_nmrpipe_2d};
use nmr_core::spectrum::{Spectrum2D, Spectrum3D};
use nmr_core::spinsystems::{build_spin_systems, SpinSystem, Tolerances};
#[cfg(not(target_arch = "wasm32"))]
use nmr_gpu::GpuEngine;
use octet_canvas::{IoFormat, MolecularPropertiesSummary, OctetBackendAdapter, OctetDrawPanel};

const UBIQUITIN_SEQ: &str =
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";

const SYNTHETIC_UBIQUITIN_STR: &str =
    include_str!("../../../data/fixtures/synthetic_ubiquitin.str");

struct PerfTimer {
    #[cfg(not(target_arch = "wasm32"))]
    start: Instant,
    #[cfg(target_arch = "wasm32")]
    start_ms: f64,
}

impl PerfTimer {
    fn start() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                start: Instant::now(),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let start_ms = web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now())
                .unwrap_or(0.0);
            Self { start_ms }
        }
    }

    fn elapsed_ms(&self) -> f64 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.start.elapsed().as_secs_f64() * 1000.0
        }
        #[cfg(target_arch = "wasm32")]
        {
            let now_ms = web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now())
                .unwrap_or(self.start_ms);
            (now_ms - self.start_ms).max(0.0)
        }
    }
}

fn build_synthetic_ubiquitin() -> Reference {
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(r) = read_star("data/fixtures/synthetic_ubiquitin.str", None) {
        return r;
    }
    if let Ok(r) = read_star_str(SYNTHETIC_UBIQUITIN_STR, None) {
        return r;
    }
    let pri = Priors::provisional();
    let mut rng = Rng::new(1);
    let seq: Vec<String> = UBIQUITIN_SEQ
        .chars()
        .map(|c| one_to_three(c).to_string())
        .collect();
    let seq_ids: Vec<i32> = (1..=(seq.len() as i32)).collect();
    let mut r = Reference::new(seq.clone(), seq_ids);
    for (k, res) in seq.iter().enumerate() {
        if let Some(atoms) = pri.table.get(res) {
            for (atom, &(mu, sd)) in atoms {
                if atom == "H" && (k == 0 || res == "PRO") {
                    continue;
                }
                r.shifts.insert((k, atom.clone()), rng.normal(mu, sd));
            }
        }
    }
    r
}

fn format_star_in_memory(
    ref_data: &Reference,
    confidence: Option<&BTreeMap<usize, f64>>,
) -> String {
    let mut lines = vec![
        "data_nmr_assign_output".to_string(),
        String::new(),
        "save_assigned_chem_shift_list_1".to_string(),
        "   _Assigned_chem_shift_list.Sf_category  assigned_chemical_shifts".to_string(),
        String::new(),
        "   loop_".to_string(),
    ];
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
        lines.push(format!(
            "      {:5} 1 {:5} {:3} {:4} {} {:9.3} 1 {}",
            i + 1,
            ref_data.seq_ids[idx],
            ref_data.sequence[idx],
            atom,
            atom_type,
            v,
            det
        ));
    }
    lines.push("   stop_".to_string());
    lines.push("save_".to_string());
    lines.push(String::new());
    lines.join("\n")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkspaceMode {
    ProteinNmr,
    OctetChem,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum GuiSolverKind {
    #[default]
    EvoGraph,
    Standard,
}

impl GuiSolverKind {
    fn label(self) -> &'static str {
        match self {
            Self::EvoGraph => "Evo-Graph",
            Self::Standard => "Standard",
        }
    }
}

struct CoreOctetBackendAdapter;

impl CoreOctetBackendAdapter {
    fn encode_hex_pretty(bytes: &[u8]) -> String {
        let mut out = String::with_capacity(bytes.len() * 3 + 64);
        out.push_str("# Binary ChemDraw CDX (VjCD0100) hex stream\n");
        for (i, b) in bytes.iter().enumerate() {
            if i > 0 {
                if i % 24 == 0 {
                    out.push('\n');
                } else {
                    out.push(' ');
                }
            }
            out.push_str(&format!("{b:02X}"));
        }
        out
    }

    fn decode_hex_bytes(input: &str) -> Result<Vec<u8>, String> {
        let mut hex_chars = Vec::new();
        for line in input.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('#') || trimmed.starts_with("//") {
                continue;
            }
            for ch in trimmed.chars() {
                if ch.is_ascii_hexdigit() {
                    hex_chars.push(ch);
                } else if !ch.is_whitespace() && ch != ',' && ch != ':' && ch != '-' {
                    return Err(format!(
                        "Unexpected non-hex character '{ch}' in CDX hex buffer"
                    ));
                }
            }
        }
        if hex_chars.len() % 2 != 0 {
            return Err("CDX hex stream must have an even number of hex digits".to_string());
        }
        let mut bytes = Vec::with_capacity(hex_chars.len() / 2);
        for pair in hex_chars.chunks_exact(2) {
            let s: String = pair.iter().collect();
            let v =
                u8::from_str_radix(&s, 16).map_err(|e| format!("Invalid hex byte '{s}': {e}"))?;
            bytes.push(v);
        }
        Ok(bytes)
    }

    fn mol_with_cip_to_scene(mut mol: Molecule) -> RenderScene {
        let _ = kekulize(&mut mol);
        let mut scene = RenderScene::from(&mol);
        let [cx, cy] = scene.atom_centroid();
        scene.translate(-cx, -cy);
        scene
    }
}

impl OctetBackendAdapter for CoreOctetBackendAdapter {
    fn clean_2d(&self, scene: &RenderScene) -> Option<RenderScene> {
        if scene.atoms.is_empty() {
            return Some(scene.clone());
        }
        let orig_centroid = scene.atom_centroid();
        let mut mol = Molecule::from(scene);
        let _ = expand_abbreviations(&mut mol);
        clean_2d(&mut mol);
        let mut cleaned = Self::mol_with_cip_to_scene(mol);
        cleaned.translate(orig_centroid[0], orig_centroid[1]);
        // Preserve non-molecular canvas annotations (arrows, brackets, orbitals, TLC, captions, embedded spectra)
        cleaned.reaction_arrows = scene.reaction_arrows.clone();
        cleaned.curved_arrows = scene.curved_arrows.clone();
        cleaned.brackets = scene.brackets.clone();
        cleaned.orbitals = scene.orbitals.clone();
        cleaned.tlc_plates = scene.tlc_plates.clone();
        cleaned.spectra_1d = scene.spectra_1d.clone();
        cleaned.texts = scene.texts.clone();
        Some(cleaned)
    }

    fn name_or_smiles_to_scene(&self, query: &str) -> Option<Result<RenderScene, String>> {
        let q = query.trim();
        if q.is_empty() {
            return Some(Err(
                "Enter a chemical name, SMILES, InChI, or peptide sequence".to_string(),
            ));
        }

        // 1. Reaction SMILES (`A.B>reag>C`)
        if q.contains(">>") || (q.matches('>').count() == 2 && !q.contains('<')) {
            if let Ok(mut doc) = parse_reaction_smiles(q) {
                let mut x_cursor = 0.0_f64;
                for m in &mut doc.molecules {
                    clean_2d(m);
                    for a in &mut m.atoms {
                        a.x += x_cursor * 1.54;
                    }
                    x_cursor += 5.5;
                }
                let mut scene = RenderScene::from(&doc);
                let [cx, cy] = scene.atom_centroid();
                scene.translate(-cx, -cy);
                return Some(Ok(scene));
            }
        }

        // 2. Standard InChI (`InChI=1S/...`)
        if q.starts_with("InChI=") {
            return Some(
                parse_inchi(q)
                    .map(|mut m| {
                        clean_2d(&mut m);
                        Self::mol_with_cip_to_scene(m)
                    })
                    .map_err(|e| format!("InChI parse error: {e}")),
            );
        }

        // 3. Biopolymer HELM / peptide 3-letter notation (`PEPTIDE1{...}` or `H-Ala-Pro-OH`)
        if q.starts_with("PEPTIDE") || q.starts_with("RNA") || q.starts_with("DNA") {
            if let Ok(mut m) = parse_biopolymer_sequence(q) {
                clean_2d(&mut m);
                return Some(Ok(Self::mol_with_cip_to_scene(m)));
            }
        }

        // 4. Offline IUPAC / common chemical name (`aspirin`, `ibuprofen`, `bicyclo[2.2.1]hept-2-ene`, `(2S)-2-aminopropanoic acid`, ...)
        if let Ok(amol) = iupac_name_to_mol(q) {
            let smi = amol.to_smiles();
            if let Ok(mut m) = parse_smiles(&smi) {
                clean_2d(&mut m);
                return Some(Ok(Self::mol_with_cip_to_scene(m)));
            }
        }

        // 5. SMILES (`CC(=O)Oc1ccccc1C(=O)O`)
        if let Ok(mut m) = parse_smiles(q) {
            clean_2d(&mut m);
            return Some(Ok(Self::mol_with_cip_to_scene(m)));
        }

        // 6. Condensed structural formula (`CH3CH2OH`, `(CH3)3COH`, `PhCOOH`) or peptide (`Ala-Pro-Tyr`)
        if let Ok(mut m) = parse_condensed_formula(q) {
            clean_2d(&mut m);
            return Some(Ok(Self::mol_with_cip_to_scene(m)));
        }
        if q.contains('-') {
            if let Ok(m) = build_peptide_2d(q) {
                return Some(Ok(Self::mol_with_cip_to_scene(m)));
            }
        }

        None
    }

    fn scene_to_name_and_smiles(&self, scene: &RenderScene) -> Option<(String, String)> {
        if scene.atoms.is_empty() {
            return Some((String::new(), String::new()));
        }
        let mut mol = Molecule::from(scene);
        let _ = expand_abbreviations(&mut mol);
        let smiles = if !scene.reaction_arrows.is_empty() {
            let doc = ChemDocument::from(scene);
            let rxn = write_reaction_smiles(&doc);
            if rxn.is_empty() {
                write_canonical_smiles(&mol)
            } else {
                rxn
            }
        } else {
            write_canonical_smiles(&mol)
        };
        let amol = AnalytMol::from(&mol);
        let name = mol_to_iupac_name(&amol).unwrap_or_else(|_| {
            compute_phys_chem_props(&amol)
                .map(|p| p.formula)
                .unwrap_or_else(|_| smiles.clone())
        });
        Some((name, smiles))
    }

    fn compute_properties(&self, scene: &RenderScene) -> Option<MolecularPropertiesSummary> {
        if scene.atoms.is_empty() {
            return Some(MolecularPropertiesSummary::default());
        }
        let mut mol = Molecule::from(scene);
        let _ = expand_abbreviations(&mut mol);
        let rings = perceive_sssr(&mol).len();
        let amol = AnalytMol::from(&mol);
        let props = compute_phys_chem_props(&amol).ok()?;
        let heavy_atoms = amol.atoms.iter().filter(|a| a.element != "H").count();
        Some(MolecularPropertiesSummary {
            formula_hill: props.formula,
            molecular_weight: props.molecular_weight,
            exact_mass: props.monoisotopic_mass,
            elemental_analysis: props.elemental_analysis_text,
            heavy_atoms,
            rings,
            h_bond_donors: props.h_bond_donors,
            h_bond_acceptors: props.h_bond_acceptors,
        })
    }

    fn predict_nmr_1h_13c(
        &self,
        scene: &RenderScene,
        _freq_mhz: f64,
    ) -> Option<(Vec<SpectrumSignal1D>, Vec<SpectrumSignal1D>)> {
        if scene.atoms.is_empty() {
            return Some((Vec::new(), Vec::new()));
        }
        let raw_mol = Molecule::from(scene);
        let has_abbrevs = raw_mol.atoms.iter().any(|a| a.superatom_label.is_some());
        let mol = if has_abbrevs {
            let mut m = raw_mol;
            let _ = expand_abbreviations(&mut m);
            m
        } else {
            raw_mol
        };
        let amol = AnalytMol::from(&mol);

        let pred_h = predict_nmr(&amol, NmrNucleus::H1, Some("CDCl3"), None);
        let pred_c = predict_nmr(&amol, NmrNucleus::C13, Some("CDCl3"), None);

        let n_scene_atoms = scene.atoms.len();
        let sigs_1h = pred_h
            .signals
            .into_iter()
            .map(|s| {
                let mult = s.multiplicity();
                let mut couplings_hz = Vec::new();
                for c in &s.couplings {
                    for _ in 0..c.count.min(12) {
                        couplings_hz.push(c.j_hz);
                    }
                }
                let atom_indices: Vec<usize> = if has_abbrevs {
                    Vec::new()
                } else {
                    s.atoms.into_iter().filter(|&i| i < n_scene_atoms).collect()
                };
                SpectrumSignal1D {
                    ppm: s.shift_ppm,
                    multiplicity: mult,
                    couplings_hz,
                    integral: s.nuclei as f64,
                    atom_indices,
                    assignment_label: Some(format!("{}H", s.nuclei)),
                }
            })
            .collect();

        let sigs_13c = pred_c
            .signals
            .into_iter()
            .map(|s| {
                let atom_indices: Vec<usize> = if has_abbrevs {
                    Vec::new()
                } else {
                    s.atoms.into_iter().filter(|&i| i < n_scene_atoms).collect()
                };
                SpectrumSignal1D {
                    ppm: s.shift_ppm,
                    multiplicity: "s".to_string(),
                    couplings_hz: Vec::new(),
                    integral: s.nuclei as f64,
                    atom_indices,
                    assignment_label: Some(format!("{}C", s.nuclei)),
                }
            })
            .collect();

        Some((sigs_1h, sigs_13c))
    }

    fn export_custom(
        &self,
        scene: &RenderScene,
        style: &DocumentStyle,
        format: IoFormat,
    ) -> Option<Result<String, String>> {
        match format {
            IoFormat::Cdxml => {
                let doc = ChemDocument::from(scene);
                Some(Ok(write_cdxml(&doc)))
            }
            IoFormat::CdxHex => {
                let doc = ChemDocument::from(scene);
                let bytes = write_cdx(&doc);
                Some(Ok(Self::encode_hex_pretty(&bytes)))
            }
            IoFormat::OctJson => {
                let doc = ChemDocument::from(scene);
                Some(write_oct(&doc).map_err(|e| format!("OCT JSON export error: {e}")))
            }
            IoFormat::MolV2000 => {
                let mut mol = Molecule::from(scene);
                let _ = expand_abbreviations(&mut mol);
                Some(write_mol_v2000(&mol).map_err(|e| format!("MOL V2000 export error: {e}")))
            }
            IoFormat::Sdf => {
                let mut mol = Molecule::from(scene);
                let _ = expand_abbreviations(&mut mol);
                let amol = AnalytMol::from(&mol);
                let mut props = BTreeMap::new();
                if let Ok(pin) = mol_to_iupac_name(&amol) {
                    props.insert("IUPAC_NAME".to_string(), pin);
                }
                props.insert("SMILES".to_string(), write_canonical_smiles(&mol));
                props.insert("INCHIKEY".to_string(), write_inchikey(&mol));
                if let Ok(pc) = compute_phys_chem_props(&amol) {
                    props.insert("FORMULA".to_string(), pc.formula);
                    props.insert(
                        "EXACT_MASS".to_string(),
                        format!("{:.5}", pc.monoisotopic_mass),
                    );
                }
                Some(Ok(write_sdf(&[SdfRecord {
                    molecule: mol,
                    properties: props,
                }])))
            }
            IoFormat::Smiles => {
                let mut mol = Molecule::from(scene);
                let _ = expand_abbreviations(&mut mol);
                if !scene.reaction_arrows.is_empty() {
                    let doc = ChemDocument::from(scene);
                    let rxn = write_reaction_smiles(&doc);
                    if !rxn.is_empty() {
                        return Some(Ok(rxn));
                    }
                }
                Some(Ok(write_canonical_smiles(&mol)))
            }
            IoFormat::InChI => {
                let mut mol = Molecule::from(scene);
                let _ = expand_abbreviations(&mut mol);
                let inchi = write_inchi(&mol);
                let key = write_inchikey(&mol);
                Some(Ok(format!("{inchi}\nInChIKey={key}")))
            }
            IoFormat::JcampDx => {
                let mut mol = Molecule::from(scene);
                let _ = expand_abbreviations(&mut mol);
                let amol = AnalytMol::from(&mol);
                let pred = predict_nmr(&amol, NmrNucleus::H1, Some("CDCl3"), None);
                let spec = synthesize_1d_nmr(&pred, &Synthesize1dOptions::default());
                Some(Ok(write_jcamp_nmr(&spec, JcampCompression::Affn)))
            }
            IoFormat::Svg => Some(Ok(render_scene_svg(scene, style))),
            IoFormat::Pdf | IoFormat::Png => None,
        }
    }

    fn import_custom(&self, format: IoFormat, input: &str) -> Option<Result<RenderScene, String>> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Some(Err("Input buffer is empty".to_string()));
        }
        match format {
            IoFormat::Cdxml => Some(
                read_cdxml(trimmed)
                    .map(|doc| {
                        let mut sc = RenderScene::from(&doc);
                        let [cx, cy] = sc.atom_centroid();
                        sc.translate(-cx, -cy);
                        sc
                    })
                    .map_err(|e| format!("CDXML import error: {e}")),
            ),
            IoFormat::CdxHex => Some(
                Self::decode_hex_bytes(trimmed)
                    .and_then(|bytes| {
                        read_cdx(&bytes).map_err(|e| format!("CDX binary error: {e}"))
                    })
                    .map(|doc| {
                        let mut sc = RenderScene::from(&doc);
                        let [cx, cy] = sc.atom_centroid();
                        sc.translate(-cx, -cy);
                        sc
                    }),
            ),
            IoFormat::OctJson => Some(
                read_oct(trimmed)
                    .map(|doc| {
                        let mut sc = RenderScene::from(&doc);
                        let [cx, cy] = sc.atom_centroid();
                        sc.translate(-cx, -cy);
                        sc
                    })
                    .map_err(|e| format!("OCT JSON import error: {e}")),
            ),
            IoFormat::MolV2000 => Some(
                read_mol(trimmed)
                    .map(Self::mol_with_cip_to_scene)
                    .map_err(|e| format!("MOL import error: {e}")),
            ),
            IoFormat::Sdf => Some(
                read_sdf(trimmed)
                    .map_err(|e| format!("SDF import error: {e}"))
                    .and_then(|recs| {
                        recs.into_iter()
                            .next()
                            .map(|r| Self::mol_with_cip_to_scene(r.molecule))
                            .ok_or_else(|| "SDF contains no molecule records".to_string())
                    }),
            ),
            IoFormat::Smiles => self.name_or_smiles_to_scene(
                trimmed
                    .lines()
                    .next()
                    .unwrap_or(trimmed)
                    .split_whitespace()
                    .next()
                    .unwrap_or(trimmed),
            ),
            IoFormat::InChI => {
                let inchi_line = trimmed
                    .lines()
                    .map(str::trim)
                    .find(|l| l.starts_with("InChI="))
                    .unwrap_or(trimmed);
                Some(
                    parse_inchi(inchi_line)
                        .map(|mut m| {
                            clean_2d(&mut m);
                            Self::mol_with_cip_to_scene(m)
                        })
                        .map_err(|e| format!("InChI import error: {e}")),
                )
            }
            IoFormat::JcampDx => {
                let blocks = match read_jcamp(trimmed) {
                    Ok(b) if !b.is_empty() => b,
                    Ok(_) => return Some(Err("No spectrum blocks found in JCAMP-DX".to_string())),
                    Err(e) => return Some(Err(format!("JCAMP-DX import error: {e}"))),
                };
                let sp = &blocks[0];
                let picked = pick_1d_multiplets(sp, 0.04, 22.0, None);
                let signals: Vec<SpectrumSignal1D> = picked
                    .into_iter()
                    .map(|m| SpectrumSignal1D {
                        ppm: m.center_ppm,
                        multiplicity: m.estimated_multiplicity,
                        couplings_hz: m.estimated_j_hz.into_iter().collect(),
                        integral: m.normalized_integral.max(0.5),
                        atom_indices: Vec::new(),
                        assignment_label: None,
                    })
                    .collect();
                let n_pts = sp.y.len();
                let mut custom_trace = Vec::with_capacity(n_pts.min(2048));
                let max_y =
                    sp.y.iter()
                        .copied()
                        .fold(0.0_f64, |acc, v| acc.max(v.abs()))
                        .max(1e-9);
                let step = (n_pts / 2048).max(1);
                for idx in (0..n_pts).step_by(step) {
                    custom_trace.push([sp.x_at(idx), sp.y[idx] / max_y]);
                }
                let ppm_min = sp.first_x.min(sp.last_x);
                let ppm_max = sp.first_x.max(sp.last_x);
                let mut scene = RenderScene::new();
                scene.spectra_1d.push(SceneSpectrum1D {
                    origin: [-5.5, -2.0],
                    width_mol: 11.0,
                    height_mol: 4.0,
                    nucleus: sp.nucleus.clone().unwrap_or_else(|| "1H".to_string()),
                    spectrometer_mhz: sp.mhz.unwrap_or(400.0).max(60.0),
                    ppm_min: if ppm_min.is_finite() { ppm_min } else { -0.5 },
                    ppm_max: if ppm_max.is_finite() { ppm_max } else { 11.0 },
                    linewidth_hz: 1.4,
                    show_integrals: !signals.is_empty(),
                    show_peak_labels: true,
                    title: Some(if sp.title.is_empty() {
                        "Imported JCAMP-DX 1D NMR".to_string()
                    } else {
                        sp.title.clone()
                    }),
                    signals,
                    custom_trace,
                });
                Some(Ok(scene))
            }
            _ => None,
        }
    }
}

struct NmrWorkbenchApp {
    workspace_mode: WorkspaceMode,
    octet_app: OctetDrawPanel,
    #[cfg(not(target_arch = "wasm32"))]
    gpu_engine: Option<GpuEngine>,
    adapter_label: Option<String>,
    solver_kind: GuiSolverKind,
    #[cfg(not(target_arch = "wasm32"))]
    use_gpu: bool,
    use_neural_picker: bool,
    status_msg: String,
    last_solve_ms: Option<f64>,

    // Active dataset
    priors: Priors,
    reference: Option<Reference>,
    sequence: Vec<String>,
    seq_ids: Vec<i32>,
    peaks: Vec<Peak>,
    truth: Option<Vec<i32>>,
    systems: Vec<SpinSystem>,
    u_mat: Option<Matrix2D>,
    p_mat: Option<Matrix2D>,
    assign_res: Option<AssignResult>,
    pred_ref: Option<Reference>,
    pred_conf: BTreeMap<usize, f64>,
    metrics: Option<Metrics>,

    // Spectra for GPU contour & 3D strip visualization
    hsqc_spec: Option<Spectrum2D>,
    hsqc_arc: Arc<Vec<f32>>,
    hsqc_gen: u64,
    hncacb_3d: Option<Spectrum3D>,

    // 2D HSQC viewport & contour settings
    view_h_left: f32,
    view_h_right: f32,
    view_n_top: f32,
    view_n_bottom: f32,
    contour_base: f32,
    contour_factor: f32,
    contour_levels: u32,
    show_negative: bool,
    show_peak_labels: bool,

    // Solver & simulator controls
    replicas: usize,
    steps_per_sys: usize,
    assign_cost: f64,
    sim_missing: f64,
    sim_spurious: f64,
    sim_seed: u64,
    picker_snr: f32,

    // Selection & File / In-Memory Text I/O
    selected_ss: Option<usize>,
    selected_res: Option<usize>,
    #[cfg(not(target_arch = "wasm32"))]
    path_input: String,
    #[cfg(not(target_arch = "wasm32"))]
    export_prefix: String,
    show_text_io_modal: bool,
    paste_text_buf: String,
    export_text_buf: String,
    #[cfg(target_arch = "wasm32")]
    pending_dropped_text: Arc<std::sync::Mutex<Option<(String, String)>>>,
}

impl NmrWorkbenchApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        let mut style = (*cc.egui_ctx.style_of(egui::Theme::Dark)).clone();
        style.visuals = egui::Visuals::dark();
        style.visuals.panel_fill = egui::Color32::from_rgb(16, 20, 28);
        style.visuals.window_fill = egui::Color32::from_rgb(20, 25, 35);
        cc.egui_ctx.set_style_of(egui::Theme::Dark, style);

        #[cfg(not(target_arch = "wasm32"))]
        let mut gpu_engine = None;
        let mut adapter_label = None;

        if let Some(rs) = &cc.wgpu_render_state {
            let contour_res = ContourPipelineResource::new(&rs.device, rs.target_format);
            rs.renderer.write().callback_resources.insert(contour_res);

            let info = rs.adapter.get_info();
            let name = if info.name.trim().is_empty() {
                "WebGPU Adapter".to_string()
            } else {
                info.name.clone()
            };
            adapter_label = Some(format!("{} ({:?})", name, info.backend));

            #[cfg(not(target_arch = "wasm32"))]
            {
                gpu_engine = Some(GpuEngine::from_device(
                    rs.device.clone(),
                    rs.queue.clone(),
                    info.name.clone(),
                    format!("{:?}", info.backend),
                ));
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        let default_replicas = 64;
        #[cfg(target_arch = "wasm32")]
        let default_replicas = 16;

        let mut octet_app = OctetDrawPanel::new();
        octet_app.set_backend_adapter(Box::new(CoreOctetBackendAdapter));

        let mut app = Self {
            workspace_mode: WorkspaceMode::ProteinNmr,
            octet_app,
            solver_kind: GuiSolverKind::default(),
            #[cfg(not(target_arch = "wasm32"))]
            use_gpu: gpu_engine.is_some(),
            #[cfg(not(target_arch = "wasm32"))]
            gpu_engine,
            use_neural_picker: true,
            adapter_label,
            status_msg: String::new(),
            last_solve_ms: None,
            priors: Priors::provisional(),
            reference: None,
            sequence: Vec::new(),
            seq_ids: Vec::new(),
            peaks: Vec::new(),
            truth: None,
            systems: Vec::new(),
            u_mat: None,
            p_mat: None,
            assign_res: None,
            pred_ref: None,
            pred_conf: BTreeMap::new(),
            metrics: None,
            hsqc_spec: None,
            hsqc_arc: Arc::new(Vec::new()),
            hsqc_gen: 1,
            hncacb_3d: None,
            view_h_left: 10.4,
            view_h_right: 6.1,
            view_n_top: 103.0,
            view_n_bottom: 135.0,
            contour_base: 8.5,
            contour_factor: 1.42,
            contour_levels: 10,
            show_negative: true,
            show_peak_labels: true,
            replicas: default_replicas,
            steps_per_sys: 1500,
            assign_cost: 0.0,
            sim_missing: 0.06,
            sim_spurious: 0.04,
            sim_seed: 0,
            picker_snr: 5.0,
            selected_ss: None,
            selected_res: None,
            #[cfg(not(target_arch = "wasm32"))]
            path_input: "data/fixtures/synthetic_ubiquitin.str".to_string(),
            #[cfg(not(target_arch = "wasm32"))]
            export_prefix: "result_gui".to_string(),
            show_text_io_modal: false,
            paste_text_buf: String::new(),
            export_text_buf: String::new(),
            #[cfg(target_arch = "wasm32")]
            pending_dropped_text: Arc::new(std::sync::Mutex::new(None)),
        };

        app.load_demo_and_solve();
        app
    }

    fn reset_view_to_spec(&mut self) {
        if let Some(spec) = &self.hsqc_spec {
            self.view_h_left = spec.axis_x.max_ppm() as f32;
            self.view_h_right = spec.axis_x.min_ppm() as f32;
            self.view_n_top = spec.axis_y.min_ppm() as f32;
            self.view_n_bottom = spec.axis_y.max_ppm() as f32;
            let noise = estimate_noise_mad(&spec.data);
            self.contour_base = (noise * 4.8).max(1.0);
        }
    }

    fn compute_matrices(&self, sys: &[SpinSystem], scfg: &ScoreConfig) -> (Matrix2D, Matrix2D) {
        #[cfg(not(target_arch = "wasm32"))]
        if self.use_gpu {
            if let Some(gpu) = &self.gpu_engine {
                if let Ok(pair) = gpu.score_matrices_gpu(sys, &self.sequence, &self.priors, scfg) {
                    return pair;
                }
            }
        }
        (
            unary_matrix(sys, &self.sequence, &self.priors, scfg),
            pair_matrix(sys, scfg),
        )
    }

    fn rebuild_systems_and_matrices(&mut self) {
        match build_spin_systems(&self.peaks, &Tolerances::default(), 3, false) {
            Ok(sys) => {
                let scfg = ScoreConfig {
                    assign_cost: self.assign_cost,
                    ..ScoreConfig::default()
                };
                let (u, p) = self.compute_matrices(&sys, &scfg);
                self.u_mat = Some(u);
                self.p_mat = Some(p);
                self.systems = sys;
                self.selected_ss = (!self.systems.is_empty()).then_some(0);
            }
            Err(e) => {
                self.status_msg = format!("Spin system build error: {e}");
            }
        }
    }

    fn regenerate_synthetic_spectra(&mut self) {
        let rcfg = SpectrumRenderConfig {
            nx_h: 512,
            ny_n: 256,
            nz_c: 96,
            seed: self.sim_seed.wrapping_add(7),
            ..SpectrumRenderConfig::default()
        };
        let hsqc = render_hsqc_2d(&self.peaks, &rcfg);
        self.hsqc_arc = Arc::new(hsqc.data.clone());
        self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
        self.hsqc_spec = Some(hsqc);
        self.hncacb_3d = Some(render_triple_3d("HNCACB", &self.peaks, &rcfg));
        self.reset_view_to_spec();
    }

    fn load_demo_and_solve(&mut self) {
        let ref_data = build_synthetic_ubiquitin();
        self.load_from_reference(ref_data);
        self.run_assignment();
    }

    fn load_from_reference(&mut self, ref_data: Reference) {
        let (peaks, truth) = simulate(
            &ref_data,
            &SimConfig {
                p_missing_peak: self.sim_missing,
                spurious_rate: self.sim_spurious,
                seed: self.sim_seed,
                ..SimConfig::default()
            },
        );
        self.sequence = ref_data.sequence.clone();
        self.seq_ids = ref_data.seq_ids.clone();
        self.octet_app
            .set_loaded_protein_sequence(self.sequence.clone());
        self.reference = Some(ref_data);
        self.peaks = peaks;
        self.truth = Some(truth);
        self.assign_res = None;
        self.pred_ref = None;
        self.pred_conf.clear();
        self.metrics = None;
        self.regenerate_synthetic_spectra();
        self.rebuild_systems_and_matrices();
        self.status_msg = format!(
            "Simulated {} peaks & {} spin systems ({} aa) + 2D HSQC / 3D HNCACB spectra",
            self.peaks.len(),
            self.systems.len(),
            self.sequence.len()
        );
    }

    fn run_assignment(&mut self) {
        if self.sequence.is_empty()
            || (self.solver_kind == GuiSolverKind::EvoGraph && self.peaks.is_empty())
            || (self.solver_kind == GuiSolverKind::Standard && self.systems.is_empty())
        {
            self.status_msg = "Need peaks/spin systems and sequence to run assignment".to_string();
            return;
        }
        let scfg = ScoreConfig {
            assign_cost: self.assign_cost,
            ..ScoreConfig::default()
        };

        let timer = PerfTimer::start();
        let (res, engine_tag, eff_priors, kept_hsqc_opt) = match self.solver_kind {
            GuiSolverKind::EvoGraph => {
                let mut evo_cfg = EvoGraphConfig {
                    score_cfg: scfg.clone(),
                    auto_reference: scfg.auto_reference,
                    structure_torsions: None,
                    seed: self.sim_seed,
                    ..EvoGraphConfig::default()
                };
                evo_cfg.population_size = self.replicas.max(4);
                evo_cfg.num_islands = evo_cfg.num_islands.min(evo_cfg.population_size).max(1);
                evo_cfg.steps_per_system_per_gen = (self.steps_per_sys * 3 / 10).max(100);
                #[cfg(target_arch = "wasm32")]
                {
                    evo_cfg.generations = 4;
                    evo_cfg.steps_per_system_per_gen = evo_cfg.steps_per_system_per_gen.min(250);
                }
                let (evo_systems, r, rep) = assign_evo_graph_from_peaks(
                    &self.peaks,
                    &self.sequence,
                    &self.priors,
                    &evo_cfg,
                );
                let shifted_pri = rep.shifted_priors(&self.priors);
                self.u_mat = Some(unary_matrix(
                    &evo_systems,
                    &self.sequence,
                    &shifted_pri,
                    &scfg,
                ));
                self.p_mat = Some(pair_matrix(&evo_systems, &scfg));
                self.systems = evo_systems;
                self.selected_ss = match self.selected_ss {
                    Some(s) if s < self.systems.len() => Some(s),
                    _ => (!self.systems.is_empty()).then_some(0),
                };
                let tag = if cfg!(target_arch = "wasm32") {
                    "WASM Evo-Graph"
                } else {
                    "CPU Evo-Graph"
                };
                (r, tag, shifted_pri, Some(rep.kept_hsqc_indices))
            }
            GuiSolverKind::Standard => {
                // Rebuild unfiltered spin systems when switching back from Evo-Graph filtered systems
                if let Ok(sys) = build_spin_systems(&self.peaks, &Tolerances::default(), 3, false) {
                    if sys.len() != self.systems.len() {
                        self.systems = sys;
                        self.p_mat = Some(pair_matrix(&self.systems, &scfg));
                        self.selected_ss = (!self.systems.is_empty()).then_some(0);
                    }
                }
                let acfg = AnnealConfig {
                    runs: self.replicas,
                    steps_per_system: self.steps_per_sys,
                    seed: self.sim_seed,
                    ..AnnealConfig::default()
                };
                #[allow(unused_mut)]
                let mut res_opt: Option<(AssignResult, &'static str)> = None;

                #[cfg(not(target_arch = "wasm32"))]
                if self.use_gpu {
                    if let Some(gpu) = &self.gpu_engine {
                        match gpu.assign_gpu(
                            &self.systems,
                            &self.sequence,
                            &self.priors,
                            &scfg,
                            &acfg,
                        ) {
                            Ok(r) => res_opt = Some((r, "WebGPU")),
                            Err(e) => {
                                self.status_msg =
                                    format!("GPU solver error ({e}), falling back to CPU");
                            }
                        }
                    }
                }

                let default_tag = if cfg!(target_arch = "wasm32") {
                    "WASM"
                } else {
                    "CPU-MT"
                };
                let (r, tag) = res_opt.unwrap_or_else(|| {
                    let u = unary_matrix(&self.systems, &self.sequence, &self.priors, &scfg);
                    let p = self
                        .p_mat
                        .clone()
                        .unwrap_or_else(|| pair_matrix(&self.systems, &scfg));
                    self.u_mat = Some(u.clone());
                    (assign_from_matrices(&u, &p, &acfg), default_tag)
                });
                (r, tag, self.priors.clone(), None)
            }
        };

        let dt_ms = timer.elapsed_ms();
        self.last_solve_ms = Some(dt_ms);

        let (pred, conf) = to_reference(
            &res,
            &self.systems,
            &self.sequence,
            &self.seq_ids,
            0.0,
            Some(&eff_priors),
        );
        if let Some(truth) = &self.truth {
            if let Some(kept) = &kept_hsqc_opt {
                let eval_truth: Vec<i32> = kept
                    .iter()
                    .map(|&idx| truth.get(idx).copied().unwrap_or(-1))
                    .collect();
                if eval_truth.len() == res.position.len() {
                    self.metrics = Some(Metrics {
                        assignable: truth.iter().filter(|&&t| t >= 0).count(),
                        ..score(&res, &eval_truth, 0.8)
                    });
                } else {
                    self.metrics = None;
                }
            } else if truth.len() == res.position.len() {
                self.metrics = Some(score(&res, truth, 0.8));
            } else {
                self.metrics = None;
            }
        }
        let n_hc = conf.values().filter(|&&c| c >= 0.8).count();
        self.status_msg = format!(
            "[{engine_tag}] Assigned {}/{} residues ({} high-conf >=0.8) in {:.0} ms",
            conf.len(),
            self.sequence.len(),
            n_hc,
            dt_ms
        );
        if let Some(ss) = self.selected_ss {
            if let Some(&k) = res.position.get(ss) {
                if k >= 0 {
                    self.selected_res = Some(k as usize);
                }
            }
        }
        self.pred_ref = Some(pred);
        self.pred_conf = conf;
        self.assign_res = Some(res);
    }

    fn load_text_in_memory(&mut self, text: &str, source_label: &str) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            self.status_msg =
                "Paste NMR-STAR (.str), NEF (.nef), Peak TSV (.tsv), CDXML, MOL, or SMILES first"
                    .to_string();
            return;
        }
        let lower_label = source_label.to_ascii_lowercase();

        // Check chemical structure formats first (`CDXML`, `OCT`, `MOL`/`SDF`, `InChI`, `.smi`)
        if trimmed.contains("<CDXML")
            || lower_label.ends_with(".cdxml")
            || lower_label.ends_with(".oct")
            || lower_label.ends_with(".mol")
            || lower_label.ends_with(".sdf")
            || lower_label.ends_with(".smi")
            || trimmed.contains("V2000")
            || trimmed.contains("V3000")
            || trimmed.starts_with("InChI=")
        {
            let fmt = if trimmed.contains("<CDXML") || lower_label.ends_with(".cdxml") {
                IoFormat::Cdxml
            } else if lower_label.ends_with(".oct") || trimmed.starts_with('{') {
                IoFormat::OctJson
            } else if lower_label.ends_with(".sdf") || trimmed.contains("$$$$") {
                IoFormat::Sdf
            } else if trimmed.starts_with("InChI=") {
                IoFormat::InChI
            } else if trimmed.contains("V2000")
                || trimmed.contains("V3000")
                || lower_label.ends_with(".mol")
            {
                IoFormat::MolV2000
            } else {
                IoFormat::Smiles
            };
            if let Some(res) = CoreOctetBackendAdapter.import_custom(fmt, trimmed) {
                match res {
                    Ok(sc) => {
                        self.octet_app.set_scene(sc);
                        self.workspace_mode = WorkspaceMode::OctetChem;
                        self.octet_app.status_msg =
                            format!("Loaded {} from {source_label}", fmt.label());
                        self.status_msg = format!("Loaded {} into Octet Studio", fmt.label());
                    }
                    Err(e) => self.status_msg = format!("Chemical file parse error: {e}"),
                }
                return;
            }
        }

        // Check NEF v1.1 (`data_nef_...` or `save_nef_molecular_system`)
        if lower_label.ends_with(".nef")
            || trimmed.contains("save_nef_molecular_system")
            || trimmed.starts_with("data_nef_")
        {
            match read_nef_str(trimmed) {
                Ok(doc) => match doc.to_reference() {
                    Ok(r) => {
                        let nef_pks = doc.to_peaks();
                        self.sequence = r.sequence.clone();
                        self.seq_ids = r.seq_ids.clone();
                        self.octet_app
                            .set_loaded_protein_sequence(self.sequence.clone());
                        self.reference = Some(r.clone());
                        if !nef_pks.is_empty() {
                            self.peaks = nef_pks;
                            self.truth = None;
                            self.metrics = None;
                            self.regenerate_synthetic_spectra();
                            self.rebuild_systems_and_matrices();
                            self.run_assignment();
                        } else {
                            self.load_from_reference(r);
                            self.run_assignment();
                        }
                        self.workspace_mode = WorkspaceMode::ProteinNmr;
                    }
                    Err(e) => self.status_msg = format!("NEF molecular system error: {e}"),
                },
                Err(e) => self.status_msg = format!("NEF parse error ({source_label}): {e}"),
            }
            return;
        }

        let looks_like_star = trimmed.contains("loop_")
            || trimmed.contains("_Atom_chem_shift")
            || trimmed.contains("_Chem_shift_value")
            || trimmed.starts_with("data_");

        if looks_like_star {
            match read_star_str(trimmed, None) {
                Ok(r) => {
                    self.load_from_reference(r);
                    self.run_assignment();
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                }
                Err(e) => self.status_msg = format!("NMR-STAR parse error ({source_label}): {e}"),
            }
        } else {
            match read_peaks_str(trimmed) {
                Ok(pks) => {
                    self.peaks = pks;
                    self.truth = None;
                    self.metrics = None;
                    self.regenerate_synthetic_spectra();
                    self.rebuild_systems_and_matrices();
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                    self.status_msg = format!(
                        "Loaded {} peaks in memory from {source_label}",
                        self.peaks.len()
                    );
                }
                Err(e) => self.status_msg = format!("Peak TSV parse error ({source_label}): {e}"),
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load_path_auto(&mut self) {
        let path_str = self.path_input.trim().to_string();
        let p = Path::new(&path_str);
        if !p.exists() {
            self.status_msg = format!("Path not found: {}", p.display());
            return;
        }
        if p.is_dir() || p.file_name().and_then(|s| s.to_str()) == Some("2rr") {
            let dir = if p.is_dir() {
                p
            } else {
                p.parent().unwrap_or(p)
            };
            match read_bruker_pdata_2d(dir) {
                Ok(spec) => {
                    self.hsqc_arc = Arc::new(spec.data.clone());
                    self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
                    self.hsqc_spec = Some(spec);
                    self.reset_view_to_spec();
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                    self.status_msg = format!("Loaded Bruker 2D pdata from {}", dir.display());
                }
                Err(e) => self.status_msg = format!("Bruker load error: {e}"),
            }
            return;
        }

        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "str" | "star" => match read_star(p, None) {
                Ok(r) => {
                    self.load_from_reference(r);
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                }
                Err(e) => self.status_msg = format!("NMR-STAR error: {e}"),
            },
            "nef" => match std::fs::read_to_string(p) {
                Ok(txt) => self.load_text_in_memory(&txt, &p.display().to_string()),
                Err(e) => self.status_msg = format!("NEF read error: {e}"),
            },
            "tsv" => match read_peaks(p) {
                Ok(pks) => {
                    self.peaks = pks;
                    self.truth = None;
                    self.metrics = None;
                    self.regenerate_synthetic_spectra();
                    self.rebuild_systems_and_matrices();
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                    self.status_msg =
                        format!("Loaded {} peaks from {}", self.peaks.len(), p.display());
                }
                Err(e) => self.status_msg = format!("Peak TSV error: {e}"),
            },
            "ft2" | "dat" | "pipe" => match read_nmrpipe_2d(p) {
                Ok(spec) => {
                    self.hsqc_arc = Arc::new(spec.data.clone());
                    self.hsqc_gen = self.hsqc_gen.wrapping_add(1);
                    self.hsqc_spec = Some(spec);
                    self.reset_view_to_spec();
                    self.workspace_mode = WorkspaceMode::ProteinNmr;
                    self.status_msg = format!("Loaded NMRPipe 2D spectrum {}", p.display());
                }
                Err(e) => self.status_msg = format!("NMRPipe load error: {e}"),
            },
            "cdx" => match std::fs::read(p) {
                Ok(bytes) => match read_cdx(&bytes) {
                    Ok(doc) => {
                        let mut sc = RenderScene::from(&doc);
                        let [cx, cy] = sc.atom_centroid();
                        sc.translate(-cx, -cy);
                        self.octet_app.set_scene(sc);
                        self.workspace_mode = WorkspaceMode::OctetChem;
                        self.octet_app.status_msg =
                            format!("Loaded binary ChemDraw CDX {}", p.display());
                        self.status_msg = format!("Loaded binary CDX {}", p.display());
                    }
                    Err(e) => self.status_msg = format!("CDX parse error: {e}"),
                },
                Err(e) => self.status_msg = format!("CDX read error: {e}"),
            },
            "cdxml" | "oct" | "mol" | "sdf" | "smi" | "jdx" | "dx" => {
                match std::fs::read_to_string(p) {
                    Ok(txt) => {
                        let fmt = match ext.as_str() {
                            "cdxml" => IoFormat::Cdxml,
                            "oct" => IoFormat::OctJson,
                            "mol" => IoFormat::MolV2000,
                            "sdf" => IoFormat::Sdf,
                            "jdx" | "dx" => IoFormat::JcampDx,
                            _ => IoFormat::Smiles,
                        };
                        if let Some(res) = CoreOctetBackendAdapter.import_custom(fmt, &txt) {
                            match res {
                                Ok(sc) => {
                                    self.octet_app.set_scene(sc);
                                    self.workspace_mode = WorkspaceMode::OctetChem;
                                    self.octet_app.status_msg =
                                        format!("Loaded {} from {}", fmt.label(), p.display());
                                    self.status_msg = format!("Loaded {}", p.display());
                                }
                                Err(e) => self.status_msg = format!("Import error: {e}"),
                            }
                        }
                    }
                    Err(e) => self.status_msg = format!("File read error: {e}"),
                }
            }
            _ => {
                self.status_msg = format!(
                    "Unrecognized file extension .{ext} (expected .str, .nef, .tsv, .ft2, pdata, .cdxml, .cdx, .oct, .mol, .sdf, .smi, .jdx)"
                );
            }
        }
    }

    fn pick_2d_roots(&mut self) {
        let Some(spec) = &self.hsqc_spec else {
            self.status_msg = "No 2D HSQC spectrum loaded".to_string();
            return;
        };
        let (picked, method_label) = if self.use_neural_picker {
            (
                pick_hsqc_2d_neural(
                    spec,
                    &NeuralPickerConfig {
                        prob_threshold: (0.25 + 0.035 * (self.picker_snr - 3.0)).clamp(0.20, 0.85),
                        ..NeuralPickerConfig::default()
                    },
                ),
                "Neural 2D CNN picker",
            )
        } else {
            (
                pick_hsqc_2d(
                    spec,
                    &PickerConfig {
                        snr_threshold: self.picker_snr,
                        ..PickerConfig::default()
                    },
                ),
                "Classical picker",
            )
        };
        let n_picked = picked.len();
        // Keep existing triple-resonance peaks while replacing HSQC roots
        let mut combined = picked;
        combined.extend(self.peaks.iter().filter(|p| p.exp != "HSQC").cloned());
        self.peaks = combined;
        self.truth = None;
        self.metrics = None;
        self.rebuild_systems_and_matrices();
        self.status_msg = format!(
            "{method_label} found {n_picked} HSQC roots -> rebuilt {} spin systems",
            self.systems.len()
        );
    }

    fn format_shifts_tsv(&self, pred: &Reference) -> String {
        let mut rows = vec!["seq_id\tresidue\tatom\tshift\tconfidence".to_string()];
        for (&(k, ref atom), &v) in &pred.shifts {
            let c = self.pred_conf.get(&k).copied().unwrap_or(0.0);
            rows.push(format!(
                "{}\t{}\t{}\t{:.3}\t{:.2}",
                self.seq_ids[k], self.sequence[k], atom, v, c
            ));
        }
        rows.join("\n") + "\n"
    }

    fn export_results(&mut self, ctx: &egui::Context) {
        let Some(pred) = &self.pred_ref else {
            self.status_msg = "Run assignment first before exporting".to_string();
            return;
        };
        let tsv_text = self.format_shifts_tsv(pred);
        let star_text = format_star_in_memory(pred, Some(&self.pred_conf));
        self.export_text_buf =
            format!("# --- NMR-STAR v3 ---\n{star_text}\n# --- Shifts TSV ---\n{tsv_text}");

        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = ctx;
            let prefix = Path::new(self.export_prefix.trim());
            let str_p = prefix.with_extension("str");
            let tsv_p = prefix.with_extension("tsv");
            if let Err(e) = write_star(&str_p, pred, Some(&self.pred_conf)) {
                self.status_msg = format!("Export failed: {e}");
                return;
            }
            let _ = std::fs::write(&tsv_p, &tsv_text);
            self.status_msg = format!("Exported {} and {}", str_p.display(), tsv_p.display());
        }

        #[cfg(target_arch = "wasm32")]
        {
            ctx.copy_text(star_text);
            self.show_text_io_modal = true;
            self.status_msg =
                "Exported NMR-STAR v3 + TSV in memory (NMR-STAR copied to clipboard)".to_string();
        }
    }

    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        #[cfg(target_arch = "wasm32")]
        {
            let ready = self
                .pending_dropped_text
                .lock()
                .ok()
                .and_then(|mut g| g.take());
            if let Some((text, label)) = ready {
                self.load_text_in_memory(&text, &label);
            }
        }

        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if let Some(file) = dropped.into_iter().next() {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let _ = ctx;
                let p = file.path();
                if p.exists() {
                    self.path_input = p.display().to_string();
                    self.load_path_auto();
                } else if let Ok(bytes) = file.bytes() {
                    let text = String::from_utf8_lossy(&bytes);
                    let label = p.display().to_string();
                    self.load_text_in_memory(&text, &label);
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                let slot = Arc::clone(&self.pending_dropped_text);
                let egui_ctx = ctx.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let label = file.path().display().to_string();
                    if let Ok(bytes) = file.bytes_async().await {
                        let text = String::from_utf8_lossy(&bytes).into_owned();
                        if let Ok(mut g) = slot.lock() {
                            *g = Some((text, label));
                        }
                        egui_ctx.request_repaint();
                    }
                });
            }
        }
    }

    fn draw_text_io_window(&mut self, ctx: &egui::Context) {
        if !self.show_text_io_modal {
            return;
        }
        let mut open = self.show_text_io_modal;
        egui::Window::new("Paste / In-Memory NMR-STAR & Peak TSV")
            .open(&mut open)
            .default_width(560.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new(
                        "Paste an NMR-STAR (.str) entry or TSV peak list below (or drag & drop a .str/.tsv file onto the canvas):",
                    )
                    .small(),
                );
                ui.horizontal(|ui| {
                    if ui.small_button("Fill Synthetic Ubiquitin .str").clicked() {
                        self.paste_text_buf = SYNTHETIC_UBIQUITIN_STR.to_string();
                    }
                    if ui.small_button("Load Pasted Text & Solve").clicked() {
                        let buf = self.paste_text_buf.clone();
                        self.load_text_in_memory(&buf, "pasted text");
                    }
                    if ui.small_button("Clear").clicked() {
                        self.paste_text_buf.clear();
                    }
                });
                egui::ScrollArea::vertical()
                    .id_salt("paste_text_scroll")
                    .max_height(180.0)
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut self.paste_text_buf)
                                .font(egui::TextStyle::Monospace)
                                .desired_rows(8)
                                .desired_width(f32::INFINITY),
                        );
                    });

                if !self.export_text_buf.is_empty() {
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Exported NMR-STAR / TSV:").small().strong());
                        if ui.small_button("Copy All to Clipboard").clicked() {
                            ui.ctx().copy_text(self.export_text_buf.clone());
                        }
                    });
                    egui::ScrollArea::vertical()
                        .id_salt("export_text_scroll")
                        .max_height(180.0)
                        .show(ui, |ui| {
                            let mut preview = self.export_text_buf.as_str();
                            ui.add(
                                egui::TextEdit::multiline(&mut preview)
                                    .font(egui::TextStyle::Monospace)
                                    .desired_rows(8)
                                    .desired_width(f32::INFINITY),
                            );
                        });
                }
            });
        self.show_text_io_modal = open;
    }

    fn draw_workspace_mode_header(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.heading(
                egui::RichText::new("nmr-assign")
                    .strong()
                    .color(egui::Color32::from_rgb(110, 220, 255)),
            );
            if let Some(label) = &self.adapter_label {
                ui.label(
                    egui::RichText::new(format!("⚡ {label}"))
                        .small()
                        .color(egui::Color32::from_rgb(130, 230, 175)),
                );
            }
            ui.separator();

            ui.selectable_value(
                &mut self.workspace_mode,
                WorkspaceMode::ProteinNmr,
                egui::RichText::new("🧬 Protein NMR & 3D Strips").strong(),
            );
            ui.selectable_value(
                &mut self.workspace_mode,
                WorkspaceMode::OctetChem,
                egui::RichText::new("⚗️ Octet Chemical Draw & 1D NMR").strong(),
            );

            if self.workspace_mode == WorkspaceMode::OctetChem {
                ui.separator();
                #[cfg(not(target_arch = "wasm32"))]
                {
                    ui.label("Path:");
                    ui.add(egui::TextEdit::singleline(&mut self.path_input).desired_width(185.0));
                    if ui.button("📂 Load (.cdxml/.cdx/.mol/.sdf/.jdx)").clicked() {
                        self.load_path_auto();
                    }
                }
                ui.label(
                    egui::RichText::new(&self.octet_app.status_msg)
                        .small()
                        .color(egui::Color32::from_rgb(190, 210, 235)),
                );
            }
        });
    }

    fn draw_top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui.button("🧬 Ubiquitin Demo").clicked() {
                self.load_demo_and_solve();
            }

            #[cfg(not(target_arch = "wasm32"))]
            {
                ui.label("Path:");
                ui.add(egui::TextEdit::singleline(&mut self.path_input).desired_width(200.0));
                if ui
                    .button("📂 Load (.str / .nef / .tsv / pdata / .ft2)")
                    .clicked()
                {
                    self.load_path_auto();
                }
            }

            if ui.button("📋 Paste .str / .nef / .tsv").clicked() {
                self.show_text_io_modal = !self.show_text_io_modal;
            }

            ui.separator();
            egui::ComboBox::from_id_salt("gui_solver_kind")
                .selected_text(format!("Solver: {}", self.solver_kind.label()))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.solver_kind,
                        GuiSolverKind::EvoGraph,
                        "Evo-Graph (CPU f64, default)",
                    );
                    ui.selectable_value(
                        &mut self.solver_kind,
                        GuiSolverKind::Standard,
                        "Standard SA (GPU / CPU)",
                    );
                });
            #[cfg(not(target_arch = "wasm32"))]
            ui.add_enabled(
                self.solver_kind == GuiSolverKind::Standard,
                egui::Checkbox::new(&mut self.use_gpu, "WebGPU Compute"),
            )
            .on_disabled_hover_text(
                "Evo-Graph runs in f64 on CPU; switch Solver to Standard SA to use WebGPU Compute",
            );
            ui.add(egui::Slider::new(&mut self.replicas, 8..=256).text("Replicas"));
            ui.add(egui::Slider::new(&mut self.steps_per_sys, 400..=4000).text("Steps/SS"));
            if ui
                .add(
                    egui::Slider::new(&mut self.assign_cost, 0.0..=4.0)
                        .text("Assign cost")
                        .fixed_decimals(2),
                )
                .changed()
                && !self.systems.is_empty()
                && !self.sequence.is_empty()
            {
                let scfg = ScoreConfig {
                    assign_cost: self.assign_cost,
                    ..ScoreConfig::default()
                };
                self.u_mat = Some(unary_matrix(
                    &self.systems,
                    &self.sequence,
                    &self.priors,
                    &scfg,
                ));
            }

            let solve_btn = egui::Button::new(
                egui::RichText::new("▶ Run Assignment")
                    .strong()
                    .color(egui::Color32::from_rgb(15, 25, 20)),
            )
            .fill(egui::Color32::from_rgb(82, 224, 152));
            if ui.add(solve_btn).clicked() {
                self.run_assignment();
            }

            if ui.button("💾 Export").clicked() {
                let ctx = ui.ctx().clone();
                self.export_results(&ctx);
            }
        });

        ui.add_space(3.0);
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("Simulator / Picker:").small().weak());
            let mut changed_sim = false;
            changed_sim |= ui
                .add(
                    egui::Slider::new(&mut self.sim_missing, 0.0..=0.25)
                        .text("Missing")
                        .fixed_decimals(2),
                )
                .changed();
            changed_sim |= ui
                .add(
                    egui::Slider::new(&mut self.sim_spurious, 0.0..=0.20)
                        .text("Spurious")
                        .fixed_decimals(2),
                )
                .changed();
            let _ = changed_sim;
            if ui.small_button("🎲 Re-simulate & Solve").clicked() {
                self.sim_seed = self.sim_seed.wrapping_add(1);
                if let Some(r) = self.reference.clone() {
                    self.load_from_reference(r);
                    self.run_assignment();
                }
            }
            ui.separator();
            ui.checkbox(&mut self.use_neural_picker, "Neural CNN")
                .on_hover_text(
                    "Toggle 2D separable residual CNN peak picker vs classical MAD picker",
                );
            ui.add(egui::Slider::new(&mut self.picker_snr, 3.0..=12.0).text("Pick SNR"));
            if ui.small_button("🔍 Pick 2D HSQC Peaks").clicked() {
                self.pick_2d_roots();
            }
            ui.separator();
            ui.label(
                egui::RichText::new(&self.status_msg)
                    .small()
                    .color(egui::Color32::from_rgb(190, 210, 235)),
            );
        });
    }

    fn draw_sequence_ribbon(&mut self, ui: &mut egui::Ui) {
        if self.sequence.is_empty() {
            return;
        }
        let mut res_to_ss: BTreeMap<usize, (usize, f64)> = BTreeMap::new();
        if let Some(res) = &self.assign_res {
            for (s, &k) in res.position.iter().enumerate() {
                if k >= 0 {
                    res_to_ss.insert(k as usize, (s, res.confidence[s]));
                }
            }
        }

        egui::ScrollArea::horizontal()
            .id_salt("seq_ribbon_scroll")
            .max_height(52.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 3.0;
                    for (k, three) in self.sequence.iter().enumerate() {
                        let one = three_to_one(three);
                        let seq_id = self.seq_ids.get(k).copied().unwrap_or((k + 1) as i32);
                        let is_pro_or_nterm = k == 0 || three == "PRO";
                        let assigned = res_to_ss.get(&k).copied();
                        let is_selected = self.selected_res == Some(k);

                        let fill = if is_pro_or_nterm {
                            egui::Color32::from_rgb(38, 43, 54)
                        } else if let Some((_, conf)) = assigned {
                            if conf >= 0.80 {
                                egui::Color32::from_rgb(28, 125, 82)
                            } else if conf >= 0.50 {
                                egui::Color32::from_rgb(148, 112, 28)
                            } else {
                                egui::Color32::from_rgb(145, 68, 42)
                            }
                        } else {
                            egui::Color32::from_rgb(48, 56, 74)
                        };

                        let stroke = if is_selected {
                            egui::Stroke::new(2.0, egui::Color32::from_rgb(120, 235, 255))
                        } else {
                            egui::Stroke::new(1.0, egui::Color32::from_rgb(65, 76, 98))
                        };

                        let label = format!("{one}{seq_id}");
                        let btn = egui::Button::new(
                            egui::RichText::new(label)
                                .monospace()
                                .size(11.5)
                                .color(egui::Color32::WHITE),
                        )
                        .fill(fill)
                        .stroke(stroke)
                        .min_size(egui::vec2(34.0, 28.0));

                        let resp = ui.add(btn);
                        if resp.clicked() {
                            self.selected_res = Some(k);
                            if let Some((ss_idx, _)) = assigned {
                                self.selected_ss = Some(ss_idx);
                            }
                        }
                        resp.on_hover_ui(|ui| {
                            ui.label(egui::RichText::new(format!("{three} {seq_id}")).strong());
                            if let Some((ss_idx, conf)) = assigned {
                                ui.label(format!(
                                    "Spin system #{ss_idx} (conf {:.0}%)",
                                    conf * 100.0
                                ));
                            } else if is_pro_or_nterm {
                                ui.label("No amide H root (N-term / Proline)");
                            } else {
                                ui.label("Unassigned");
                            }
                            if let Some(pred) = &self.pred_ref {
                                for atom in ["H", "N", "CA", "CB", "C"] {
                                    if let Some(&v) = pred.shifts.get(&(k, atom.to_string())) {
                                        ui.monospace(format!("  {atom:2}: {v:7.3} ppm"));
                                    }
                                }
                            }
                        });
                    }
                });
            });
    }

    fn draw_hsqc_canvas(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("2D ¹H–¹⁵N HSQC (WebGPU Shader)").strong());
            ui.add(
                egui::Slider::new(&mut self.contour_base, 0.5..=60.0)
                    .logarithmic(true)
                    .text("Base"),
            );
            ui.add(egui::Slider::new(&mut self.contour_factor, 1.15..=2.0).text("Step"));
            ui.add(egui::Slider::new(&mut self.contour_levels, 4..=16).text("Levels"));
            ui.checkbox(&mut self.show_negative, "Neg");
            ui.checkbox(&mut self.show_peak_labels, "Labels");
            if ui.small_button("Reset Zoom").clicked() {
                self.reset_view_to_spec();
            }
        });

        let avail = ui.available_size();
        let canvas_size = egui::vec2(avail.x.max(200.0), (avail.y - 4.0).max(200.0));
        let (rect, response) = ui.allocate_exact_size(canvas_size, egui::Sense::click_and_drag());

        // Pan with drag
        if response.dragged() {
            let delta = response.drag_delta();
            let span_h = self.view_h_right - self.view_h_left;
            let span_n = self.view_n_bottom - self.view_n_top;
            let dh = -delta.x / rect.width().max(1.0) * span_h;
            let dn = -delta.y / rect.height().max(1.0) * span_n;
            self.view_h_left += dh;
            self.view_h_right += dh;
            self.view_n_top += dn;
            self.view_n_bottom += dn;
        }
        if response.double_clicked() {
            self.reset_view_to_spec();
        }

        // Scroll-wheel zoom around pointer
        if let Some(hover_pos) = response.hover_pos() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.1 {
                let zoom = (1.0 - scroll * 0.0025).clamp(0.75, 1.35);
                let u = ((hover_pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                let v = ((hover_pos.y - rect.top()) / rect.height()).clamp(0.0, 1.0);
                let anchor_h = self.view_h_left + u * (self.view_h_right - self.view_h_left);
                let anchor_n = self.view_n_top + v * (self.view_n_bottom - self.view_n_top);
                self.view_h_left = anchor_h + (self.view_h_left - anchor_h) * zoom;
                self.view_h_right = anchor_h + (self.view_h_right - anchor_h) * zoom;
                self.view_n_top = anchor_n + (self.view_n_top - anchor_n) * zoom;
                self.view_n_bottom = anchor_n + (self.view_n_bottom - anchor_n) * zoom;
            }
        }

        let painter = ui.painter_at(rect);
        if let Some(spec) = &self.hsqc_spec {
            let uniforms = ContourUniforms {
                view_ppm: [
                    self.view_h_left,
                    self.view_h_right,
                    self.view_n_top,
                    self.view_n_bottom,
                ],
                axis_ppm: [
                    spec.axis_x.first_ppm as f32,
                    spec.axis_x.step_ppm as f32,
                    spec.axis_y.first_ppm as f32,
                    spec.axis_y.step_ppm as f32,
                ],
                dims: [
                    spec.nx() as f32,
                    spec.ny() as f32,
                    rect.width(),
                    rect.height(),
                ],
                levels: [
                    self.contour_base,
                    self.contour_factor,
                    self.contour_levels as f32,
                    if self.show_negative { 1.0 } else { 0.0 },
                ],
            };
            let cb = egui_wgpu::Callback::new_paint_callback(
                rect,
                ContourCallback {
                    uniforms,
                    spec_gen: self.hsqc_gen,
                    spec_data: Arc::clone(&self.hsqc_arc),
                },
            );
            painter.add(cb);
        } else {
            painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(14, 18, 25));
        }

        let ppm_to_screen = |h: f64, n: f64| -> egui::Pos2 {
            let u = ((h as f32) - self.view_h_left) / (self.view_h_right - self.view_h_left);
            let v = ((n as f32) - self.view_n_top) / (self.view_n_bottom - self.view_n_top);
            egui::pos2(
                rect.left() + u * rect.width(),
                rect.top() + v * rect.height(),
            )
        };

        // Click to select nearest spin system
        if response.clicked() {
            if let Some(click_pos) = response.interact_pointer_pos() {
                let mut best: Option<(usize, f32)> = None;
                for (s, ss) in self.systems.iter().enumerate() {
                    let pt = ppm_to_screen(ss.h, ss.n);
                    let dist = pt.distance(click_pos);
                    if dist <= 18.0 && best.is_none_or(|(_, bd)| dist < bd) {
                        best = Some((s, dist));
                    }
                }
                if let Some((s, _)) = best {
                    self.selected_ss = Some(s);
                    if let Some(res) = &self.assign_res {
                        if res.position[s] >= 0 {
                            self.selected_res = Some(res.position[s] as usize);
                        }
                    }
                }
            }
        }

        // Draw spin system peak markers and residue labels
        for (s, ss) in self.systems.iter().enumerate() {
            let pt = ppm_to_screen(ss.h, ss.n);
            if !rect.contains(pt) {
                continue;
            }
            let is_sel = self.selected_ss == Some(s);
            let (assigned_k, conf) = self
                .assign_res
                .as_ref()
                .map(|r| (r.position[s], r.confidence[s]))
                .unwrap_or((-1, 0.0));

            let marker_col = if assigned_k >= 0 {
                if conf >= 0.80 {
                    egui::Color32::from_rgb(95, 245, 165)
                } else {
                    egui::Color32::from_rgb(255, 205, 95)
                }
            } else {
                egui::Color32::from_rgb(155, 175, 205)
            };

            let sz = if is_sel { 6.5 } else { 3.8 };
            painter.line_segment(
                [egui::pos2(pt.x - sz, pt.y), egui::pos2(pt.x + sz, pt.y)],
                egui::Stroke::new(if is_sel { 2.0 } else { 1.2 }, marker_col),
            );
            painter.line_segment(
                [egui::pos2(pt.x, pt.y - sz), egui::pos2(pt.x, pt.y + sz)],
                egui::Stroke::new(if is_sel { 2.0 } else { 1.2 }, marker_col),
            );
            if is_sel {
                painter.circle_stroke(
                    pt,
                    10.5,
                    egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 240, 120)),
                );
            }

            if self.show_peak_labels {
                let text = if assigned_k >= 0 && (assigned_k as usize) < self.sequence.len() {
                    let k = assigned_k as usize;
                    format!("{}{}", three_to_one(&self.sequence[k]), self.seq_ids[k])
                } else {
                    format!("#{s}")
                };
                painter.text(
                    egui::pos2(pt.x + 6.0, pt.y - 6.0),
                    egui::Align2::LEFT_BOTTOM,
                    text,
                    egui::FontId::monospace(10.5),
                    marker_col,
                );
            }
        }

        // Draw axis ticks and live cursor ppm readout
        painter.rect_stroke(
            rect,
            2.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(65, 80, 105)),
            egui::StrokeKind::Inside,
        );
        for i in 0..=5 {
            let t = (i as f32) / 5.0;
            let h_val = self.view_h_left + t * (self.view_h_right - self.view_h_left);
            let x = rect.left() + t * rect.width();
            painter.text(
                egui::pos2(x, rect.bottom() - 3.0),
                egui::Align2::CENTER_BOTTOM,
                format!("{h_val:.2} ¹H"),
                egui::FontId::monospace(10.0),
                egui::Color32::from_rgb(170, 188, 215),
            );

            let n_val = self.view_n_top + t * (self.view_n_bottom - self.view_n_top);
            let y = rect.top() + t * rect.height();
            painter.text(
                egui::pos2(rect.left() + 4.0, y),
                egui::Align2::LEFT_CENTER,
                format!("{n_val:.1}"),
                egui::FontId::monospace(10.0),
                egui::Color32::from_rgb(170, 188, 215),
            );
        }

        if let Some(hp) = response.hover_pos() {
            let u = (hp.x - rect.left()) / rect.width();
            let v = (hp.y - rect.top()) / rect.height();
            let cur_h = self.view_h_left + u * (self.view_h_right - self.view_h_left);
            let cur_n = self.view_n_top + v * (self.view_n_bottom - self.view_n_top);
            painter.text(
                egui::pos2(rect.right() - 8.0, rect.top() + 6.0),
                egui::Align2::RIGHT_TOP,
                format!("¹H {cur_h:.3} ppm | ¹⁵N {cur_n:.2} ppm"),
                egui::FontId::monospace(11.0),
                egui::Color32::from_rgb(220, 235, 255),
            );
        }
    }

    fn draw_right_inspector(&mut self, ui: &mut egui::Ui) {
        // 1. Metrics summary card if reference truth is present
        if let Some(m) = &self.metrics {
            egui::Frame::group(ui.style())
                .fill(egui::Color32::from_rgb(22, 29, 40))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Benchmark vs Truth").strong());
                        if let Some(ms) = self.last_solve_ms {
                            ui.label(
                                egui::RichText::new(format!("{ms:.0} ms"))
                                    .monospace()
                                    .color(egui::Color32::from_rgb(120, 225, 255)),
                            );
                        }
                    });
                    ui.columns(3, |cols| {
                        cols[0].vertical_centered(|ui| {
                            ui.label(egui::RichText::new("Accuracy").small().weak());
                            ui.label(
                                egui::RichText::new(format!("{:.1}%", m.accuracy() * 100.0))
                                    .strong()
                                    .size(16.0),
                            );
                        });
                        cols[1].vertical_centered(|ui| {
                            ui.label(egui::RichText::new("Recall").small().weak());
                            ui.label(
                                egui::RichText::new(format!("{:.1}%", m.recall() * 100.0))
                                    .strong()
                                    .size(16.0),
                            );
                        });
                        cols[2].vertical_centered(|ui| {
                            ui.label(egui::RichText::new("HC Prec (>=0.8)").small().weak());
                            ui.label(
                                egui::RichText::new(format!(
                                    "{:.1}% ({}/{})",
                                    m.hc_precision() * 100.0,
                                    m.hc_correct,
                                    m.hc_assigned
                                ))
                                .strong()
                                .size(15.0)
                                .color(egui::Color32::from_rgb(95, 240, 165)),
                            );
                        });
                    });
                });
            ui.add_space(6.0);
        }

        // 2. 3D Triple-Resonance Sequential Walk Strip Plot (k-1, k, k+1)
        ui.label(egui::RichText::new("3D HNCACB / CBCA(CO)NH Sequential Walk Strips").strong());
        ui.label(
            egui::RichText::new("Cyan = +Cα, Coral = −Cβ, Gold ◆ = (i−1), Dashed = sequential tie")
                .small()
                .weak(),
        );

        let Some(sel_s) = self.selected_ss.filter(|&s| s < self.systems.len()) else {
            ui.label("Click any HSQC peak or assigned residue to inspect 3D strips.");
            return;
        };

        // Determine predecessor and successor spin systems (either from sequence assignment or best P[s,t])
        let (pred_s, succ_s) = self.find_walk_neighbors(sel_s);
        let strip_list: Vec<(Option<usize>, &str)> = vec![
            (pred_s, "i − 1"),
            (Some(sel_s), "Selected (i)"),
            (succ_s, "i + 1"),
        ];

        let strip_h = 275.0f32;
        let (strip_rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width().max(240.0), strip_h),
            egui::Sense::hover(),
        );
        self.paint_sequential_strips(ui.painter_at(strip_rect), strip_rect, &strip_list);

        ui.add_space(8.0);
        ui.separator();

        // 3. Detailed Spin System Candidate & LLR / Connectivity Inspector
        egui::ScrollArea::vertical()
            .id_salt("ss_inspector_scroll")
            .show(ui, |ui| {
                let ss = &self.systems[sel_s];
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "Spin System #{sel_s}  (¹H {:.3}, ¹⁵N {:.2})",
                            ss.h, ss.n
                        ))
                        .strong(),
                    );
                    if let Some(res) = &self.assign_res {
                        let k = res.position[sel_s];
                        if k >= 0 && (k as usize) < self.sequence.len() {
                            let ku = k as usize;
                            ui.label(
                                egui::RichText::new(format!(
                                    "→ {}{} ({:.0}%)",
                                    self.sequence[ku],
                                    self.seq_ids[ku],
                                    res.confidence[sel_s] * 100.0
                                ))
                                .color(egui::Color32::from_rgb(95, 240, 165))
                                .strong(),
                            );
                        }
                    }
                });

                egui::Grid::new("cands_grid")
                    .striped(true)
                    .min_col_width(58.0)
                    .show(ui, |ui| {
                        ui.strong("Atom");
                        ui.strong("Offset");
                        ui.strong("Candidates (ppm × support)");
                        ui.end_row();

                        for atom in ["CA", "CB", "C"] {
                            for (off, label) in [(0, "i"), (-1, "i − 1")] {
                                if let Some(cands) = ss.cands.get(&(atom.to_string(), off)) {
                                    ui.monospace(atom);
                                    ui.monospace(label);
                                    let text = cands
                                        .iter()
                                        .map(|(v, sup)| format!("{v:.2} (×{sup})"))
                                        .collect::<Vec<_>>()
                                        .join(", ");
                                    ui.monospace(text);
                                    ui.end_row();
                                }
                            }
                        }
                    });

                if let Some(u_mat) = &self.u_mat {
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new("Top Residue Typing LLR U(s, k):")
                            .small()
                            .strong(),
                    );
                    let mut ranked: Vec<(usize, f64)> = (0..u_mat.cols)
                        .map(|k| (k, u_mat.get(sel_s, k)))
                        .filter(|&(_, v)| v > FORBID * 0.5)
                        .collect();
                    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
                    let top_txt = if ranked.is_empty() {
                        "0 observed carbons (excluded from backbone)".to_string()
                    } else {
                        ranked
                            .iter()
                            .take(5)
                            .map(|&(k, sc)| {
                                format!("{}{}: {:+.1}", self.sequence[k], self.seq_ids[k], sc)
                            })
                            .collect::<Vec<_>>()
                            .join("  |  ")
                    };
                    ui.monospace(top_txt);
                }

                if let Some(p_mat) = &self.p_mat {
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("Best Sequential Successors P(s → t):")
                            .small()
                            .strong(),
                    );
                    let mut succs: Vec<(usize, f64)> = (0..p_mat.cols)
                        .filter(|&t| t != sel_s)
                        .map(|t| (t, p_mat.get(sel_s, t)))
                        .filter(|&(_, v)| v > -20.0 && v.abs() > 1e-6)
                        .collect();
                    succs.sort_by(|a, b| b.1.total_cmp(&a.1));
                    for (t, sc) in succs.into_iter().take(4) {
                        if ui
                            .small_button(format!("Jump → SS #{t} (P = {sc:+.2})"))
                            .clicked()
                        {
                            self.selected_ss = Some(t);
                        }
                    }
                }
            });
    }

    fn find_walk_neighbors(&self, sel_s: usize) -> (Option<usize>, Option<usize>) {
        if let Some(res) = &self.assign_res {
            let k = res.position[sel_s];
            if k >= 0 {
                let pred = res.position.iter().position(|&pk| pk == k - 1);
                let succ = res.position.iter().position(|&pk| pk == k + 1);
                return (pred, succ);
            }
        }
        if let Some(p_mat) = &self.p_mat {
            let pred = (0..p_mat.rows)
                .filter(|&a| a != sel_s)
                .max_by(|&a, &b| p_mat.get(a, sel_s).total_cmp(&p_mat.get(b, sel_s)))
                .filter(|&a| p_mat.get(a, sel_s) > 0.0);
            let succ = (0..p_mat.cols)
                .filter(|&b| b != sel_s)
                .max_by(|&a, &b| p_mat.get(sel_s, a).total_cmp(&p_mat.get(sel_s, b)))
                .filter(|&b| p_mat.get(sel_s, b) > 0.0);
            return (pred, succ);
        }
        (None, None)
    }

    fn paint_sequential_strips(
        &self,
        painter: egui::Painter,
        rect: egui::Rect,
        strips: &[(Option<usize>, &str)],
    ) {
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(13, 17, 24));
        let c_top = 13.0f64;
        let c_bot = 74.0f64;
        let top_pad = 22.0f32;
        let bot_pad = 8.0f32;
        let left_axis_w = 34.0f32;

        let plot_top = rect.top() + top_pad;
        let plot_bot = rect.bottom() - bot_pad;
        let c_to_y = |c_ppm: f64| -> f32 {
            let t = ((c_ppm - c_top) / (c_bot - c_top)) as f32;
            plot_top + t.clamp(0.0, 1.0) * (plot_bot - plot_top)
        };

        // 13C ppm scale on left
        for c_tick in [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 72.0] {
            let y = c_to_y(c_tick);
            painter.text(
                egui::pos2(rect.left() + 3.0, y),
                egui::Align2::LEFT_CENTER,
                format!("{c_tick:.0}"),
                egui::FontId::monospace(9.5),
                egui::Color32::from_rgb(145, 165, 190),
            );
            painter.line_segment(
                [
                    egui::pos2(rect.left() + left_axis_w, y),
                    egui::pos2(rect.right() - 4.0, y),
                ],
                egui::Stroke::new(0.5, egui::Color32::from_rgb(28, 35, 48)),
            );
        }

        let total_w = (rect.width() - left_axis_w - 6.0).max(90.0);
        let col_w = total_w / (strips.len() as f32);

        let mut col_centers = Vec::with_capacity(strips.len());

        for (idx, &(ss_opt, role)) in strips.iter().enumerate() {
            let x0 = rect.left() + left_axis_w + (idx as f32) * col_w + 3.0;
            let x1 = x0 + col_w - 6.0;
            let cx = 0.5 * (x0 + x1);
            col_centers.push(cx);

            let strip_box =
                egui::Rect::from_min_max(egui::pos2(x0, plot_top), egui::pos2(x1, plot_bot));
            painter.rect_stroke(
                strip_box,
                2.0,
                egui::Stroke::new(
                    if idx == 1 { 1.5 } else { 1.0 },
                    if idx == 1 {
                        egui::Color32::from_rgb(95, 200, 255)
                    } else {
                        egui::Color32::from_rgb(50, 62, 82)
                    },
                ),
                egui::StrokeKind::Inside,
            );

            let Some(s) = ss_opt else {
                painter.text(
                    egui::pos2(cx, rect.top() + 10.0),
                    egui::Align2::CENTER_CENTER,
                    format!("{role}: —"),
                    egui::FontId::monospace(10.0),
                    egui::Color32::GRAY,
                );
                continue;
            };
            let ss = &self.systems[s];

            // Header title (assigned residue or SS#)
            let title = self
                .assign_res
                .as_ref()
                .and_then(|r| (r.position[s] >= 0).then(|| r.position[s] as usize))
                .map(|k| {
                    format!(
                        "{}{} (#{s})",
                        three_to_one(&self.sequence[k]),
                        self.seq_ids[k]
                    )
                })
                .unwrap_or_else(|| format!("SS #{s}"));
            painter.text(
                egui::pos2(cx, rect.top() + 10.0),
                egui::Align2::CENTER_CENTER,
                title,
                egui::FontId::monospace(10.5),
                egui::Color32::from_rgb(225, 238, 255),
            );

            // Render 3D HNCACB strip slice intensity heatmap if available
            if let Some(s3) = &self.hncacb_3d {
                let slice = s3.extract_strip(ss.h, ss.n, 0.04);
                let max_v = slice.max_abs().max(1.0);
                for ic in 0..slice.ny() {
                    let c_ppm = slice.axis_y.index_to_ppm(ic as f64);
                    if !(c_top..=c_bot).contains(&c_ppm) {
                        continue;
                    }
                    let y = c_to_y(c_ppm);
                    let mid_x = slice.nx() / 2;
                    let val = slice.get(mid_x, ic);
                    let norm = (val.abs() / max_v).clamp(0.0, 1.0);
                    if norm > 0.14 {
                        let col = if val >= 0.0 {
                            egui::Color32::from_rgba_unmultiplied(
                                40,
                                195,
                                255,
                                (norm * 130.0) as u8,
                            )
                        } else {
                            egui::Color32::from_rgba_unmultiplied(
                                255,
                                95,
                                105,
                                (norm * 130.0) as u8,
                            )
                        };
                        let half_bar = norm * (strip_box.width() * 0.36);
                        painter.line_segment(
                            [egui::pos2(cx - half_bar, y), egui::pos2(cx + half_bar, y)],
                            egui::Stroke::new(2.2, col),
                        );
                    }
                }
            }

            // Plot resolved spin-system candidates in strip: (i) solid circles, (i-1) diamonds
            for atom in ["CA", "CB"] {
                let base_col = if atom == "CA" {
                    egui::Color32::from_rgb(80, 230, 255)
                } else {
                    egui::Color32::from_rgb(255, 115, 120)
                };
                if let Some(cands_i) = ss.cands.get(&(atom.to_string(), 0)) {
                    if let Some(&(ppm, sup)) = cands_i.first() {
                        if sup > 0 {
                            let y = c_to_y(ppm);
                            painter.circle_filled(egui::pos2(cx - 8.0, y), 4.2, base_col);
                            painter.text(
                                egui::pos2(cx - 14.0, y),
                                egui::Align2::RIGHT_CENTER,
                                format!("{ppm:.1}"),
                                egui::FontId::monospace(8.8),
                                base_col,
                            );
                        }
                    }
                }
                if let Some(cands_im1) = ss.cands.get(&(atom.to_string(), -1)) {
                    if let Some(&(ppm, _)) = cands_im1.first() {
                        let y = c_to_y(ppm);
                        let d_col = egui::Color32::from_rgb(250, 205, 90);
                        let pts = vec![
                            egui::pos2(cx + 8.0, y - 4.2),
                            egui::pos2(cx + 12.2, y),
                            egui::pos2(cx + 8.0, y + 4.2),
                            egui::pos2(cx + 3.8, y),
                        ];
                        painter.add(egui::Shape::convex_polygon(pts, d_col, egui::Stroke::NONE));
                    }
                }
            }
        }

        // Draw sequential connectivity tie-lines between strip[w] (offset 0) and strip[w+1] (offset -1)
        for w in 0..strips.len().saturating_sub(1) {
            let (Some(sa), Some(sb)) = (strips[w].0, strips[w + 1].0) else {
                continue;
            };
            for atom in ["CA", "CB"] {
                let Some(va) = self.systems[sa]
                    .cands
                    .get(&(atom.to_string(), 0))
                    .and_then(|c| c.first())
                    .filter(|c| c.1 > 0)
                    .map(|c| c.0)
                else {
                    continue;
                };
                let Some(vb) = self.systems[sb]
                    .cands
                    .get(&(atom.to_string(), -1))
                    .and_then(|c| c.first())
                    .map(|c| c.0)
                else {
                    continue;
                };
                if (va - vb).abs() <= 0.45 {
                    let ya = c_to_y(va);
                    let yb = c_to_y(vb);
                    let x_start = col_centers[w] - 4.0;
                    let x_end = col_centers[w + 1] + 4.0;
                    let col = if atom == "CA" {
                        egui::Color32::from_rgba_unmultiplied(110, 245, 255, 210)
                    } else {
                        egui::Color32::from_rgba_unmultiplied(255, 145, 145, 210)
                    };
                    // Dashed horizontal connector
                    let segs = 9;
                    for j in (0..segs).step_by(2) {
                        let t0 = (j as f32) / (segs as f32);
                        let t1 = ((j + 1) as f32) / (segs as f32);
                        painter.line_segment(
                            [
                                egui::pos2(x_start + t0 * (x_end - x_start), ya + t0 * (yb - ya)),
                                egui::pos2(x_start + t1 * (x_end - x_start), ya + t1 * (yb - ya)),
                            ],
                            egui::Stroke::new(1.5, col),
                        );
                    }
                }
            }
        }
    }
}

impl eframe::App for NmrWorkbenchApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.handle_dropped_files(&ctx);
        self.draw_text_io_window(&ctx);

        egui::Panel::top("workspace_mode_bar").show(ui, |ui| {
            self.draw_workspace_mode_header(ui);
        });

        match self.workspace_mode {
            WorkspaceMode::ProteinNmr => {
                egui::Panel::top("top_controls_panel").show(ui, |ui| {
                    self.draw_top_bar(ui);
                    ui.add_space(4.0);
                    self.draw_sequence_ribbon(ui);
                });

                egui::Panel::right("right_strip_inspector")
                    .default_size(390.0)
                    .min_size(320.0)
                    .show(ui, |ui| {
                        self.draw_right_inspector(ui);
                    });

                egui::CentralPanel::default().show(ui, |ui| {
                    self.draw_hsqc_canvas(ui);
                });
            }
            WorkspaceMode::OctetChem => {
                self.octet_app.ui(ui);
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(format!(
                "nmr-assign v{} — WebGPU Protein NMR & Octet Chemical Studio",
                nmr_core::VERSION
            ))
            .with_inner_size([1380.0, 860.0])
            .with_min_inner_size([960.0, 620.0]),
        ..Default::default()
    };
    eframe::run_native(
        "nmr-assign",
        native_options,
        Box::new(|cc| Ok(Box::new(NmrWorkbenchApp::new(cc)))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use wasm_bindgen::JsCast as _;

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("no browser Window")
            .document()
            .expect("no browser Document");

        let canvas = document
            .get_element_by_id("nmr_canvas")
            .expect("missing <canvas id=\"nmr_canvas\"> element")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("nmr_canvas is not an HtmlCanvasElement");

        let web_options = eframe::WebOptions::default();
        let start_result = eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(NmrWorkbenchApp::new(cc)))),
            )
            .await;

        if let Some(loading) = document.get_element_by_id("nmr_loading") {
            match start_result {
                Ok(()) => loading.remove(),
                Err(e) => {
                    loading.set_inner_html(&format!(
                        "<p style=\"color:#ff7b7b;max-width:520px;text-align:center\">Failed to start WebGPU workbench: {:?}<br>\
                         Ensure WebGPU is enabled in your browser (Chrome/Edge 113+, Safari 18+, Firefox Nightly).</p>",
                        e
                    ));
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::GuiSolverKind;

    #[test]
    fn test_gui_default_solver_is_evo_graph() {
        assert_eq!(GuiSolverKind::default(), GuiSolverKind::EvoGraph);
        assert_eq!(GuiSolverKind::default().label(), "Evo-Graph");
    }
}
