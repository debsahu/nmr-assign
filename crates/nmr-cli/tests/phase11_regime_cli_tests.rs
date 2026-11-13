//! End-to-end CLI integration tests for Phase 11 (`T11.1`–`T11.4`) solver regimes
//! (`--regime globular|idp-13c|paramagnetic`), `nmrassign propensity` (`SSP` & 4-state `δ2D`),
//! `nmrassign report --idp-propensity`, and `P1-12` non-destructive aromatic user ring preservation.
//!
//! Provenance: Clean-room integration test suite citing peer-reviewed biomolecular NMR literature:
//! - Bermel, Bertini, Felli, Piccioli & Pierattelli (*Prog. Nucl. Magn. Reson. Spectrosc.* 2006,
//!   48, 25–45) and Bermel et al. (*J. Am. Chem. Soc.* 2006, 128, 3918–3919; *J. Am. Chem. Soc.*
//!   2009, 131, 15339–15345 — protonless `13C`-detected `CON`, `CACO`, `CANCO`, `CBCACON`,
//!   `CCCON`, `NCACO` assignment through proline).
//! - Marsh, Singh, Jia & Forman-Kay (*Protein Sci.* 2006, 15, 2795–2804 — `SSP`) and Camilloni,
//!   De Simone, Vranken & Vendruscolo (*Biochemistry* 2012, 51, 2224–2231 — 4-state `δ2D`).
//! - Bertini, Luchinat & Parigi (*Prog. Nucl. Magn. Reson. Spectrosc.* 2002, 40, 249–273) and
//!   Schmitz, Stanton-Cook, Su, Otting & Huber (*J. Biomol. NMR* 2008, 41, 179–189 — 5-parameter
//!   traceless symmetric `Δχ` pseudocontact shift SVD fitting).
//! - Yamazaki, Forman-Kay & Kay (*J. Am. Chem. Soc.* 1993, 115, 11054–11055) and Pelton et al.
//!   (*Protein Sci.* 1993, 2, 543–558 — aromatic ring assignment & His tautomers).

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

use nmr_core::aromatic::{
    default_his_tautomer_map, simulate_aromatic_peaks, synthesize_aromatic_reference,
    AromaticSimConfig,
};
use nmr_core::c13_detected::{
    pick_caco_2d, pick_con_2d, pick_con_anchored_3d, simulate_idp_13c_cubes, C13DetectedExp,
    C13PickedDataset, C13PickerConfig, IdpCubeSimConfig,
};
use nmr_core::nmrstar::{read_star_all_atoms, write_star, Reference};
use nmr_core::noesy::parse_pdb_first_model;
use nmr_core::paramagnetic::predict_pcs_spherical_ppm;
use nmr_core::peaks::write_peaks;
use nmr_core::priors::three_to_one;
use nmr_core::sidechain::{
    simulate_sidechain_peaks, synthesize_all_atom_reference, SidechainSimConfig,
};
use nmr_core::simulate::{simulate, SimConfig};
use nmr_core::spectrum::{write_ucsf_2d, write_ucsf_3d};
use nmr_core::topology::SidechainPriors;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn nmrassign_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nmrassign"))
}

fn make_temp_dir(tag: &str) -> PathBuf {
    let dir = env::temp_dir().join(format!("nmr_p11_cli_{tag}_{}", process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp test dir");
    dir
}

fn sequence_one_letter(seq: &[String]) -> String {
    seq.iter().map(|s| three_to_one(s)).collect()
}

fn write_c13_dataset_tsv(path: &Path, ds: &C13PickedDataset) {
    let mut lines = vec![
        "# exp\tdim1_ppm\tdim2_ppm\tdim3_ppm\tsign\theight\tvolume\tanchor_root_idx".to_string(),
    ];
    for r in &ds.con_roots {
        lines.push(format!(
            "CON\t{:.4}\t{:.4}\t.\t1\t{:.2}\t{:.2}\t.",
            r.c_prev_ppm, r.n_curr_ppm, r.height, r.volume
        ));
    }
    for p in &ds.caco_peaks {
        lines.push(format!(
            "CACO\t{:.4}\t{:.4}\t.\t{}\t{:.2}\t{:.2}\t.",
            p.c_curr_ppm, p.ca_curr_ppm, p.sign, p.height, p.volume
        ));
    }
    for p in &ds.strip_peaks {
        let anc = p
            .anchor_root_idx
            .map(|i| i.to_string())
            .unwrap_or_else(|| ".".to_string());
        lines.push(format!(
            "{}\t{:.4}\t{:.4}\t{:.4}\t{}\t{:.2}\t{:.2}\t{}",
            p.exp.as_str(),
            p.c_prime_ppm,
            p.n_ppm,
            p.c_indirect_ppm,
            p.sign,
            p.height,
            p.volume,
            anc
        ));
    }
    fs::write(path, format!("{}\n", lines.join("\n"))).expect("write 13C TSV");
}

fn pick_simulated_c13_bundle(truth_ref: &Reference, seed: u64) -> C13PickedDataset {
    let sim_cfg = IdpCubeSimConfig {
        proton_mhz: 700.0,
        temperature_k: 298.15,
        ph: 6.0,
        snr_2d: 55.0,
        snr_3d: 36.0,
        con_grid: (512, 512),
        caco_grid: (256, 256),
        cube_3d_grid: (128, 128, 128),
        experiments: vec![
            C13DetectedExp::Con2D,
            C13DetectedExp::Caco2D,
            C13DetectedExp::Canco3D,
            C13DetectedExp::Cbcaco3D,
            C13DetectedExp::Cccon3D,
            C13DetectedExp::Ncaco3D,
        ],
        seed,
        ..IdpCubeSimConfig::default()
    };
    let cubes = simulate_idp_13c_cubes(truth_ref, &sim_cfg);
    let pcfg = C13PickerConfig::default();
    let con_roots = pick_con_2d(&cubes.con_2d, &pcfg);
    let caco_peaks = cubes
        .caco_2d
        .as_ref()
        .map(|s| pick_caco_2d(s, &pcfg))
        .unwrap_or_default();
    let mut strip_peaks = Vec::new();
    for (exp, s3) in &cubes.cubes_3d {
        strip_peaks.extend(pick_con_anchored_3d(s3, *exp, &con_roots, &pcfg));
    }
    C13PickedDataset {
        con_roots,
        caco_peaks,
        strip_peaks,
    }
}

fn evaluate_idp_backbone_shift_accuracy(
    truth: &Reference,
    pred: &Reference,
) -> (f64, usize, usize) {
    let mut matched = 0usize;
    let mut correct = 0usize;
    for (&(k, ref atom), &v_pred) in &pred.shifts {
        let tol = match atom.as_str() {
            "N" => 0.55,
            "C" => 0.35,
            "CA" => 0.40,
            "CB" => 0.50,
            _ => continue,
        };
        if let Some(&v_true) = truth.shifts.get(&(k, atom.clone())) {
            matched += 1;
            if (v_pred - v_true).abs() <= tol {
                correct += 1;
            }
        }
    }
    let acc = if matched > 0 {
        (correct as f64) / (matched as f64)
    } else {
        0.0
    };
    (acc, correct, matched)
}

fn parse_star_ambiguity_codes(star_text: &str) -> BTreeMap<(i32, String), u8> {
    let mut map = BTreeMap::new();
    let mut in_shift_loop = false;
    for raw in star_text.lines() {
        let line = raw.trim();
        if line == "_Atom_chem_shift.Details" {
            in_shift_loop = true;
            continue;
        }
        if in_shift_loop {
            if line == "stop_" {
                break;
            }
            if line.is_empty() || line.starts_with('#') || line.starts_with('_') {
                continue;
            }
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() >= 8 {
                if let (Ok(seq_id), Ok(amb)) = (cols[2].parse::<i32>(), cols[7].parse::<u8>()) {
                    map.insert((seq_id, cols[4].to_string()), amb);
                }
            }
        }
    }
    map
}

#[test]
fn test_cli_assign_regime_idp_13c_cpu_and_auto_with_proline_shifts() {
    let root = repo_root();
    let idp_star = root.join("data/fixtures/bmr51984_idp_p53_tad_prr.str");
    let truth_ref = read_star_all_atoms(&idp_star, None).expect("read bmr51984 p53 IDP fixture");
    assert_eq!(truth_ref.sequence.len(), 102);

    let dir = make_temp_dir("idp_13c");
    let ds = pick_simulated_c13_bundle(&truth_ref, 42);
    let c13_tsv = dir.join("p53_c13_peaks.tsv");
    write_c13_dataset_tsv(&c13_tsv, &ds);
    let seq_str = sequence_one_letter(&truth_ref.sequence);

    for engine in ["cpu", "auto"] {
        let out_star = dir.join(format!("p53_assigned_{engine}.str"));
        let output = Command::new(nmrassign_bin())
            .args([
                "assign",
                "--regime",
                "idp-13c",
                "--seq",
                &seq_str,
                "--peaks",
                c13_tsv.to_str().unwrap(),
                "--temp-k",
                "298.15",
                "--ph",
                "6.0",
                "--engine",
                engine,
                "--seed",
                "42",
                "--out",
                out_star.to_str().unwrap(),
            ])
            .output()
            .expect("run nmrassign assign --regime idp-13c");

        assert!(
            output.status.success(),
            "assign --regime idp-13c --engine {engine} failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let star_text = fs::read_to_string(&out_star).expect("read assigned IDP star");
        assert!(
            star_text.contains("conf="),
            "expected per-residue confidence annotations in output NMR-STAR"
        );

        let pred_ref = read_star_all_atoms(&out_star, None).expect("parse assigned IDP star");
        let (acc, correct, matched) = evaluate_idp_backbone_shift_accuracy(&truth_ref, &pred_ref);
        assert!(
            matched >= 240 && acc >= 0.92,
            "expected >=92% 13C-IDP accuracy on bmr51984 (--engine {engine}), got {:.2}% ({correct}/{matched})",
            acc * 100.0
        );

        // Verify Proline N, CA, CB, C shifts are assigned directly through Prolines (including Pro-Pro)
        let mut pro_n_count = 0usize;
        let mut pro_ca_count = 0usize;
        let mut pro_cb_count = 0usize;
        let mut pro_c_count = 0usize;
        for (k, rname) in pred_ref.sequence.iter().enumerate() {
            if rname == "PRO" {
                if let Some(&n_ppm) = pred_ref.shifts.get(&(k, "N".to_string())) {
                    assert!(
                        (130.0..=145.0).contains(&n_ppm),
                        "expected Pro imide 15N in [130, 145] ppm, got {n_ppm}"
                    );
                    pro_n_count += 1;
                }
                if pred_ref.shifts.contains_key(&(k, "CA".to_string())) {
                    pro_ca_count += 1;
                }
                if pred_ref.shifts.contains_key(&(k, "CB".to_string())) {
                    pro_cb_count += 1;
                }
                if pred_ref.shifts.contains_key(&(k, "C".to_string())) {
                    pro_c_count += 1;
                }
            }
        }
        assert!(
            pro_n_count >= 16 && pro_ca_count >= 16 && pro_cb_count >= 14 && pro_c_count >= 16,
            "expected rich Pro N/CA/CB/C shift recovery (--engine {engine}), got N={pro_n_count}, CA={pro_ca_count}, CB={pro_cb_count}, C={pro_c_count}"
        );
    }

    // Also verify direct processed 2D/3D .ucsf spectrum ingestion (--con / --caco / --canco / --cbcacon)
    // on the 22-residue proline-rich N-terminal p53(1..22) segment (containing PRO6, PRO10, PRO14, PRO15).
    let sub_len = 22usize;
    let mut sub_ref = Reference::new(
        truth_ref.sequence[..sub_len].to_vec(),
        truth_ref.seq_ids[..sub_len].to_vec(),
    );
    for (&(k, ref atom), &v) in &truth_ref.shifts {
        if k < sub_len {
            sub_ref.shifts.insert((k, atom.clone()), v);
        }
    }
    let cubes = simulate_idp_13c_cubes(
        &sub_ref,
        &IdpCubeSimConfig {
            snr_2d: 55.0,
            snr_3d: 36.0,
            con_grid: (256, 256),
            caco_grid: (128, 128),
            cube_3d_grid: (96, 96, 96),
            experiments: vec![
                C13DetectedExp::Con2D,
                C13DetectedExp::Caco2D,
                C13DetectedExp::Canco3D,
                C13DetectedExp::Cbcaco3D,
            ],
            seed: 7,
            ..IdpCubeSimConfig::default()
        },
    );
    let con_ucsf = dir.join("con.ucsf");
    let caco_ucsf = dir.join("caco.ucsf");
    let canco_ucsf = dir.join("canco.ucsf");
    let cbcacon_ucsf = dir.join("cbcacon.ucsf");
    write_ucsf_2d(&con_ucsf, &cubes.con_2d).expect("write CON ucsf");
    write_ucsf_2d(&caco_ucsf, cubes.caco_2d.as_ref().unwrap()).expect("write CACO ucsf");
    for (exp, c3) in &cubes.cubes_3d {
        match exp {
            C13DetectedExp::Canco3D => write_ucsf_3d(&canco_ucsf, c3).unwrap(),
            C13DetectedExp::Cbcaco3D => write_ucsf_3d(&cbcacon_ucsf, c3).unwrap(),
            _ => {}
        }
    }

    let ucsf_out = dir.join("p53_sub_ucsf_assigned.str");
    let sub_seq_str = sequence_one_letter(&sub_ref.sequence);
    let out_ucsf_cmd = Command::new(nmrassign_bin())
        .args([
            "assign",
            "--regime",
            "idp-13c",
            "--seq",
            &sub_seq_str,
            "--con",
            con_ucsf.to_str().unwrap(),
            "--caco",
            caco_ucsf.to_str().unwrap(),
            "--canco",
            canco_ucsf.to_str().unwrap(),
            "--cbcacon",
            cbcacon_ucsf.to_str().unwrap(),
            "--engine",
            "cpu",
            "--seed",
            "7",
            "--out",
            ucsf_out.to_str().unwrap(),
        ])
        .output()
        .expect("run assign --regime idp-13c on .ucsf spectra");
    assert!(
        out_ucsf_cmd.status.success(),
        "assign --regime idp-13c .ucsf failed:\n{}",
        String::from_utf8_lossy(&out_ucsf_cmd.stderr)
    );
    let sub_pred = read_star_all_atoms(&ucsf_out, None).unwrap();
    let (sub_acc, _, sub_matched) = evaluate_idp_backbone_shift_accuracy(&sub_ref, &sub_pred);
    assert!(
        sub_matched >= 50 && sub_acc >= 0.92,
        "expected >=92% accuracy from direct .ucsf 13C-IDP picking, got {:.2}% ({sub_matched} shifts)",
        sub_acc * 100.0
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_cli_assign_regime_paramagnetic_metal_ligands_and_pcs_svd() {
    let root = repo_root();
    let star_in = root.join("data/fixtures/synthetic_ubiquitin.str");
    let pdb_in = root.join("data/pdb_refs/1d3z.pdb");
    let truth_ref = read_star_all_atoms(&star_in, None).expect("read synthetic ubiquitin");
    let pdb_chain = parse_pdb_first_model(&pdb_in).expect("parse 1d3z.pdb");

    let dir = make_temp_dir("paramagnetic_pcs");

    // Place a paramagnetic lanthanide HETATM site 7.5 Å outside His68 CA and synthesize exact 5-param PCS
    let his68 = pdb_chain
        .residues
        .iter()
        .find(|r| r.seq_id == 68)
        .expect("His68 in 1d3z.pdb");
    let metal_pos = [his68.ca[0] + 5.5, his68.ca[1] + 4.5, his68.ca[2] + 3.0];

    let mut pdb_with_metal = fs::read_to_string(&pdb_in).expect("read 1d3z.pdb text");
    if let Some(end_idx) = pdb_with_metal.find("ENDMDL") {
        // Prepend decoy HETATMs (selenomethionine CA with and without col 77-78 element, plus
        // closed-shell diamagnetic CA/MG/ZN/LA/LU ions) BEFORE the true paramagnetic DY ion so any
        // naive first-HETATM or atom_name=="CA" matcher would lock onto the wrong site!
        let het_block = format!(
            "HETATM99990  CA  MSE A 990     -25.000 -25.000 -25.000  1.00 10.00           C\n\
             HETATM99991  CA  MSE A 991     -23.000 -25.000 -25.000  1.00 10.00\n\
             HETATM99992 MG    MG A 992     -20.000  20.000 -20.000  1.00 10.00          MG\n\
             HETATM99993 CA    CA A 993      20.000 -20.000 -20.000  1.00 10.00          CA\n\
             HETATM99994 ZN    ZN A 994      20.000  20.000 -20.000  1.00 10.00          ZN\n\
             HETATM99995 LA    LA A 995     -20.000 -20.000  20.000  1.00 10.00          LA\n\
             HETATM99996 LU    LU A 996      20.000 -20.000  20.000  1.00 10.00          LU\n\
             HETATM99999 DY    DY A 999    {:8.3}{:8.3}{:8.3}  1.00 10.00          DY\n",
            metal_pos[0], metal_pos[1], metal_pos[2]
        );
        pdb_with_metal.insert_str(end_idx, &het_block);
    }
    let para_pdb = dir.join("ubiquitin_dy_tag.pdb");
    fs::write(&para_pdb, &pdb_with_metal).expect("write tagged PDB");

    let true_dchi_ax = 24.5_f64;
    let true_dchi_rh = 6.2_f64;
    let true_euler = [0.42_f64, 0.85_f64, 1.15_f64];

    let mut pcs_lines = vec!["# seq_id\tatom\tpcs_ppm\tsigma_ppm".to_string()];
    for r in &pdb_chain.residues {
        if r.res_name == "PRO" || r.seq_id == 1 {
            continue;
        }
        let coord = r.h.unwrap_or(r.n);
        let dx = coord[0] - metal_pos[0];
        let dy = coord[1] - metal_pos[1];
        let dz = coord[2] - metal_pos[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist >= 9.0 {
            let pcs =
                predict_pcs_spherical_ppm(coord, metal_pos, true_dchi_ax, true_dchi_rh, true_euler);
            pcs_lines.push(format!("{}\tH\t{:.5}\t0.020", r.seq_id, pcs));
        }
    }
    let pcs_tsv = dir.join("amide_pcs.tsv");
    fs::write(&pcs_tsv, format!("{}\n", pcs_lines.join("\n"))).expect("write PCS TSV");

    // Simulate backbone peaks with bleached spin systems near His68
    let (mut peaks, spin_map) = simulate(
        &truth_ref,
        &SimConfig {
            p_missing_peak: 0.01,
            spurious_rate: 0.01,
            seed: 11,
            ..SimConfig::default()
        },
    );
    // Drop spin systems within the immediate blind zone around His68 (residues 67..=69 -> idx 66..=68)
    let bleached_roots: Vec<(f64, f64)> = peaks
        .iter()
        .filter(|p| p.exp == "HSQC")
        .enumerate()
        .filter_map(|(ss_idx, p)| {
            let res_idx = spin_map.get(ss_idx).copied().unwrap_or(-1);
            if (66..=68).contains(&res_idx) {
                Some((p.h, p.n))
            } else {
                None
            }
        })
        .collect();
    peaks.retain(|p| {
        !bleached_roots
            .iter()
            .any(|&(bh, bn)| (p.h - bh).abs() < 1e-4 && (p.n - bn).abs() < 1e-4)
    });
    let peaks_tsv = dir.join("para_peaks.tsv");
    write_peaks(&peaks_tsv, &peaks).expect("write paramagnetic peaks");
    let seq_str = sequence_one_letter(&truth_ref.sequence);

    for solver in ["standard", "evo-graph"] {
        let out_star = dir.join(format!("para_out_{solver}.str"));
        let output = Command::new(nmrassign_bin())
            .args([
                "assign",
                "--regime",
                "paramagnetic",
                "--solver",
                solver,
                "--seq",
                &seq_str,
                "--peaks",
                peaks_tsv.to_str().unwrap(),
                "--metal-ligands",
                "HIS68",
                "--pdb",
                para_pdb.to_str().unwrap(),
                "--pcs",
                pcs_tsv.to_str().unwrap(),
                "--seed",
                "11",
                "--out",
                out_star.to_str().unwrap(),
            ])
            .output()
            .expect("run assign --regime paramagnetic");

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "assign --regime paramagnetic --solver {solver} failed:\nstdout: {stdout}\nstderr: {stderr}"
        );
        assert!(
            stdout.contains("PCS 5-param Δχ SVD fit")
                && stdout.contains("Δχ_ax=+24.500")
                && (stdout.contains("Δχ_rh=+6.200") || stdout.contains("Δχ_rh=-6.200"))
                && stdout.contains("rhombicity=0.253")
                && stdout.contains("PCS Q-factor=0.0000")
                && stdout.contains("Euler ZYZ="),
            "expected exact 5-param Δχ SVD tensor summary in stdout for {solver}, got:\n{stdout}"
        );

        let pred = read_star_all_atoms(&out_star, None).expect("read paramagnetic output .str");
        assert!(
            pred.shifts.len() >= 220,
            "expected diamagnetic scaffold shifts assigned cleanly ({solver}), got {}",
            pred.shifts.len()
        );
    }

    // Regression check (P1): tandem multi-cluster [2Fe-2S] FES cofactors (>15 Å apart) with FE1/FE2
    // (~2.7 Å intra-cluster) + S1/S2 sulfurs + HETATM CA MSE decoy must preserve BOTH cluster sites
    // (2 clusters) without truncating to 1 cluster or matching non-metal S/C atoms.
    let glu24 = pdb_chain
        .residues
        .iter()
        .find(|r| r.seq_id == 24)
        .expect("Glu24 in 1d3z.pdb");
    let c1 = [glu24.ca[0] - 4.0, glu24.ca[1] - 3.0, glu24.ca[2]];
    let c2 = [his68.ca[0] + 5.5, his68.ca[1] + 4.5, his68.ca[2] + 3.0];
    let inter_dist =
        ((c1[0] - c2[0]).powi(2) + (c1[1] - c2[1]).powi(2) + (c1[2] - c2[2]).powi(2)).sqrt();
    assert!(
        inter_dist > 15.0,
        "expected >15 Å separation between tandem FES sites, got {inter_dist:.2}"
    );
    let mut tandem_pdb_txt = fs::read_to_string(&pdb_in).expect("read 1d3z.pdb text");
    if let Some(end_idx) = tandem_pdb_txt.find("ENDMDL") {
        let tandem_het = format!(
            "HETATM99800  CA  MSE A 980     -30.000 -30.000 -30.000  1.00 10.00           C\n\
             HETATM99801 FE1  FES A 981    {:8.3}{:8.3}{:8.3}  1.00 10.00          FE\n\
             HETATM99802 FE2  FES A 981    {:8.3}{:8.3}{:8.3}  1.00 10.00          FE\n\
             HETATM99803  S1  FES A 981    {:8.3}{:8.3}{:8.3}  1.00 10.00           S\n\
             HETATM99804  S2  FES A 981    {:8.3}{:8.3}{:8.3}  1.00 10.00           S\n\
             HETATM99805 FE1  FES A 982    {:8.3}{:8.3}{:8.3}  1.00 10.00\n\
             HETATM99806 FE2  FES A 982    {:8.3}{:8.3}{:8.3}  1.00 10.00\n\
             HETATM99807  S1  FES A 982    {:8.3}{:8.3}{:8.3}  1.00 10.00\n\
             HETATM99808  S2  FES A 982    {:8.3}{:8.3}{:8.3}  1.00 10.00\n",
            c1[0] - 1.35,
            c1[1],
            c1[2],
            c1[0] + 1.35,
            c1[1],
            c1[2],
            c1[0],
            c1[1] + 1.80,
            c1[2],
            c1[0],
            c1[1] - 1.80,
            c1[2],
            c2[0] - 1.35,
            c2[1],
            c2[2],
            c2[0] + 1.35,
            c2[1],
            c2[2],
            c2[0],
            c2[1] + 1.80,
            c2[2],
            c2[0],
            c2[1] - 1.80,
            c2[2],
        );
        tandem_pdb_txt.insert_str(end_idx, &tandem_het);
    }
    let tandem_pdb = dir.join("ubiquitin_tandem_fes.pdb");
    fs::write(&tandem_pdb, &tandem_pdb_txt).expect("write tandem FES PDB");
    let tandem_out_prefix = dir.join("para_tandem_out");
    let tandem_run = Command::new(nmrassign_bin())
        .args([
            "assign",
            "--regime",
            "paramagnetic",
            "--solver",
            "standard",
            "--seq",
            &seq_str,
            "--peaks",
            peaks_tsv.to_str().unwrap(),
            "--metal-ligands",
            "GLU24,HIS68",
            "--pdb",
            tandem_pdb.to_str().unwrap(),
            "--seed",
            "11",
            "--out",
            tandem_out_prefix.to_str().unwrap(),
        ])
        .output()
        .expect("run assign --regime paramagnetic with tandem [2Fe-2S] PDB");
    let tandem_stdout = String::from_utf8_lossy(&tandem_run.stdout);
    let tandem_stderr = String::from_utf8_lossy(&tandem_run.stderr);
    assert!(
        tandem_run.status.success(),
        "tandem [2Fe-2S] paramagnetic assign failed:\nstdout: {tandem_stdout}\nstderr: {tandem_stderr}"
    );
    assert!(
        tandem_stdout.contains("regime=paramagnetic | 2 clusters,"),
        "expected both tandem [2Fe-2S] clusters preserved (2 clusters), got:\n{tandem_stdout}"
    );
    let tandem_star_txt =
        fs::read_to_string(tandem_out_prefix.with_extension("str")).expect("read tandem .str");
    assert!(
        tandem_star_txt.contains("clusters=2,"),
        "expected clusters=2 in output NMR-STAR header, got:\n{}",
        &tandem_star_txt[..tandem_star_txt.len().min(600)]
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_cli_propensity_tsv_json_and_report_idp_propensity() {
    let root = repo_root();
    let bmr17769 = {
        let p_fix = root.join("data/fixtures/bmr17769.str");
        if p_fix.exists() {
            p_fix
        } else {
            root.join("data/benchmark/bmr17769.str")
        }
    };
    let bmr51984 = root.join("data/fixtures/bmr51984_idp_p53_tad_prr.str");
    let dir = make_temp_dir("propensity_report");

    // 1. TSV output on BMRB 17769 (stdout and -o)
    let out_tsv = Command::new(nmrassign_bin())
        .args([
            "propensity",
            "--shifts",
            bmr17769.to_str().unwrap(),
            "--format",
            "tsv",
        ])
        .output()
        .expect("run nmrassign propensity --format tsv");
    assert!(
        out_tsv.status.success(),
        "propensity tsv failed: {}",
        String::from_utf8_lossy(&out_tsv.stderr)
    );
    let tsv_text = String::from_utf8_lossy(&out_tsv.stdout);
    let mut data_rows = 0usize;
    for line in tsv_text.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') || t.starts_with("seq_id\t") {
            continue;
        }
        let cols: Vec<&str> = t.split('\t').collect();
        assert!(cols.len() >= 10, "unexpected propensity TSV row: {t}");
        if cols[4] != "NA" {
            let ssp: f64 = cols[4].parse().unwrap();
            assert!((-1.0 - 1e-6..=1.0 + 1e-6).contains(&ssp));
        }
        let pa: f64 = cols[5].parse().unwrap();
        let pb: f64 = cols[6].parse().unwrap();
        let pp: f64 = cols[7].parse().unwrap();
        let pc: f64 = cols[8].parse().unwrap();
        assert!(
            ((pa + pb + pp + pc) - 1.0).abs() < 1e-4,
            "4-state δ2D populations must sum to 1.0, got {}",
            pa + pb + pp + pc
        );
        data_rows += 1;
    }
    assert_eq!(data_rows, 76, "expected 76 ubiquitin residues in TSV");

    // 2. JSON output via -o on BMRB 51984 p53 IDP with explicit --temp-k and --ph
    let json_path = dir.join("p53_propensity.json");
    let out_json = Command::new(nmrassign_bin())
        .args([
            "propensity",
            "--shifts",
            bmr51984.to_str().unwrap(),
            "--temp-k",
            "298.15",
            "--ph",
            "6.0",
            "--format",
            "json",
            "-o",
            json_path.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign propensity --format json");
    assert!(
        out_json.status.success(),
        "propensity json failed: {}",
        String::from_utf8_lossy(&out_json.stderr)
    );
    let json_text = fs::read_to_string(&json_path).expect("read propensity JSON");
    assert!(
        json_text.trim_start().starts_with('{')
            && json_text.trim_end().ends_with('}')
            && json_text.contains("\"is_predominantly_disordered\": true")
            && json_text.contains("\"temperature_kelvin\": 298.15")
            && json_text.contains("\"effective_ph\": 6.0")
            && json_text.matches("\"seq_id\":").count() == 102,
        "unexpected propensity JSON payload:\n{}",
        &json_text[..json_text.len().min(600)]
    );

    // 3. HTML quality report with --idp-propensity
    let html_path = dir.join("p53_report.html");
    let out_rep = Command::new(nmrassign_bin())
        .args([
            "report",
            "--star",
            bmr51984.to_str().unwrap(),
            "--idp-propensity",
            "--temp-k",
            "298.15",
            "--ph",
            "6.0",
            "--out",
            html_path.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign report --idp-propensity");
    assert!(
        out_rep.status.success(),
        "report --idp-propensity failed: {}",
        String::from_utf8_lossy(&out_rep.stderr)
    );
    let html = fs::read_to_string(&html_path).expect("read report HTML");
    assert!(
        html.contains("section-idp-propensity")
            && html.contains("svg-idp-propensity-track")
            && html.contains("p_PPII"),
        "expected dedicated #section-idp-propensity SVG panel in HTML report"
    );

    // 4. Pipeline --regime idp-13c auto-enables IDP propensity report & propensity.tsv
    let truth_p53 = read_star_all_atoms(&bmr51984, None).unwrap();
    let mut mini_ref = Reference::new(
        truth_p53.sequence[..18].to_vec(),
        truth_p53.seq_ids[..18].to_vec(),
    );
    for (&(k, ref atom), &v) in &truth_p53.shifts {
        if k < 18 {
            mini_ref.shifts.insert((k, atom.clone()), v);
        }
    }
    let mini_star = dir.join("mini_p53.str");
    write_star(&mini_star, &mini_ref, None).unwrap();
    let pipe_dir = dir.join("pipe_idp_out");
    let out_pipe = Command::new(nmrassign_bin())
        .args([
            "pipeline",
            "--regime",
            "idp-13c",
            "--star",
            mini_star.to_str().unwrap(),
            "--engine",
            "cpu",
            "--out-dir",
            pipe_dir.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign pipeline --regime idp-13c");
    assert!(
        out_pipe.status.success(),
        "pipeline --regime idp-13c failed:\n{}",
        String::from_utf8_lossy(&out_pipe.stderr)
    );
    assert!(pipe_dir.join("propensity.tsv").exists());
    let pipe_html = fs::read_to_string(pipe_dir.join("report.html")).unwrap();
    assert!(pipe_html.contains("section-idp-propensity"));

    // 5. NEF input parity: nmrassign propensity --shifts <result.nef> --format tsv
    let nef_path = pipe_dir.join("result.nef");
    assert!(nef_path.exists(), "expected pipeline to emit result.nef");
    let out_nef_prop = Command::new(nmrassign_bin())
        .args([
            "propensity",
            "--shifts",
            nef_path.to_str().unwrap(),
            "--format",
            "tsv",
        ])
        .output()
        .expect("run nmrassign propensity --shifts result.nef");
    assert!(
        out_nef_prop.status.success(),
        "propensity on .nef failed: {}",
        String::from_utf8_lossy(&out_nef_prop.stderr)
    );
    let nef_tsv_rows = String::from_utf8_lossy(&out_nef_prop.stdout)
        .lines()
        .filter(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with('#') && !t.starts_with("seq_id\t")
        })
        .count();
    assert_eq!(
        nef_tsv_rows, 18,
        "expected 18 residue propensity rows from result.nef"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_cli_sidechain_aromatic_preserves_unassigned_user_ring_shifts_p1_12() {
    let root = repo_root();
    let star_in = root.join("data/fixtures/synthetic_ubiquitin.str");
    let raw_ref = read_star_all_atoms(&star_in, None).expect("read ubiquitin star");
    let sc_priors = SidechainPriors::empirical_bmrb();
    let seed = 2026u64;

    let all_ref = synthesize_all_atom_reference(&raw_ref, &sc_priors, seed);
    let his_map = default_his_tautomer_map(&all_ref.sequence, seed);
    let truth_ref = synthesize_aromatic_reference(&all_ref, &sc_priors, &his_map, seed);

    let dir = make_temp_dir("p1_12_aromatic_preserve");

    // Simulate aromatic peaks ONLY for PHE4 (residue index 3, Seq_ID 4), stripping all ring shifts
    // on PHE45 (index 44, Seq_ID 45), TYR59 (index 58), and HIS68 (index 67, Seq_ID 68) before simulating
    // so PHE45 and HIS68 remain completely unobserved by the aromatic solver!
    let mut partial_truth = truth_ref.clone();
    partial_truth.shifts.retain(|&(k, ref a), _| {
        if k == 44 || k == 58 || k == 67 {
            matches!(
                a.as_str(),
                "H" | "N" | "CA" | "CB" | "C" | "HA" | "HB2" | "HB3"
            )
        } else {
            true
        }
    });
    let (arom_pks, _) = simulate_aromatic_peaks(
        &partial_truth,
        &BTreeMap::new(),
        &AromaticSimConfig {
            seed,
            ..AromaticSimConfig::default()
        },
    );
    let mut c13_hsqc = Vec::new();
    let mut hd_pks = Vec::new();
    let mut he_pks = Vec::new();
    for p in arom_pks {
        match p.exp.as_str() {
            "AROM_C13HSQC" => c13_hsqc.push(p),
            "HBCBCGCDHD" => hd_pks.push(p),
            "HBCBCGCDCEHE" => he_pks.push(p),
            _ => {}
        }
    }

    let arom_hsqc_tsv = dir.join("arom_c13hsqc.tsv");
    let hd_tsv = dir.join("hbcbgcdhd.tsv");
    let he_tsv = dir.join("hbcbgcdce.tsv");
    write_peaks(&arom_hsqc_tsv, &c13_hsqc).unwrap();
    write_peaks(&hd_tsv, &hd_pks).unwrap();
    write_peaks(&he_tsv, &he_pks).unwrap();

    // Build user input .str containing backbone + beta shifts PLUS pre-existing user-curated ring shifts
    // and custom Ambiguity_codes on unassigned PHE45 (idx 44) and HIS68 (idx 67)!
    let mut user_in_ref = all_ref.clone();
    // Ensure PHE4 has no prior ring shifts so we verify the solver assigns PHE4 fresh
    user_in_ref.shifts.retain(|&(k, ref a), _| {
        !(k == 3
            && matches!(
                a.as_str(),
                "CD1" | "CD2" | "HD1" | "HD2" | "CE1" | "CE2" | "HE1" | "HE2" | "CZ" | "HZ"
            ))
    });
    user_in_ref.shifts.insert((44, "CD1".to_string()), 131.425);
    user_in_ref.shifts.insert((44, "HD1".to_string()), 7.265);
    user_in_ref.shifts.insert((67, "CE1".to_string()), 137.855);
    user_in_ref.shifts.insert((67, "HE1".to_string()), 8.145);

    user_in_ref
        .ambiguity_codes
        .insert((44, "CD1".to_string()), 2u8);
    user_in_ref
        .ambiguity_codes
        .insert((44, "HD1".to_string()), 2u8);
    user_in_ref
        .ambiguity_codes
        .insert((67, "CE1".to_string()), 1u8);
    user_in_ref
        .ambiguity_codes
        .insert((67, "HE1".to_string()), 1u8);

    let user_star = dir.join("user_curated_input.str");
    write_star(&user_star, &user_in_ref, None).expect("write user curated star");

    // Case A: Standalone aromatic mode (--aromatic without aliphatic --peaks)
    let out_star_a = dir.join("out_arom_only.str");
    let res_a = Command::new(nmrassign_bin())
        .args([
            "sidechain",
            "--star",
            user_star.to_str().unwrap(),
            "--aromatic",
            "--arom-c13hsqc",
            arom_hsqc_tsv.to_str().unwrap(),
            "--hbcbgcdhd",
            hd_tsv.to_str().unwrap(),
            "--hbcbgcdce",
            he_tsv.to_str().unwrap(),
            "--out",
            out_star_a.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign sidechain --aromatic");
    assert!(
        res_a.status.success(),
        "sidechain --aromatic failed:\n{}",
        String::from_utf8_lossy(&res_a.stderr)
    );

    // Case B: Combined aliphatic --peaks + --aromatic mode
    let aliph_peaks = simulate_sidechain_peaks(
        &all_ref,
        &SidechainSimConfig {
            include_aromatics: false,
            seed,
            ..SidechainSimConfig::default()
        },
    );
    let aliph_tsv = dir.join("aliph_peaks.tsv");
    write_peaks(&aliph_tsv, &aliph_peaks).unwrap();
    let out_star_b = dir.join("out_aliph_plus_arom.str");
    let res_b = Command::new(nmrassign_bin())
        .args([
            "sidechain",
            "--star",
            user_star.to_str().unwrap(),
            "--peaks",
            aliph_tsv.to_str().unwrap(),
            "--aromatic",
            "--arom-c13hsqc",
            arom_hsqc_tsv.to_str().unwrap(),
            "--hbcbgcdhd",
            hd_tsv.to_str().unwrap(),
            "--hbcbgcdce",
            he_tsv.to_str().unwrap(),
            "--out",
            out_star_b.to_str().unwrap(),
        ])
        .output()
        .expect("run nmrassign sidechain --peaks + --aromatic");
    assert!(
        res_b.status.success(),
        "sidechain --peaks + --aromatic failed:\n{}",
        String::from_utf8_lossy(&res_b.stderr)
    );

    for (label, out_p) in [("arom-only", &out_star_a), ("aliph+arom", &out_star_b)] {
        let out_ref = read_star_all_atoms(out_p, None).unwrap();
        let out_txt = fs::read_to_string(out_p).unwrap();
        let amb_map = parse_star_ambiguity_codes(&out_txt);

        // Observed PHE4 (idx 3) was assigned by the aromatic solver
        assert!(
            out_ref.shifts.contains_key(&(3, "CD1".to_string()))
                && out_ref.shifts.contains_key(&(3, "HD1".to_string())),
            "[{label}] expected observed PHE4 ring shifts to be assigned"
        );

        // Unobserved PHE45 (idx 44, Seq_ID 45) & HIS68 (idx 67, Seq_ID 68) preserved 100% intact!
        assert!(
            (out_ref.shifts[&(44, "CD1".to_string())] - 131.425).abs() < 1e-2,
            "[{label}] PHE45 CD1 user shift wiped or altered!"
        );
        assert!(
            (out_ref.shifts[&(44, "HD1".to_string())] - 7.265).abs() < 1e-2,
            "[{label}] PHE45 HD1 user shift wiped or altered!"
        );
        assert!(
            (out_ref.shifts[&(67, "CE1".to_string())] - 137.855).abs() < 1e-2,
            "[{label}] HIS68 CE1 user shift wiped or altered!"
        );
        assert!(
            (out_ref.shifts[&(67, "HE1".to_string())] - 8.145).abs() < 1e-2,
            "[{label}] HIS68 HE1 user shift wiped or altered!"
        );
        assert_eq!(amb_map.get(&(45, "CD1".to_string())).copied(), Some(2));
        assert_eq!(amb_map.get(&(45, "HD1".to_string())).copied(), Some(2));
        assert_eq!(amb_map.get(&(68, "CE1".to_string())).copied(), Some(1));
        assert_eq!(amb_map.get(&(68, "HE1".to_string())).copied(), Some(1));
    }

    let _ = fs::remove_dir_all(&dir);
}
