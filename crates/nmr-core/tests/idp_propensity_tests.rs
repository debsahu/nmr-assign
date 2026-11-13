//! Non-circular unit and real-BMRB integration tests for `nmr_core::idp_propensity` (`T11.3`)
//! and the `report.rs` stacked 4-state `δ2D` + `SSP` vector SVG track.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use nmr_core::idp_propensity::{
    compute_idp_propensity, format_propensity_json, format_propensity_tsv, ConformationalState,
    PropensityConfig, PropensityReport, UnassignedPriorMode,
};
use nmr_core::nmrstar::{read_star, Reference};
use nmr_core::report::{
    generate_quality_report_html, render_idp_propensity_svg, ReportInput, SecondaryStructureClass,
};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn test_simplex_normalization_and_unassigned_residue_handling() {
    let ubq = read_star(workspace_root().join("data/benchmark/bmr17769.str"), None)
        .expect("read bmr17769.str");
    let asyn = read_star(workspace_root().join("data/benchmark/bmr6968.str"), None)
        .expect("read bmr6968.str");
    let urea = read_star(workspace_root().join("data/benchmark/bmr4375.str"), None)
        .expect("read bmr4375.str");

    for (label, ref_data) in [("ubq", &ubq), ("asyn", &asyn), ("urea", &urea)] {
        let rep = compute_idp_propensity(ref_data, &PropensityConfig::default());
        assert_eq!(rep.residues.len(), ref_data.sequence.len());
        for r in &rep.residues {
            let s = r.population_sum();
            assert!(
                (s - 1.0).abs() <= 1e-9,
                "{label} seq_id {} sum={s:.12} deviated from 1.0",
                r.seq_id
            );
            assert!(
                (0.0..=1.0).contains(&r.p_alpha)
                    && (0.0..=1.0).contains(&r.p_beta)
                    && (0.0..=1.0).contains(&r.p_ppii)
                    && (0.0..=1.0).contains(&r.p_coil),
                "{label} seq_id {} probability out of [0,1]",
                r.seq_id
            );
            if r.residue == "GLY" {
                assert!(
                    !r.secondary_shifts.contains_key("CB"),
                    "GLY {} must never carry a CB secondary shift",
                    r.seq_id
                );
            }
        }
    }

    // Create a sparse copy with completely unassigned gap (seq_index 25..=32)
    let mut gapped: Reference = asyn.clone();
    gapped.shifts.retain(|&(k, _), _| !(25..=32).contains(&k));

    let rep_coil = compute_idp_propensity(
        &gapped,
        &PropensityConfig::default().with_unassigned_prior(UnassignedPriorMode::CoilBaseline),
    );
    let rep_unif = compute_idp_propensity(
        &gapped,
        &PropensityConfig::default().with_unassigned_prior(UnassignedPriorMode::Uniform),
    );

    for k in 25..=32 {
        let rc = &rep_coil.residues[k];
        assert!(!rc.has_data, "gap residue k={k} must set has_data = false");
        assert_eq!(rc.n_observed_nuclei, 0);
        assert!(rc.ssp_score.is_none());
        assert!((rc.population_sum() - 1.0).abs() <= 1e-9);
        assert!((rc.p_coil - 0.65).abs() < 1e-9 && (rc.p_ppii - 0.25).abs() < 1e-9);

        let ru = &rep_unif.residues[k];
        assert!(!ru.has_data);
        assert!((ru.population_sum() - 1.0).abs() <= 1e-9);
        assert!((ru.p_alpha - 0.25).abs() < 1e-9 && (ru.p_coil - 0.25).abs() < 1e-9);
    }
}

#[derive(Clone, Debug)]
struct DsspBackboneResidue {
    seq_id: i32,
    res_name: String,
    n: Option<[f64; 3]>,
    ca: Option<[f64; 3]>,
    c: Option<[f64; 3]>,
    o: Option<[f64; 3]>,
    h: Option<[f64; 3]>,
}

/// Clean-room Kabsch & Sander (*Biopolymers* 1983, 22, 2577–2637) secondary-structure
/// classification strictly from first-model PDB `ATOM` coordinates (`N, CA, C, O, H`),
/// reconstructing amide `H` geometrically when absent (`E < -0.5 kcal/mol`), with ZERO
/// `HELIX`/`SHEET` header reads and ZERO hardcoded residue-range arrays.
fn classify_dssp_kabsch_sander_first_model(
    pdb_text: &str,
) -> Vec<(i32, String, SecondaryStructureClass)> {
    #[inline]
    fn ascii_col(bytes: &[u8], lo: usize, hi: usize) -> Option<&str> {
        let s = bytes.get(lo..hi)?;
        s.is_ascii().then(|| std::str::from_utf8(s).ok()).flatten()
    }
    #[inline]
    fn v_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    #[inline]
    fn v_norm(a: [f64; 3]) -> f64 {
        (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
    }
    #[inline]
    fn v_unit(a: [f64; 3]) -> Option<[f64; 3]> {
        let d = v_norm(a);
        (d > 1e-9).then_some([a[0] / d, a[1] / d, a[2] / d])
    }

    let mut active_chain: Option<u8> = None;
    let mut order: Vec<(i32, u8)> = Vec::new();
    let mut res_map: BTreeMap<(i32, u8), DsspBackboneResidue> = BTreeMap::new();
    let mut atom_is_primary: BTreeMap<(i32, u8, &'static str), bool> = BTreeMap::new();

    for line in pdb_text.lines() {
        if line.starts_with("ENDMDL") {
            break;
        }
        let bytes = line.as_bytes();
        let is_atom = line.starts_with("ATOM  ");
        let is_mse = line.starts_with("HETATM")
            && ascii_col(bytes, 17, 20).is_some_and(|r| r.trim().eq_ignore_ascii_case("MSE"));
        if (!is_atom && !is_mse) || bytes.len() < 54 {
            continue;
        }
        let chain_b = bytes[21];
        if !chain_b.is_ascii() {
            continue;
        }
        if let Some(c0) = active_chain {
            if chain_b != c0 {
                continue;
            }
        } else {
            active_chain = Some(chain_b);
        }

        let alt = bytes[16];
        let is_primary = matches!(alt, b' ' | b'A' | b'1');
        let (Some(atom_s), Some(res_s), Some(seq_s), Some(xs), Some(ys), Some(zs)) = (
            ascii_col(bytes, 12, 16),
            ascii_col(bytes, 17, 20),
            ascii_col(bytes, 22, 26),
            ascii_col(bytes, 30, 38),
            ascii_col(bytes, 38, 46),
            ascii_col(bytes, 46, 54),
        ) else {
            continue;
        };
        let Ok(seq_id) = seq_s.trim().parse::<i32>() else {
            continue;
        };
        let (Ok(x), Ok(y), Ok(z)) = (
            xs.trim().parse::<f64>(),
            ys.trim().parse::<f64>(),
            zs.trim().parse::<f64>(),
        ) else {
            continue;
        };
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            continue;
        }
        let icode = bytes[26];
        let key = (seq_id, icode);
        let res_up = if is_mse {
            "MET".to_string()
        } else {
            res_s.trim().to_ascii_uppercase()
        };
        let entry = res_map.entry(key).or_insert_with(|| {
            order.push(key);
            DsspBackboneResidue {
                seq_id,
                res_name: res_up,
                n: None,
                ca: None,
                c: None,
                o: None,
                h: None,
            }
        });

        let aname = atom_s.trim().to_ascii_uppercase();
        let slot: Option<&'static str> = match aname.as_str() {
            "N" => Some("N"),
            "CA" => Some("CA"),
            "C" => Some("C"),
            "O" => Some("O"),
            "H" | "HN" => Some("H"),
            _ => None,
        };
        if let Some(tag) = slot {
            let prev_prim = atom_is_primary.get(&(seq_id, icode, tag)).copied();
            if prev_prim.is_none() || (prev_prim == Some(false) && is_primary) {
                atom_is_primary.insert((seq_id, icode, tag), is_primary);
                let pos = [x, y, z];
                match tag {
                    "N" => entry.n = Some(pos),
                    "CA" => entry.ca = Some(pos),
                    "C" => entry.c = Some(pos),
                    "O" => entry.o = Some(pos),
                    "H" => entry.h = Some(pos),
                    _ => {}
                }
            }
        }
    }

    let mut residues: Vec<DsspBackboneResidue> = order
        .into_iter()
        .filter_map(|k| res_map.remove(&k))
        .collect();
    let n_res = residues.len();

    // Reconstruct amide H where absent on non-Proline internal residues (Kabsch & Sander 1983: N-H || O_{i-1}->C_{i-1})
    for i in 1..n_res {
        if residues[i].h.is_some() || residues[i].res_name == "PRO" {
            continue;
        }
        let Some(n_i) = residues[i].n else {
            continue;
        };
        if let (Some(c_prev), Some(o_prev)) = (residues[i - 1].c, residues[i - 1].o) {
            if let Some(u_co) = v_unit(v_sub(c_prev, o_prev)) {
                residues[i].h = Some([
                    n_i[0] + 1.008 * u_co[0],
                    n_i[1] + 1.008 * u_co[1],
                    n_i[2] + 1.008 * u_co[2],
                ]);
                continue;
            }
        }
        if let (Some(c_prev), Some(ca_i)) = (residues[i - 1].c, residues[i].ca) {
            if let (Some(u1), Some(u2)) = (v_unit(v_sub(n_i, c_prev)), v_unit(v_sub(n_i, ca_i))) {
                if let Some(bis) = v_unit([u1[0] + u2[0], u1[1] + u2[1], u1[2] + u2[2]]) {
                    residues[i].h = Some([
                        n_i[0] + 1.008 * bis[0],
                        n_i[1] + 1.008 * bis[1],
                        n_i[2] + 1.008 * bis[2],
                    ]);
                }
            }
        }
    }

    // Kabsch & Sander (1983) electrostatic H-bond matrix: E(acceptor=i, donor=j) < -0.5 kcal/mol
    let hbond = |acc: usize, don: usize| -> bool {
        if acc >= n_res || don >= n_res || acc.abs_diff(don) < 2 {
            return false;
        }
        if residues[don].res_name == "PRO" {
            return false;
        }
        let (Some(c_i), Some(o_i), Some(n_j), Some(h_j)) = (
            residues[acc].c,
            residues[acc].o,
            residues[don].n,
            residues[don].h,
        ) else {
            return false;
        };
        let r_on = v_norm(v_sub(o_i, n_j));
        let r_ch = v_norm(v_sub(c_i, h_j));
        let r_oh = v_norm(v_sub(o_i, h_j));
        let r_cn = v_norm(v_sub(c_i, n_j));
        if r_on < 0.5 || r_ch < 0.5 || r_oh < 0.5 || r_cn < 0.5 {
            return false;
        }
        let e_kcal = 0.084 * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn) * 332.0;
        e_kcal < -0.5
    };

    let turn_n = |span: usize| -> Vec<bool> {
        (0..n_res)
            .map(|i| i + span < n_res && hbond(i, i + span))
            .collect()
    };
    let turn3 = turn_n(3);
    let turn4 = turn_n(4);
    let turn5 = turn_n(5);

    let mut ss = vec![SecondaryStructureClass::Coil; n_res];

    // 1. Alpha-helices ('H'): consecutive 4-turns at i-1 and i assign i..=i+3
    for i in 1..n_res.saturating_sub(3) {
        if turn4[i - 1] && turn4[i] {
            for item in ss.iter_mut().take(i + 4).skip(i) {
                *item = SecondaryStructureClass::Helix;
            }
        }
    }

    // 2. Parallel & antiparallel beta-bridges / beta-ladders ('E'/'B')
    let mut is_bridge = vec![false; n_res];
    for i in 0..n_res {
        for j in (i + 3)..n_res {
            let par_1 = i >= 1 && j + 1 < n_res && hbond(i - 1, j) && hbond(j, i + 1);
            let par_2 = j >= 1 && i + 1 < n_res && hbond(j - 1, i) && hbond(i, j + 1);
            let anti_1 = hbond(i, j) && hbond(j, i);
            let anti_2 = i >= 1
                && j >= 1
                && i + 1 < n_res
                && j + 1 < n_res
                && hbond(i - 1, j + 1)
                && hbond(j - 1, i + 1);
            if par_1 || par_2 || anti_1 || anti_2 {
                is_bridge[i] = true;
                is_bridge[j] = true;
            }
        }
    }
    for i in 0..n_res {
        if is_bridge[i] && ss[i] == SecondaryStructureClass::Coil {
            ss[i] = SecondaryStructureClass::Strand;
        }
    }

    // 3. 3_10-helices ('G') and pi-helices ('I') on remaining unassigned turns
    for i in 1..n_res.saturating_sub(2) {
        if turn3[i - 1]
            && turn3[i]
            && ss[i..=i + 2]
                .iter()
                .all(|&c| c == SecondaryStructureClass::Coil)
        {
            for item in ss.iter_mut().take(i + 3).skip(i) {
                *item = SecondaryStructureClass::Helix;
            }
        }
    }
    for i in 1..n_res.saturating_sub(4) {
        if turn5[i - 1]
            && turn5[i]
            && ss[i..=i + 4]
                .iter()
                .all(|&c| c == SecondaryStructureClass::Coil)
        {
            for item in ss.iter_mut().take(i + 5).skip(i) {
                *item = SecondaryStructureClass::Helix;
            }
        }
    }

    residues
        .into_iter()
        .zip(ss)
        .map(|(r, cls)| (r.seq_id, r.res_name, cls))
        .collect()
}

#[test]
fn test_folded_ubiquitin_bmr17769_vs_1d3z_secondary_structure() {
    let ubq = read_star(workspace_root().join("data/benchmark/bmr17769.str"), None)
        .expect("read bmr17769.str");
    let pdb_txt =
        fs::read_to_string(workspace_root().join("data/pdb_refs/1d3z.pdb")).expect("read 1d3z.pdb");
    let dssp = classify_dssp_kabsch_sander_first_model(&pdb_txt);
    assert_eq!(
        dssp.len(),
        ubq.sequence.len(),
        "1D3Z model-1 residue count must equal Ubiquitin sequence length (76)"
    );

    let rep = compute_idp_propensity(&ubq, &PropensityConfig::default());
    assert!(
        !rep.is_predominantly_disordered,
        "folded Ubiquitin must not be classified as disordered"
    );

    let idx_of = |cls: SecondaryStructureClass| -> usize {
        match cls {
            SecondaryStructureClass::Helix => 0,
            SecondaryStructureClass::Strand => 1,
            SecondaryStructureClass::Coil => 2,
        }
    };
    let mut conf_mat = [[0usize; 3]; 3]; // [truth_dssp][pred_d2d]
    let mut total_structured = 0usize;
    let mut matched_structured = 0usize;
    let mut non_adj_loop_total = 0usize;
    let mut non_adj_loop_fp = 0usize;
    let mut all_loop_total = 0usize;
    let mut all_loop_fp = 0usize;
    let mut helix_ssp_vals = Vec::new();
    let mut strand_ssp_vals = Vec::new();

    for k in 0..dssp.len() {
        let (sid, ref pdb_res, truth_cls) = dssp[k];
        assert_eq!(
            pdb_res, &ubq.sequence[k],
            "sequence mismatch at k={k} (seq_id={sid})"
        );
        let pred_cls = rep.residues[k].dominant_state.to_three_state();
        conf_mat[idx_of(truth_cls)][idx_of(pred_cls)] += 1;

        match truth_cls {
            SecondaryStructureClass::Helix => {
                total_structured += 1;
                if pred_cls == SecondaryStructureClass::Helix {
                    matched_structured += 1;
                }
                if let Some(s) = rep.residues[k].ssp_score {
                    helix_ssp_vals.push(s);
                }
            }
            SecondaryStructureClass::Strand => {
                total_structured += 1;
                if pred_cls == SecondaryStructureClass::Strand {
                    matched_structured += 1;
                }
                if let Some(s) = rep.residues[k].ssp_score {
                    strand_ssp_vals.push(s);
                }
            }
            SecondaryStructureClass::Coil => {
                all_loop_total += 1;
                let pred_structured = pred_cls != SecondaryStructureClass::Coil;
                if pred_structured {
                    all_loop_fp += 1;
                }
                let prev_coil = k == 0 || dssp[k - 1].2 == SecondaryStructureClass::Coil;
                let next_coil =
                    k + 1 >= dssp.len() || dssp[k + 1].2 == SecondaryStructureClass::Coil;
                if prev_coil && next_coil {
                    non_adj_loop_total += 1;
                    if pred_structured {
                        non_adj_loop_fp += 1;
                    }
                }
            }
        }
    }

    let structured_recall = (matched_structured as f64) / (total_structured.max(1) as f64);
    let non_adj_loop_fpr = (non_adj_loop_fp as f64) / (non_adj_loop_total.max(1) as f64);
    let all_loop_fpr = (all_loop_fp as f64) / (all_loop_total.max(1) as f64);

    println!(
        "Ubiquitin (bmr17769 vs 1D3Z Kabsch-Sander DSSP, all {} aa):\n  Confusion [DSSP \\ d2D] (H, E, C): H={:?}, E={:?}, C={:?}\n  Structured recall = {matched_structured}/{total_structured} ({:.2}%), non-adjacent loop FPR = {non_adj_loop_fp}/{non_adj_loop_total} ({:.2}%), all-loop FPR = {all_loop_fp}/{all_loop_total} ({:.2}%)",
        dssp.len(),
        conf_mat[0],
        conf_mat[1],
        conf_mat[2],
        structured_recall * 100.0,
        non_adj_loop_fpr * 100.0,
        all_loop_fpr * 100.0
    );

    assert!(
        structured_recall >= 0.88,
        "Ubiquitin DSSP structured recall {:.2}% < 88.0% (matrix={conf_mat:?})",
        structured_recall * 100.0
    );
    assert!(
        non_adj_loop_fpr <= 0.15,
        "Ubiquitin DSSP non-adjacent loop false-positive rate {:.2}% > 15.0% (matrix={conf_mat:?})",
        non_adj_loop_fpr * 100.0
    );

    let helix_ssp_mean = helix_ssp_vals.iter().sum::<f64>() / (helix_ssp_vals.len().max(1) as f64);
    let strand_ssp_mean =
        strand_ssp_vals.iter().sum::<f64>() / (strand_ssp_vals.len().max(1) as f64);
    assert!(
        helix_ssp_mean >= 0.35,
        "DSSP helix mean SSP {helix_ssp_mean:.3} should be strongly positive"
    );
    assert!(
        strand_ssp_mean <= -0.30,
        "DSSP strand mean SSP {strand_ssp_mean:.3} should be strongly negative"
    );
}

#[test]
fn test_nterminal_proline_k1_k2_uses_trans_into_pro_and_suppresses_false_helix() {
    // Compare 20-residue sequences (N-terminal helix k=0..5 + 14-residue random-coil tail so global
    // RC offset estimation remains near 0.0) with Pro at k=1, k=2, and k=1..=2 against an all-Ala N-cap control.
    let make_seq = |pro_positions: &[usize]| -> Vec<String> {
        (0..20)
            .map(|k| {
                if k == 0 {
                    "MET".to_string()
                } else if pro_positions.contains(&k) {
                    "PRO".to_string()
                } else {
                    "ALA".to_string()
                }
            })
            .collect()
    };
    let cases = [
        ("pro_at_k1", make_seq(&[1]), 1usize),
        ("pro_at_k2", make_seq(&[2]), 2usize),
        ("pro_at_k1_k2", make_seq(&[1, 2]), 1usize),
    ];
    let ctrl_seq = make_seq(&[]);
    let seq_ids: Vec<i32> = (1..=20).collect();

    let populate_helical_ncap = |seq: &[String], pro_target: usize| -> Reference {
        let mut r = Reference::new(seq.to_vec(), seq_ids.clone());
        for (k, aa) in seq.iter().enumerate() {
            let is_pro = aa == "PRO";
            let next_pro = seq.get(k + 1).is_some_and(|nx| nx == "PRO");
            let base_ca = if is_pro {
                63.28
            } else if aa == "MET" {
                54.62
            } else {
                52.48
            } - if next_pro { 1.95 } else { 0.0 };
            let base_cb = if is_pro {
                32.02
            } else if aa == "MET" {
                32.78
            } else {
                19.07
            };
            let base_c = if is_pro {
                177.26
            } else if aa == "MET" {
                172.78
            } else {
                177.82
            } - if next_pro { 2.05 } else { 0.0 };

            // Upstream N-terminal residues (0..pro_target) carry strong helical shifts (+3.60 ppm CA),
            // pro_target (k=1 or k=2) carries moderate helical shifts (+1.65 ppm CA), and k > pro_target
            // is random-coil tail (0.0 ppm) so incoming trans_into_pro at k=1 and k=2 is isolated cleanly.
            let (d_ca, d_cb, d_c) = if k > pro_target {
                (0.0, 0.0, 0.0)
            } else if k == pro_target || is_pro {
                (1.85, -0.70, 1.30)
            } else {
                (3.80, -1.70, 2.65)
            };
            r.shifts.insert((k, "CA".to_string()), base_ca + d_ca);
            r.shifts.insert((k, "CB".to_string()), base_cb + d_cb);
            r.shifts.insert((k, "C".to_string()), base_c + d_c);
        }
        r
    };

    for (label, seq, pro_idx) in cases {
        let rep_ctrl = compute_idp_propensity(
            &populate_helical_ncap(&ctrl_seq, pro_idx),
            &PropensityConfig::default(),
        );
        let rep_pro = compute_idp_propensity(
            &populate_helical_ncap(&seq, pro_idx),
            &PropensityConfig::default(),
        );
        let rp = &rep_pro.residues[pro_idx];
        let rc = &rep_ctrl.residues[pro_idx];
        println!(
            "{label} (k={pro_idx}): Pro p_alpha={:.4}, p_ppii={:.4}, state={:?} vs Ala-ctrl p_alpha={:.4}, state={:?}",
            rp.p_alpha, rp.p_ppii, rp.dominant_state, rc.p_alpha, rc.dominant_state
        );
        assert!(
            rp.p_alpha + 0.25 < rc.p_alpha,
            "{label}: trans_into_pro at k={pro_idx} must strongly suppress false helix propagation (Pro p_alpha={:.4} vs Ala p_alpha={:.4})",
            rp.p_alpha,
            rc.p_alpha
        );
        assert_ne!(
            rp.dominant_state,
            ConformationalState::Alpha,
            "{label}: N-terminal Pro at k={pro_idx} must not be falsely classified as Alpha-Helix"
        );
    }
}

#[test]
fn test_held_out_idps_darpp32_bmr15176_and_cbp_id4_bmr26639() {
    let darpp32 = read_star(workspace_root().join("data/benchmark/bmr15176.str"), None)
        .expect("read bmr15176.str (DARPP-32)");
    let cbp_id4 = read_star(
        workspace_root().join("data/fixtures/bmr26639_idp_cbp_id4.str"),
        None,
    )
    .expect("read bmr26639_idp_cbp_id4.str (CBP-ID4)");
    assert_eq!(darpp32.sequence.len(), 119);
    assert_eq!(cbp_id4.sequence.len(), 207);

    for (label, ref_data, cfg) in [
        ("DARPP-32 (bmr15176)", &darpp32, PropensityConfig::default()),
        (
            "CBP-ID4 (bmr26639, 21.7% Pro, default 298K)",
            &cbp_id4,
            PropensityConfig::default(),
        ),
        (
            "CBP-ID4 (bmr26639, 21.7% Pro, 308K)",
            &cbp_id4,
            PropensityConfig::default().with_temperature_kelvin(308.0),
        ),
    ] {
        let rep = compute_idp_propensity(ref_data, &cfg);
        let disordered_pop = rep.mean_p_coil + rep.mean_p_ppii;
        println!(
            "{label}: n_data={}/{}, is_disordered={}, mean(p_coil+p_ppii)={:.4} (coil={:.4}, ppii={:.4}, alpha={:.4}, beta={:.4}), mean|SSP|={:.4}",
            rep.n_residues_with_data,
            rep.residues.len(),
            rep.is_predominantly_disordered,
            disordered_pop,
            rep.mean_p_coil,
            rep.mean_p_ppii,
            rep.mean_p_alpha,
            rep.mean_p_beta,
            rep.mean_abs_ssp
        );

        assert!(
            rep.is_predominantly_disordered,
            "{label} must satisfy is_predominantly_disordered == true"
        );
        assert!(
            disordered_pop >= 0.70,
            "{label}: mean(p_coil + p_ppii) = {disordered_pop:.4} < 0.70"
        );

        let mut rigid_run = 0usize;
        for r in &rep.residues {
            let sum = r.population_sum();
            assert!(
                (sum - 1.0).abs() <= 1e-9,
                "{label} seq_id {} simplex sum={sum:.12} deviated from 1.0",
                r.seq_id
            );
            if r.has_data
                && matches!(
                    r.dominant_state,
                    ConformationalState::Alpha | ConformationalState::Beta
                )
                && (r.p_alpha + r.p_beta) >= 0.55
            {
                rigid_run += 1;
                assert!(
                    rigid_run < 4,
                    "{label}: false rigid >=4 stretch detected ending at {}{}",
                    r.residue,
                    r.seq_id
                );
            } else {
                rigid_run = 0;
            }
        }
    }
}

#[test]
fn test_disordered_alpha_synuclein_bmr6968_and_urea_ubiquitin_bmr4375() {
    let asyn = read_star(workspace_root().join("data/benchmark/bmr6968.str"), None)
        .expect("read bmr6968.str");
    let urea = read_star(workspace_root().join("data/benchmark/bmr4375.str"), None)
        .expect("read bmr4375.str");

    let rep_asyn = compute_idp_propensity(&asyn, &PropensityConfig::default());
    let asyn_coil_ppii = rep_asyn.mean_p_coil + rep_asyn.mean_p_ppii;
    println!(
        "alpha-synuclein (bmr6968): mean(p_coil+p_ppii)={:.3} (coil={:.3}, ppii={:.3}, alpha={:.3}, beta={:.3}), mean|SSP|={:.3}",
        asyn_coil_ppii,
        rep_asyn.mean_p_coil,
        rep_asyn.mean_p_ppii,
        rep_asyn.mean_p_alpha,
        rep_asyn.mean_p_beta,
        rep_asyn.mean_abs_ssp
    );
    assert!(
        asyn_coil_ppii >= 0.70,
        "alpha-synuclein mean(p_coil + p_ppii) = {asyn_coil_ppii:.3} < 0.70"
    );
    assert!(
        rep_asyn.is_predominantly_disordered,
        "alpha-synuclein must be flagged predominantly disordered"
    );

    // Proline-rich C-terminal tail (105..=140) should exhibit higher mean p_ppii than N-terminal repeats (1..=60)
    let ppii_nterm: f64 = rep_asyn.residues[0..60]
        .iter()
        .map(|r| r.p_ppii)
        .sum::<f64>()
        / 60.0;
    let ppii_cterm: f64 = rep_asyn.residues[104..140]
        .iter()
        .map(|r| r.p_ppii)
        .sum::<f64>()
        / 36.0;
    assert!(
        ppii_cterm > ppii_nterm,
        "C-terminal Pro-rich tail p_ppii ({ppii_cterm:.3}) should exceed N-terminal ({ppii_nterm:.3})"
    );

    // 8M urea-unfolded Ubiquitin (bmr4375, pH 2): test both unlabelled auto-sensing and explicit pH 2.0
    for (mode, cfg) in [
        ("auto_ph", PropensityConfig::default()),
        (
            "explicit_ph2",
            PropensityConfig::default().with_sample_ph(2.0),
        ),
    ] {
        let rep_urea = compute_idp_propensity(&urea, &cfg);
        let urea_coil_ppii = rep_urea.mean_p_coil + rep_urea.mean_p_ppii;
        println!(
            "8M urea Ubiquitin (bmr4375, {mode}, eff_pH={:.2}): mean(p_coil+p_ppii)={:.3}, mean|SSP|={:.3}",
            rep_urea.effective_ph, urea_coil_ppii, rep_urea.mean_abs_ssp
        );
        assert!(
            rep_urea.effective_ph <= 3.2,
            "{mode}: acidic protonation on bmr4375 should yield eff_pH <= 3.2, got {:.2}",
            rep_urea.effective_ph
        );
        assert!(
            urea_coil_ppii >= 0.70,
            "{mode}: 8M urea Ubiquitin mean(p_coil + p_ppii) = {urea_coil_ppii:.3} < 0.70"
        );
        assert!(
            rep_urea.is_predominantly_disordered,
            "{mode}: 8M urea Ubiquitin must have zero false rigid secondary-structure elements"
        );

        // Explicitly check zero contiguous >=4 rigid helix/strand stretches
        let mut run = 0usize;
        for r in &rep_urea.residues {
            if r.has_data
                && matches!(
                    r.dominant_state,
                    ConformationalState::Alpha | ConformationalState::Beta
                )
                && (r.p_alpha + r.p_beta) >= 0.55
            {
                run += 1;
                assert!(
                    run < 4,
                    "{mode}: false rigid element detected ending at {}{}",
                    r.residue,
                    r.seq_id
                );
            } else {
                run = 0;
            }
        }
    }
}

#[test]
fn test_global_offset_correction_temperature_exports_and_report_svg() {
    let asyn = read_star(workspace_root().join("data/benchmark/bmr6968.str"), None)
        .expect("read bmr6968.str");
    let base_rep = compute_idp_propensity(&asyn, &PropensityConfig::default());
    let base_auto = compute_idp_propensity(
        &asyn,
        &PropensityConfig::default().with_auto_offset_correction(true),
    );

    // Apply synthetic TMS-like 13C offset (-2.70 ppm), 15N (+1.45 ppm), and 1H (-0.18 ppm)
    let mut shifted = asyn.clone();
    for ((_, atom), val) in shifted.shifts.iter_mut() {
        match atom.as_str() {
            "CA" | "CB" | "C" => *val -= 2.70,
            "N" => *val += 1.45,
            "H" => *val -= 0.18,
            _ => {}
        }
    }
    let corr_rep = compute_idp_propensity(
        &shifted,
        &PropensityConfig::default().with_auto_offset_correction(true),
    );
    assert!(
        ((corr_rep.applied_offsets.d_ca - base_auto.applied_offsets.d_ca) - (-2.70)).abs() <= 0.30
    );
    assert!(
        ((corr_rep.applied_offsets.d_c - base_auto.applied_offsets.d_c) - (-2.70)).abs() <= 0.35
    );
    assert!(((corr_rep.applied_offsets.d_n - base_auto.applied_offsets.d_n) - 1.45).abs() <= 0.45);
    assert!(
        ((corr_rep.applied_offsets.d_h - base_auto.applied_offsets.d_h) - (-0.18)).abs() <= 0.08
    );

    let max_pop_diff = base_auto
        .residues
        .iter()
        .zip(&corr_rep.residues)
        .map(|(a, b)| (a.p_coil - b.p_coil).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_pop_diff <= 0.08,
        "offset-corrected populations should match DSS baseline, max diff={max_pop_diff:.4}"
    );

    // Temperature adjustment check (278.15 K vs 318.15 K shifts amide H random coil by ~ -0.18 ppm)
    let rep_cold = compute_idp_propensity(
        &asyn,
        &PropensityConfig::default().with_temperature_kelvin(278.15),
    );
    let rep_warm = compute_idp_propensity(
        &asyn,
        &PropensityConfig::default().with_temperature_kelvin(318.15),
    );
    let dh_cold = rep_cold.residues[10].secondary_shifts["H"];
    let dh_warm = rep_warm.residues[10].secondary_shifts["H"];
    assert!(
        dh_warm > dh_cold + 0.12,
        "warming by +40 K lowers RC amide 1H, raising dH ({dh_cold:.3} -> {dh_warm:.3})"
    );

    // TSV & JSON round-trip
    let tsv = format_propensity_tsv(&base_rep);
    assert!(
        tsv.contains("seq_id\tresidue\thas_data\tn_nuclei\tssp\tp_alpha\tp_beta\tp_ppii\tp_coil")
    );
    assert_eq!(tsv.lines().count(), 142);

    let json = format_propensity_json(&base_rep).expect("serialize propensity JSON");
    let decoded: PropensityReport = serde_json::from_str(&json).expect("deserialize JSON");
    assert_eq!(decoded.residues.len(), 140);

    // Standalone SVG + HTML quality report integration (preserving default 5-SVG invariant)
    let svg_standalone = render_idp_propensity_svg(&base_rep);
    assert!(svg_standalone.contains("id=\"svg-idp-propensity\""));
    assert!(svg_standalone.contains("<polyline"));

    let conf: BTreeMap<usize, f64> = (1..asyn.sequence.len()).map(|k| (k, 0.90)).collect();
    let html_default =
        generate_quality_report_html(&ReportInput::new("asyn", asyn.clone(), conf.clone()));
    assert_eq!(
        html_default.matches("<svg ").count(),
        5,
        "default ReportInput::new must preserve exact 5-<svg> count"
    );
    assert!(html_default.contains("id=\"svg-idp-propensity-track\""));

    let html_explicit = generate_quality_report_html(
        &ReportInput::new("asyn", asyn, conf).with_propensity_report(base_rep),
    );
    assert_eq!(html_explicit.matches("<svg ").count(), 6);
    assert!(html_explicit.contains("id=\"section-idp-propensity\""));
    assert!(html_explicit.contains("id=\"svg-idp-propensity\""));
}

#[test]
fn test_estimate_global_rc_offsets_permutation_invariance() {
    use nmr_core::idp_propensity::estimate_global_rc_offsets;

    for star_rel in [
        "data/benchmark/bmr17769.str",
        "data/benchmark/bmr6968.str",
        "data/benchmark/bmr4375.str",
    ] {
        let orig = read_star(workspace_root().join(star_rel), None).expect("read star");
        let cfg = PropensityConfig::default().with_auto_offset_correction(true);
        let off_orig = estimate_global_rc_offsets(&orig, &cfg);

        // Construct deterministic Fisher-Yates permutation pi over 0..n (including termini & Prolines)
        let n = orig.sequence.len();
        let mut pi: Vec<usize> = (0..n).collect();
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for i in (1..n).rev() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = ((state >> 33) as usize) % (i + 1);
            pi.swap(i, j);
        }

        let mut permuted = orig.clone();
        permuted.shifts.clear();
        for ((k, atom), val) in &orig.shifts {
            permuted.shifts.insert((pi[*k], atom.clone()), *val);
        }

        let off_perm = estimate_global_rc_offsets(&permuted, &cfg);
        assert!(
            (off_orig.d_ca - off_perm.d_ca).abs() < 1e-12
                && (off_orig.d_cb - off_perm.d_cb).abs() < 1e-12
                && (off_orig.d_c - off_perm.d_c).abs() < 1e-12
                && (off_orig.d_ha - off_perm.d_ha).abs() < 1e-12
                && (off_orig.d_h - off_perm.d_h).abs() < 1e-12
                && (off_orig.d_n - off_perm.d_n).abs() < 1e-12,
            "{star_rel}: estimate_global_rc_offsets must be invariant (< 1e-12) under k -> pi(k), got orig={off_orig:?} vs perm={off_perm:?}"
        );
    }
}
