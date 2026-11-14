use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use nmr_core::dihedrals::{align_by_res_names, periodic_angle_abs_diff_deg};
use nmr_core::nef::{read_nef_str, write_nef_str, NefDocument};
use nmr_core::nmrstar::{read_star_all_atoms, Reference};
use nmr_core::torsion_db::{
    apply_homolog_firewall, classify_rama_basin, estimate_global_shift_offsets,
    format_torsions_json, format_torsions_json_with_options, format_torsions_tsv,
    format_torsions_tsv_with_options, is_ramachandran_outlier, needleman_wunsch_stats,
    predict_torsions, predict_torsions_with_firewall, sha256_hex, shares_7mer_within_1_mismatch,
    smith_waterman_stats, tm_score_ca, HomologFirewall, HomologFirewallQuery, TorsionDb,
    TorsionExportOptions, TorsionPredictionClass,
};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[derive(Clone, Debug)]
struct BenchRefResidue {
    seq_id: i32,
    res_name: String,
    phi_deg: Option<f64>,
    psi_deg: Option<f64>,
    chi1_deg: Option<f64>,
    ca_coord: [f64; 3],
}

#[derive(Clone, Debug)]
struct BenchRefEntry {
    bmrb_id: String,
    pdb_id: String,
    uniprot_id: Option<String>,
    family_id: Option<String>,
    topology_id: Option<String>,
    is_folded: bool,
    residues: Vec<BenchRefResidue>,
}

fn load_benchmark_refs() -> BTreeMap<String, BenchRefEntry> {
    let path = workspace_root().join("data/torsion_db/benchmark_refs.tsv");
    let text = fs::read_to_string(&path).expect("read benchmark_refs.tsv");
    let mut map: BTreeMap<String, BenchRefEntry> = BTreeMap::new();

    let parse_opt = |s: &str| -> Option<f64> {
        if s == "." || s.is_empty() {
            None
        } else {
            Some(s.parse::<f64>().expect("valid float in benchmark_refs.tsv"))
        }
    };
    let opt_str = |s: &str| -> Option<String> {
        if s == "." || s.is_empty() {
            None
        } else {
            Some(s.to_string())
        }
    };

    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = t.split('\t').collect();
        match cols[0] {
            "@BENCH" => {
                let bmrb_id = cols[1].to_string();
                map.insert(
                    bmrb_id.clone(),
                    BenchRefEntry {
                        bmrb_id,
                        pdb_id: cols[2].to_string(),
                        uniprot_id: opt_str(cols[3]),
                        family_id: opt_str(cols[4]),
                        topology_id: opt_str(cols[5]),
                        is_folded: cols[6] == "1",
                        residues: Vec::new(),
                    },
                );
            }
            "BRES" => {
                let entry = map.get_mut(cols[1]).expect("BRES after @BENCH");
                entry.residues.push(BenchRefResidue {
                    seq_id: cols[2].parse().unwrap(),
                    res_name: cols[3].to_string(),
                    phi_deg: parse_opt(cols[4]),
                    psi_deg: parse_opt(cols[5]),
                    chi1_deg: parse_opt(cols[6]),
                    ca_coord: [
                        cols[8].parse().unwrap(),
                        cols[9].parse().unwrap(),
                        cols[10].parse().unwrap(),
                    ],
                });
            }
            other => panic!("unexpected tag in benchmark_refs.tsv: {other}"),
        }
    }
    map
}

#[test]
fn test_sha256_and_corpus_provenance_integrity() {
    // NIST FIPS 180-4 test vectors
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );

    let root = workspace_root();
    let corpus_bytes = fs::read(root.join("data/torsion_db/torsion_corpus.tsv")).unwrap();
    let refs_bytes = fs::read(root.join("data/torsion_db/benchmark_refs.tsv")).unwrap();
    let prov_text = fs::read_to_string(root.join("data/torsion_db/PROVENANCE.md")).unwrap();

    let corpus_sha = sha256_hex(&corpus_bytes);
    let refs_sha = sha256_hex(&refs_bytes);
    assert!(
        prov_text.contains(&corpus_sha),
        "PROVENANCE.md must record exact SHA-256 of torsion_corpus.tsv ({corpus_sha})"
    );
    assert!(
        prov_text.contains(&refs_sha),
        "PROVENANCE.md must record exact SHA-256 of benchmark_refs.tsv ({refs_sha})"
    );

    let db = TorsionDb::embedded();
    assert!(
        db.chains.len() >= 120,
        "corpus must contain >= 120 non-redundant chains, got {}",
        db.chains.len()
    );
    assert_eq!(TorsionDb::embedded_sha256(), corpus_sha);

    let mut n_c1 = 0usize;
    let mut n_c2 = 0usize;
    let mut n_c3 = 0usize;
    for ch in &db.chains {
        if ch.topology_id.starts_with("1.") {
            n_c1 += 1;
        } else if ch.topology_id.starts_with("2.") {
            n_c2 += 1;
        } else if ch.topology_id.starts_with("3.") {
            n_c3 += 1;
        }
    }
    assert!(n_c1 >= 30 && n_c2 >= 30 && n_c3 >= 40);

    // Adversarial corpus parser checks: non-ASCII, NaN, Inf
    assert!(TorsionDb::from_tsv_str("@CHAIN\t1UBQ_A\t1UBQ\tA\t5387\tP0CG48\t3.10.20.90\t3.10.20\nRES\t1\tM\t.\t-40.0\t.\t0.0\t0.0\t0.0\t.\t.\tNaN\t.\t.\t.\n").is_err());
    assert!(TorsionDb::from_tsv_str("# non-ascii \u{00e9}\n").is_err());
    assert!(TorsionDb::from_tsv_str("@CHAIN\t1UBQ_A\t1UBQ\tA\t5387\tP0CG48\t3.10.20.90\t3.10.20\nRES\t1\tM\t.\tinf\t.\t0.0\t0.0\t0.0\t.\t.\t.\t.\t.\t.\n").is_err());
}

#[test]
fn test_homolog_firewall_all_five_layers_independently() {
    let db = TorsionDb::embedded();
    let dummy_seq = b"ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY";

    // Layer 1a: PDB ID match
    let fw_pdb = HomologFirewallQuery {
        query_pdb_ids: vec!["1UBQ".to_string()],
        ..Default::default()
    };
    let (_, ex_pdb, sha_pdb, _) = apply_homolog_firewall(db, dummy_seq, &fw_pdb);
    assert!(ex_pdb
        .iter()
        .any(|e| e.pdb_id == "1UBQ" && e.reasons.iter().any(|r| r == "L1:same_pdb")));
    assert_eq!(sha_pdb.len(), 64);

    // Layer 1b: BMRB ID match (with "bmr" prefix)
    let fw_bmrb = HomologFirewallQuery {
        query_bmrb_ids: vec!["bmr5387".to_string()],
        ..Default::default()
    };
    let (_, ex_bmrb, _, _) = apply_homolog_firewall(db, dummy_seq, &fw_bmrb);
    assert!(ex_bmrb
        .iter()
        .any(|e| e.bmrb_id == "5387" && e.reasons.iter().any(|r| r == "L1:same_bmrb")));

    // Layer 1c: UniProt accession match
    let fw_unp = HomologFirewallQuery {
        query_uniprot_id: Some("P0CG48".to_string()),
        ..Default::default()
    };
    let (_, ex_unp, _, _) = apply_homolog_firewall(db, dummy_seq, &fw_unp);
    assert!(ex_unp
        .iter()
        .any(|e| e.pdb_id == "1UBQ" && e.reasons.iter().any(|r| r == "L1:same_uniprot")));

    // Layer 2: CATH/ECOD topology fold match ("3.10.20" beta-grasp fold) with zero L1/L3/L4 overlap
    let poly_w_40 = vec![b'W'; 40];
    let fw_top = HomologFirewallQuery {
        query_topology_id: Some("3.10.20".to_string()),
        ..Default::default()
    };
    let (ret_top, ex_top, _, _) = apply_homolog_firewall(db, &poly_w_40, &fw_top);
    assert!(
        !ex_top.is_empty()
            && ex_top
                .iter()
                .all(|e| e.reasons.iter().any(|r| r == "L2:same_topology_fold"))
    );
    assert!(ret_top.iter().all(|c| c.topology_id != "3.10.20"));

    // Layer 3a: Global NW & Local SW > 25% sequence identity in BOTH query coverage & template coverage directions
    let ubq_chain = db.chains.iter().find(|c| c.pdb_id == "1UBQ").unwrap();
    // Keep every 3rd residue identical (34.2% identity -> at most 3/7 in any 7-mer!),
    // substituting non-identical positions with neutral non-matching amino acids so BLOSUM62 score > 0
    let neutral = *b"ASND";
    let mut seq_33pct = Vec::with_capacity(ubq_chain.seq1.len());
    for (idx, &aa) in ubq_chain.seq1.iter().enumerate() {
        if idx % 3 == 0 {
            seq_33pct.push(aa);
        } else {
            let mut sub = neutral[idx % neutral.len()];
            if sub == aa {
                sub = b'G';
            }
            if sub == aa {
                sub = b'H';
            }
            seq_33pct.push(sub);
        }
    }
    assert!(!shares_7mer_within_1_mismatch(&seq_33pct, &ubq_chain.seq1));
    let nw_33 = needleman_wunsch_stats(&seq_33pct, &ubq_chain.seq1);
    assert!(nw_33.exceeds_firewall(seq_33pct.len(), ubq_chain.seq1.len()));

    // Asymmetric template-coverage direction: 220-residue query embedding the 76-residue 33%-id domain
    let mut long_query = vec![b'C'; 70];
    long_query.extend_from_slice(&seq_33pct);
    long_query.extend(vec![b'C'; 74]);
    let sw_asym = smith_waterman_stats(&long_query, &ubq_chain.seq1);
    assert!(
        sw_asym.exceeds_firewall(long_query.len(), ubq_chain.seq1.len()),
        "SW must catch >25% identity when >=50% of template is covered inside a much longer query"
    );
    let (_, ex_l3_asym, _, _) =
        apply_homolog_firewall(db, &long_query, &HomologFirewallQuery::default());
    assert!(ex_l3_asym
        .iter()
        .any(|e| e.pdb_id == "1UBQ" && e.reasons.iter().any(|r| r.starts_with("L3:sw_seq_id"))));

    // Layer 3a-adversarial: 34.2% sequence identity (26/76 identical, 0 shared 6/7-mers) with harsh
    // non-conservative W/R/D substitutions so full-diagonal BLOSUM62 score is strictly negative (< 0)!
    let harsh = *b"WRD";
    let mut seq_34pct_wrd = Vec::with_capacity(ubq_chain.seq1.len());
    for (idx, &aa) in ubq_chain.seq1.iter().enumerate() {
        if idx % 3 == 0 {
            seq_34pct_wrd.push(aa);
        } else {
            let mut sub = harsh[idx % harsh.len()];
            if sub == aa {
                sub = b'W';
            }
            if sub == aa {
                sub = b'D';
            }
            seq_34pct_wrd.push(sub);
        }
    }
    assert!(!shares_7mer_within_1_mismatch(
        &seq_34pct_wrd,
        &ubq_chain.seq1
    ));
    let (_, ex_wrd, _, max_ret_wrd) =
        apply_homolog_firewall(db, &seq_34pct_wrd, &HomologFirewallQuery::default());
    assert!(
        ex_wrd.iter().any(|e| e.pdb_id == "1UBQ"
            && e.reasons
                .iter()
                .any(|r| r.starts_with("L3:nw_seq_id") || r.starts_with("L3:sw_seq_id"))),
        "Layer 3 must exclude 34.2%-id 1UBQ homolog even under harsh W/R/D substitutions with negative BLOSUM62!"
    );
    assert!(
        max_ret_wrd <= 0.25 + 1e-12,
        "max_retained_seq_id {max_ret_wrd} exceeded 0.25"
    );

    // Also verify embedded harsh W/R/D 34.2%-id 1UBQ domain inside a 210-aa query is caught by Layer 3
    let mut long_wrd = vec![b'P'; 65];
    long_wrd.extend_from_slice(&seq_34pct_wrd);
    long_wrd.extend(vec![b'P'; 69]);
    let (_, ex_long_wrd, _, max_ret_long_wrd) =
        apply_homolog_firewall(db, &long_wrd, &HomologFirewallQuery::default());
    assert!(
        ex_long_wrd.iter().any(|e| e.pdb_id == "1UBQ"
            && e.reasons
                .iter()
                .any(|r| r.starts_with("L3:nw_seq_id") || r.starts_with("L3:sw_seq_id"))),
        "Layer 3 must exclude embedded 34.2%-id W/R/D domain inside 210-aa query!"
    );
    assert!(max_ret_long_wrd <= 0.25 + 1e-12);

    // Layer 3b: 7-mer <= 1 mismatch (>= 6/7) filter on an otherwise 0%-identity sequence
    let mut seq_7mer = vec![b'W'; 60];
    // Copy residues 10..17 of 1UBQ with 1 mismatch at offset 3
    seq_7mer[20..27].copy_from_slice(&ubq_chain.seq1[10..17]);
    seq_7mer[23] = b'C';
    assert!(shares_7mer_within_1_mismatch(&seq_7mer, &ubq_chain.seq1));
    let (_, ex_7mer, _, _) =
        apply_homolog_firewall(db, &seq_7mer, &HomologFirewallQuery::default());
    assert!(ex_7mer
        .iter()
        .any(|e| e.pdb_id == "1UBQ" && e.reasons.iter().any(|r| r == "L3:7mer_6of7")));

    // Layer 4: Clean-room Zhang-Skolnick TM-score >= 0.50 on rotated/translated/perturbed coordinates
    // with completely unrelated poly-Trp sequence and blank IDs!
    let ubq_ca: Vec<[f64; 3]> = ubq_chain.residues.iter().map(|r| r.ca_coord).collect();
    assert!((tm_score_ca(&ubq_ca, &ubq_ca) - 1.0).abs() < 1e-12);

    // Apply arbitrary proper 3D rotation + translation + deterministic 0.6 A perturbation
    let (sa, ca_ang) = (0.73f64).sin_cos();
    let (sb, cb_ang) = (-0.51f64).sin_cos();
    let rot_pert_ca: Vec<[f64; 3]> = ubq_ca
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let x1 = ca_ang * p[0] - sa * p[1];
            let y1 = sa * p[0] + ca_ang * p[1];
            let z1 = p[2];
            let x2 = cb_ang * x1 + sb * z1;
            let z2 = -sb * x1 + cb_ang * z1;
            let jit = ((i as f64) * 1.7).sin() * 0.45;
            [x2 + 42.5 + jit, y1 - 19.3 - jit, z2 + 88.1 + 0.5 * jit]
        })
        .collect();
    let tm_pert = tm_score_ca(&rot_pert_ca, &ubq_ca);
    assert!(
        tm_pert > 0.80 && tm_pert < 1.0,
        "expected perturbed ubiquitin TM-score in (0.80, 1.0), got {tm_pert}"
    );

    let poly_w = vec![b'W'; ubq_ca.len()];
    let fw_tm = HomologFirewallQuery {
        query_ca_coords: Some(rot_pert_ca.clone()),
        ..Default::default()
    };
    let (_, ex_tm, sha_tm1, _) = apply_homolog_firewall(db, &poly_w, &fw_tm);
    let (_, _, sha_tm2, _) = apply_homolog_firewall(db, &poly_w, &fw_tm);
    assert_eq!(
        sha_tm1, sha_tm2,
        "Audit Digest & Identity Stratification SHA-256 exclusion digest must be deterministic"
    );
    assert!(
        ex_tm
            .iter()
            .any(|e| e.pdb_id == "1UBQ" && e.reasons.iter().any(|r| r.starts_with("L4:tm_score"))),
        "Layer 4 must exclude structural analog with TM-score >= 0.50 even at 0% sequence identity"
    );

    // Layer 4b: Perturbed 76-aa 1UBQ embedded at offset 62 inside a 200-aa multi-domain query
    let mut embedded_200_ca = Vec::with_capacity(200);
    for k in 0..62 {
        let s = k as f64;
        embedded_200_ca.push([
            -80.0 + 1.5 * s,
            50.0 + 2.3 * (0.9 * s).cos(),
            -40.0 + 2.3 * (0.9 * s).sin(),
        ]);
    }
    embedded_200_ca.extend_from_slice(&rot_pert_ca);
    for k in 0..62 {
        let s = k as f64;
        embedded_200_ca.push([
            160.0 + 1.5 * s,
            -70.0 + 2.3 * (1.1 * s).cos(),
            120.0 + 2.3 * (1.1 * s).sin(),
        ]);
    }
    assert_eq!(embedded_200_ca.len(), 200);
    let tm_emb_200 = tm_score_ca(&embedded_200_ca, &ubq_ca);
    assert!(
        tm_emb_200 >= 0.80,
        "expected embedded 76-aa 1UBQ in 200-aa query to achieve TM >= 0.80 (min(Lq,Lt) norm), got {tm_emb_200}"
    );
    let (_, ex_tm_emb, _, _) = apply_homolog_firewall(
        db,
        &[b'W'; 200],
        &HomologFirewallQuery {
            query_ca_coords: Some(embedded_200_ca),
            ..Default::default()
        },
    );
    assert!(
        ex_tm_emb
            .iter()
            .any(|e| e.pdb_id == "1UBQ" && e.reasons.iter().any(|r| r.starts_with("L4:tm_score"))),
        "Layer 4 must exclude 76-aa 1UBQ domain embedded inside a 200-aa query!"
    );

    // Layer 4c: Perturbed 76-aa 1UBQ preceded by a 5-aa N-terminal cloning tag (`sh = 5`)
    let mut tagged_81_ca = Vec::with_capacity(81);
    for k in 0..5 {
        let s = (5 - k) as f64;
        tagged_81_ca.push([
            rot_pert_ca[0][0] - 3.8 * s,
            rot_pert_ca[0][1] + 0.9 * s,
            rot_pert_ca[0][2] - 0.7 * s,
        ]);
    }
    tagged_81_ca.extend_from_slice(&rot_pert_ca);
    assert_eq!(tagged_81_ca.len(), 81);
    let tm_tag_81 = tm_score_ca(&tagged_81_ca, &ubq_ca);
    assert!(
        tm_tag_81 >= 0.80,
        "expected 1UBQ with 5-aa N-terminal tag to achieve TM >= 0.80, got {tm_tag_81}"
    );
    let (_, ex_tm_tag, _, _) = apply_homolog_firewall(
        db,
        &[b'W'; 81],
        &HomologFirewallQuery {
            query_ca_coords: Some(tagged_81_ca),
            ..Default::default()
        },
    );
    assert!(
        ex_tm_tag
            .iter()
            .any(|e| e.pdb_id == "1UBQ" && e.reasons.iter().any(|r| r.starts_with("L4:tm_score"))),
        "Layer 4 must exclude 1UBQ with a 5-aa N-terminal cloning tag!"
    );
}

#[test]
fn test_unlabelled_global_shift_offset_recovery() {
    let root = workspace_root();
    let bench_refs = load_benchmark_refs();

    for (bmrb_id, star_name) in [("bmr15536", "bmr15536.str"), ("bmr17769", "bmr17769.str")] {
        let orig = read_star_all_atoms(root.join("data/benchmark").join(star_name), None).unwrap();
        let mut shifted: Reference = orig.clone();
        for ((_, atom), val) in shifted.shifts.iter_mut() {
            match atom.as_str() {
                "CA" | "CB" | "C" => *val += 2.35,
                "N" => *val -= 1.85,
                "H" | "HA" | "HA2" | "HA3" => *val += 0.26,
                _ => {}
            }
        }

        let est = estimate_global_shift_offsets(&shifted);
        assert!(
            (est.dca - 2.35).abs() < 0.65 && (est.dcb - 2.35).abs() < 0.65,
            "{bmrb_id}: failed to recover +2.35 ppm 13C offset: {est:?}"
        );
        assert!(
            (est.dn - (-1.85)).abs() < 1.10,
            "{bmrb_id}: failed to recover -1.85 ppm 15N offset: {est:?}"
        );

        // Verify sequence-order / residue-label permutation invariance of `estimate_global_shift_offsets`
        // across all non-Proline and non-pre-Proline positions!
        let n_seq = shifted.sequence.len();
        let free_idxs: Vec<usize> = (0..n_seq)
            .filter(|&i| {
                let is_pro = shifted.sequence[i].trim().eq_ignore_ascii_case("PRO");
                let pre_pro =
                    i + 1 < n_seq && shifted.sequence[i + 1].trim().eq_ignore_ascii_case("PRO");
                !is_pro && !pre_pro
            })
            .collect();
        let mut perm_map: Vec<usize> = (0..n_seq).collect();
        for (k, &old_i) in free_idxs.iter().enumerate() {
            // Deterministic non-trivial reversal + stride permutation
            let new_i = free_idxs[free_idxs.len() - 1 - k];
            perm_map[new_i] = old_i;
        }
        let mut permuted = Reference {
            sequence: (0..n_seq)
                .map(|new_i| shifted.sequence[perm_map[new_i]].clone())
                .collect(),
            seq_ids: (0..n_seq).map(|new_i| (new_i + 1) as i32).collect(),
            shifts: std::collections::BTreeMap::new(),
            ambiguity_codes: std::collections::BTreeMap::new(),
        };
        let mut inv_perm = vec![0usize; n_seq];
        for (new_i, &old_i) in perm_map.iter().enumerate() {
            inv_perm[old_i] = new_i;
        }
        for ((old_i, atom), &val) in &shifted.shifts {
            permuted
                .shifts
                .insert((inv_perm[*old_i], atom.clone()), val);
        }
        let est_perm = estimate_global_shift_offsets(&permuted);
        assert!(
            (est_perm.dh - est.dh).abs() < 1e-12
                && (est_perm.dn - est.dn).abs() < 1e-12
                && (est_perm.dca - est.dca).abs() < 1e-12
                && (est_perm.dcb - est.dcb).abs() < 1e-12
                && (est_perm.dc - est.dc).abs() < 1e-12
                && (est_perm.dha - est.dha).abs() < 1e-12,
            "{bmrb_id}: estimate_global_shift_offsets is not permutation-invariant: {est:?} vs {est_perm:?}"
        );

        let bref = &bench_refs[bmrb_id];
        let fw = HomologFirewallQuery {
            query_pdb_ids: vec![bref.pdb_id.clone()],
            query_bmrb_ids: vec![bref.bmrb_id.clone()],
            query_uniprot_id: bref.uniprot_id.clone(),
            query_family_id: bref.family_id.clone(),
            query_topology_id: bref.topology_id.clone(),
            query_ca_coords: Some(bref.residues.iter().map(|r| r.ca_coord).collect()),
        };
        let rep = predict_torsions_with_firewall(&shifted, &fw);

        let p_keys: Vec<(i32, &str)> = rep
            .residues
            .iter()
            .map(|r| (r.seq_id, r.res_name.as_str()))
            .collect();
        let t_keys: Vec<(i32, &str)> = bref
            .residues
            .iter()
            .map(|r| (r.seq_id, r.res_name.as_str()))
            .collect();
        let pairs = align_by_res_names(&p_keys, &t_keys);

        let mut errs = Vec::new();
        for (pi, ti) in pairs {
            let pr = &rep.residues[pi];
            let tr = &bref.residues[ti];
            if pr.class == TorsionPredictionClass::Good {
                if let (Some(pp), Some(ps), Some(tp), Some(ts)) =
                    (pr.phi_deg, pr.psi_deg, tr.phi_deg, tr.psi_deg)
                {
                    errs.push(
                        0.5 * (periodic_angle_abs_diff_deg(pp, tp)
                            + periodic_angle_abs_diff_deg(ps, ts)),
                    );
                }
            }
        }
        let mae = errs.iter().sum::<f64>() / (errs.len() as f64);
        assert!(
            mae <= 15.0,
            "{bmrb_id}: mis-referenced shift prediction MAE too high: {mae:.2} deg"
        );
    }
}

#[test]
fn test_12_protein_bmrb_suite_with_q5_firewall_and_stratification() {
    let root = workspace_root();
    let manifest_text = fs::read_to_string(root.join("data/benchmark/manifest.tsv")).unwrap();
    let bench_refs = load_benchmark_refs();

    let mut folded_total_res = 0usize;
    let mut folded_struct_res = 0usize;
    let mut folded_good_res = 0usize;
    let mut folded_struct_good_res = 0usize;
    let mut folded_rama_outliers = 0usize;
    let mut folded_phi_errs = Vec::new();
    let mut folded_psi_errs = Vec::new();
    let mut folded_circ_errs = Vec::new();
    let mut folded_chi1_errs = Vec::new();
    let mut wrong_basin_vs_truth_count = 0usize;
    let mut n_good_compared_truth = 0usize;

    // Sequence identity stratification bins: [0,0.10), [0.10,0.15), [0.15,0.20), [0.20,0.25]
    let mut strat_bins: [Vec<f64>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

    println!(
        "\n=== T12.3 + Q5 12-PROTEIN BMRB BENCHMARK (4-LAYER HOMOLOG FIREWALL + AUDIT DIGEST & IDENTITY STRATIFICATION) ==="
    );
    println!(
        "{:<10} {:<6} {:<6} {:>4} {:>5} {:>6} {:>6} {:>6} {:>9} {:>7} {:>7} {:>7} {:>8} {:>16}",
        "BMRB",
        "PDB",
        "State",
        "N",
        "Excl",
        "MaxId",
        "Good%",
        "Dyn%",
        "WrongBsn%",
        "PhiMAE",
        "PsiMAE",
        "CircMAE",
        "Chi1MAE",
        "ExclSHA256[..12]"
    );

    for (line_idx, line) in manifest_text.lines().enumerate() {
        if line_idx == 0 || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        let bmrb_id = cols[0];
        let rel_path = cols[5];
        let shifts = read_star_all_atoms(root.join(rel_path), None).expect("read benchmark STAR");
        let bref = bench_refs
            .get(bmrb_id)
            .unwrap_or_else(|| panic!("missing {bmrb_id} in benchmark_refs.tsv"));

        let fw = HomologFirewallQuery {
            query_pdb_ids: vec![bref.pdb_id.clone()],
            query_bmrb_ids: vec![bref.bmrb_id.clone()],
            query_uniprot_id: bref.uniprot_id.clone(),
            query_family_id: bref.family_id.clone(),
            query_topology_id: bref.topology_id.clone(),
            query_ca_coords: if bref.is_folded {
                Some(bref.residues.iter().map(|r| r.ca_coord).collect())
            } else {
                None
            },
        };

        let rep = predict_torsions_with_firewall(&shifts, &fw);
        assert!(
            rep.max_retained_seq_identity <= 0.25 + 1e-12,
            "{bmrb_id}: retained homolog with sequence identity {:.3} > 0.25!",
            rep.max_retained_seq_identity
        );
        assert!(
            rep.nearest_template_seq_identities
                .iter()
                .all(|&sid| sid <= 0.25 + 1e-12),
            "{bmrb_id}: nearest_template_seq_identities exceeded 0.25: {:?}",
            rep.nearest_template_seq_identities
        );
        assert_eq!(rep.exclusion_list_sha256.len(), 64);

        let n_res = rep.residues.len();
        let n_good = rep
            .residues
            .iter()
            .filter(|r| r.class == TorsionPredictionClass::Good)
            .count();
        let n_dyn = rep
            .residues
            .iter()
            .filter(|r| r.class == TorsionPredictionClass::Dynamic)
            .count();

        if !bref.is_folded {
            let abstain_pct = 100.0 * ((n_res - n_good) as f64) / (n_res as f64);
            println!(
                "{:<10} {:<6} {:<6} {:>4} {:>5} {:>5.1}% {:>5.1}% {:>5.1}% {:>9} {:>7} {:>7} {:>7} {:>8} {}",
                bmrb_id,
                bref.pdb_id,
                "IDP/UF",
                n_res,
                rep.excluded_template_ids.len(),
                rep.max_retained_seq_identity * 100.0,
                100.0 * (n_good as f64) / (n_res as f64),
                100.0 * (n_dyn as f64) / (n_res as f64),
                "-",
                "-",
                "-",
                "-",
                "-",
                &rep.exclusion_list_sha256[..12]
            );
            assert!(
                abstain_pct >= 90.0,
                "{bmrb_id}: unfolded/IDP entry must abstain on >= 90% of residues, got {abstain_pct:.1}%"
            );
            continue;
        }

        let p_keys: Vec<(i32, &str)> = rep
            .residues
            .iter()
            .map(|r| (r.seq_id, r.res_name.as_str()))
            .collect();
        let t_keys: Vec<(i32, &str)> = bref
            .residues
            .iter()
            .map(|r| (r.seq_id, r.res_name.as_str()))
            .collect();
        let pairs = align_by_res_names(&p_keys, &t_keys);

        let mut prot_phi = Vec::new();
        let mut prot_psi = Vec::new();
        let mut prot_circ = Vec::new();
        let mut prot_chi1 = Vec::new();
        let mut prot_wrong_basin = 0usize;
        let mut prot_good_truth = 0usize;

        for r in &rep.residues {
            folded_total_res += 1;
            if r.rci_s2 >= 0.65 {
                folded_struct_res += 1;
                if r.class == TorsionPredictionClass::Good {
                    folded_struct_good_res += 1;
                }
            }
            if r.class == TorsionPredictionClass::Good {
                folded_good_res += 1;
                if is_ramachandran_outlier(&r.res_name, r.phi_deg.unwrap(), r.psi_deg.unwrap()) {
                    folded_rama_outliers += 1;
                }
            }
        }

        for (pi, ti) in pairs {
            let pr = &rep.residues[pi];
            let tr = &bref.residues[ti];
            let nn_seq_id = rep.nearest_template_seq_identities[pi];
            let bin_idx = if nn_seq_id < 0.10 {
                0
            } else if nn_seq_id < 0.15 {
                1
            } else if nn_seq_id < 0.20 {
                2
            } else {
                3
            };
            if pr.class == TorsionPredictionClass::Good {
                if let (Some(pp), Some(ps), Some(tp), Some(ts)) =
                    (pr.phi_deg, pr.psi_deg, tr.phi_deg, tr.psi_deg)
                {
                    let ep = periodic_angle_abs_diff_deg(pp, tp);
                    let es = periodic_angle_abs_diff_deg(ps, ts);
                    let ec = 0.5 * (ep + es);
                    folded_phi_errs.push(ep);
                    folded_psi_errs.push(es);
                    folded_circ_errs.push(ec);
                    prot_phi.push(ep);
                    prot_psi.push(es);
                    prot_circ.push(ec);
                    strat_bins[bin_idx].push(ec);

                    n_good_compared_truth += 1;
                    prot_good_truth += 1;
                    if classify_rama_basin(pp, ps) != classify_rama_basin(tp, ts) {
                        wrong_basin_vs_truth_count += 1;
                        prot_wrong_basin += 1;
                    }
                }
                if let (Some(pc), Some(tc)) = (pr.chi1_deg, tr.chi1_deg) {
                    let ech = periodic_angle_abs_diff_deg(pc, tc);
                    folded_chi1_errs.push(ech);
                    prot_chi1.push(ech);
                }
            }
        }

        let p_phi_mae = prot_phi.iter().sum::<f64>() / (prot_phi.len().max(1) as f64);
        let p_psi_mae = prot_psi.iter().sum::<f64>() / (prot_psi.len().max(1) as f64);
        let p_mae = prot_circ.iter().sum::<f64>() / (prot_circ.len().max(1) as f64);
        let c_mae_str = if prot_chi1.is_empty() {
            "-".to_string()
        } else {
            format!(
                "{:.2}",
                prot_chi1.iter().sum::<f64>() / (prot_chi1.len() as f64)
            )
        };
        println!(
            "{:<10} {:<6} {:<6} {:>4} {:>5} {:>5.1}% {:>5.1}% {:>5.1}% {:>8.2}% {:>7.2} {:>7.2} {:>7.2} {:>8} {}",
            bmrb_id,
            bref.pdb_id,
            "Folded",
            n_res,
            rep.excluded_template_ids.len(),
            rep.max_retained_seq_identity * 100.0,
            100.0 * (n_good as f64) / (n_res as f64),
            100.0 * (n_dyn as f64) / (n_res as f64),
            100.0 * (prot_wrong_basin as f64) / (prot_good_truth.max(1) as f64),
            p_phi_mae,
            p_psi_mae,
            p_mae,
            c_mae_str,
            &rep.exclusion_list_sha256[..12]
        );
    }

    let overall_good_pct = 100.0 * (folded_good_res as f64) / (folded_total_res as f64);
    let struct_good_pct = 100.0 * (folded_struct_good_res as f64) / (folded_struct_res as f64);
    let rama_outlier_pct = 100.0 * (folded_rama_outliers as f64) / (folded_good_res as f64);
    let wrong_basin_pct =
        100.0 * (wrong_basin_vs_truth_count as f64) / (n_good_compared_truth.max(1) as f64);
    let phi_mae = folded_phi_errs.iter().sum::<f64>() / (folded_phi_errs.len() as f64);
    let psi_mae = folded_psi_errs.iter().sum::<f64>() / (folded_psi_errs.len() as f64);
    let circ_mae = folded_circ_errs.iter().sum::<f64>() / (folded_circ_errs.len() as f64);
    let chi1_mae = folded_chi1_errs.iter().sum::<f64>() / (folded_chi1_errs.len().max(1) as f64);

    println!(
        "\n[SUMMARY 9 FOLDED PROTEINS] Good overall = {}/{} ({:.1}%), Good structured (RCI S^2>=0.65) = {}/{} ({:.1}%)",
        folded_good_res, folded_total_res, overall_good_pct,
        folded_struct_good_res, folded_struct_res, struct_good_pct
    );
    println!(
        "[SUMMARY ACCURACY] phi_mae_deg = {:.2} deg, psi_mae_deg = {:.2} deg, combined Circular (phi,psi) MAE = {:.2} deg, chi1_mae_deg = {:.2} deg (n={}), wrong_basin_vs_truth = {}/{} ({:.2}%), pred Rama outliers = {}/{} ({:.2}%)",
        phi_mae, psi_mae, circ_mae, chi1_mae, folded_chi1_errs.len(),
        wrong_basin_vs_truth_count, n_good_compared_truth, wrong_basin_pct,
        folded_rama_outliers, folded_good_res, rama_outlier_pct
    );

    let bin_labels = ["[0-10%)", "[10-15%)", "[15-20%)", "[20-25%]"];
    println!("\n[AUDIT DIGEST & IDENTITY STRATIFICATION]");
    for (b, label) in bin_labels.iter().enumerate() {
        let n = strat_bins[b].len();
        let m = if n > 0 {
            strat_bins[b].iter().sum::<f64>() / (n as f64)
        } else {
            f64::NAN
        };
        println!("  Bin {label:<9}: n_good_pairs = {n:>4}, circular_MAE = {m:.2} deg");
    }

    assert!(
        overall_good_pct >= 68.0,
        "overall Good coverage {overall_good_pct:.1}% below 68%"
    );
    assert!(
        struct_good_pct >= 78.0,
        "structured Good coverage {struct_good_pct:.1}% below 78%"
    );
    assert!(
        rama_outlier_pct < 2.5,
        "Ramachandran outliers {rama_outlier_pct:.2}% >= 2.5% gate!"
    );
    assert!(
        wrong_basin_pct < 8.0,
        "Wrong Ramachandran basin vs PDB ground truth {wrong_basin_pct:.2}% >= 8.0%!"
    );
    assert!(
        phi_mae <= 13.5 && psi_mae <= 17.0 && circ_mae <= 14.0,
        "Torsion MAE regression: phi={phi_mae:.2}, psi={psi_mae:.2}, circ={circ_mae:.2} deg"
    );
}

#[test]
fn test_torsions_tsv_json_and_nef_roundtrip() {
    let root = workspace_root();
    let shifts = read_star_all_atoms(root.join("data/benchmark/bmr17769.str"), None).unwrap();
    let rep = predict_torsions(&shifts);

    // 1. Default export (`include_chi1: false`) MUST omit CHI1 across TSV, JSON, DihedralRestraint, and NEF!
    let tsv = format_torsions_tsv(&rep);
    assert!(tsv.contains("# exclusion_list_sha256:"));
    assert!(tsv.contains("seq_id\tres_name\tclass\tphi_deg"));
    for line in tsv.lines().skip(5) {
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(
            cols[7], ".",
            "default TSV must omit chi1_deg, got {}",
            cols[7]
        );
        assert_eq!(
            cols[8], ".",
            "default TSV must omit chi1_std_deg, got {}",
            cols[8]
        );
    }

    let js = format_torsions_json(&rep).unwrap();
    assert!(js.contains("\"exclusion_list_sha256\""));
    assert!(
        !js.lines()
            .any(|l| l.contains("\"chi1_deg\":") && !l.contains("null")),
        "default JSON must omit non-null chi1_deg unless include_chi1=true"
    );

    let dihs = rep.to_dihedral_restraints();
    assert_eq!(dihs.len(), shifts.sequence.len());
    assert!(
        dihs.iter()
            .all(|d| !d.restrained_chi1 && d.chi1_deg.is_none()),
        "default to_dihedral_restraints() must omit CHI1"
    );

    let nef_list = rep.to_nef_dihedral_restraint_list("A", "torsion_db_restraints");
    assert!(nef_list.restraints.iter().any(|r| r.name == "PHI"));
    assert!(nef_list.restraints.iter().any(|r| r.name == "PSI"));
    assert!(
        !nef_list.restraints.iter().any(|r| r.name == "CHI1"),
        "default NEF export must omit CHI1 unless include_chi1=true"
    );

    // 2. Opt-in `include_chi1: true` emits CHI1 across TSV, JSON, DihedralRestraint, and NEF!
    let opt_chi1 = TorsionExportOptions { include_chi1: true };
    let tsv_chi1 = format_torsions_tsv_with_options(&rep, opt_chi1);
    assert!(tsv_chi1
        .lines()
        .skip(5)
        .any(|l| l.split('\t').nth(7).is_some_and(|c| c != ".")));
    let js_chi1 = format_torsions_json_with_options(&rep, opt_chi1).unwrap();
    assert!(js_chi1
        .lines()
        .any(|l| l.contains("\"chi1_deg\":") && !l.contains("null")));
    let dihs_chi1 = rep.to_dihedral_restraints_with_options(opt_chi1);
    assert!(dihs_chi1
        .iter()
        .any(|d| d.restrained_chi1 && d.chi1_deg.is_some()));
    let nef_chi1 =
        rep.to_nef_dihedral_restraint_list_with_options("A", "torsion_db_restraints", opt_chi1);
    assert!(nef_chi1.restraints.iter().any(|r| r.name == "CHI1"));

    let mut doc = NefDocument::new("ubq_torsions");
    doc.dihedral_restraint_lists.push(nef_list.clone());
    let nef_str = write_nef_str(&doc);
    let parsed = read_nef_str(&nef_str).unwrap();
    assert_eq!(parsed.dihedral_restraint_lists.len(), 1);
    assert_eq!(
        parsed.dihedral_restraint_lists[0].restraints.len(),
        nef_list.restraints.len()
    );
}

#[test]
fn test_build_torsion_corpus_script_verify_and_spot_check() {
    let root = workspace_root();
    let script = root.join("scripts/build_torsion_corpus.py");
    assert!(
        script.exists(),
        "scripts/build_torsion_corpus.py must exist"
    );

    let output = std::process::Command::new("python3")
        .arg(&script)
        .arg("--verify")
        .arg(root.join("data/torsion_db/torsion_corpus.tsv"))
        .arg("--spot-check-pdb")
        .arg(root.join("data/fixtures/ubiquitin_1ubq_xray.pdb"))
        .arg("--spot-check-star")
        .arg(root.join("data/benchmark/bmr5387.str"))
        .arg("--chain")
        .arg("1UBQ_A")
        .output()
        .expect("invoke python3 scripts/build_torsion_corpus.py");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "build_torsion_corpus.py --verify failed: status={:?}\nstdout={stdout}\nstderr={stderr}",
        output.status
    );
    assert!(
        stdout.contains("SPOT_CHECK_OK chain=1UBQ_A residues=76"),
        "expected SPOT_CHECK_OK on 1UBQ_A, got: {stdout}"
    );
    assert!(
        stdout.contains(
            "VERIFY_OK chains=129 residues=13647 lines=13781 sha256=e34d3001d4c6989564d6bdfacff9dc401f92566aa77b499a23933b202045d88b"
        ),
        "expected VERIFY_OK with exact corpus SHA-256, got: {stdout}"
    );
}

#[test]
fn test_torsion_db_20pct_vs_25pct_seq_identity_cutoff_sensitivity() {
    let root = workspace_root();
    let manifest_text = fs::read_to_string(root.join("data/benchmark/manifest.tsv")).unwrap();
    let bench_refs = load_benchmark_refs();

    #[derive(Default)]
    struct CutoffSummary {
        folded_total_res: usize,
        folded_struct_res: usize,
        folded_good_res: usize,
        folded_struct_good_res: usize,
        folded_rama_outliers: usize,
        phi_errs: Vec<f64>,
        psi_errs: Vec<f64>,
        circ_errs: Vec<f64>,
        chi1_errs: Vec<f64>,
        wrong_basin_count: usize,
        good_truth_count: usize,
        idp_total_res: usize,
        idp_good_res: usize,
        max_retained_sid: f64,
    }

    let mut summaries: Vec<(f64, CutoffSummary)> = Vec::new();

    println!("\n=== P2 HOMOLOG FIREWALL SEQUENCE-IDENTITY CUTOFF SENSITIVITY (20% vs 25%) ===");
    for cutoff in [0.25f64, 0.20f64] {
        let mut sum = CutoffSummary::default();
        println!(
            "\n--- Cutoff max_seq_identity <= {:.0}% ---",
            cutoff * 100.0
        );
        for (line_idx, line) in manifest_text.lines().enumerate() {
            if line_idx == 0 || line.trim().is_empty() {
                continue;
            }
            let cols: Vec<&str> = line.split('\t').collect();
            let bmrb_id = cols[0];
            let rel_path = cols[5];
            let shifts = read_star_all_atoms(root.join(rel_path), None).expect("read STAR");
            let bref = &bench_refs[bmrb_id];
            let query = HomologFirewallQuery {
                query_pdb_ids: vec![bref.pdb_id.clone()],
                query_bmrb_ids: vec![bref.bmrb_id.clone()],
                query_uniprot_id: bref.uniprot_id.clone(),
                query_family_id: bref.family_id.clone(),
                query_topology_id: bref.topology_id.clone(),
                query_ca_coords: if bref.is_folded {
                    Some(bref.residues.iter().map(|r| r.ca_coord).collect())
                } else {
                    None
                },
            };
            let firewall = HomologFirewall::new(query).with_max_seq_identity(cutoff);
            let rep = firewall.predict(&shifts);

            assert!(
                rep.max_retained_seq_identity <= cutoff + 1e-12,
                "{bmrb_id} @ cutoff {cutoff}: retained template seq identity {:.4} > {cutoff}",
                rep.max_retained_seq_identity
            );
            assert!(
                rep.nearest_template_seq_identities
                    .iter()
                    .all(|&sid| sid <= cutoff + 1e-12),
                "{bmrb_id} @ cutoff {cutoff}: nearest_template_seq_identities exceeded {cutoff}"
            );
            sum.max_retained_sid = sum.max_retained_sid.max(rep.max_retained_seq_identity);

            let n_res = rep.residues.len();
            let n_good = rep
                .residues
                .iter()
                .filter(|r| r.class == TorsionPredictionClass::Good)
                .count();
            if !bref.is_folded {
                sum.idp_total_res += n_res;
                sum.idp_good_res += n_good;
                let abstain_pct = 100.0 * ((n_res - n_good) as f64) / (n_res as f64);
                assert!(
                    abstain_pct >= 90.0,
                    "{bmrb_id} @ cutoff {cutoff}: IDP abstention {abstain_pct:.1}% < 90%"
                );
                continue;
            }

            let p_keys: Vec<(i32, &str)> = rep
                .residues
                .iter()
                .map(|r| (r.seq_id, r.res_name.as_str()))
                .collect();
            let t_keys: Vec<(i32, &str)> = bref
                .residues
                .iter()
                .map(|r| (r.seq_id, r.res_name.as_str()))
                .collect();
            let pairs = align_by_res_names(&p_keys, &t_keys);

            let mut prot_circ = Vec::new();
            let mut prot_wb = 0usize;
            let mut prot_gt = 0usize;
            for r in &rep.residues {
                sum.folded_total_res += 1;
                if r.rci_s2 >= 0.65 {
                    sum.folded_struct_res += 1;
                    if r.class == TorsionPredictionClass::Good {
                        sum.folded_struct_good_res += 1;
                    }
                }
                if r.class == TorsionPredictionClass::Good {
                    sum.folded_good_res += 1;
                    if is_ramachandran_outlier(&r.res_name, r.phi_deg.unwrap(), r.psi_deg.unwrap())
                    {
                        sum.folded_rama_outliers += 1;
                    }
                }
            }
            for (pi, ti) in pairs {
                let pr = &rep.residues[pi];
                let tr = &bref.residues[ti];
                if pr.class == TorsionPredictionClass::Good {
                    if let (Some(pp), Some(ps), Some(tp), Some(ts)) =
                        (pr.phi_deg, pr.psi_deg, tr.phi_deg, tr.psi_deg)
                    {
                        let ep = periodic_angle_abs_diff_deg(pp, tp);
                        let es = periodic_angle_abs_diff_deg(ps, ts);
                        let ec = 0.5 * (ep + es);
                        sum.phi_errs.push(ep);
                        sum.psi_errs.push(es);
                        sum.circ_errs.push(ec);
                        prot_circ.push(ec);
                        sum.good_truth_count += 1;
                        prot_gt += 1;
                        if classify_rama_basin(pp, ps) != classify_rama_basin(tp, ts) {
                            sum.wrong_basin_count += 1;
                            prot_wb += 1;
                        }
                    }
                    if let (Some(pc), Some(tc)) = (pr.chi1_deg, tr.chi1_deg) {
                        sum.chi1_errs.push(periodic_angle_abs_diff_deg(pc, tc));
                    }
                }
            }
            println!(
                "  {:<10} {:<5} Excl={:>3} MaxSid={:>5.1}% Good={:>5.1}% CircMAE={:>6.2} deg WrongBasin={:>5.2}%",
                bmrb_id,
                bref.pdb_id,
                rep.excluded_template_ids.len(),
                rep.max_retained_seq_identity * 100.0,
                100.0 * (n_good as f64) / (n_res as f64),
                prot_circ.iter().sum::<f64>() / (prot_circ.len().max(1) as f64),
                100.0 * (prot_wb as f64) / (prot_gt.max(1) as f64),
            );
        }

        let good_pct = 100.0 * (sum.folded_good_res as f64) / (sum.folded_total_res as f64);
        let sgood_pct =
            100.0 * (sum.folded_struct_good_res as f64) / (sum.folded_struct_res as f64);
        let phi_mae = sum.phi_errs.iter().sum::<f64>() / (sum.phi_errs.len() as f64);
        let psi_mae = sum.psi_errs.iter().sum::<f64>() / (sum.psi_errs.len() as f64);
        let circ_mae = sum.circ_errs.iter().sum::<f64>() / (sum.circ_errs.len() as f64);
        let chi1_mae = sum.chi1_errs.iter().sum::<f64>() / (sum.chi1_errs.len().max(1) as f64);
        let wb_pct = 100.0 * (sum.wrong_basin_count as f64) / (sum.good_truth_count.max(1) as f64);
        let idp_leak_pct = 100.0 * (sum.idp_good_res as f64) / (sum.idp_total_res as f64);
        println!(
            "[CUTOFF <= {:.0}%] MaxRetainedSid={:.2}%, Good={}/{} ({:.1}%), StructGood={}/{} ({:.1}%), PhiMAE={:.2}, PsiMAE={:.2}, CircMAE={:.2}, Chi1MAE={:.2} (n={}), WrongBasin={}/{} ({:.2}%), IDPLeak={}/{} ({:.2}%)",
            cutoff * 100.0,
            sum.max_retained_sid * 100.0,
            sum.folded_good_res,
            sum.folded_total_res,
            good_pct,
            sum.folded_struct_good_res,
            sum.folded_struct_res,
            sgood_pct,
            phi_mae,
            psi_mae,
            circ_mae,
            chi1_mae,
            sum.chi1_errs.len(),
            sum.wrong_basin_count,
            sum.good_truth_count,
            wb_pct,
            sum.idp_good_res,
            sum.idp_total_res,
            idp_leak_pct,
        );

        assert!(
            good_pct >= 65.0 && sgood_pct >= 75.0 && circ_mae <= 14.5 && wb_pct < 8.5,
            "cutoff {cutoff} sensitivity check out of bounds: Good={good_pct:.1}%, StructGood={sgood_pct:.1}%, CircMAE={circ_mae:.2}, WrongBasin={wb_pct:.2}%"
        );
        summaries.push((cutoff, sum));
    }

    assert!(summaries[1].1.max_retained_sid <= 0.20 + 1e-12);
}
