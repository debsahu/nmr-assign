//! Comprehensive integration test suite for `octet_analyt`:
//! ¹H & ¹³C NMR prediction + multiplet splitting, 1D Lorentzian synthesis & JCAMP-DX I/O,
//! exact mass / Cl & Br isotopic envelopes / ESI adducts / MS fragmentation / physicochemical properties,
//! 2D reaction perception & laboratory stoichiometry tables, and offline bidirectional IUPAC naming.
//!
//! Provenance & Licence:
//! Ported/adapted from Octet-Draw by Sam Ellis (<https://github.com/SamChemMan/Octet-Draw>, Apache-2.0)
//! and published chemical shift additivity / IUPAC nomenclature rules.

pub use nmr_core::*;

use octet_analyt::iupac_naming::{iupac_name_to_mol, iupac_name_to_smiles, mol_to_iupac_name};
use octet_analyt::jcamp_spec1d::{
    pick_1d_multiplets, read_jcamp, synthesize_1d_nmr, write_jcamp_ms, write_jcamp_nmr,
    JcampCompression, SpectrumKind, Synthesize1dOptions,
};
use octet_analyt::mass_props::{
    calculate_standard_ms_adducts, compute_phys_chem_props, crossed_bonds_along_path,
    fragments_from_bond_cuts, hrms_report_line, isotope_envelope_for_mol, predict_ms_fragments,
    FragmentationMode, MsIonKind,
};
use octet_analyt::nmr_predict::{predict_nmr, NmrNucleus};
use octet_analyt::stoichiometry::{
    check_reaction_balance, format_stoich_amount, parse_stoich_amount, perceive_reaction_canvas,
    solve_stoichiometry, ReactionArrow, ReactionCaption, ReactionRole, StoichComponent,
    StoichField,
};
use octet_analyt::{AnalytAtom, AnalytBond, AnalytMol};

// ============================================================================
// 1. ¹H & ¹³C NMR Shift, Equivalence, J-Coupling & ACS String Benchmark Tests
// ============================================================================

#[test]
fn test_nmr_ethanol_ethyl_acetate_and_d2o_exchange() {
    // Ethanol: CH3-CH2-OH -> 3 ¹H signals in CDCl3 (CH2 quartet ~3.6–3.8 ppm, OH broad singlet, CH3 triplet ~1.1–1.3 ppm)
    // and 2 ¹³C signals (~58 ppm CH2O, ~18 ppm CH3)
    let etoh = AnalytMol::from_smiles("CCO").expect("valid ethanol SMILES");
    let h1 = predict_nmr(&etoh, NmrNucleus::H1, Some("CDCl3"), None);
    assert_eq!(
        h1.signals.len(),
        3,
        "Ethanol in CDCl3 has CH2, OH, CH3 signals"
    );

    let ch2 = h1
        .signals
        .iter()
        .find(|s| s.nuclei == 2)
        .expect("CH2 2H signal");
    assert!(
        (ch2.shift_ppm - 3.65).abs() < 0.35,
        "CH2 shift = {}",
        ch2.shift_ppm
    );
    assert_eq!(ch2.multiplicity(), "q", "CH2 coupled to CH3 is a quartet");
    assert_eq!(
        ch2.multiplet_lines.len(),
        4,
        "Quartet has 4 lines (1:3:3:1)"
    );

    let ch3 = h1
        .signals
        .iter()
        .find(|s| s.nuclei == 3)
        .expect("CH3 3H signal");
    assert!(
        (ch3.shift_ppm - 1.20).abs() < 0.30,
        "CH3 shift = {}",
        ch3.shift_ppm
    );
    assert_eq!(ch3.multiplicity(), "t", "CH3 coupled to CH2 is a triplet");
    assert_eq!(ch3.multiplet_lines.len(), 3, "Triplet has 3 lines (1:2:1)");

    // In D2O, labile OH exchanges out -> only 2 ¹H signals remain!
    let h1_d2o = predict_nmr(&etoh, NmrNucleus::H1, Some("D2O"), None);
    assert_eq!(h1_d2o.signals.len(), 2, "Labile OH disappears in D2O");

    let c13 = predict_nmr(&etoh, NmrNucleus::C13, Some("CDCl3"), None);
    assert_eq!(c13.signals.len(), 2);
    assert!(c13.signals[0].shift_ppm > 50.0 && c13.signals[0].shift_ppm < 66.0);
    assert!(c13.signals[1].shift_ppm > 12.0 && c13.signals[1].shift_ppm < 24.0);

    // Ethyl acetate: CH3-C(=O)-O-CH2-CH3
    let etoac = AnalytMol::from_smiles("CC(=O)OCC").expect("valid EtOAc SMILES");
    let h_ea = predict_nmr(&etoac, NmrNucleus::H1, Some("CDCl3"), None);
    assert_eq!(
        h_ea.signals.len(),
        3,
        "EtOAc has OCH2 (q), COCH3 (s), CH3 (t)"
    );
    let och2 = h_ea.signals.iter().find(|s| s.nuclei == 2).unwrap();
    assert!(
        (och2.shift_ppm - 4.12).abs() < 0.35,
        "OCH2 = {}",
        och2.shift_ppm
    );
    assert_eq!(och2.multiplicity(), "q");
    let ac_ch3 = h_ea
        .signals
        .iter()
        .find(|s| s.nuclei == 3 && s.couplings.is_empty())
        .expect("acetyl singlet 3H");
    assert!(
        (ac_ch3.shift_ppm - 2.05).abs() < 0.30,
        "Ac CH3 = {}",
        ac_ch3.shift_ppm
    );
    assert_eq!(ac_ch3.multiplicity(), "s");

    let c_ea = predict_nmr(&etoac, NmrNucleus::C13, Some("CDCl3"), None);
    assert_eq!(c_ea.signals.len(), 4);
    assert!(
        (c_ea.signals[0].shift_ppm - 171.0).abs() < 6.0,
        "Ester C=O ~171 ppm, got {}",
        c_ea.signals[0].shift_ppm
    );
    let acs_h = h_ea.acs_text(400.0);
    assert!(acs_h.starts_with("¹H NMR (400 MHz, CDCl3) δ"));
    assert!(acs_h.contains("(q, J = 7.0 Hz, 2H)"));
    let acs_c = c_ea.acs_text(400.0);
    assert!(acs_c.starts_with("¹³C NMR (101 MHz, CDCl3) δ"));
}

#[test]
fn test_nmr_4_bromoacetophenone_ibuprofen_alanine_and_alkene_j() {
    // 4-Bromoacetophenone: para-disubstituted AA'BB' benzene symmetry ->
    // 3 ¹H signals (2H ortho-to-C=O d ~7.8 ppm, 2H ortho-to-Br d ~7.6 ppm, 3H CH3 s ~2.6 ppm)
    // and 6 ¹³C signals (C=O ~197 ppm, 4 aromatic C classes, CH3 ~26 ppm)!
    let bap = AnalytMol::from_smiles("CC(=O)c1ccc(Br)cc1").unwrap();
    let h_bap = predict_nmr(&bap, NmrNucleus::H1, Some("CDCl3"), None);
    assert_eq!(
        h_bap.signals.len(),
        3,
        "Symmetry must group 2x2 aromatic CH + 1 CH3: {:?}",
        h_bap.signals
    );
    let arom_doublets: Vec<_> = h_bap
        .signals
        .iter()
        .filter(|s| s.nuclei == 2 && s.shift_ppm > 7.0)
        .collect();
    assert_eq!(arom_doublets.len(), 2);
    for sig in arom_doublets {
        assert_eq!(sig.multiplicity(), "d");
        assert!((sig.couplings[0].j_hz - 8.0).abs() < 0.6);
    }
    let c_bap = predict_nmr(&bap, NmrNucleus::C13, Some("CDCl3"), None);
    assert_eq!(
        c_bap.signals.len(),
        6,
        "4-Bromoacetophenone has 6 equivalent ¹³C environments"
    );
    assert!(
        (c_bap.signals[0].shift_ppm - 197.0).abs() < 6.0,
        "Ketone C=O = {}",
        c_bap.signals[0].shift_ppm
    );

    // Ibuprofen: CC(C)Cc1ccc(C(C)C(=O)O)cc1 -> C13H18O2 (total 18 ¹H, 10 symmetry-distinct ¹³C signals!)
    let ibu = AnalytMol::from_smiles("CC(C)Cc1ccc(C(C)C(=O)O)cc1").unwrap();
    let h_ibu = predict_nmr(&ibu, NmrNucleus::H1, Some("CDCl3"), None);
    let total_h: usize = h_ibu.signals.iter().map(|s| s.nuclei).sum();
    assert_eq!(total_h, 18, "Ibuprofen integrates to 18 protons");
    let cooh = &h_ibu.signals[0];
    assert!(
        cooh.shift_ppm > 10.5 && cooh.broad,
        "Carboxylic OH downfield broad signal"
    );
    let isobutyl_me = h_ibu
        .signals
        .iter()
        .find(|s| s.nuclei == 6)
        .expect("gem-dimethyl 6H doublet");
    assert_eq!(isobutyl_me.multiplicity(), "d");
    assert!(isobutyl_me.shift_ppm > 0.75 && isobutyl_me.shift_ppm < 1.10);

    let c_ibu = predict_nmr(&ibu, NmrNucleus::C13, Some("CDCl3"), None);
    assert_eq!(
        c_ibu.signals.len(),
        10,
        "Ibuprofen has 10 symmetry-equivalent ¹³C signals"
    );
    let total_c: usize = c_ibu.signals.iter().map(|s| s.nuclei).sum();
    assert_eq!(total_c, 13);

    // L-Alanine in D2O: N[C@@H](C)C(=O)O -> exchangeable NH2 & COOH gone -> CH quartet (1H) + CH3 doublet (3H)!
    let ala = AnalytMol::from_smiles("N[C@@H](C)C(=O)O").unwrap();
    let h_ala = predict_nmr(&ala, NmrNucleus::H1, Some("D2O"), None);
    assert_eq!(h_ala.signals.len(), 2);
    assert_eq!(h_ala.signals[0].nuclei, 1);
    assert_eq!(h_ala.signals[0].multiplicity(), "q");
    assert_eq!(h_ala.signals[1].nuclei, 3);
    assert_eq!(h_ala.signals[1].multiplicity(), "d");

    // Cis vs Trans alkene ³J coupling + Aldehyde ³J (~2.5 Hz)
    let trans_cinnam = AnalytMol::from_smiles("O=C/C=C/c1ccccc1").unwrap();
    let h_tr = predict_nmr(&trans_cinnam, NmrNucleus::H1, Some("CDCl3"), None);
    assert!(
        h_tr.signals[0].shift_ppm > 9.4,
        "Aldehyde CHO ~9.7 ppm, got {}",
        h_tr.signals[0].shift_ppm
    );
    assert!(
        h_tr.signals
            .iter()
            .any(|s| s.couplings.iter().any(|c| (c.j_hz - 15.5).abs() < 1.0)),
        "Trans alkene must exhibit ~15.5 Hz ³J coupling"
    );

    let cis_alk = AnalytMol::from_smiles("Cl/C=C\\Br").unwrap();
    let h_cis = predict_nmr(&cis_alk, NmrNucleus::H1, Some("CDCl3"), None);
    assert!(
        h_cis
            .signals
            .iter()
            .all(|s| s.couplings.iter().any(|c| (c.j_hz - 10.5).abs() < 1.0)),
        "Cis alkene must exhibit ~10.5 Hz ³J coupling"
    );
}

// ============================================================================
// 2. 1D Lorentzian/Voigt Synthesis & JCAMP-DX (AFFN + DIFDUP + MS) Round-Trip
// ============================================================================

#[test]
fn test_1d_nmr_synthesis_integration_picking_and_jcamp_roundtrip() {
    let etoac = AnalytMol::from_smiles("CC(=O)OCC").unwrap();
    let pred = predict_nmr(&etoac, NmrNucleus::H1, Some("CDCl3"), None);

    let opts = Synthesize1dOptions {
        mhz_1h: 500.0,
        n_points: 16384,
        fwhm_hz: 1.0,
        gaussian_fraction: 0.15,
        title: "Ethyl Acetate 500 MHz 1H".to_string(),
        ..Default::default()
    };
    let spec = synthesize_1d_nmr(&pred, &opts);
    assert_eq!(spec.y.len(), 16384);
    assert_eq!(spec.kind, SpectrumKind::Nmr);

    // Total cumulative integral should equal 8.0 protons (OCH2 2H + Ac 3H + CH3 3H)
    let int_curve = spec.integral.as_ref().expect("synthesized integral curve");
    let total_int = *int_curve.last().unwrap();
    assert!(
        (total_int - 8.0).abs() < 0.05,
        "Cumulative integral = {total_int}, expected 8.0"
    );

    // Regional integration around OCH2 (~4.1 ppm -> 2H)
    let och2_area = spec.integrate_region(3.8, 4.4);
    assert!(
        (och2_area - 2.0).abs() < 0.08,
        "OCH2 integral = {och2_area}, expected ~2.0"
    );

    // Multiplet picker recovers all 3 multiplets (q, s, t)
    let mults = pick_1d_multiplets(&spec, 0.02, 20.0, Some(8.0));
    assert_eq!(
        mults.len(),
        3,
        "Expected 3 multiplets for EtOAc, got {mults:?}"
    );
    assert_eq!(
        mults[0].peaks.len(),
        4,
        "Downfield quartet has 4 picked sub-peaks"
    );
    assert_eq!(mults[1].peaks.len(), 1, "Acetyl singlet has 1 picked peak");
    assert_eq!(
        mults[2].peaks.len(),
        3,
        "Upfield triplet has 3 picked sub-peaks"
    );

    // JCAMP-DX AFFN round-trip
    let jdx_affn = write_jcamp_nmr(&spec, JcampCompression::Affn);
    assert!(jdx_affn.contains("##TITLE= Ethyl Acetate 500 MHz 1H"));
    assert!(jdx_affn.contains("##.OBSERVE FREQUENCY="));
    assert!(jdx_affn.contains("##XYDATA= (X++(Y..Y))"));
    let parsed_affn = read_jcamp(&jdx_affn).expect("valid AFFN JCAMP");
    assert_eq!(parsed_affn.len(), 1);
    assert_eq!(parsed_affn[0].y.len(), spec.y.len());
    let max_diff_affn = spec
        .y
        .iter()
        .zip(parsed_affn[0].y.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(max_diff_affn < 1e-4, "AFFN max error = {max_diff_affn}");

    // JCAMP-DX DIFDUP (SQZ + DIF + DUP + Y-check) round-trip
    let jdx_dif = write_jcamp_nmr(&spec, JcampCompression::DifDup);
    assert!(
        jdx_dif.len() < jdx_affn.len(),
        "DIFDUP should compress continuous baseline"
    );
    let parsed_dif = read_jcamp(&jdx_dif).expect("valid DIFDUP JCAMP");
    assert_eq!(parsed_dif[0].y.len(), spec.y.len());
    let max_y = spec.y.iter().copied().fold(0.0f64, f64::max);
    let max_rel_err = spec
        .y
        .iter()
        .zip(parsed_dif[0].y.iter())
        .map(|(a, b)| (a - b).abs() / max_y)
        .fold(0.0f64, f64::max);
    assert!(max_rel_err < 2e-5, "DIFDUP relative error = {max_rel_err}");

    // MS JCAMP-DX PEAK TABLE round-trip
    let bap = AnalytMol::from_smiles("CC(=O)c1ccc(Br)cc1").unwrap();
    let env = isotope_envelope_for_mol(&bap, 0.5).unwrap();
    let sticks: Vec<(f64, f64)> = env.iter().map(|p| (p.mz, p.rel_abundance)).collect();
    let jdx_ms = write_jcamp_ms("4-Bromoacetophenone EI-MS", &sticks);
    assert!(jdx_ms.contains("##DATA TYPE= MASS SPECTRUM"));
    assert!(jdx_ms.contains("##PEAK TABLE= (XY..XY)"));
    let parsed_ms = read_jcamp(&jdx_ms).expect("valid MS JCAMP");
    assert_eq!(parsed_ms[0].kind, SpectrumKind::Ms);
    assert!(parsed_ms[0].sticks);
    assert_eq!(parsed_ms[0].y.len(), sticks.len());
}

// ============================================================================
// 3. Exact Mass, Cl/Br Isotopic Envelopes, ESI Adducts, Fragmentation & Props
// ============================================================================

#[test]
fn test_mass_props_isotopes_adducts_and_fragmentation() {
    // Ibuprofen (`C13H18O2`): exact mass = 206.130680 u, MW = 206.28 g/mol, RDBE = 5, HBD = 1, HBA = 2, RotB = 4
    let ibu = AnalytMol::from_smiles("CC(C)Cc1ccc(C(C)C(=O)O)cc1").unwrap();
    let props = compute_phys_chem_props(&ibu).expect("valid ibuprofen props");
    assert_eq!(props.formula, "C13H18O2");
    assert!((props.monoisotopic_mass - 206.13068).abs() < 1e-4);
    assert!((props.molecular_weight - 206.28).abs() < 0.05);
    assert!((props.rdbe - 5.0).abs() < 1e-6);
    assert_eq!(props.h_bond_donors, 1);
    assert_eq!(props.h_bond_acceptors, 2);
    assert_eq!(props.rotatable_bonds, 4);
    assert!(
        (props.tpsa - 37.30).abs() < 0.5,
        "Ibuprofen Ertl TPSA = {}",
        props.tpsa
    );
    assert!(
        props.clogp > 3.0 && props.clogp < 4.3,
        "Ibuprofen Crippen cLogP = {}",
        props.clogp
    );
    assert!(props.lipinski_ro5_pass);

    // Chlorobenzene (`C6H5Cl`): [M] : [M+2] ~ 100 : 32.0 (³⁵Cl 75.76% / ³⁷Cl 24.24%)
    let cl_bz = AnalytMol::from_smiles("Clc1ccccc1").unwrap();
    let cl_env = isotope_envelope_for_mol(&cl_bz, 0.5).unwrap();
    assert!((cl_env[0].mz - 112.00798).abs() < 1e-3);
    assert!((cl_env[0].rel_abundance - 100.0).abs() < 1e-6);
    let cl_m2 = cl_env
        .iter()
        .find(|p| (p.mz - 114.005).abs() < 0.05)
        .expect("M+2 peak");
    assert!(
        (cl_m2.rel_abundance - 32.0).abs() < 1.2,
        "Chlorobenzene M+2 = {}, expected ~32%",
        cl_m2.rel_abundance
    );

    // 4-Bromoacetophenone (`C8H7BrO`): [M] (197.968) : [M+2] (199.966) ~ 100 : 97.5 (⁷⁹Br 50.69% / ⁸¹Br 49.31%)
    let br_ap = AnalytMol::from_smiles("CC(=O)c1ccc(Br)cc1").unwrap();
    let br_env = isotope_envelope_for_mol(&br_ap, 0.5).unwrap();
    let br_m0 = &br_env[0];
    let br_m2 = br_env
        .iter()
        .find(|p| (p.mz - (br_m0.mz + 2.0)).abs() < 0.05)
        .expect("Br M+2");
    assert!(
        (br_m2.rel_abundance - 97.5).abs() < 2.0,
        "Bromide 1:1 doublet M+2 = {}",
        br_m2.rel_abundance
    );

    // Dibromomethane (`CH2Br2`): classic 1 : 2 : 1 triplet -> M+2 is base peak (100%), M and M+4 ~ 51% & 49%!
    let ch2br2 = AnalytMol::from_smiles("BrCBr").unwrap();
    let br2_env = isotope_envelope_for_mol(&ch2br2, 0.5).unwrap();
    let br2_m0 = &br2_env[0];
    let br2_m2 = br2_env
        .iter()
        .find(|p| (p.mz - (br2_m0.mz + 2.0)).abs() < 0.05)
        .expect("CH2Br2 M+2 peak");
    let br2_m4 = br2_env
        .iter()
        .find(|p| (p.mz - (br2_m0.mz + 4.0)).abs() < 0.05)
        .expect("CH2Br2 M+4 peak");
    assert!(
        (br2_m2.rel_abundance - 100.0).abs() < 1e-6,
        "CH2Br2 M+2 is 100% base peak"
    );
    assert!((br2_m0.rel_abundance - 51.4).abs() < 2.0);
    assert!((br2_m4.rel_abundance - 48.6).abs() < 2.0);

    // ESI/MALDI Adducts on Ibuprofen (`[M+H]+`, `[M+Na]+`, `[M+K]+`, `[M+NH4]+`, `[2M+Na]+`, `[M−H]−`)
    let adducts = calculate_standard_ms_adducts(&ibu);
    let mh = adducts
        .iter()
        .find(|a| a.kind == MsIonKind::MPlusH)
        .unwrap();
    let mna = adducts
        .iter()
        .find(|a| a.kind == MsIonKind::MPlusNa)
        .unwrap();
    let mk = adducts
        .iter()
        .find(|a| a.kind == MsIonKind::MPlusK)
        .unwrap();
    let mnh4 = adducts
        .iter()
        .find(|a| a.kind == MsIonKind::MPlusNH4)
        .unwrap();
    let m2na = adducts
        .iter()
        .find(|a| a.kind == MsIonKind::TwoMPlusNa)
        .unwrap();
    let m_h = adducts
        .iter()
        .find(|a| a.kind == MsIonKind::MMinusH)
        .unwrap();

    assert!((mh.monoisotopic_mz - 207.1380).abs() < 1e-3);
    assert!((mna.monoisotopic_mz - 229.1199).abs() < 1e-3);
    assert!((mk.monoisotopic_mz - 245.0938).abs() < 1e-3);
    assert!((mnh4.monoisotopic_mz - 224.1645).abs() < 1e-3);
    assert!((m2na.monoisotopic_mz - 435.2506).abs() < 1e-3);
    assert!((m_h.monoisotopic_mz - 205.1234).abs() < 1e-3);
    let hrms = hrms_report_line("ESI", mna, Some(229.1201), 4);
    assert!(hrms.contains("[M+Na]+ calcd for C13H18NaO2 229.1199; found 229.1201."));

    // Single-bond MS cleavage on 4-bromoacetophenone: acylium `[CH3CO]+` (m/z 43.018) & `[BrC6H4CO]+` (m/z 182.944)
    let mut bap_2d = br_ap.clone();
    bap_2d.generate_2d_coords_if_missing();
    let frags = predict_ms_fragments(&bap_2d);
    assert!(
        frags
            .iter()
            .any(|f| f.formula == "C2H3O" && (f.cation_mz - 43.018).abs() < 0.01),
        "Expected CH3CO+ acylium fragment (43.018), got {frags:?}"
    );
    assert!(
        frags
            .iter()
            .any(|f| f.formula == "C7H4BrO" && (f.cation_mz - 182.944).abs() < 0.01),
        "Expected BrC6H4CO+ benzoylium fragment (182.944)"
    );

    // 2D cut-path helper test across bond 0
    let b0 = &bap_2d.bonds[0];
    let (ax, ay) = (bap_2d.atoms[b0.a].x, bap_2d.atoms[b0.a].y);
    let (bx, by) = (bap_2d.atoms[b0.b].x, bap_2d.atoms[b0.b].y);
    let (mx, my) = (0.5 * (ax + bx), 0.5 * (ay + by));
    let (nx, ny) = (-(by - ay), bx - ax);
    let crossed = crossed_bonds_along_path(&bap_2d, &[(mx - nx, my - ny), (mx + nx, my + ny)]);
    assert!(crossed.iter().any(|&(bi, _)| bi == 0));
    let crossed_indices: Vec<usize> = crossed.into_iter().map(|(bi, _)| bi).collect();
    let cut_frags =
        fragments_from_bond_cuts(&bap_2d, &[crossed_indices], FragmentationMode::Simultaneous);
    assert!(!cut_frags.is_empty());
}

// ============================================================================
// 4. 2D Reaction Arrow Perception, Balance & Laboratory Stoichiometry Engine
// ============================================================================

#[test]
fn test_reaction_perception_balance_and_stoichiometry_table() {
    // Fischer esterification / acetylation on a 2D canvas:
    // salicylic acid (left of arrow) + acetic anhydride (left of arrow)
    // --[H2SO4 (above), 80 °C, N2 (below)]--> aspirin (right of arrow, "88% yield") + acetic acid (right)
    let mut salic = AnalytMol::from_smiles("Oc1ccccc1C(=O)O").unwrap();
    salic.generate_2d_coords_if_missing();
    for a in &mut salic.atoms {
        a.x -= 12.0;
    }
    let mut ac2o = AnalytMol::from_smiles("CC(=O)OC(=O)C").unwrap();
    ac2o.generate_2d_coords_if_missing();
    for a in &mut ac2o.atoms {
        a.x -= 6.0;
    }
    let mut dmap_cat = AnalytMol::from_smiles("CN(C)c1ccncc1").unwrap();
    dmap_cat.generate_2d_coords_if_missing();
    for a in &mut dmap_cat.atoms {
        a.y += 3.5;
    }
    let mut aspirin = AnalytMol::from_smiles("CC(=O)Oc1ccccc1C(=O)O").unwrap();
    aspirin.generate_2d_coords_if_missing();
    for a in &mut aspirin.atoms {
        a.x += 9.0;
    }
    let mut acoh = AnalytMol::from_smiles("CC(=O)O").unwrap();
    acoh.generate_2d_coords_if_missing();
    for a in &mut acoh.atoms {
        a.x += 16.0;
    }

    let arrow = ReactionArrow {
        id: 1,
        from: (-2.5, 0.0),
        to: (4.5, 0.0),
    };
    let captions = vec![
        ReactionCaption {
            id: 10,
            text: "N2".to_string(),
            bbox: [0.0, -2.5, 1.5, -1.5],
            pinned_on_arrow: Some((1, false)),
        },
        ReactionCaption {
            id: 11,
            text: "88% yield".to_string(),
            bbox: [8.0, -3.0, 10.5, -2.0],
            pinned_on_arrow: None,
        },
    ];

    let steps = perceive_reaction_canvas(
        &[
            salic.clone(),
            ac2o.clone(),
            dmap_cat,
            aspirin.clone(),
            acoh.clone(),
        ],
        &captions,
        &[arrow],
    );
    assert_eq!(steps.len(), 1);
    let step = &steps[0];
    assert_eq!(step.reactants.len(), 2);
    assert_eq!(step.reagents_above.len(), 1);
    assert_eq!(step.reagents_above[0].role, ReactionRole::ReagentAbove);
    assert_eq!(step.products.len(), 2);
    assert_eq!(step.products[0].attached_yield_percent, Some(88.0));
    assert!(step.condition_texts[0].is_inert_atmosphere);

    // Atom balance check: Salicylic acid + Ac2O == Aspirin + AcOH (100% atom-balanced!)
    let bal_ok = check_reaction_balance(&[(&salic, 1), (&ac2o, 1)], &[(&aspirin, 1), (&acoh, 1)]);
    assert!(
        bal_ok.balanced,
        "Full reaction is atom-balanced: {}",
        bal_ok.summary
    );

    // Without AcOH byproduct drawn, delta is `-C2H4O2`
    let bal_cond = check_reaction_balance(&[(&salic, 1), (&ac2o, 1)], &[(&aspirin, 1)]);
    assert!(!bal_cond.balanced);
    assert_eq!(bal_cond.summary, "-C2H4O2");

    // Laboratory Stoichiometry Table:
    // - Salicylic acid (MW 138.12): 1.3812 g -> 10.00 mmol (limiting reagent = 1.00 eq!)
    // - Acetic anhydride (MW 102.09, density 1.08 g/mL): 1.50 mL -> 15.87 mmol (1.59 eq, excess!)
    // - DMAP catalyst (MW 122.17): typed as 0.05 eq -> 0.50 mmol -> 61.1 mg!
    // - Aspirin product (MW 180.16): theoretical = 10.00 mmol = 1.8016 g; actual isolated = 1.5854 g -> 88.0% yield!
    let mut c_sal = StoichComponent::from_mol("sal", "Salicylic acid", &salic, false);
    c_sal.typed = c_sal.typed.with_mass_g(1.3812);

    let mut c_anh = StoichComponent::from_mol("anh", "Acetic anhydride", &ac2o, false);
    c_anh.typed = c_anh.typed.with_volume_ml(1.50).with_density(1.08);

    let mut c_cat = StoichComponent::from_formula("dmap", "C7H10N2", false).unwrap();
    c_cat.can_be_limiting = false;
    c_cat.typed = c_cat.typed.with_equiv(0.05);

    let mut c_asp = StoichComponent::from_mol("asp", "Aspirin", &aspirin, true);
    c_asp.typed = c_asp.typed.with_actual_mass_g(1.5854);

    let table = solve_stoichiometry(&[c_sal, c_anh, c_cat, c_asp], None);
    assert_eq!(
        table.limiting_key.as_deref(),
        Some("sal"),
        "Salicylic acid auto-detected as limiting"
    );

    let r_sal = table.row("sal").unwrap();
    assert!(r_sal.is_limiting);
    assert!((r_sal.mmol.unwrap() - 10.0).abs() < 0.02);
    assert!((r_sal.equiv.unwrap() - 1.0).abs() < 1e-4);

    let r_anh = table.row("anh").unwrap();
    assert!(!r_anh.is_limiting);
    assert!((r_anh.mmol.unwrap() - 15.87).abs() < 0.05);
    assert!((r_anh.equiv.unwrap() - 1.587).abs() < 0.02);
    assert!((r_anh.mass_g.unwrap() - 1.62).abs() < 0.01);

    let r_cat = table.row("dmap").unwrap();
    assert!((r_cat.mmol.unwrap() - 0.50).abs() < 0.01);
    assert!((r_cat.mass_mg.unwrap() - 61.08).abs() < 0.2);

    let r_asp = table.row("asp").unwrap();
    assert!((r_asp.theoretical_mmol.unwrap() - 10.0).abs() < 0.02);
    assert!((r_asp.theoretical_mass_g.unwrap() - 1.8016).abs() < 0.01);
    assert!((r_asp.percent_yield.unwrap() - 88.0).abs() < 0.2);

    // Unit parser & formatter sanity checks
    assert!((parse_stoich_amount("250 mg", StoichField::Mass).unwrap() - 0.250).abs() < 1e-9);
    assert!((parse_stoich_amount("500 uL", StoichField::Volume).unwrap() - 0.500).abs() < 1e-9);
    assert_eq!(format_stoich_amount(0.250, StoichField::Mass), "250 mg");
}

// ============================================================================
// 5. Offline Bidirectional IUPAC Structure ↔ Name Tests
// ============================================================================

#[test]
fn test_bidirectional_iupac_structure_and_name() {
    // 1. Exact PIN generation from benchmark structures
    let cases_mol_to_name = [
        ("CCO", "ethanol"),
        ("CC(C)(C)O", "2-methylpropan-2-ol"),
        ("CC(=O)OCC", "ethyl ethanoate"),
        ("Nc1ccc(C(=O)O)cc1", "4-aminobenzoic acid"),
        ("CC(=O)c1ccc(Br)cc1", "1-(4-bromophenyl)ethan-1-one"),
        (
            "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
            "2-[4-(2-methylpropyl)phenyl]propanoic acid",
        ),
        ("N[C@@H](C)C(=O)O", "(2S)-2-aminopropanoic acid"),
        ("C/C=C/C", "(2E)-but-2-ene"),
        ("C/C=C\\C", "(2Z)-but-2-ene"),
        ("C1=CC2CCC1C2", "bicyclo[2.2.1]hept-2-ene"),
        ("C1CC2CCC1O2", "7-oxabicyclo[2.2.1]heptane"),
        ("C1CCC2(C1)CCCCC2", "spiro[4.5]decane"),
        ("C1C2CC3CC1CC(C2)C3", "tricyclo[3.3.1.1³,⁷]decane"),
        (
            "OC(=O)c1ccc(Oc2ccc(C(=O)O)cc2)cc1",
            "4,4'-oxydibenzoic acid",
        ),
        ("c1ccncc1", "pyridine"),
        ("C1CCOC1", "oxolane"),
        ("c1ccoc1", "furan"),
        ("c1ccc2[nH]ccc2c1", "1H-indole"),
        ("O", "oxidane"),
        ("N", "azane"),
        ("[SiH4]", "silane"),
    ];

    for (smi, expected_pin) in cases_mol_to_name {
        let mol = AnalytMol::from_smiles(smi).unwrap_or_else(|e| panic!("SMILES {smi}: {e}"));
        let pin = mol_to_iupac_name(&mol).unwrap_or_else(|e| panic!("mol_to_iupac({smi}): {e}"));
        assert_eq!(pin, expected_pin, "PIN mismatch for SMILES {smi}");
    }

    // 2. Systematic & dictionary Name -> Structure -> PIN round-trip!
    let roundtrip_names = [
        "2-methylpropan-2-ol",
        "4-aminobenzoic acid",
        "bicyclo[2.2.1]hept-2-ene",
        "(2S)-2-aminopropanoic acid",
        "(2E)-but-2-ene",
        "(2Z)-but-2-ene",
        "ethyl ethanoate",
        "1-(4-bromophenyl)ethan-1-one",
        "2-[4-(2-methylpropyl)phenyl]propanoic acid",
        "4,4'-oxydibenzoic acid",
        "7-oxabicyclo[2.2.1]heptane",
        "spiro[4.5]decane",
        "pyridine",
        "oxolane",
        "1H-indole",
        "oxidane",
        "silane",
    ];

    for name in roundtrip_names {
        let parsed = iupac_name_to_mol(name).unwrap_or_else(|e| panic!("parse '{name}': {e}"));
        let regen = mol_to_iupac_name(&parsed).unwrap_or_else(|e| panic!("regen '{name}': {e}"));
        assert_eq!(
            regen, name,
            "Bidirectional IUPAC round-trip failed for '{name}'"
        );
    }

    // 3. Common reagents / drugs / amino-acids dictionary resolution
    for (common, expected_formula) in [
        ("ibuprofen", "C13H18O2"),
        ("ethyl acetate", "C4H8O2"),
        ("4-bromoacetophenone", "C8H7BrO"),
        ("L-alanine", "C3H7NO2"),
        ("THF", "C4H8O"),
        ("DMF", "C3H7NO"),
        ("caffeine", "C8H10N4O2"),
        ("aspirin", "C9H8O4"),
    ] {
        let mol = iupac_name_to_mol(common).unwrap();
        let props = compute_phys_chem_props(&mol).unwrap();
        assert_eq!(
            props.formula, expected_formula,
            "Formula mismatch for {common}"
        );
        assert!(!iupac_name_to_smiles(common).unwrap().is_empty());
    }

    // Verify manual `AnalytMol` graph construction (zero-friction integration path for `octet_mol::Molecule`)
    let mut manual = AnalytMol::new();
    let c0 = manual.add_atom(AnalytAtom::new("C").with_coords(0.0, 0.0));
    let c1 = manual.add_atom(AnalytAtom::new("C").with_coords(1.5, 0.0));
    let o2 = manual.add_atom(AnalytAtom::new("O").with_coords(2.25, 1.3));
    manual.add_bond(AnalytBond::single(c0, c1));
    manual.add_bond(AnalytBond::single(c1, o2));
    assert_eq!(mol_to_iupac_name(&manual).unwrap(), "ethanol");
}
