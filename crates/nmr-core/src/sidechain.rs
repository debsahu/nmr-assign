//! Sidechain expected-peak models, 2D/3D simulators & pickers, and joint backbone + sidechain
//! graph resonance assignment optimizer.
//!
//! Provenance: Clean-room original implementation based on published heteronuclear TOCSY
//! pulse-sequence physics and IUPAC covalent spin-system topologies:
//! - `C(CCO)NH` & `H(CCCO)NH` (`i-1` aliphatic sidechain `13C`/`1H` to amide `i` relay):
//!   Grzesiek, Anglister & Bax, *J. Magn. Reson. B* 101, 114-119 (1993);
//!   Montelione, Lyons, Emerson, Tashiro & Farmer, *J. Am. Chem. Soc.* 114, 10974-10975 (1992).
//! - 3D `HCCH-TOCSY` (intra-residue aliphatic `1H-13C-13C-1H` isotropic TOCSY relay):
//!   Bax, Clore & Gronenborn, *J. Magn. Reson.* 88, 425-431 (1990);
//!   Kay, Ikura & Bax, *J. Am. Chem. Soc.* 112, 888-889 (1990).
//!
//! ### Experiment & Axis Conventions
//! - `C13HSQC` (2D `1H-13C` one-bond correlation):
//!   - `Peak`: `exp = "C13HSQC"`, `h = H_x`, `n = 0.0`, `c = Some(C_x)`, `sign = +1` (or `-1` for
//!     `CH2` when multiplicity editing is enabled), `h2 = None`.
//!   - `Spectrum2D`: `axis_x = "1H"` (direct), `axis_y = "13C"` (indirect).
//! - `CCCONH` (3D `(H)CC(CO)NH`-TOCSY):
//!   - `Peak`: `exp = "CCCONH"`, `h = H_N[i]`, `n = N[i]`, `c = Some(C_aliph[i-1])`, `sign = 1`, `h2 = None`.
//!   - `Spectrum3D`: `axis_x = "1H"` (`H_N`), `axis_y = "15N"` (`N`), `axis_z = "13C"` (`C_aliph`).
//! - `HCCCONH` (3D `H(CCCO)NH`-TOCSY):
//!   - `Peak`: `exp = "HCCCONH"`, `h = H_N[i]`, `n = N[i]`, `c = None`, `sign = 1`, `h2 = Some(H_aliph[i-1])`.
//!   - `Spectrum3D`: `axis_x = "1H"` (`H_N`), `axis_y = "15N"` (`N`), `axis_z = "1H_aliph"` (`H_aliph`).
//! - `HCCHTOCSY` (3D `HCCH-TOCSY`):
//!   - `Peak`: `exp = "HCCHTOCSY"`, `h = H_attached[i]`, `n = 0.0`, `c = Some(C_attached[i])`,
//!     `sign = 1`, `h2 = Some(H_tocsy[i])` (intra-residue proton relayed across the continuous
//!     `13C-13C` TOCSY spin system, including the diagonal auto-peak `H_tocsy == H_attached`).
//!   - `Spectrum3D`: `axis_x = "1H"` (`H_attached`), `axis_y = "13C"` (`C_attached`),
//!     `axis_z = "1H_tocsy"` (`H_tocsy`).

use std::collections::{BTreeMap, BTreeSet};

use crate::evaluate::to_reference;
use crate::nmrstar::Reference;
use crate::peaks::Peak;
use crate::picker::estimate_noise_mad;
use crate::priors::Priors;
use crate::rng::Rng;
use crate::solver::{
    assign_from_matrices, pair_matrix, unary_matrix, AnnealConfig, AssignResult, Matrix2D,
    ScoreConfig, FORBID,
};
use crate::spectrum::{AxisPpm, Spectrum2D, Spectrum3D};
use crate::spinsystems::SpinSystem;
use crate::topology::{
    residue_topology, CarbonRegion, ChGroupSpec, ChMultiplicity, ProchiralGroup, ResidueTopology,
    SidechainPriors,
};

pub const SIDECHAIN_EXPERIMENTS: [&str; 4] = ["C13HSQC", "CCCONH", "HCCCONH", "HCCHTOCSY"];

// ============================================================================
// 1. Expected-Peak Models, Synthetic All-Atom Builder & Spectrum Simulators
// ============================================================================

/// Distinct observable `(H, C)` one-bond resonance within a residue after collapsing
/// magnetically equivalent methyl protons (`HB1..3`, `HG11..13`, etc.) and degenerate geminal/ring pairs.
#[derive(Clone, Debug)]
pub struct ObservableChResonance {
    pub residue_idx: usize,
    pub carbon_atom: &'static str,
    pub proton_atoms: Vec<&'static str>,
    pub c_ppm: f64,
    pub h_ppm: f64,
    pub multiplicity: ChMultiplicity,
    pub region: CarbonRegion,
    pub steps_from_ca: Option<u8>,
    /// Number of equivalent protons contributing to this peak's intensity (`1`, `2`, or `3`).
    pub equivalent_protons: usize,
}

/// Extract deduplicated observable `(H, C)` resonances for residue `k` from `ref_data`.
pub fn observable_ch_resonances(ref_data: &Reference, k: usize) -> Vec<ObservableChResonance> {
    let Some(res_name) = ref_data.sequence.get(k) else {
        return Vec::new();
    };
    let Some(topo) = residue_topology(res_name) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    let mut handled_carbons: BTreeSet<&'static str> = BTreeSet::new();

    // Handle symmetric aromatic pairs (PHE/TYR CD1==CD2, CE1==CE2) if their shifts coincide
    for pg in topo.prochiral_groups {
        if let ProchiralGroup::SymmetricAromatic {
            carbon_a,
            proton_a,
            carbon_b,
            proton_b,
        } = pg
        {
            let ca = ref_data.shifts.get(&(k, (*carbon_a).to_string())).copied();
            let ha = ref_data.shifts.get(&(k, (*proton_a).to_string())).copied();
            let cb = ref_data.shifts.get(&(k, (*carbon_b).to_string())).copied();
            let hb = ref_data.shifts.get(&(k, (*proton_b).to_string())).copied();
            if let (Some(ca_v), Some(ha_v), Some(cb_v), Some(hb_v)) = (ca, ha, cb, hb) {
                if (ca_v - cb_v).abs() < 0.02 && (ha_v - hb_v).abs() < 0.005 {
                    out.push(ObservableChResonance {
                        residue_idx: k,
                        carbon_atom: carbon_a,
                        proton_atoms: vec![proton_a, proton_b],
                        c_ppm: 0.5 * (ca_v + cb_v),
                        h_ppm: 0.5 * (ha_v + hb_v),
                        multiplicity: ChMultiplicity::Methine,
                        region: CarbonRegion::Aromatic,
                        steps_from_ca: None,
                        equivalent_protons: 2,
                    });
                    handled_carbons.insert(carbon_a);
                    handled_carbons.insert(carbon_b);
                }
            }
        }
    }

    for chg in topo.ch_groups {
        if handled_carbons.contains(chg.carbon) {
            continue;
        }
        let Some(&c_ppm) = ref_data.shifts.get(&(k, chg.carbon.to_string())) else {
            continue;
        };
        let mut present_h: Vec<(&'static str, f64)> = Vec::new();
        for &p_name in chg.protons {
            if let Some(&h_ppm) = ref_data.shifts.get(&(k, p_name.to_string())) {
                present_h.push((p_name, h_ppm));
            }
        }
        if present_h.is_empty() {
            continue;
        }

        match chg.multiplicity {
            ChMultiplicity::Methyl => {
                let mean_h = present_h.iter().map(|x| x.1).sum::<f64>() / (present_h.len() as f64);
                out.push(ObservableChResonance {
                    residue_idx: k,
                    carbon_atom: chg.carbon,
                    proton_atoms: chg.protons.to_vec(),
                    c_ppm,
                    h_ppm: mean_h,
                    multiplicity: ChMultiplicity::Methyl,
                    region: chg.region,
                    steps_from_ca: chg.steps_from_ca,
                    equivalent_protons: 3,
                });
            }
            ChMultiplicity::Methine => {
                out.push(ObservableChResonance {
                    residue_idx: k,
                    carbon_atom: chg.carbon,
                    proton_atoms: vec![present_h[0].0],
                    c_ppm,
                    h_ppm: present_h[0].1,
                    multiplicity: ChMultiplicity::Methine,
                    region: chg.region,
                    steps_from_ca: chg.steps_from_ca,
                    equivalent_protons: 1,
                });
            }
            ChMultiplicity::Methylene => {
                if present_h.len() >= 2 && (present_h[0].1 - present_h[1].1).abs() < 0.008 {
                    // Degenerate geminal CH2 pair -> single coalesced peak of 2-proton intensity
                    out.push(ObservableChResonance {
                        residue_idx: k,
                        carbon_atom: chg.carbon,
                        proton_atoms: vec![present_h[0].0, present_h[1].0],
                        c_ppm,
                        h_ppm: 0.5 * (present_h[0].1 + present_h[1].1),
                        multiplicity: ChMultiplicity::Methylene,
                        region: chg.region,
                        steps_from_ca: chg.steps_from_ca,
                        equivalent_protons: 2,
                    });
                } else {
                    for (p_name, h_ppm) in present_h {
                        out.push(ObservableChResonance {
                            residue_idx: k,
                            carbon_atom: chg.carbon,
                            proton_atoms: vec![p_name],
                            c_ppm,
                            h_ppm,
                            multiplicity: ChMultiplicity::Methylene,
                            region: chg.region,
                            steps_from_ca: chg.steps_from_ca,
                            equivalent_protons: 1,
                        });
                    }
                }
            }
        }
    }

    out
}

/// Complete a backbone-only `Reference` into a realistic all-atom `Reference` by sampling
/// missing sidechain and `HA` shifts from `SidechainPriors` (respecting methyl & ring degeneracies).
pub fn synthesize_all_atom_reference(
    base: &Reference,
    priors: &SidechainPriors,
    seed: u64,
) -> Reference {
    let mut rng = Rng::new(seed ^ 0x51DE_C4A1_9E37_79B9);
    let mut out = base.clone();

    for (k, res) in base.sequence.iter().enumerate() {
        let Some(topo) = residue_topology(res) else {
            continue;
        };

        // Sample or preserve carbon and proton shifts per CH_n group
        for chg in topo.ch_groups {
            if let std::collections::btree_map::Entry::Vacant(e) =
                out.shifts.entry((k, chg.carbon.to_string()))
            {
                if let Some((mu_c, sd_c)) = priors.prior(res, chg.carbon) {
                    let c_val = mu_c + rng.normal(0.0, (sd_c * 0.65).min(1.1));
                    e.insert(c_val);
                }
            }
            match chg.multiplicity {
                ChMultiplicity::Methyl => {
                    let existing = chg
                        .protons
                        .iter()
                        .find_map(|p| out.shifts.get(&(k, (*p).to_string())).copied());
                    let h_val = if let Some(v) = existing {
                        v
                    } else if let Some((mu_h, sd_h)) = priors.prior(res, chg.protons[0]) {
                        mu_h + rng.normal(0.0, (sd_h * 0.65).min(0.22))
                    } else {
                        continue;
                    };
                    for &p in chg.protons {
                        out.shifts.insert((k, p.to_string()), h_val);
                    }
                }
                ChMultiplicity::Methine => {
                    let p = chg.protons[0];
                    if let std::collections::btree_map::Entry::Vacant(e) =
                        out.shifts.entry((k, p.to_string()))
                    {
                        if let Some((mu_h, sd_h)) = priors.prior(res, p) {
                            let h_val = mu_h + rng.normal(0.0, (sd_h * 0.65).min(0.30));
                            e.insert(h_val);
                        }
                    }
                }
                ChMultiplicity::Methylene => {
                    // ~20% of methylenes are near-degenerate in globular proteins; 80% split
                    let is_deg = rng.f64() < 0.20;
                    if let Some((mu0, sd0)) = priors.prior(res, chg.protons[0]) {
                        let center = mu0 + rng.normal(0.0, (sd0 * 0.55).min(0.22));
                        let split = if is_deg { 0.0 } else { rng.uniform(0.05, 0.32) };
                        let h_hi = center + 0.5 * split;
                        let h_lo = center - 0.5 * split;
                        out.shifts
                            .entry((k, chg.protons[0].to_string()))
                            .or_insert(h_hi);
                        if chg.protons.len() > 1 {
                            out.shifts
                                .entry((k, chg.protons[1].to_string()))
                                .or_insert(h_lo);
                        }
                    }
                }
            }
        }

        // Enforce fast-flip aromatic symmetry on PHE and TYR
        for pg in topo.prochiral_groups {
            if let ProchiralGroup::SymmetricAromatic {
                carbon_a,
                proton_a,
                carbon_b,
                proton_b,
            } = pg
            {
                if let Some(&ca) = out.shifts.get(&(k, (*carbon_a).to_string())) {
                    out.shifts.insert((k, (*carbon_b).to_string()), ca);
                }
                if let Some(&ha) = out.shifts.get(&(k, (*proton_a).to_string())) {
                    out.shifts.insert((k, (*proton_b).to_string()), ha);
                }
            }
        }
    }

    out
}

/// Configuration for simulating sidechain peak lists (`C13HSQC`, `CCCONH`, `HCCCONH`, `HCCHTOCSY`).
#[derive(Clone, Debug)]
pub struct SidechainSimConfig {
    pub experiments: Vec<String>,
    pub include_aromatics: bool,
    pub multiplicity_edited_hsqc: bool,
    pub noise_h: f64,
    pub noise_n: f64,
    pub noise_c: f64,
    pub p_missing_peak: f64,
    pub spurious_rate: f64,
    pub include_intensities: bool,
    pub seed: u64,
}

impl Default for SidechainSimConfig {
    fn default() -> Self {
        Self {
            experiments: SIDECHAIN_EXPERIMENTS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            include_aromatics: true,
            multiplicity_edited_hsqc: false,
            noise_h: 0.008,
            noise_n: 0.06,
            noise_c: 0.04,
            p_missing_peak: 0.05,
            spurious_rate: 0.02,
            include_intensities: true,
            seed: 42,
        }
    }
}

/// Simulate blind sidechain peak lists (`C13HSQC`, `CCCONH`, `HCCCONH`, `HCCHTOCSY`) from an
/// all-atom `Reference`.
pub fn simulate_sidechain_peaks(ref_data: &Reference, cfg: &SidechainSimConfig) -> Vec<Peak> {
    let mut rng = Rng::new(cfg.seed);
    let mut peaks: Vec<Peak> = Vec::new();

    let want_c13 = cfg.experiments.iter().any(|e| e == "C13HSQC");
    let want_ccconh = cfg.experiments.iter().any(|e| e == "CCCONH");
    let want_hccconh = cfg.experiments.iter().any(|e| e == "HCCCONH");
    let want_hcch = cfg.experiments.iter().any(|e| e == "HCCHTOCSY");

    let n_res = ref_data.sequence.len();
    let per_res_obs: Vec<Vec<ObservableChResonance>> = (0..n_res)
        .map(|k| observable_ch_resonances(ref_data, k))
        .collect();

    // 1. 2D C13HSQC and 3D HCCHTOCSY (intra-residue)
    for (k, obs_list) in per_res_obs.iter().enumerate() {
        let Some(topo) = residue_topology(&ref_data.sequence[k]) else {
            continue;
        };

        for (idx_u, u) in obs_list.iter().enumerate() {
            if !cfg.include_aromatics && u.region == CarbonRegion::Aromatic {
                continue;
            }

            if want_c13 && rng.f64() >= cfg.p_missing_peak * 0.5 {
                let h = u.h_ppm + rng.normal(0.0, cfg.noise_h);
                let c = u.c_ppm + rng.normal(0.0, cfg.noise_c);
                let sign = if cfg.multiplicity_edited_hsqc && u.region.is_aliphatic() {
                    u.multiplicity.edited_hsqc_sign()
                } else {
                    1
                };
                let mut pk = Peak::c13_hsqc(h, c, sign);
                if cfg.include_intensities {
                    let mult_scale = match u.equivalent_protons {
                        3 => 2.6,
                        2 => 1.85,
                        _ => 1.0,
                    };
                    let ht = 110.0 * mult_scale * (rng.normal(0.0, 0.15)).exp();
                    pk = pk.with_intensity(ht, Some(ht * 4.5));
                }
                peaks.push(pk);
            }

            // 3D HCCH-TOCSY: correlate (H_u, C_u) to all protons H_v in the SAME spin-system component
            if want_hcch && (u.region.is_aliphatic() || cfg.include_aromatics) {
                for (idx_v, v) in obs_list.iter().enumerate() {
                    if u.region != v.region {
                        continue;
                    }
                    // Isolated methyl (MET CE) only correlates with itself (no 13C-13C relay across SD)
                    if u.region == CarbonRegion::AliphaticIsolatedMethyl && idx_u != idx_v {
                        continue;
                    }
                    // For aromatics, only relay within coupled CH sub-networks (e.g. PHE/TYR/TRP 6-ring, not HIS isolated CHs)
                    if u.region == CarbonRegion::Aromatic
                        && idx_u != idx_v
                        && ref_data.sequence[k] == "HIS"
                    {
                        continue;
                    }

                    let is_diag = idx_u == idx_v;
                    let miss_prob = if is_diag {
                        cfg.p_missing_peak * 0.4
                    } else {
                        cfg.p_missing_peak
                    };
                    if rng.f64() < miss_prob {
                        continue;
                    }

                    let h1 = u.h_ppm + rng.normal(0.0, cfg.noise_h);
                    let c1 = u.c_ppm + rng.normal(0.0, cfg.noise_c);
                    let h2 = v.h_ppm + rng.normal(0.0, cfg.noise_h);

                    let mut pk = Peak {
                        exp: "HCCHTOCSY".to_string(),
                        h: h1,
                        n: 0.0,
                        c: Some(c1),
                        sign: 1,
                        height: None,
                        volume: None,
                        h2: Some(h2),
                    };

                    if cfg.include_intensities {
                        let bonded = u.carbon_atom == v.carbon_atom
                            || topo.ch_group(u.carbon_atom).is_some_and(|g| {
                                g.covalent_carbon_neighbors.contains(&v.carbon_atom)
                            });
                        let relay_atten = if is_diag {
                            1.25
                        } else if bonded {
                            0.85
                        } else {
                            0.55
                        };
                        let mult_boost = (v.equivalent_protons as f64).sqrt();
                        let ht = 85.0 * relay_atten * mult_boost * (rng.normal(0.0, 0.18)).exp();
                        pk = pk.with_intensity(ht, Some(ht * 5.2));
                    }
                    peaks.push(pk);
                }
            }
        }
    }

    // 2. 3D C(CCO)NH and H(CCCO)NH anchored on amide i (k >= 1, non-Pro) reporting AliphaticCaConnected of k-1
    let mut amide_roots: Vec<(f64, f64)> = Vec::new();
    for k in 1..n_res {
        if ref_data.sequence[k] == "PRO" {
            continue;
        }
        let Some(&hn) = ref_data.shifts.get(&(k, "H".to_string())) else {
            continue;
        };
        let Some(&n_ppm) = ref_data.shifts.get(&(k, "N".to_string())) else {
            continue;
        };
        amide_roots.push((hn, n_ppm));

        let prev_obs = &per_res_obs[k - 1];

        if want_ccconh {
            // One peak per distinct AliphaticCaConnected carbon of residue k-1
            let mut seen_carbons: BTreeSet<&'static str> = BTreeSet::new();
            for u in prev_obs {
                if u.region != CarbonRegion::AliphaticCaConnected {
                    continue;
                }
                if !seen_carbons.insert(u.carbon_atom) {
                    continue;
                }
                if rng.f64() < cfg.p_missing_peak {
                    continue;
                }
                let ph = hn + rng.normal(0.0, cfg.noise_h);
                let pn = n_ppm + rng.normal(0.0, cfg.noise_n);
                let pc = u.c_ppm + rng.normal(0.0, cfg.noise_c);
                let mut pk = Peak::triple("CCCONH", ph, pn, pc, 1);
                if cfg.include_intensities {
                    let steps = u.steps_from_ca.unwrap_or(2) as f64;
                    let atten = (0.86_f64).powf(steps);
                    let ht = 80.0 * atten * (rng.normal(0.0, 0.18)).exp();
                    pk = pk.with_intensity(ht, Some(ht * 5.5));
                }
                peaks.push(pk);
            }
        }

        if want_hccconh {
            // One peak per distinct AliphaticCaConnected proton resonance of residue k-1
            for u in prev_obs {
                if u.region != CarbonRegion::AliphaticCaConnected {
                    continue;
                }
                if rng.f64() < cfg.p_missing_peak {
                    continue;
                }
                let ph = hn + rng.normal(0.0, cfg.noise_h);
                let pn = n_ppm + rng.normal(0.0, cfg.noise_n);
                let ph2 = u.h_ppm + rng.normal(0.0, cfg.noise_h);
                let mut pk = Peak {
                    exp: "HCCCONH".to_string(),
                    h: ph,
                    n: pn,
                    c: None,
                    sign: 1,
                    height: None,
                    volume: None,
                    h2: Some(ph2),
                };
                if cfg.include_intensities {
                    let steps = u.steps_from_ca.unwrap_or(2) as f64;
                    let atten = (0.85_f64).powf(steps) * (u.equivalent_protons as f64).sqrt();
                    let ht = 75.0 * atten * (rng.normal(0.0, 0.18)).exp();
                    pk = pk.with_intensity(ht, Some(ht * 5.5));
                }
                peaks.push(pk);
            }
        }
    }

    // 3. Spurious noise artifacts
    if cfg.spurious_rate > 0.0 {
        if want_c13 {
            let n_c13 = peaks.iter().filter(|p| p.exp == "C13HSQC").count();
            let n_spur = rng.poisson(cfg.spurious_rate * (n_c13 as f64));
            for _ in 0..n_spur {
                let mut pk = Peak::c13_hsqc(rng.uniform(-0.3, 5.6), rng.uniform(11.0, 72.0), 1);
                if cfg.include_intensities {
                    let ht = 25.0 * (rng.normal(0.0, 0.25)).exp();
                    pk = pk.with_intensity(ht, Some(ht * 4.0));
                }
                peaks.push(pk);
            }
        }
        if !amide_roots.is_empty() && want_ccconh {
            let n_cc = peaks.iter().filter(|p| p.exp == "CCCONH").count();
            let n_spur = rng.poisson(cfg.spurious_rate * (n_cc as f64));
            for _ in 0..n_spur {
                let (rh, rn) = amide_roots[rng.usize(amide_roots.len())];
                let mut pk = Peak::triple(
                    "CCCONH",
                    rh + rng.normal(0.0, cfg.noise_h),
                    rn + rng.normal(0.0, cfg.noise_n),
                    rng.uniform(12.0, 72.0),
                    1,
                );
                if cfg.include_intensities {
                    let ht = 22.0 * (rng.normal(0.0, 0.25)).exp();
                    pk = pk.with_intensity(ht, Some(ht * 4.5));
                }
                peaks.push(pk);
            }
        }
        if !amide_roots.is_empty() && want_hccconh {
            let n_hcc = peaks.iter().filter(|p| p.exp == "HCCCONH").count();
            let n_spur = rng.poisson(cfg.spurious_rate * (n_hcc as f64));
            for _ in 0..n_spur {
                let (rh, rn) = amide_roots[rng.usize(amide_roots.len())];
                let mut pk = Peak {
                    exp: "HCCCONH".to_string(),
                    h: rh + rng.normal(0.0, cfg.noise_h),
                    n: rn + rng.normal(0.0, cfg.noise_n),
                    c: None,
                    sign: 1,
                    height: None,
                    volume: None,
                    h2: Some(rng.uniform(0.2, 5.4)),
                };
                if cfg.include_intensities {
                    let ht = 22.0 * (rng.normal(0.0, 0.25)).exp();
                    pk = pk.with_intensity(ht, Some(ht * 4.5));
                }
                peaks.push(pk);
            }
        }
    }

    rng.shuffle(&mut peaks);
    peaks
}

/// Configuration for rendering synthetic 2D `13C-HSQC` and 3D sidechain TOCSY spectra.
#[derive(Clone, Debug)]
pub struct SidechainRenderConfig {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub h_aliph_high_ppm: f64,
    pub h_aliph_low_ppm: f64,
    pub c_aliph_high_ppm: f64,
    pub c_aliph_low_ppm: f64,
    pub hn_high_ppm: f64,
    pub hn_low_ppm: f64,
    pub n_high_ppm: f64,
    pub n_low_ppm: f64,
    pub lw_h_ppm: f64,
    pub lw_n_ppm: f64,
    pub lw_c_ppm: f64,
    pub noise_level: f32,
    pub gaussian_mix: f64,
    pub seed: u64,
}

impl Default for SidechainRenderConfig {
    fn default() -> Self {
        Self {
            nx: 256,
            ny: 128,
            nz: 160,
            h_aliph_high_ppm: 6.2,
            h_aliph_low_ppm: -0.8,
            c_aliph_high_ppm: 76.0,
            c_aliph_low_ppm: 8.0,
            hn_high_ppm: 10.6,
            hn_low_ppm: 5.8,
            n_high_ppm: 136.0,
            n_low_ppm: 102.0,
            lw_h_ppm: 0.024,
            lw_n_ppm: 0.24,
            lw_c_ppm: 0.30,
            noise_level: 1.4,
            gaussian_mix: 0.70,
            seed: 101,
        }
    }
}

#[inline]
fn pseudo_voigt_1d(delta_ppm: f64, lw_ppm: f64, gauss_mix: f64) -> f64 {
    let u = delta_ppm / (0.5 * lw_ppm.max(1e-6));
    let g = (-std::f64::consts::LN_2 * u * u).exp();
    let l = 1.0 / (1.0 + u * u);
    gauss_mix * g + (1.0 - gauss_mix) * l
}

/// Render a synthetic 2D `1H-13C` HSQC spectrum from `C13HSQC` peaks.
/// Automatically expands spectral bounds to `[h_aliph_low..8.8, c_aliph_low..144.0]` if aromatic
/// `C13HSQC` peaks (`C > 90 ppm`) are present in `peaks`.
pub fn simulate_c13_hsqc_spectrum(peaks: &[Peak], cfg: &SidechainRenderConfig) -> Spectrum2D {
    let has_arom = peaks
        .iter()
        .any(|p| p.exp == "C13HSQC" && p.c.is_some_and(|c| c > 90.0));
    let (h_hi, c_hi, nx, ny) = if has_arom {
        (
            cfg.h_aliph_high_ppm.max(8.8),
            cfg.c_aliph_high_ppm.max(144.0),
            cfg.nx.max(512),
            cfg.ny.max(384),
        )
    } else {
        (
            cfg.h_aliph_high_ppm,
            cfg.c_aliph_high_ppm,
            cfg.nx.max(384),
            cfg.ny.max(256),
        )
    };

    let axis_x = AxisPpm::from_bounds("1H", nx, h_hi, cfg.h_aliph_low_ppm, 600.13);
    let axis_y = AxisPpm::from_bounds("13C", ny, c_hi, cfg.c_aliph_low_ppm, 150.90);
    let mut rng = Rng::new(cfg.seed);

    let mut data = vec![0.0f32; ny * nx];
    if cfg.noise_level > 0.0 {
        for v in &mut data {
            *v = rng.normal(0.0, cfg.noise_level as f64) as f32;
        }
    }

    let rad_x = ((3.5 * cfg.lw_h_ppm / axis_x.step_ppm.abs()).ceil() as isize).max(3);
    let rad_y = ((3.5 * cfg.lw_c_ppm / axis_y.step_ppm.abs()).ceil() as isize).max(3);

    for p in peaks.iter().filter(|p| p.exp == "C13HSQC") {
        let Some(c_ppm) = p.c else { continue };
        let cx = axis_x.ppm_to_index(p.h).round() as isize;
        let cy = axis_y.ppm_to_index(c_ppm).round() as isize;
        let amp = p.height.unwrap_or(100.0) * (p.sign as f64);

        let x0 = (cx - rad_x).max(0) as usize;
        let x1 = (cx + rad_x).min(nx as isize - 1).max(0) as usize;
        let y0 = (cy - rad_y).max(0) as usize;
        let y1 = (cy + rad_y).min(ny as isize - 1).max(0) as usize;

        for iy in y0..=y1 {
            let py = axis_y.index_to_ppm(iy as f64);
            let wy = pseudo_voigt_1d(py - c_ppm, cfg.lw_c_ppm, cfg.gaussian_mix);
            for ix in x0..=x1 {
                let px = axis_x.index_to_ppm(ix as f64);
                let wx = pseudo_voigt_1d(px - p.h, cfg.lw_h_ppm, cfg.gaussian_mix);
                data[iy * nx + ix] += (amp * wx * wy) as f32;
            }
        }
    }

    Spectrum2D {
        name: "Synthetic 1H-13C HSQC".to_string(),
        axis_x,
        axis_y,
        data,
    }
}

/// Render a synthetic 3D sidechain spectrum (`CCCONH`, `HCCCONH`, or `HCCHTOCSY`) from `peaks`.
pub fn simulate_sidechain_3d_spectrum(
    exp: &str,
    peaks: &[Peak],
    cfg: &SidechainRenderConfig,
) -> Spectrum3D {
    let exp_up = exp.trim().to_ascii_uppercase().replace(['(', ')', '-'], "");
    let (axis_x, axis_y, axis_z, lw_x, lw_y, lw_z) = match exp_up.as_str() {
        "CCCONH" => (
            AxisPpm::from_bounds("1H", cfg.nx, cfg.hn_high_ppm, cfg.hn_low_ppm, 600.13),
            AxisPpm::from_bounds("15N", cfg.ny, cfg.n_high_ppm, cfg.n_low_ppm, 60.81),
            AxisPpm::from_bounds(
                "13C",
                cfg.nz,
                cfg.c_aliph_high_ppm,
                cfg.c_aliph_low_ppm,
                150.90,
            ),
            cfg.lw_h_ppm,
            cfg.lw_n_ppm,
            cfg.lw_c_ppm,
        ),
        "HCCCONH" => (
            AxisPpm::from_bounds("1H", cfg.nx, cfg.hn_high_ppm, cfg.hn_low_ppm, 600.13),
            AxisPpm::from_bounds("15N", cfg.ny, cfg.n_high_ppm, cfg.n_low_ppm, 60.81),
            AxisPpm::from_bounds(
                "1H_aliph",
                cfg.nz,
                cfg.h_aliph_high_ppm,
                cfg.h_aliph_low_ppm,
                600.13,
            ),
            cfg.lw_h_ppm,
            cfg.lw_n_ppm,
            cfg.lw_h_ppm,
        ),
        _ => (
            AxisPpm::from_bounds(
                "1H",
                cfg.nx,
                cfg.h_aliph_high_ppm,
                cfg.h_aliph_low_ppm,
                600.13,
            ),
            AxisPpm::from_bounds(
                "13C",
                cfg.ny,
                cfg.c_aliph_high_ppm,
                cfg.c_aliph_low_ppm,
                150.90,
            ),
            AxisPpm::from_bounds(
                "1H_tocsy",
                cfg.nz,
                cfg.h_aliph_high_ppm,
                cfg.h_aliph_low_ppm,
                600.13,
            ),
            cfg.lw_h_ppm,
            cfg.lw_c_ppm,
            cfg.lw_h_ppm,
        ),
    };

    let (nx, ny, nz) = (axis_x.size, axis_y.size, axis_z.size);
    let mut rng = Rng::new(cfg.seed.wrapping_add(7919));
    let mut data = vec![0.0f32; nz * ny * nx];
    if cfg.noise_level > 0.0 {
        for v in &mut data {
            *v = rng.normal(0.0, cfg.noise_level as f64) as f32;
        }
    }

    let rad_x = ((3.0 * lw_x / axis_x.step_ppm.abs()).ceil() as isize).max(2);
    let rad_y = ((3.0 * lw_y / axis_y.step_ppm.abs()).ceil() as isize).max(2);
    let rad_z = ((3.0 * lw_z / axis_z.step_ppm.abs()).ceil() as isize).max(2);

    for p in peaks.iter().filter(|p| p.exp == exp_up) {
        let (vx, vy, vz) = match exp_up.as_str() {
            "CCCONH" => {
                let Some(c) = p.c else { continue };
                (p.h, p.n, c)
            }
            "HCCCONH" => {
                let Some(h2) = p.h2.or(p.c) else { continue };
                (p.h, p.n, h2)
            }
            _ => {
                let (Some(c), Some(h2)) = (p.c, p.h2) else {
                    continue;
                };
                (p.h, c, h2)
            }
        };

        // Skip aromatic peaks when rendering an aliphatic-window 3D HCCH-TOCSY grid
        if vx > axis_x.max_ppm()
            || vx < axis_x.min_ppm()
            || vy > axis_y.max_ppm()
            || vy < axis_y.min_ppm()
            || vz > axis_z.max_ppm()
            || vz < axis_z.min_ppm()
        {
            continue;
        }

        let cx = axis_x.ppm_to_index(vx).round() as isize;
        let cy = axis_y.ppm_to_index(vy).round() as isize;
        let cz = axis_z.ppm_to_index(vz).round() as isize;
        let amp = p.height.unwrap_or(80.0) * (p.sign as f64);

        let x0 = (cx - rad_x).max(0) as usize;
        let x1 = (cx + rad_x).min(nx as isize - 1).max(0) as usize;
        let y0 = (cy - rad_y).max(0) as usize;
        let y1 = (cy + rad_y).min(ny as isize - 1).max(0) as usize;
        let z0 = (cz - rad_z).max(0) as usize;
        let z1 = (cz + rad_z).min(nz as isize - 1).max(0) as usize;

        for iz in z0..=z1 {
            let pz = axis_z.index_to_ppm(iz as f64);
            let wz = pseudo_voigt_1d(pz - vz, lw_z, cfg.gaussian_mix);
            for iy in y0..=y1 {
                let py = axis_y.index_to_ppm(iy as f64);
                let wy = pseudo_voigt_1d(py - vy, lw_y, cfg.gaussian_mix);
                for ix in x0..=x1 {
                    let px = axis_x.index_to_ppm(ix as f64);
                    let wx = pseudo_voigt_1d(px - vx, lw_x, cfg.gaussian_mix);
                    data[(iz * ny + iy) * nx + ix] += (amp * wx * wy * wz) as f32;
                }
            }
        }
    }

    Spectrum3D {
        name: format!("Synthetic 3D {exp_up}"),
        exp: exp_up,
        axis_x,
        axis_y,
        axis_z,
        data,
    }
}

// ============================================================================
// 2. 2D & 3D Local-Extrema Sidechain Peak Pickers
// ============================================================================

/// Configuration for 2D `C13HSQC` and 3D `CCCONH` / `HCCCONH` / `HCCHTOCSY` peak picking.
#[derive(Clone, Debug)]
pub struct SidechainPickerConfig {
    pub snr_threshold: f32,
    pub allow_negative: bool,
    pub strip_tol_h: f64,
    pub strip_tol_n: f64,
    pub strip_tol_c: f64,
}

impl Default for SidechainPickerConfig {
    fn default() -> Self {
        Self {
            snr_threshold: 5.0,
            allow_negative: false,
            strip_tol_h: 0.04,
            strip_tol_n: 0.40,
            strip_tol_c: 0.45,
        }
    }
}

#[inline]
fn subvoxel_3pt(ym1: f32, y0: f32, yp1: f32) -> f64 {
    let a = (ym1.abs()) as f64;
    let b = (y0.abs()) as f64;
    let c = (yp1.abs()) as f64;
    if a > 0.0
        && b > a
        && b > c
        && c > 0.0
        && ym1.signum() == y0.signum()
        && yp1.signum() == y0.signum()
    {
        let la = a.ln();
        let lb = b.ln();
        let lc = c.ln();
        let denom = la - 2.0 * lb + lc;
        if denom.abs() > 1e-12 {
            return (0.5 * (la - lc) / denom).clamp(-0.5, 0.5);
        }
    }
    let denom = a - 2.0 * b + c;
    if denom.abs() > 1e-12 {
        (0.5 * (a - c) / denom).clamp(-0.5, 0.5)
    } else {
        0.0
    }
}

/// Pick 2D `C13HSQC` `(1H, 13C)` peaks from a 2D spectrum using MAD noise & 8-neighbor local extrema.
pub fn pick_c13_hsqc(spec: &Spectrum2D, cfg: &SidechainPickerConfig) -> Vec<Peak> {
    let (nx, ny) = (spec.nx(), spec.ny());
    if nx < 3 || ny < 3 {
        return Vec::new();
    }
    let noise = estimate_noise_mad(&spec.data);
    let thresh = cfg.snr_threshold * noise;
    let mut out = Vec::new();

    for iy in 1..ny - 1 {
        for ix in 1..nx - 1 {
            let v = spec.get(ix, iy);
            if v.abs() < thresh || (!cfg.allow_negative && v < 0.0) {
                continue;
            }
            let sign = if v >= 0.0 { 1 } else { -1 };
            let mut is_ext = true;
            'nbr: for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nv = spec.get((ix as isize + dx) as usize, (iy as isize + dy) as usize);
                    if (sign > 0 && nv >= v) || (sign < 0 && nv <= v) {
                        is_ext = false;
                        break 'nbr;
                    }
                }
            }
            if !is_ext {
                continue;
            }
            let dx = subvoxel_3pt(spec.get(ix - 1, iy), v, spec.get(ix + 1, iy));
            let dy = subvoxel_3pt(spec.get(ix, iy - 1), v, spec.get(ix, iy + 1));
            let h = spec.axis_x.index_to_ppm(ix as f64 + dx);
            let c = spec.axis_y.index_to_ppm(iy as f64 + dy);
            let mut vol = 0.0f64;
            for wy in -1isize..=1 {
                for wx in -1isize..=1 {
                    let sv = spec.get((ix as isize + wx) as usize, (iy as isize + wy) as usize);
                    if sv.signum() == v.signum() {
                        vol += (sv.abs()) as f64;
                    }
                }
            }
            out.push(Peak {
                exp: "C13HSQC".to_string(),
                h,
                n: 0.0,
                c: Some(c),
                sign,
                height: Some((v.abs()) as f64),
                volume: Some(vol),
                h2: None,
            });
        }
    }
    out
}

/// Pick 3D sidechain peaks (`CCCONH`, `HCCCONH`, or `HCCHTOCSY`) from `spec`.
/// - For `CCCONH` and `HCCCONH`, optional `anchor_roots` are `15N-HSQC` amide peaks `(H_N, N)`.
/// - For `HCCHTOCSY`, optional `anchor_roots` are `C13HSQC` peaks `(H_x, C_x)`.
pub fn pick_sidechain_3d(
    spec: &Spectrum3D,
    anchor_roots: Option<&[Peak]>,
    cfg: &SidechainPickerConfig,
) -> Vec<Peak> {
    let (nx, ny, nz) = (spec.nx(), spec.ny(), spec.nz());
    if nx < 3 || ny < 3 || nz < 3 {
        return Vec::new();
    }
    let exp_up = spec
        .exp
        .trim()
        .to_ascii_uppercase()
        .replace(['(', ')', '-'], "");
    let is_hcch = exp_up == "HCCHTOCSY";
    let is_hccconh = exp_up == "HCCCONH";

    let noise = estimate_noise_mad(&spec.data);
    let thresh = cfg.snr_threshold * noise;
    let roots = anchor_roots.unwrap_or(&[]);

    let tol_y = if is_hcch {
        cfg.strip_tol_c
    } else {
        cfg.strip_tol_n
    };

    let mut out = Vec::new();
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            let y_grid = spec.axis_y.index_to_ppm(iy as f64);
            if !roots.is_empty() {
                let y_match = if is_hcch {
                    roots
                        .iter()
                        .any(|r| r.c.is_some_and(|rc| (rc - y_grid).abs() <= tol_y * 1.5))
                } else {
                    roots.iter().any(|r| (r.n - y_grid).abs() <= tol_y * 1.5)
                };
                if !y_match {
                    continue;
                }
            }

            for ix in 1..nx - 1 {
                let v = spec.get(ix, iy, iz);
                if v.abs() < thresh || (!cfg.allow_negative && v < 0.0) {
                    continue;
                }
                let x_grid = spec.axis_x.index_to_ppm(ix as f64);
                if !roots.is_empty() {
                    let xy_match = if is_hcch {
                        roots.iter().any(|r| {
                            (r.h - x_grid).abs() <= cfg.strip_tol_h
                                && r.c.is_some_and(|rc| (rc - y_grid).abs() <= tol_y)
                        })
                    } else {
                        roots.iter().any(|r| {
                            (r.h - x_grid).abs() <= cfg.strip_tol_h && (r.n - y_grid).abs() <= tol_y
                        })
                    };
                    if !xy_match {
                        continue;
                    }
                }

                let sign = if v >= 0.0 { 1 } else { -1 };
                let mut is_ext = true;
                'nbr: for dz in -1isize..=1 {
                    for dy in -1isize..=1 {
                        for dx in -1isize..=1 {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }
                            let nv = spec.get(
                                (ix as isize + dx) as usize,
                                (iy as isize + dy) as usize,
                                (iz as isize + dz) as usize,
                            );
                            if (sign > 0 && nv >= v) || (sign < 0 && nv <= v) {
                                is_ext = false;
                                break 'nbr;
                            }
                        }
                    }
                }
                if !is_ext {
                    continue;
                }

                let dx = subvoxel_3pt(spec.get(ix - 1, iy, iz), v, spec.get(ix + 1, iy, iz));
                let dy = subvoxel_3pt(spec.get(ix, iy - 1, iz), v, spec.get(ix, iy + 1, iz));
                let dz = subvoxel_3pt(spec.get(ix, iy, iz - 1), v, spec.get(ix, iy, iz + 1));

                let vx = spec.axis_x.index_to_ppm(ix as f64 + dx);
                let vy = spec.axis_y.index_to_ppm(iy as f64 + dy);
                let vz = spec.axis_z.index_to_ppm(iz as f64 + dz);

                let mut vol = 0.0f64;
                for wz in -1isize..=1 {
                    for wy in -1isize..=1 {
                        for wx in -1isize..=1 {
                            let sv = spec.get(
                                (ix as isize + wx) as usize,
                                (iy as isize + wy) as usize,
                                (iz as isize + wz) as usize,
                            );
                            if sv.signum() == v.signum() {
                                vol += (sv.abs()) as f64;
                            }
                        }
                    }
                }

                let pk = if is_hcch {
                    Peak {
                        exp: "HCCHTOCSY".to_string(),
                        h: vx,
                        n: 0.0,
                        c: Some(vy),
                        sign,
                        height: Some((v.abs()) as f64),
                        volume: Some(vol),
                        h2: Some(vz),
                    }
                } else if is_hccconh {
                    Peak {
                        exp: "HCCCONH".to_string(),
                        h: vx,
                        n: vy,
                        c: None,
                        sign,
                        height: Some((v.abs()) as f64),
                        volume: Some(vol),
                        h2: Some(vz),
                    }
                } else {
                    Peak {
                        exp: exp_up.clone(),
                        h: vx,
                        n: vy,
                        c: Some(vz),
                        sign,
                        height: Some((v.abs()) as f64),
                        volume: Some(vol),
                        h2: None,
                    }
                };
                out.push(pk);
            }
        }
    }
    out
}

pub fn pick_ccconh_3d(
    spec: &Spectrum3D,
    hsqc_roots: &[Peak],
    cfg: &SidechainPickerConfig,
) -> Vec<Peak> {
    pick_sidechain_3d(spec, Some(hsqc_roots), cfg)
}

pub fn pick_hccconh_3d(
    spec: &Spectrum3D,
    hsqc_roots: &[Peak],
    cfg: &SidechainPickerConfig,
) -> Vec<Peak> {
    pick_sidechain_3d(spec, Some(hsqc_roots), cfg)
}

pub fn pick_hcch_tocsy_3d(
    spec: &Spectrum3D,
    c13_roots: Option<&[Peak]>,
    cfg: &SidechainPickerConfig,
) -> Vec<Peak> {
    pick_sidechain_3d(spec, c13_roots, cfg)
}

// ============================================================================
// 3. Sidechain & Joint Backbone + Sidechain Graph Assignment Optimizer
// ============================================================================

/// Configuration for sidechain and joint backbone+sidechain optimization.
#[derive(Clone, Debug)]
pub struct SidechainSolverConfig {
    /// Amide `1H` matching tolerance (ppm) for `CCCONH` / `HCCCONH` strip attachment.
    pub tol_hn_h: f64,
    /// Amide `15N` matching tolerance (ppm) for `CCCONH` / `HCCCONH` strip attachment.
    pub tol_hn_n: f64,
    /// Aliphatic/aromatic `1H` matching tolerance (ppm).
    pub tol_h: f64,
    /// Aliphatic/aromatic `13C` matching tolerance (ppm).
    pub tol_c: f64,
    /// Maximum `13C` discrepancy (ppm) between two geminal protons on the same `CH2` carbon.
    pub tol_c_geminal: f64,
    /// Minimum backbone confidence to anchor sequential `i-1` `CCCONH` / `HCCCONH` strips.
    pub min_anchor_conf: f64,
    /// Number of global cross-residue soft-exclusivity coordinate-ascent passes.
    pub auction_passes: usize,
    /// Weight applied to sidechain spin-type LLR boosts during joint backbone+sidechain solving.
    pub sidechain_boost_weight: f64,
}

impl Default for SidechainSolverConfig {
    fn default() -> Self {
        Self {
            tol_hn_h: 0.035,
            tol_hn_n: 0.35,
            tol_h: 0.035,
            tol_c: 0.28,
            tol_c_geminal: 0.16,
            min_anchor_conf: 0.50,
            auction_passes: 4,
            sidechain_boost_weight: 0.45,
        }
    }
}

/// Consolidated one-bond `(H_x, C_x)` spin pair with its attached `HCCH-TOCSY` proton ladder.
#[derive(Clone, Debug)]
pub struct HcNode {
    pub h: f64,
    pub c: f64,
    pub sign: i32,
    pub height: f64,
    /// Deduplicated `1H` shifts observed in `HCCHTOCSY` at `(h, c)`.
    pub tocsy_h: Vec<f64>,
    pub from_c13_hsqc: bool,
}

impl HcNode {
    #[inline]
    pub fn is_aliphatic(&self) -> bool {
        self.c < 90.0
    }

    #[inline]
    pub fn is_aromatic(&self) -> bool {
        self.c >= 105.0
    }
}

fn cluster_1d(mut vals: Vec<f64>, tol: f64) -> Vec<f64> {
    if vals.is_empty() {
        return Vec::new();
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let mut clusters: Vec<Vec<f64>> = Vec::new();
    for v in vals {
        if let Some(last) = clusters.last_mut() {
            let mean = last.iter().sum::<f64>() / (last.len() as f64);
            if (v - mean).abs() <= tol {
                last.push(v);
                continue;
            }
        }
        clusters.push(vec![v]);
    }
    clusters
        .into_iter()
        .map(|c| c.iter().sum::<f64>() / (c.len() as f64))
        .collect()
}

/// Consolidate `C13HSQC` peaks and `HCCHTOCSY` strip anchors into `HcNode`s.
pub fn build_hc_nodes(peaks: &[Peak], cfg: &SidechainSolverConfig) -> Vec<HcNode> {
    let mut nodes: Vec<HcNode> = Vec::new();

    for p in peaks.iter().filter(|p| p.exp == "C13HSQC") {
        let Some(c) = p.c else { continue };
        // Merge duplicate C13HSQC picks within tight tolerance
        if let Some(existing) = nodes.iter_mut().find(|n| {
            ((n.h - p.h) / (cfg.tol_h * 0.75)).hypot((n.c - c) / (cfg.tol_c * 0.65)) <= 1.0
        }) {
            let w1 = existing.height.max(1.0);
            let w2 = p.height.unwrap_or(50.0).max(1.0);
            existing.h = (existing.h * w1 + p.h * w2) / (w1 + w2);
            existing.c = (existing.c * w1 + c * w2) / (w1 + w2);
            existing.height += p.height.unwrap_or(0.0);
        } else {
            nodes.push(HcNode {
                h: p.h,
                c,
                sign: p.sign,
                height: p.height.unwrap_or(100.0),
                tocsy_h: Vec::new(),
                from_c13_hsqc: true,
            });
        }
    }

    // Recover missing C13HSQC nodes from well-populated HCCH-TOCSY strip anchors (>= 2 relays)
    let mut orphan_hcch: Vec<(f64, f64, Vec<f64>, f64)> = Vec::new();
    for p in peaks.iter().filter(|p| p.exp == "HCCHTOCSY") {
        let (Some(c), Some(h2)) = (p.c, p.h2) else {
            continue;
        };
        let mut attached = false;
        for n in &mut nodes {
            let d = ((n.h - p.h) / cfg.tol_h).hypot((n.c - c) / cfg.tol_c);
            if d <= 1.15 {
                n.tocsy_h.push(h2);
                attached = true;
            }
        }
        if !attached {
            if let Some(cl) = orphan_hcch
                .iter_mut()
                .find(|o| ((o.0 - p.h) / cfg.tol_h).hypot((o.1 - c) / cfg.tol_c) <= 1.0)
            {
                let cnt = cl.2.len() as f64;
                cl.0 = (cl.0 * cnt + p.h) / (cnt + 1.0);
                cl.1 = (cl.1 * cnt + c) / (cnt + 1.0);
                cl.2.push(h2);
                cl.3 = cl.3.max(p.height.unwrap_or(60.0));
            } else {
                orphan_hcch.push((p.h, c, vec![h2], p.height.unwrap_or(60.0)));
            }
        }
    }

    for (oh, oc, h2s, max_ht) in orphan_hcch {
        if h2s.len() >= 2 {
            nodes.push(HcNode {
                h: oh,
                c: oc,
                sign: 1,
                height: max_ht,
                tocsy_h: h2s,
                from_c13_hsqc: false,
            });
        }
    }

    for n in &mut nodes {
        n.tocsy_h = cluster_1d(std::mem::take(&mut n.tocsy_h), cfg.tol_h);
    }

    nodes
}

#[derive(Clone, Debug)]
enum GroupCand {
    Unassigned,
    Single(usize),
    GeminalPair(usize, usize),
    CarbonOnly(f64),
}

impl GroupCand {
    fn node_indices(&self) -> (Option<usize>, Option<usize>) {
        match *self {
            Self::Single(u) => (Some(u), None),
            Self::GeminalPair(u, v) => (Some(u), Some(v)),
            Self::Unassigned | Self::CarbonOnly(_) => (None, None),
        }
    }

    fn carbon_ppm(&self, nodes: &[HcNode]) -> Option<f64> {
        match *self {
            Self::Unassigned => None,
            Self::Single(u) => Some(nodes[u].c),
            Self::GeminalPair(u, v) => Some(0.5 * (nodes[u].c + nodes[v].c)),
            Self::CarbonOnly(c) => Some(c),
        }
    }
}

#[inline]
fn has_proton_match(ladder: &[f64], target_h: f64, tol_h: f64) -> Option<f64> {
    ladder
        .iter()
        .map(|&x| (x - target_h).abs())
        .filter(|&d| d <= tol_h)
        .min_by(|a, b| a.total_cmp(b))
}

#[inline]
fn has_carbon_match(ladder: &[f64], target_c: f64, tol_c: f64) -> Option<f64> {
    ladder
        .iter()
        .map(|&x| (x - target_c).abs())
        .filter(|&d| d <= tol_c)
        .min_by(|a, b| a.total_cmp(b))
}

/// Compute `HCCH-TOCSY` intra-residue connectivity score between two `HcNode`s `u` and `v`.
fn hcch_pair_score(
    u: &HcNode,
    v: &HcNode,
    directly_bonded: bool,
    tol_h: f64,
    has_hcch_data: bool,
) -> f64 {
    if !has_hcch_data {
        return 0.0;
    }
    let uv = has_proton_match(&u.tocsy_h, v.h, tol_h * 1.25).is_some();
    let vu = has_proton_match(&v.tocsy_h, u.h, tol_h * 1.25).is_some();

    // Ladder overlap (shared relayed protons)
    let mut shared = 0usize;
    for &hu in &u.tocsy_h {
        if has_proton_match(&v.tocsy_h, hu, tol_h * 1.25).is_some() {
            shared += 1;
        }
    }
    let union = (u.tocsy_h.len() + v.tocsy_h.len()).saturating_sub(shared);
    let jaccard = if union > 0 {
        (shared as f64) / (union as f64)
    } else {
        0.0
    };

    let mut sc = 0.0;
    let relay_w = if directly_bonded { 2.6 } else { 1.7 };
    if uv {
        sc += relay_w;
    }
    if vu {
        sc += relay_w;
    }
    sc += 2.2 * jaccard;

    // Anti-clique penalty if both nodes have rich TOCSY ladders yet completely fail to communicate
    if !uv && !vu && u.tocsy_h.len() >= 2 && v.tocsy_h.len() >= 2 && jaccard < 0.18 {
        sc -= if directly_bonded { 5.2 } else { 3.0 };
    }
    sc
}

#[derive(Clone, Debug, Default)]
struct ResidueEvidence {
    ca_anchor: Option<f64>,
    cb_anchor: Option<f64>,
    ccconh_c: Vec<f64>,
    hccconh_h: Vec<f64>,
    /// Number of backbone amides overlapping within `(tol_hn_h, tol_hn_n)` at `k+1` (`1` = clean).
    next_amide_overlap: usize,
    anchor_conf: f64,
}

/// Output of sidechain resonance assignment across a protein sequence.
#[derive(Clone, Debug)]
pub struct SidechainAssignResult {
    /// Complete assigned chemical shifts (backbone + sidechain), keyed by `(residue_idx, atom)`.
    pub reference: Reference,
    /// Per-atom posterior assignment confidence in `[0.0, 1.0]`.
    pub atom_confidence: BTreeMap<(usize, String), f64>,
    /// Per-residue best sidechain graph log-score.
    pub residue_scores: Vec<f64>,
}

/// Harvest per-residue `CA/CB` anchors and sequential `i-1` `CCCONH`/`HCCCONH` ladders.
fn collect_residue_evidence(
    backbone_ref: &Reference,
    backbone_conf: Option<&BTreeMap<usize, f64>>,
    sidechain_peaks: &[Peak],
    cfg: &SidechainSolverConfig,
) -> Vec<ResidueEvidence> {
    let k_len = backbone_ref.sequence.len();
    let mut ev = vec![ResidueEvidence::default(); k_len];

    // Precompute amide roots (k >= 1) to detect overlapped amides
    let mut amides: Vec<(usize, f64, f64, f64)> = Vec::new();
    for (k, ev_k) in ev.iter_mut().enumerate().take(k_len) {
        let c_conf = backbone_conf
            .and_then(|m| m.get(&k).copied())
            .unwrap_or(1.0);
        ev_k.anchor_conf = c_conf;
        if c_conf >= cfg.min_anchor_conf {
            ev_k.ca_anchor = backbone_ref.shifts.get(&(k, "CA".to_string())).copied();
            ev_k.cb_anchor = backbone_ref.shifts.get(&(k, "CB".to_string())).copied();
        }
        if k >= 1 && backbone_ref.sequence[k] != "PRO" && c_conf >= cfg.min_anchor_conf {
            if let (Some(&h), Some(&n)) = (
                backbone_ref.shifts.get(&(k, "H".to_string())),
                backbone_ref.shifts.get(&(k, "N".to_string())),
            ) {
                amides.push((k, h, n, c_conf));
            }
        }
    }

    for &(k_next, hn, nn, c_conf) in &amides {
        let target_k = k_next - 1;
        let overlap_cnt = amides
            .iter()
            .filter(|&&(_, oh, on, _)| {
                ((oh - hn) / (cfg.tol_hn_h * 1.25)).hypot((on - nn) / (cfg.tol_hn_n * 1.25)) <= 1.0
            })
            .count()
            .max(1);

        let mut c_vals = Vec::new();
        let mut h_vals = Vec::new();
        for p in sidechain_peaks {
            if p.exp == "CCCONH" {
                let d = ((p.h - hn) / cfg.tol_hn_h).hypot((p.n - nn) / cfg.tol_hn_n);
                if d <= 1.15 {
                    if let Some(c) = p.c {
                        c_vals.push(c);
                    }
                }
            } else if p.exp == "HCCCONH" {
                let d = ((p.h - hn) / cfg.tol_hn_h).hypot((p.n - nn) / cfg.tol_hn_n);
                if d <= 1.15 {
                    if let Some(h2) = p.h2.or(p.c) {
                        h_vals.push(h2);
                    }
                }
            }
        }

        ev[target_k].ccconh_c = cluster_1d(c_vals, cfg.tol_c * 0.85);
        ev[target_k].hccconh_h = cluster_1d(h_vals, cfg.tol_h * 0.85);
        ev[target_k].next_amide_overlap = overlap_cnt;
        if ev[target_k].anchor_conf < c_conf {
            ev[target_k].anchor_conf = c_conf;
        }
    }

    ev
}

/// Build shortlist of candidate assignments for a single `AliphaticCaConnected` `ChGroupSpec`.
#[allow(clippy::too_many_arguments)]
fn build_group_candidates(
    res: &str,
    chg: &ChGroupSpec,
    ev: &ResidueEvidence,
    nodes: &[HcNode],
    priors: &SidechainPriors,
    med_c13_height: f64,
    multiplicity_edited: bool,
    node_owner: &[Option<(usize, f64)>],
    k_self: usize,
    cfg: &SidechainSolverConfig,
) -> Vec<(GroupCand, f64)> {
    let Some((mu_c, sd_c)) = priors.prior(res, chg.carbon) else {
        return vec![(GroupCand::Unassigned, -1.5)];
    };
    let (mu_h0, sd_h0) = priors.prior(res, chg.protons[0]).unwrap_or((2.0, 0.5));

    let anchor_c = match chg.carbon {
        "CA" => ev.ca_anchor,
        "CB" => ev.cb_anchor,
        _ => None,
    };

    let c_gate = (6.5 * sd_c).max(8.5);
    let h_gate = (6.5 * sd_h0).max(1.85);

    let overlap_discount = if ev.next_amide_overlap > 1 { 0.55 } else { 1.0 };
    let has_cc = !ev.ccconh_c.is_empty();
    let has_hcc = !ev.hccconh_h.is_empty();

    let ownership_penalty = |u: usize| -> f64 {
        if let Some((owner_k, margin)) = node_owner[u] {
            if owner_k != k_self {
                // Allow shared use with mild penalty if peak amplitude suggests 2-residue overlap
                let expected_mult = chg.multiplicity.proton_count().min(3) as f64;
                if med_c13_height > 0.0 && nodes[u].height >= 1.75 * expected_mult * med_c13_height
                {
                    return -1.2;
                }
                return -(3.5 + 0.55 * margin.clamp(0.0, 10.0));
            }
        }
        0.0
    };

    let score_single_node = |u: usize| -> Option<f64> {
        let n = &nodes[u];
        if !n.is_aliphatic() {
            return None;
        }
        if let Some(ac) = anchor_c {
            if (n.c - ac).abs() > cfg.tol_c * 1.75 {
                return None;
            }
        } else if (n.c - mu_c).abs() > c_gate || (n.h - mu_h0).abs() > h_gate {
            return None;
        }

        if multiplicity_edited && n.sign != 0 && n.sign != chg.multiplicity.edited_hsqc_sign() {
            return None;
        }

        let mut sc = priors.llr_c13(res, chg.carbon, chg.region, n.c)
            + priors.llr_h1(res, chg.protons[0], chg.region, n.h);

        if let Some(ac) = anchor_c {
            let z = (n.c - ac) / (cfg.tol_c * 0.45);
            sc += 7.5 - 0.5 * z * z;
        }

        if has_cc {
            if let Some(dc) = has_carbon_match(&ev.ccconh_c, n.c, cfg.tol_c * 1.2) {
                let z = dc / (cfg.tol_c * 0.5);
                sc += overlap_discount * (4.8 - 0.5 * z * z);
            } else if ev.next_amide_overlap <= 1 {
                sc -= 1.35;
            }
        }

        if has_hcc {
            if let Some(dh) = has_proton_match(&ev.hccconh_h, n.h, cfg.tol_h * 1.2) {
                let z = dh / (cfg.tol_h * 0.5);
                sc += overlap_discount * (4.4 - 0.5 * z * z);
            } else if ev.next_amide_overlap <= 1 {
                sc -= 1.15;
            }
        }

        if chg.multiplicity == ChMultiplicity::Methyl
            && med_c13_height > 0.0
            && n.height >= 1.65 * med_c13_height
        {
            sc += 0.75;
        }

        sc += ownership_penalty(u);
        Some(sc)
    };

    let mut single_scored: Vec<(usize, f64)> = (0..nodes.len())
        .filter_map(|u| score_single_node(u).map(|sc| (u, sc)))
        .collect();
    single_scored.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut cands: Vec<(GroupCand, f64)> = Vec::new();

    // For CH2 methylenes, build resolved GeminalPair(u, v) candidates from compatible single nodes
    if chg.multiplicity == ChMultiplicity::Methylene {
        let pool_len = single_scored.len().min(18);
        let mut pair_cands: Vec<(GroupCand, f64)> = Vec::new();
        for ia in 0..pool_len {
            let (u, scu) = single_scored[ia];
            for &(v, scv) in &single_scored[(ia + 1)..pool_len] {
                let dc = (nodes[u].c - nodes[v].c).abs();
                let dh = (nodes[u].h - nodes[v].h).abs();
                if dc <= cfg.tol_c_geminal && dh >= 0.012 {
                    let c_avg = 0.5 * (nodes[u].c + nodes[v].c);
                    // Combine carbon score once + both proton LLRs + geminal bonus
                    let (h_hi, h_lo) = if nodes[u].h >= nodes[v].h {
                        (nodes[u].h, nodes[v].h)
                    } else {
                        (nodes[v].h, nodes[u].h)
                    };
                    let p1_name = chg.protons.get(1).copied().unwrap_or(chg.protons[0]);
                    let base_pair = 0.5 * (scu + scv)
                        + 0.5
                            * (priors.llr_h1(res, chg.protons[0], chg.region, h_hi)
                                + priors.llr_h1(res, p1_name, chg.region, h_lo))
                        - 0.5 * priors.llr_c13(res, chg.carbon, chg.region, c_avg)
                        + 1.85
                        - 2.0 * (dc / cfg.tol_c_geminal).powi(2);

                    let gem_relay =
                        if has_proton_match(&nodes[u].tocsy_h, nodes[v].h, cfg.tol_h * 1.25)
                            .is_some()
                            || has_proton_match(&nodes[v].tocsy_h, nodes[u].h, cfg.tol_h * 1.25)
                                .is_some()
                        {
                            1.4
                        } else {
                            0.0
                        };

                    let both_hcc = if has_hcc
                        && has_proton_match(&ev.hccconh_h, nodes[u].h, cfg.tol_h * 1.2).is_some()
                        && has_proton_match(&ev.hccconh_h, nodes[v].h, cfg.tol_h * 1.2).is_some()
                    {
                        overlap_discount * 2.0
                    } else {
                        0.0
                    };

                    pair_cands.push((
                        GroupCand::GeminalPair(u.min(v), u.max(v)),
                        base_pair + gem_relay + both_hcc,
                    ));
                }
            }
        }
        pair_cands.sort_by(|a, b| b.1.total_cmp(&a.1));
        pair_cands.truncate(8);
        cands.extend(pair_cands);
    }

    for &(u, sc) in single_scored.iter().take(10) {
        cands.push((GroupCand::Single(u), sc));
    }

    // Fallback CarbonOnly candidate if backbone or CCCONH sees the carbon when C13HSQC missed it
    if let Some(ac) = anchor_c {
        let fallback_sc = priors.llr_c13(res, chg.carbon, chg.region, ac) + 2.2;
        cands.push((GroupCand::CarbonOnly(ac), fallback_sc));
    } else if has_cc && ev.next_amide_overlap == 1 {
        for &cc in &ev.ccconh_c {
            if (cc - mu_c).abs() <= c_gate {
                let sc = priors.llr_c13(res, chg.carbon, chg.region, cc) + 1.2;
                cands.push((GroupCand::CarbonOnly(cc), sc));
            }
        }
    }

    cands.push((GroupCand::Unassigned, -1.6));
    cands.sort_by(|a, b| b.1.total_cmp(&a.1));
    cands.truncate(12);
    cands
}

/// Exact branch-and-bound / enumeration over all `L <= 5` aliphatic `ChGroupSpec`s of residue `k`,
/// returning `(best_choice_per_group, best_score, per_group_posterior_confidence, second_best_margin)`.
#[allow(clippy::too_many_arguments)]
fn solve_residue_aliphatic_clique(
    topo: &ResidueTopology,
    groups: &[&'static ChGroupSpec],
    group_cands: &[Vec<(GroupCand, f64)>],
    nodes: &[HcNode],
    has_hcch_data: bool,
    cfg: &SidechainSolverConfig,
) -> (Vec<usize>, f64, Vec<f64>, Vec<f64>) {
    let l = groups.len();
    if l == 0 {
        return (Vec::new(), 0.0, Vec::new(), Vec::new());
    }

    // Precompute whether group pairs (a, b) are covalently bonded carbons
    let mut bonded = vec![vec![false; l]; l];
    let mut allow_iso_share = vec![vec![false; l]; l];
    for a in 0..l {
        for b in (a + 1)..l {
            if groups[a]
                .covalent_carbon_neighbors
                .contains(&groups[b].carbon)
            {
                bonded[a][b] = true;
                bonded[b][a] = true;
            }
            for pg in topo.prochiral_groups {
                if let ProchiralGroup::IsopropylMethyls {
                    carbon_a, carbon_b, ..
                } = pg
                {
                    if (groups[a].carbon == *carbon_a && groups[b].carbon == *carbon_b)
                        || (groups[a].carbon == *carbon_b && groups[b].carbon == *carbon_a)
                    {
                        allow_iso_share[a][b] = true;
                        allow_iso_share[b][a] = true;
                    }
                }
            }
        }
    }

    // Precompute pairwise TOCSY interaction tables between group a's candidates and group b's candidates
    let mut pair_tables: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); l]; l];
    for a in 0..l {
        for b in (a + 1)..l {
            let na = group_cands[a].len();
            let nb = group_cands[b].len();
            let mut tab = vec![0.0; na * nb];
            for ia in 0..na {
                let (ua1, ua2) = group_cands[a][ia].0.node_indices();
                for ib in 0..nb {
                    let (ub1, ub2) = group_cands[b][ib].0.node_indices();
                    // Disallow reusing the same HcNode across two different carbons in the same residue
                    // (except strictly degenerate VAL CG1==CG2 or LEU CD1==CD2 with a degeneracy cost).
                    let collides = [ua1, ua2]
                        .into_iter()
                        .flatten()
                        .any(|x| Some(x) == ub1 || Some(x) == ub2);
                    if collides {
                        tab[ia * nb + ib] = if allow_iso_share[a][b] { -2.5 } else { -1.0e6 };
                        continue;
                    }

                    let mut psc = 0.0;
                    let mut cnt = 0usize;
                    for u in [ua1, ua2].into_iter().flatten() {
                        for v in [ub1, ub2].into_iter().flatten() {
                            psc += hcch_pair_score(
                                &nodes[u],
                                &nodes[v],
                                bonded[a][b],
                                cfg.tol_h,
                                has_hcch_data,
                            );
                            cnt += 1;
                        }
                    }
                    if cnt > 0 {
                        psc /= (cnt as f64).sqrt();
                    }
                    tab[ia * nb + ib] = psc;
                }
            }
            pair_tables[a][b] = tab;
        }
    }

    let mut cur_choice = vec![0usize; l];
    let mut best_choice = vec![0usize; l];
    let mut best_score = f64::NEG_INFINITY;
    let mut states: Vec<([usize; 6], f64)> = Vec::with_capacity(512);

    #[allow(clippy::too_many_arguments)]
    fn dfs(
        depth: usize,
        l: usize,
        running_score: f64,
        cur_choice: &mut [usize],
        group_cands: &[Vec<(GroupCand, f64)>],
        pair_tables: &[Vec<Vec<f64>>],
        best_score: &mut f64,
        best_choice: &mut [usize],
        states: &mut Vec<([usize; 6], f64)>,
    ) {
        if depth == l {
            if running_score > *best_score {
                *best_score = running_score;
                best_choice.copy_from_slice(cur_choice);
            }
            let mut arr = [0usize; 6];
            arr[..l].copy_from_slice(cur_choice);
            states.push((arr, running_score));
            return;
        }

        let cands = &group_cands[depth];
        for (ic, &(_, u_sc)) in cands.iter().enumerate() {
            let mut delta = u_sc;
            let mut invalid = false;
            for prev in 0..depth {
                let nb = cands.len();
                let psc = pair_tables[prev][depth][cur_choice[prev] * nb + ic];
                if psc < -1.0e5 {
                    invalid = true;
                    break;
                }
                delta += psc;
            }
            if invalid {
                continue;
            }
            cur_choice[depth] = ic;
            dfs(
                depth + 1,
                l,
                running_score + delta,
                cur_choice,
                group_cands,
                pair_tables,
                best_score,
                best_choice,
                states,
            );
        }
    }

    dfs(
        0,
        l,
        0.0,
        &mut cur_choice,
        group_cands,
        &pair_tables,
        &mut best_score,
        &mut best_choice,
        &mut states,
    );

    if states.is_empty() {
        return (vec![0; l], -10.0, vec![0.0; l], vec![0.0; l]);
    }

    // Marginal Boltzmann posterior confidence & margin per group (temperature T_post = 2.0)
    let t_post = 2.0;
    let max_s = best_score;
    let z_tot: f64 = states
        .iter()
        .map(|&(_, s)| ((s - max_s) / t_post).exp())
        .sum::<f64>()
        .max(1e-30);

    let mut confs = vec![0.0; l];
    let mut margins = vec![0.0; l];
    for g in 0..l {
        let win_ic = best_choice[g];
        let win_node_set = group_cands[g][win_ic].0.node_indices();
        let mut z_agree = 0.0;
        let mut alt_best = f64::NEG_INFINITY;
        for &(ref arr, s) in &states {
            // Treat prochiral isopropyl swaps on VAL/LEU as agreeing on node usage when computing margin
            if arr[g] == win_ic || group_cands[g][arr[g]].0.node_indices() == win_node_set {
                z_agree += ((s - max_s) / t_post).exp();
            } else if s > alt_best {
                alt_best = s;
            }
        }
        confs[g] = (z_agree / z_tot).clamp(0.0, 1.0);
        margins[g] = if alt_best.is_finite() {
            (best_score - alt_best).max(0.0)
        } else {
            10.0
        };
    }

    (best_choice, best_score, confs, margins)
}

/// Assign all aliphatic and aromatic sidechain `13C` and `1H` resonances given a backbone
/// `Reference` (with `H, N, CA, CB, C`) and sidechain peak lists.
pub fn assign_sidechains(
    backbone_ref: &Reference,
    backbone_conf: Option<&BTreeMap<usize, f64>>,
    sidechain_peaks: &[Peak],
    sc_priors: &SidechainPriors,
    cfg: &SidechainSolverConfig,
) -> SidechainAssignResult {
    let nodes = build_hc_nodes(sidechain_peaks, cfg);
    let has_hcch = sidechain_peaks.iter().any(|p| p.exp == "HCCHTOCSY");
    let multiplicity_edited = nodes.iter().any(|n| n.sign < 0);

    let mut c13_heights: Vec<f64> = nodes
        .iter()
        .filter(|n| n.is_aliphatic() && n.height > 0.0)
        .map(|n| n.height)
        .collect();
    c13_heights.sort_by(|a, b| a.total_cmp(b));
    let med_c13_height = if c13_heights.is_empty() {
        0.0
    } else {
        c13_heights[c13_heights.len() / 2]
    };

    let evidence = collect_residue_evidence(backbone_ref, backbone_conf, sidechain_peaks, cfg);
    let k_len = backbone_ref.sequence.len();

    // Order residues so strongly anchored ones (both backbone + CCCONH/HCCCONH) claim nodes first
    let mut res_order: Vec<usize> = (0..k_len).collect();
    res_order.sort_by_key(|&k| {
        let e = &evidence[k];
        let tier = if !e.ccconh_c.is_empty() && e.next_amide_overlap == 1 && e.ca_anchor.is_some() {
            0
        } else if !e.ccconh_c.is_empty() {
            1
        } else if e.ca_anchor.is_some() {
            2
        } else {
            3
        };
        (tier, k)
    });

    let mut node_owner: Vec<Option<(usize, f64)>> = vec![None; nodes.len()];
    let mut per_res_choices: Vec<Vec<(GroupCand, f64)>> = vec![Vec::new(); k_len];
    let mut per_res_best_idx: Vec<Vec<usize>> = vec![Vec::new(); k_len];
    let mut per_res_confs: Vec<Vec<f64>> = vec![Vec::new(); k_len];
    let mut residue_scores = vec![0.0; k_len];

    let n_passes = cfg.auction_passes.max(1);
    for _pass in 0..n_passes {
        for &k in &res_order {
            let Some(topo) = residue_topology(&backbone_ref.sequence[k]) else {
                continue;
            };
            // Release previous ownership held by residue k before re-solving k
            for own in &mut node_owner {
                if own.is_some_and(|(ok, _)| ok == k) {
                    *own = None;
                }
            }

            let groups: Vec<&'static ChGroupSpec> = topo.aliphatic_ca_groups().collect();
            let group_cands: Vec<Vec<(GroupCand, f64)>> = groups
                .iter()
                .map(|chg| {
                    build_group_candidates(
                        topo.three_letter,
                        chg,
                        &evidence[k],
                        &nodes,
                        sc_priors,
                        med_c13_height,
                        multiplicity_edited,
                        &node_owner,
                        k,
                        cfg,
                    )
                })
                .collect();

            let (best_idx, best_sc, confs, margins) =
                solve_residue_aliphatic_clique(topo, &groups, &group_cands, &nodes, has_hcch, cfg);

            for (g, &ic) in best_idx.iter().enumerate() {
                let (u1, u2) = group_cands[g][ic].0.node_indices();
                for u in [u1, u2].into_iter().flatten() {
                    node_owner[u] = Some((k, margins[g]));
                }
            }

            per_res_choices[k] = groups
                .iter()
                .enumerate()
                .map(|(g, _)| group_cands[g][best_idx[g]].clone())
                .collect();
            per_res_best_idx[k] = best_idx;
            per_res_confs[k] = confs;
            residue_scores[k] = best_sc;
        }
    }

    // Assemble output Reference starting from backbone shifts
    let mut out_ref = backbone_ref.clone();
    let mut atom_confidence: BTreeMap<(usize, String), f64> = BTreeMap::new();
    for &(k, ref atom) in backbone_ref.shifts.keys() {
        let bc = backbone_conf
            .and_then(|m| m.get(&k).copied())
            .unwrap_or(1.0);
        atom_confidence.insert((k, atom.clone()), bc);
    }

    for k in 0..k_len {
        let Some(topo) = residue_topology(&backbone_ref.sequence[k]) else {
            continue;
        };
        let groups: Vec<&'static ChGroupSpec> = topo.aliphatic_ca_groups().collect();
        let base_conf = evidence[k].anchor_conf.clamp(0.35, 1.0);

        for (g, chg) in groups.iter().enumerate() {
            let Some((ref cand, _)) = per_res_choices[k].get(g) else {
                continue;
            };
            let g_conf = (per_res_confs[k][g] * base_conf).clamp(0.0, 1.0);

            // Preserve existing high-precision backbone CA/CB when already known, else populate from sidechain
            if let Some(c_val) = cand.carbon_ppm(&nodes) {
                if let std::collections::btree_map::Entry::Vacant(e) =
                    out_ref.shifts.entry((k, chg.carbon.to_string()))
                {
                    e.insert(c_val);
                    atom_confidence.insert((k, chg.carbon.to_string()), g_conf);
                }
            }

            match *cand {
                GroupCand::Single(u) => {
                    let h_val = nodes[u].h;
                    for &p_name in chg.protons {
                        out_ref.shifts.insert((k, p_name.to_string()), h_val);
                        atom_confidence.insert((k, p_name.to_string()), g_conf);
                    }
                }
                GroupCand::GeminalPair(u, v) => {
                    let (h_hi, h_lo) = if nodes[u].h >= nodes[v].h {
                        (nodes[u].h, nodes[v].h)
                    } else {
                        (nodes[v].h, nodes[u].h)
                    };
                    out_ref.shifts.insert((k, chg.protons[0].to_string()), h_hi);
                    atom_confidence.insert((k, chg.protons[0].to_string()), g_conf);
                    if let Some(&p1) = chg.protons.get(1) {
                        out_ref.shifts.insert((k, p1.to_string()), h_lo);
                        atom_confidence.insert((k, p1.to_string()), g_conf);
                    }
                }
                GroupCand::Unassigned | GroupCand::CarbonOnly(_) => {}
            }
        }

        // Canonical downfield-first ordering for prochiral isopropyl methyl branches (VAL CG1>=CG2, LEU CD1>=CD2)
        // while keeping each methyl's bonded (C, H#) pair strictly intact!
        for pg in topo.prochiral_groups {
            if let ProchiralGroup::IsopropylMethyls {
                carbon_a,
                protons_a,
                carbon_b,
                protons_b,
            } = pg
            {
                let ca = out_ref.shifts.get(&(k, (*carbon_a).to_string())).copied();
                let cb = out_ref.shifts.get(&(k, (*carbon_b).to_string())).copied();
                let ha = out_ref.shifts.get(&(k, protons_a[0].to_string())).copied();
                let hb = out_ref.shifts.get(&(k, protons_b[0].to_string())).copied();
                if let (Some(ca_v), Some(cb_v), Some(ha_v), Some(hb_v)) = (ca, cb, ha, hb) {
                    if cb_v > ca_v {
                        out_ref.shifts.insert((k, (*carbon_a).to_string()), cb_v);
                        out_ref.shifts.insert((k, (*carbon_b).to_string()), ca_v);
                        for &pa in protons_a {
                            out_ref.shifts.insert((k, pa.to_string()), hb_v);
                        }
                        for &pb in protons_b {
                            out_ref.shifts.insert((k, pb.to_string()), ha_v);
                        }
                    }
                }
            }
        }
    }

    // Assign isolated MET CE / HE1..3 methyls from unclaimed aliphatic methyl nodes
    assign_isolated_met_methyls(
        &mut out_ref,
        &mut atom_confidence,
        &nodes,
        &mut node_owner,
        sc_priors,
    );

    // Assign aromatic ring CH groups (PHE, TYR, HIS, TRP) from aromatic HcNodes
    assign_aromatic_sidechains(
        &mut out_ref,
        &mut atom_confidence,
        &nodes,
        sc_priors,
        has_hcch,
        cfg,
    );

    SidechainAssignResult {
        reference: out_ref,
        atom_confidence,
        residue_scores,
    }
}

fn assign_isolated_met_methyls(
    out_ref: &mut Reference,
    atom_confidence: &mut BTreeMap<(usize, String), f64>,
    nodes: &[HcNode],
    node_owner: &mut [Option<(usize, f64)>],
    sc_priors: &SidechainPriors,
) {
    let met_positions: Vec<usize> = out_ref
        .sequence
        .iter()
        .enumerate()
        .filter_map(|(k, r)| (r == "MET").then_some(k))
        .collect();
    if met_positions.is_empty() {
        return;
    }

    for k in met_positions {
        let mut best: Option<(usize, f64)> = None;
        let mut second = f64::NEG_INFINITY;
        for (u, n) in nodes.iter().enumerate() {
            if !n.is_aliphatic() || node_owner[u].is_some() {
                continue;
            }
            if !(11.0..=23.5).contains(&n.c) || !(0.8..=2.9).contains(&n.h) {
                continue;
            }
            let mut sc = sc_priors.llr_c13("MET", "CE", CarbonRegion::AliphaticIsolatedMethyl, n.c)
                + sc_priors.llr_h1("MET", "HE1", CarbonRegion::AliphaticIsolatedMethyl, n.h);
            // True MET CE has at most 1 self-peak in HCCH-TOCSY (isolated by thioether SD)
            if n.tocsy_h.len() <= 1 {
                sc += 3.0;
            } else {
                sc -= 2.5 * ((n.tocsy_h.len() - 1) as f64);
            }
            if best.is_none_or(|(_, bs)| sc > bs) {
                if let Some((_, bs)) = best {
                    second = second.max(bs);
                }
                best = Some((u, sc));
            } else if sc > second {
                second = sc;
            }
        }
        if let Some((u, bs)) = best {
            if bs > -2.5 {
                let conf = if second.is_finite() {
                    1.0 / (1.0 + (-(bs - second) / 2.0).exp())
                } else {
                    0.92
                };
                node_owner[u] = Some((k, (bs - second).max(2.0)));
                out_ref.shifts.insert((k, "CE".to_string()), nodes[u].c);
                atom_confidence.insert((k, "CE".to_string()), conf);
                for p in ["HE1", "HE2", "HE3"] {
                    out_ref.shifts.insert((k, p.to_string()), nodes[u].h);
                    atom_confidence.insert((k, p.to_string()), conf);
                }
            }
        }
    }
}

fn assign_aromatic_sidechains(
    out_ref: &mut Reference,
    atom_confidence: &mut BTreeMap<(usize, String), f64>,
    nodes: &[HcNode],
    sc_priors: &SidechainPriors,
    has_hcch: bool,
    cfg: &SidechainSolverConfig,
) {
    let arom_idxs: Vec<usize> = nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| n.is_aromatic().then_some(i))
        .collect();
    if arom_idxs.is_empty() {
        return;
    }

    let mut used_arom = vec![false; nodes.len()];
    // Order distinctive aromatic residue types first (HIS CE1 ~138, TYR CE ~118, TRP CZ2 ~114, then PHE)
    let mut arom_res: Vec<usize> = out_ref
        .sequence
        .iter()
        .enumerate()
        .filter_map(|(k, r)| matches!(r.as_str(), "HIS" | "TYR" | "TRP" | "PHE").then_some(k))
        .collect();
    arom_res.sort_by_key(|&k| match out_ref.sequence[k].as_str() {
        "HIS" => 0,
        "TYR" => 1,
        "TRP" => 2,
        _ => 3,
    });

    for k in arom_res {
        let res = out_ref.sequence[k].as_str();
        let Some(topo) = residue_topology(res) else {
            continue;
        };

        // Group symmetric PHE/TYR pairs so CD1/CD2 and CE1/CE2 share the degenerate ring-flip node
        let mut sym_handled: BTreeSet<&'static str> = BTreeSet::new();
        let mut chosen_ring_nodes: Vec<usize> = Vec::new();

        for pg in topo.prochiral_groups {
            if let ProchiralGroup::SymmetricAromatic {
                carbon_a,
                proton_a,
                carbon_b,
                proton_b,
            } = pg
            {
                sym_handled.insert(carbon_a);
                sym_handled.insert(carbon_b);

                let mut best: Option<(usize, f64)> = None;
                for &u in &arom_idxs {
                    if used_arom[u] {
                        continue;
                    }
                    let mut sc =
                        sc_priors.llr_c13(res, carbon_a, CarbonRegion::Aromatic, nodes[u].c)
                            + sc_priors.llr_h1(res, proton_a, CarbonRegion::Aromatic, nodes[u].h);
                    if has_hcch {
                        for &prev_u in &chosen_ring_nodes {
                            sc += hcch_pair_score(&nodes[prev_u], &nodes[u], true, cfg.tol_h, true);
                        }
                    }
                    if best.is_none_or(|(_, bs)| sc > bs) {
                        best = Some((u, sc));
                    }
                }
                if let Some((u, bs)) = best {
                    if bs > -3.0 {
                        used_arom[u] = true;
                        chosen_ring_nodes.push(u);
                        for &(ca, pa) in &[(carbon_a, proton_a), (carbon_b, proton_b)] {
                            out_ref.shifts.insert((k, (*ca).to_string()), nodes[u].c);
                            out_ref.shifts.insert((k, (*pa).to_string()), nodes[u].h);
                            atom_confidence.insert((k, (*ca).to_string()), 0.85);
                            atom_confidence.insert((k, (*pa).to_string()), 0.85);
                        }
                    }
                }
            }
        }

        for chg in topo.aromatic_groups() {
            if sym_handled.contains(chg.carbon) {
                continue;
            }
            let p_name = chg.protons[0];
            let mut best: Option<(usize, f64)> = None;
            for &u in &arom_idxs {
                if used_arom[u] {
                    continue;
                }
                let mut sc = sc_priors.llr_c13(res, chg.carbon, CarbonRegion::Aromatic, nodes[u].c)
                    + sc_priors.llr_h1(res, p_name, CarbonRegion::Aromatic, nodes[u].h);
                if has_hcch && res != "HIS" {
                    for &prev_u in &chosen_ring_nodes {
                        sc += hcch_pair_score(&nodes[prev_u], &nodes[u], true, cfg.tol_h, true);
                    }
                }
                if best.is_none_or(|(_, bs)| sc > bs) {
                    best = Some((u, sc));
                }
            }
            if let Some((u, bs)) = best {
                if bs > -3.0 {
                    used_arom[u] = true;
                    chosen_ring_nodes.push(u);
                    out_ref
                        .shifts
                        .insert((k, chg.carbon.to_string()), nodes[u].c);
                    out_ref.shifts.insert((k, p_name.to_string()), nodes[u].h);
                    atom_confidence.insert((k, chg.carbon.to_string()), 0.85);
                    atom_confidence.insert((k, p_name.to_string()), 0.85);
                }
            }
        }
    }
}

// ============================================================================
// 4. Joint Backbone + Sidechain Feedback Scoring & Solver
// ============================================================================

/// Score how well an observed `i-1` `(CCCONH_carbons, HCCCONH_protons)` ladder on spin system `s`
/// matches candidate amino acid type `prev_res` (`sequence[k - 1]`).
fn score_im1_ladder_against_aa(
    prev_res: &str,
    c_ladder: &[f64],
    h_ladder: &[f64],
    nodes: &[HcNode],
    sc_priors: &SidechainPriors,
    cfg: &SidechainSolverConfig,
) -> f64 {
    if c_ladder.is_empty() && h_ladder.is_empty() {
        return 0.0;
    }
    let Some(topo) = residue_topology(prev_res) else {
        return 0.0;
    };
    let groups: Vec<&'static ChGroupSpec> = topo.aliphatic_ca_groups().collect();
    let expected_c = groups.len();

    let mut score = 0.0;

    // 1. Carbon ladder compatibility (greedy injective assignment of expected groups to c_ladder)
    if !c_ladder.is_empty() {
        let mut used_c = vec![false; c_ladder.len()];
        for chg in &groups {
            let mut best_j: Option<(usize, f64)> = None;
            for (j, &cv) in c_ladder.iter().enumerate() {
                if used_c[j] {
                    continue;
                }
                // Extra bonus if (H, C) exists in C13HSQC and H also matches h_ladder!
                let mut llr = sc_priors.llr_c13(prev_res, chg.carbon, chg.region, cv);
                if !h_ladder.is_empty() {
                    let hc_pair_bonus = nodes
                        .iter()
                        .filter(|n| (n.c - cv).abs() <= cfg.tol_c)
                        .filter_map(|n| {
                            has_proton_match(h_ladder, n.h, cfg.tol_h * 1.2).map(|_| {
                                sc_priors
                                    .llr_h1(prev_res, chg.protons[0], chg.region, n.h)
                                    .clamp(-3.0, 4.5)
                            })
                        })
                        .fold(f64::NEG_INFINITY, f64::max);
                    if hc_pair_bonus.is_finite() {
                        llr += 0.65 * hc_pair_bonus;
                    }
                }
                if best_j.is_none_or(|(_, b)| llr > b) {
                    best_j = Some((j, llr));
                }
            }
            if let Some((j, b)) = best_j {
                if b > -5.5 {
                    used_c[j] = true;
                    score += b;
                } else {
                    score -= 2.0;
                }
            } else {
                // Expected carbon missing from CCCONH ladder
                score -= 1.1;
            }
        }

        // Penalize unexplained excess carbons in CCCONH (e.g. GLY/ALA/SER cannot explain a 4-carbon LYS/ILE ladder!)
        let unmatched_obs = used_c.iter().filter(|&&u| !u).count();
        if unmatched_obs > 0 && c_ladder.len() > expected_c {
            score -= 2.4 * ((c_ladder.len() - expected_c) as f64);
        }
    } else if !h_ladder.is_empty() {
        for chg in &groups {
            let best_h = h_ladder
                .iter()
                .map(|&hv| sc_priors.llr_h1(prev_res, chg.protons[0], chg.region, hv))
                .fold(f64::NEG_INFINITY, f64::max);
            if best_h.is_finite() {
                score += 0.6 * best_h.max(-5.0);
            }
        }
    }

    score.clamp(-14.0, 18.0)
}

/// Compute `(n_systems x seq_len)` spin-type LLR boost matrix `delta_U` and `(n_systems x n_systems)`
/// sequential connectivity boost matrix `delta_P` from sidechain peak lists.
pub fn compute_sidechain_spin_type_boosts(
    systems: &[SpinSystem],
    sequence: &[String],
    sidechain_peaks: &[Peak],
    sc_priors: &SidechainPriors,
    cfg: &SidechainSolverConfig,
) -> (Matrix2D, Matrix2D) {
    let s_len = systems.len();
    let k_len = sequence.len();
    let mut delta_u = Matrix2D::filled(s_len, k_len, 0.0);
    let mut delta_p = Matrix2D::filled(s_len, s_len, 0.0);

    if s_len == 0 || k_len == 0 || sidechain_peaks.is_empty() {
        return (delta_u, delta_p);
    }

    let nodes = build_hc_nodes(sidechain_peaks, cfg);

    // Collect CCCONH and HCCCONH ladders per backbone SpinSystem s
    let mut sys_c_ladders: Vec<Vec<f64>> = vec![Vec::new(); s_len];
    let mut sys_h_ladders: Vec<Vec<f64>> = vec![Vec::new(); s_len];
    let mut sys_overlap: Vec<usize> = vec![1; s_len];

    for s in 0..s_len {
        let (hn, nn) = (systems[s].h, systems[s].n);
        sys_overlap[s] = systems
            .iter()
            .filter(|o| {
                ((o.h - hn) / (cfg.tol_hn_h * 1.25)).hypot((o.n - nn) / (cfg.tol_hn_n * 1.25))
                    <= 1.0
            })
            .count()
            .max(1);

        let mut cs = Vec::new();
        let mut hs = Vec::new();
        for p in sidechain_peaks {
            if p.exp == "CCCONH" {
                if ((p.h - hn) / cfg.tol_hn_h).hypot((p.n - nn) / cfg.tol_hn_n) <= 1.1 {
                    if let Some(c) = p.c {
                        cs.push(c);
                    }
                }
            } else if p.exp == "HCCCONH"
                && ((p.h - hn) / cfg.tol_hn_h).hypot((p.n - nn) / cfg.tol_hn_n) <= 1.1
            {
                if let Some(h2) = p.h2.or(p.c) {
                    hs.push(h2);
                }
            }
        }
        sys_c_ladders[s] = cluster_1d(cs, cfg.tol_c * 0.85);
        sys_h_ladders[s] = cluster_1d(hs, cfg.tol_h * 0.85);
    }

    // 1. Unary boosts: score SpinSystem s's (i-1) sidechain ladder against each unique AA type
    for s in 0..s_len {
        if sys_c_ladders[s].is_empty() && sys_h_ladders[s].is_empty() {
            continue;
        }
        // Only apply strong unary typing boost when amide root is non-overlapped
        let ov_scale = if sys_overlap[s] == 1 { 1.0 } else { 0.25 };
        let mut aa_cache: BTreeMap<&str, f64> = BTreeMap::new();
        for k in 1..k_len {
            if sequence[k] == "PRO" {
                continue;
            }
            let prev_aa = sequence[k - 1].as_str();
            let raw = *aa_cache.entry(prev_aa).or_insert_with(|| {
                score_im1_ladder_against_aa(
                    prev_aa,
                    &sys_c_ladders[s],
                    &sys_h_ladders[s],
                    &nodes,
                    sc_priors,
                    cfg,
                )
            });
            delta_u.set(s, k, cfg.sidechain_boost_weight * ov_scale * raw);
        }
    }

    // 2. Pairwise sequential connectivity boost: SpinSystem s (at k) CA(i)/CB(i) matching SpinSystem t (at k+1) CCCONH(i-1)
    for t in 0..s_len {
        if sys_c_ladders[t].is_empty() {
            continue;
        }
        let ov_scale = if sys_overlap[t] == 1 { 1.0 } else { 0.4 };
        for (s, sys_s) in systems.iter().enumerate().take(s_len) {
            if s == t {
                continue;
            }
            let mut p_boost = 0.0;
            for atom in ["CA", "CB"] {
                if let Some(cands_i) = sys_s.cands.get(&(atom.to_string(), 0)) {
                    let matched = cands_i.iter().any(|&(cv, sup)| {
                        sup > 0 && has_carbon_match(&sys_c_ladders[t], cv, cfg.tol_c).is_some()
                    });
                    if matched {
                        p_boost += 1.4;
                    } else if cands_i.iter().any(|&(_, sup)| sup > 0) && sys_overlap[t] == 1 {
                        p_boost -= 0.8;
                    }
                }
            }
            if p_boost != 0.0 {
                delta_p.set(s, t, ov_scale * p_boost);
            }
        }
    }

    (delta_u, delta_p)
}

/// Complete output of `solve_joint_backbone_sidechain`.
#[derive(Clone, Debug)]
pub struct JointAssignOutput {
    pub backbone_result: AssignResult,
    pub backbone_confidence: BTreeMap<usize, f64>,
    pub sidechain_result: SidechainAssignResult,
    pub spin_type_boosts: Matrix2D,
}

/// Jointly optimize backbone spin-system placement and all-atom sidechain assignments!
/// Sidechain `CCCONH` / `HCCCONH` / `C13HSQC` spin-type LLRs and sequential carbon-ladder links
/// feed directly into the backbone `(U, P)` matrices to resolve ambiguous backbone placements,
/// followed by global per-residue sidechain graph optimization.
#[allow(clippy::too_many_arguments)]
pub fn solve_joint_backbone_sidechain(
    systems: &[SpinSystem],
    sequence: &[String],
    seq_ids: &[i32],
    backbone_priors: &Priors,
    sidechain_priors: &SidechainPriors,
    sidechain_peaks: &[Peak],
    score_cfg: &ScoreConfig,
    anneal_cfg: &AnnealConfig,
    sc_cfg: &SidechainSolverConfig,
) -> JointAssignOutput {
    let mut u = unary_matrix(systems, sequence, backbone_priors, score_cfg);
    let mut p = pair_matrix(systems, score_cfg);

    let (du, dp) = compute_sidechain_spin_type_boosts(
        systems,
        sequence,
        sidechain_peaks,
        sidechain_priors,
        sc_cfg,
    );

    for r in 0..u.rows {
        for c in 0..u.cols {
            let base = u.get(r, c);
            if base > FORBID * 0.5 {
                u.set(r, c, base + du.get(r, c));
            }
        }
    }
    for r in 0..p.rows {
        for c in 0..p.cols {
            let base = p.get(r, c);
            if base > FORBID * 0.5 {
                p.set(r, c, base + dp.get(r, c));
            }
        }
    }

    let bb_res = assign_from_matrices(&u, &p, anneal_cfg);
    let (bb_ref, bb_conf) = to_reference(
        &bb_res,
        systems,
        sequence,
        seq_ids,
        0.0,
        Some(backbone_priors),
    );

    let sc_res = assign_sidechains(
        &bb_ref,
        Some(&bb_conf),
        sidechain_peaks,
        sidechain_priors,
        sc_cfg,
    );

    JointAssignOutput {
        backbone_result: bb_res,
        backbone_confidence: bb_conf,
        sidechain_result: sc_res,
        spin_type_boosts: du,
    }
}

// ============================================================================
// 5. Sidechain Assignment Evaluation Metrics (Modulo Prochiral Symmetry)
// ============================================================================

/// Detailed accuracy and recall breakdown for sidechain chemical shift assignments.
#[derive(Clone, Debug, Default)]
pub struct SidechainMetrics {
    /// Assignable aliphatic sidechain carbons (`CG*`, `CD*`, `CE*` — strictly beyond `CA`/`CB`).
    pub sc_c_assignable: usize,
    pub sc_c_assigned: usize,
    pub sc_c_correct: usize,
    /// Assignable aliphatic protons (`HA*`, `HB*`, `HG*`, `HD*`, `HE*`, deduplicated per methyl).
    pub aliph_h_assignable: usize,
    pub aliph_h_assigned: usize,
    pub aliph_h_correct: usize,
    /// Combined aliphatic sidechain resonances (`sc_c` + `aliph_h`).
    pub total_aliph_assignable: usize,
    pub total_aliph_assigned: usize,
    pub total_aliph_correct: usize,
    /// Aromatic `CH` shifts (`CD*`, `CE*`, `CZ*`, `CH2` and attached protons).
    pub arom_assignable: usize,
    pub arom_assigned: usize,
    pub arom_correct: usize,
}

impl SidechainMetrics {
    pub fn sc_carbon_accuracy(&self) -> f64 {
        if self.sc_c_assigned > 0 {
            (self.sc_c_correct as f64) / (self.sc_c_assigned as f64)
        } else {
            0.0
        }
    }

    pub fn sc_carbon_recall(&self) -> f64 {
        if self.sc_c_assignable > 0 {
            (self.sc_c_correct as f64) / (self.sc_c_assignable as f64)
        } else {
            0.0
        }
    }

    pub fn aliph_proton_accuracy(&self) -> f64 {
        if self.aliph_h_assigned > 0 {
            (self.aliph_h_correct as f64) / (self.aliph_h_assigned as f64)
        } else {
            0.0
        }
    }

    pub fn aliph_proton_recall(&self) -> f64 {
        if self.aliph_h_assignable > 0 {
            (self.aliph_h_correct as f64) / (self.aliph_h_assignable as f64)
        } else {
            0.0
        }
    }

    pub fn aliphatic_accuracy(&self) -> f64 {
        if self.total_aliph_assigned > 0 {
            (self.total_aliph_correct as f64) / (self.total_aliph_assigned as f64)
        } else {
            0.0
        }
    }

    pub fn aliphatic_recall(&self) -> f64 {
        if self.total_aliph_assignable > 0 {
            (self.total_aliph_correct as f64) / (self.total_aliph_assignable as f64)
        } else {
            0.0
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "Aliphatic Sidechain C (CG/CD/CE): acc {:.1}% ({}/{}) | recall {:.1}% ({}/{})\n\
             Aliphatic Sidechain H (HA..HE)  : acc {:.1}% ({}/{}) | recall {:.1}% ({}/{})\n\
             Combined Aliphatic Sidechain    : acc {:.1}% ({}/{}) | recall {:.1}% ({}/{})\n\
             Aromatic Ring C/H               : acc {:.1}% ({}/{})",
            self.sc_carbon_accuracy() * 100.0,
            self.sc_c_correct,
            self.sc_c_assigned,
            self.sc_carbon_recall() * 100.0,
            self.sc_c_correct,
            self.sc_c_assignable,
            self.aliph_proton_accuracy() * 100.0,
            self.aliph_h_correct,
            self.aliph_h_assigned,
            self.aliph_proton_recall() * 100.0,
            self.aliph_h_correct,
            self.aliph_h_assignable,
            self.aliphatic_accuracy() * 100.0,
            self.total_aliph_correct,
            self.total_aliph_assigned,
            self.aliphatic_recall() * 100.0,
            self.total_aliph_correct,
            self.total_aliph_assignable,
            if self.arom_assigned > 0 {
                100.0 * (self.arom_correct as f64) / (self.arom_assigned as f64)
            } else {
                0.0
            },
            self.arom_correct,
            self.arom_assigned,
        )
    }
}

/// Evaluate predicted sidechain chemical shifts against ground-truth `truth` (`tol_c = 0.40 ppm`,
/// `tol_h = 0.06 ppm`), accounting for non-stereospecific prochiral geminal methylene pairs and
/// coupled isopropyl methyl branches (`NMR-STAR Ambiguity_code = 2`).
pub fn evaluate_sidechain_shifts(
    pred: &Reference,
    truth: &Reference,
    tol_c: f64,
    tol_h: f64,
) -> SidechainMetrics {
    let mut m = SidechainMetrics::default();
    let k_len = truth.sequence.len().min(pred.sequence.len());

    // Canonicalize a copy of `pred` per residue modulo prochiral swaps against `truth`
    let mut aligned_pred = pred.shifts.clone();

    for k in 0..k_len {
        let Some(topo) = residue_topology(&truth.sequence[k]) else {
            continue;
        };

        for pg in topo.prochiral_groups {
            match pg {
                ProchiralGroup::Methylene {
                    protons: [p0, p1], ..
                } => {
                    let t0 = truth.shifts.get(&(k, (*p0).to_string())).copied();
                    let t1 = truth.shifts.get(&(k, (*p1).to_string())).copied();
                    let a0 = aligned_pred.get(&(k, (*p0).to_string())).copied();
                    let a1 = aligned_pred.get(&(k, (*p1).to_string())).copied();
                    if let (Some(tv0), Some(tv1), Some(av0), Some(av1)) = (t0, t1, a0, a1) {
                        let err_keep = (tv0 - av0).abs() + (tv1 - av1).abs();
                        let err_swap = (tv0 - av1).abs() + (tv1 - av0).abs();
                        if err_swap < err_keep {
                            aligned_pred.insert((k, (*p0).to_string()), av1);
                            aligned_pred.insert((k, (*p1).to_string()), av0);
                        }
                    }
                }
                ProchiralGroup::IsopropylMethyls {
                    carbon_a,
                    protons_a,
                    carbon_b,
                    protons_b,
                } => {
                    let tca = truth.shifts.get(&(k, (*carbon_a).to_string())).copied();
                    let tcb = truth.shifts.get(&(k, (*carbon_b).to_string())).copied();
                    let tha = truth.shifts.get(&(k, protons_a[0].to_string())).copied();
                    let thb = truth.shifts.get(&(k, protons_b[0].to_string())).copied();

                    let pca = aligned_pred.get(&(k, (*carbon_a).to_string())).copied();
                    let pcb = aligned_pred.get(&(k, (*carbon_b).to_string())).copied();
                    let pha = aligned_pred.get(&(k, protons_a[0].to_string())).copied();
                    let phb = aligned_pred.get(&(k, protons_b[0].to_string())).copied();

                    if let (Some(tca), Some(tcb), Some(pca), Some(pcb)) = (tca, tcb, pca, pcb) {
                        let mut err_keep =
                            ((tca - pca) / tol_c).abs() + ((tcb - pcb) / tol_c).abs();
                        let mut err_swap =
                            ((tca - pcb) / tol_c).abs() + ((tcb - pca) / tol_c).abs();
                        if let (Some(tha), Some(thb), Some(pha), Some(phb)) = (tha, thb, pha, phb) {
                            err_keep += ((tha - pha) / tol_h).abs() + ((thb - phb) / tol_h).abs();
                            err_swap += ((tha - phb) / tol_h).abs() + ((thb - pha) / tol_h).abs();
                        }
                        if err_swap < err_keep {
                            aligned_pred.insert((k, (*carbon_a).to_string()), pcb);
                            aligned_pred.insert((k, (*carbon_b).to_string()), pca);
                            if let (Some(pha), Some(phb)) = (pha, phb) {
                                for &pa in protons_a {
                                    aligned_pred.insert((k, pa.to_string()), phb);
                                }
                                for &pb in protons_b {
                                    aligned_pred.insert((k, pb.to_string()), pha);
                                }
                            }
                        }
                    }
                }
                ProchiralGroup::SymmetricAromatic { .. } => {}
            }
        }

        for chg in topo.ch_groups {
            if chg.region.is_aliphatic() {
                // Score sidechain carbons strictly beyond backbone CA/CB
                if chg.carbon != "CA" && chg.carbon != "CB" {
                    if let Some(&tc) = truth.shifts.get(&(k, chg.carbon.to_string())) {
                        m.sc_c_assignable += 1;
                        if let Some(&pc) = aligned_pred.get(&(k, chg.carbon.to_string())) {
                            m.sc_c_assigned += 1;
                            if (tc - pc).abs() <= tol_c {
                                m.sc_c_correct += 1;
                            }
                        }
                    }
                }

                // Score distinct aliphatic proton resonances (count each 3-proton methyl once)
                let rep_protons: &[&str] = match chg.multiplicity {
                    ChMultiplicity::Methyl => &chg.protons[..1],
                    _ => chg.protons,
                };
                for &p_name in rep_protons {
                    if let Some(&th) = truth.shifts.get(&(k, p_name.to_string())) {
                        m.aliph_h_assignable += 1;
                        if let Some(&ph) = aligned_pred.get(&(k, p_name.to_string())) {
                            m.aliph_h_assigned += 1;
                            if (th - ph).abs() <= tol_h {
                                m.aliph_h_correct += 1;
                            }
                        }
                    }
                }
            } else {
                // Aromatic CH
                if let Some(&tc) = truth.shifts.get(&(k, chg.carbon.to_string())) {
                    m.arom_assignable += 1;
                    if let Some(&pc) = aligned_pred.get(&(k, chg.carbon.to_string())) {
                        m.arom_assigned += 1;
                        if (tc - pc).abs() <= tol_c * 1.25 {
                            m.arom_correct += 1;
                        }
                    }
                }
                let p_name = chg.protons[0];
                if let Some(&th) = truth.shifts.get(&(k, p_name.to_string())) {
                    m.arom_assignable += 1;
                    if let Some(&ph) = aligned_pred.get(&(k, p_name.to_string())) {
                        m.arom_assigned += 1;
                        if (th - ph).abs() <= tol_h * 1.25 {
                            m.arom_correct += 1;
                        }
                    }
                }
            }
        }
    }

    m.total_aliph_assignable = m.sc_c_assignable + m.aliph_h_assignable;
    m.total_aliph_assigned = m.sc_c_assigned + m.aliph_h_assigned;
    m.total_aliph_correct = m.sc_c_correct + m.aliph_h_correct;
    m
}
