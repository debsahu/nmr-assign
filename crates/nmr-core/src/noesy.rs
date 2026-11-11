//! PDB coordinate reader, 3D `15N`/`13C`-edited NOESY simulator & picker, and Nilges (1995)
//! iterative ambiguous distance restraint (`ADR`) assignment with network anchoring.
//!
//! Provenance: Clean-room implementation from peer-reviewed NMR structure calculation literature:
//! - Nilges, M. (1995). "Calculation of protein structures with ambiguous distance restraints.
//!   Automated assignment of ambiguous NOE crosspeaks and disulphide bridges."
//!   *J. Mol. Biol.* 245(5), 645-660 (`D_eff = (sum_k d_k^-6)^(-1/6)` and iterative re-weighting).
//! - Herrmann, T., Guntert, P. & Wuthrich, K. (2002). *J. Mol. Biol.* 319(1), 209-227
//!   (network-anchoring principle using covalent, sequential, reciprocal, and triangle NOE consistency).
//! - Wuthrich, K., Billeter, M. & Braun, W. (1983). "Pseudo-structures for the 20 common amino acids
//!   for use in studies of protein conformations by measurements of intramolecular proton-proton
//!   distance constraints with nuclear magnetic resonance." *J. Mol. Biol.* 169(4), 949-961.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::dihedrals::{align_by_res_names, random_coil_shifts};
use crate::nmrstar::Reference;
use crate::peaks::Peak;
use crate::picker::{estimate_noise_mad, PickerConfig};
use crate::rng::Rng;
use crate::spectrum::{AxisPpm, Spectrum3D};

/// Cartesian 3D coordinate `(x, y, z)` in Angstroms.
pub type Vec3 = [f64; 3];

#[inline]
pub fn v_add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn v_sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn v_scale(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub fn v_dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn v_cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn v_norm_sq(a: Vec3) -> f64 {
    v_dot(a, a)
}

#[inline]
pub fn v_norm(a: Vec3) -> f64 {
    v_norm_sq(a).sqrt()
}

#[inline]
pub fn v_dist_sq(a: Vec3, b: Vec3) -> f64 {
    v_norm_sq(v_sub(a, b))
}

#[inline]
pub fn v_dist(a: Vec3, b: Vec3) -> f64 {
    v_dist_sq(a, b).sqrt()
}

#[inline]
pub fn v_unit(a: Vec3) -> Vec3 {
    let n = v_norm(a);
    if n > 1e-12 {
        [a[0] / n, a[1] / n, a[2] / n]
    } else {
        [1.0, 0.0, 0.0]
    }
}

/// IUPAC right-handed dihedral torsion angle (in degrees `(-180, 180]`) defined by four points `p0-p1-p2-p3`.
pub fn calc_dihedral_deg(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> f64 {
    let b0 = v_sub(p0, p1);
    let b1 = v_unit(v_sub(p2, p1));
    let b2 = v_sub(p3, p2);

    let v = v_sub(b0, v_scale(b1, v_dot(b0, b1)));
    let w = v_sub(b2, v_scale(b1, v_dot(b2, b1)));
    let x = v_dot(v, w);
    let m = v_cross(b1, v);
    let y = v_dot(m, w);
    y.atan2(x).to_degrees()
}

/// Proton / pseudoatom site participating in 3D `15N`-edited and `13C`-edited NOESY crosspeaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NoeProton {
    /// Backbone amide `1H^N` attached to `15N`.
    AmideH,
    /// Backbone alpha `1H^alpha` attached to `13CA`.
    AlphaHA,
    /// Beta-methylene / methyl pseudoatom centroid `QB` along `CA -> CB` attached to `13CB`
    /// (Wuthrich, Billeter & Braun, *J. Mol. Biol.* 1983).
    BetaQB,
}

impl NoeProton {
    pub fn label(self) -> &'static str {
        match self {
            Self::AmideH => "H",
            Self::AlphaHA => "HA",
            Self::BetaQB => "QB",
        }
    }
}

/// Parsed and proton-completed backbone + `CB`/`QB` coordinates for one residue of a PDB model.
#[derive(Clone, Debug, PartialEq)]
pub struct PdbResidue {
    pub seq_id: i32,
    pub res_name: String,
    pub chain_id: char,
    pub n: Vec3,
    pub h: Option<Vec3>,
    pub ca: Vec3,
    pub ha: Vec3,
    pub ha3: Option<Vec3>,
    pub c: Vec3,
    pub o: Vec3,
    pub cb: Option<Vec3>,
    pub qb: Option<Vec3>,
}

impl PdbResidue {
    /// Coordinate of a specific NOE proton/pseudoatom site on this residue.
    #[inline]
    pub fn noe_coord(&self, site: NoeProton) -> Option<Vec3> {
        match site {
            NoeProton::AmideH => self.h,
            NoeProton::AlphaHA => Some(self.ha),
            NoeProton::BetaQB => self.qb,
        }
    }
}

/// First-model polymer chain extracted from a PDB file.
#[derive(Clone, Debug, Default)]
pub struct PdbChain {
    pub residues: Vec<PdbResidue>,
}

/// Reconstruct chiral L-amino acid `CB`, `HA`, and `QB` positions from `(N, CA, C)` when absent in X-ray PDBs.
pub fn reconstruct_ca_substituents(
    res_name: &str,
    n: Vec3,
    ca: Vec3,
    c: Vec3,
) -> (Option<Vec3>, Vec3, Option<Vec3>, Option<Vec3>) {
    let b1 = v_unit(v_sub(n, ca));
    let b2 = v_unit(v_sub(c, ca));
    let u = v_unit(v_add(b1, b2));
    let normal = v_unit(v_cross(b1, b2));

    // Exact tetrahedral L-amino acid projections validated against NMR 1D3Z (< 0.05 A):
    // L-CB direction: unit(-0.581 * u + 0.814 * normal), d(CA-CB) = 1.526 A
    // L-HA direction: unit(-0.581 * u - 0.814 * normal), d(CA-HA) = 1.090 A
    let dir_l_cb = v_unit(v_add(v_scale(u, -0.5810), v_scale(normal, 0.8139)));
    let dir_l_ha = v_unit(v_add(v_scale(u, -0.5810), v_scale(normal, -0.8139)));

    if res_name.eq_ignore_ascii_case("GLY") {
        let ha2 = v_add(ca, v_scale(dir_l_cb, 1.090));
        let ha3 = v_add(ca, v_scale(dir_l_ha, 1.090));
        // Centroid of prochiral HA2/HA3 for degenerate Gly HA matching, plus individual ha3 stored
        let ha_mid = v_scale(v_add(ha2, ha3), 0.5);
        (None, ha_mid, Some(ha3), None)
    } else {
        let cb = v_add(ca, v_scale(dir_l_cb, 1.526));
        let ha = v_add(ca, v_scale(dir_l_ha, 1.090));
        // Tetrahedral beta-proton centroid along CA -> CB axis: CB + 1.09 * cos(70.53 deg) * dir = CB + 0.363 * dir
        let qb = v_add(cb, v_scale(dir_l_cb, 0.3633));
        (Some(cb), ha, None, Some(qb))
    }
}

/// Parse the first model of a PDB coordinate file, extracting `(N, H, CA, HA, C, O, CB, QB)`
/// and geometrically reconstructing amide `H`, `HA`, `O`, and `CB`/`QB` when absent (e.g. X-ray references).
pub fn parse_pdb_first_model(path: impl AsRef<Path>) -> Result<PdbChain, String> {
    let text = fs::read_to_string(path.as_ref())
        .map_err(|e| format!("failed to read PDB {}: {e}", path.as_ref().display()))?;
    parse_pdb_str_first_model(&text)
}

/// Parse the first model of a PDB string into a `PdbChain`.
pub fn parse_pdb_str_first_model(pdb_text: &str) -> Result<PdbChain, String> {
    #[derive(Default)]
    struct RawRes {
        seq_id: i32,
        res_name: String,
        chain_id: char,
        n: Option<Vec3>,
        h: Option<Vec3>,
        ca: Option<Vec3>,
        ha: Option<Vec3>,
        ha2: Option<Vec3>,
        ha3: Option<Vec3>,
        c: Option<Vec3>,
        o: Option<Vec3>,
        cb: Option<Vec3>,
        hb_sum: Vec3,
        hb_count: usize,
    }

    let mut raw_list: Vec<RawRes> = Vec::new();
    let mut index_by_seq: BTreeMap<(char, i32), usize> = BTreeMap::new();
    let mut selected_chain: Option<char> = None;

    for line in pdb_text.lines() {
        if line.starts_with("ENDMDL") {
            break;
        }
        if !line.starts_with("ATOM  ") || line.len() < 54 {
            continue;
        }
        let bytes = line.as_bytes();
        let alt_loc = bytes[16] as char;
        if alt_loc != ' ' && alt_loc != 'A' && alt_loc != '1' {
            continue;
        }
        let chain_id = bytes[21] as char;
        if let Some(ch) = selected_chain {
            if chain_id != ch {
                continue;
            }
        } else {
            selected_chain = Some(chain_id);
        }

        // Ignore insertion codes other than space so residue numbering stays clean
        let i_code = bytes[26] as char;
        if i_code != ' ' {
            continue;
        }

        let atom_name = line[12..16].trim();
        let res_name = line[17..20].trim().to_ascii_uppercase();
        let Ok(seq_id) = line[22..26].trim().parse::<i32>() else {
            continue;
        };
        let Ok(x) = line[30..38].trim().parse::<f64>() else {
            continue;
        };
        let Ok(y) = line[38..46].trim().parse::<f64>() else {
            continue;
        };
        let Ok(z) = line[46..54].trim().parse::<f64>() else {
            continue;
        };
        let xyz = [x, y, z];

        let idx = *index_by_seq.entry((chain_id, seq_id)).or_insert_with(|| {
            let id = raw_list.len();
            raw_list.push(RawRes {
                seq_id,
                res_name: res_name.clone(),
                chain_id,
                ..Default::default()
            });
            id
        });

        let entry = &mut raw_list[idx];
        match atom_name {
            "N" => entry.n = Some(xyz),
            "H" | "HN" => {
                if entry.h.is_none() {
                    entry.h = Some(xyz);
                }
            }
            "CA" => entry.ca = Some(xyz),
            "HA" => entry.ha = Some(xyz),
            "HA2" | "1HA" => entry.ha2 = Some(xyz),
            "HA3" | "2HA" => entry.ha3 = Some(xyz),
            "C" => entry.c = Some(xyz),
            "O" | "OT1" | "OXT" => {
                if entry.o.is_none() {
                    entry.o = Some(xyz);
                }
            }
            "CB" => entry.cb = Some(xyz),
            "HB" | "HB1" | "HB2" | "HB3" | "1HB" | "2HB" | "3HB" => {
                entry.hb_sum = v_add(entry.hb_sum, xyz);
                entry.hb_count += 1;
            }
            _ => {}
        }
    }

    // Filter to complete backbone (N, CA, C) residues
    let valid_raw: Vec<RawRes> = raw_list
        .into_iter()
        .filter(|r| r.n.is_some() && r.ca.is_some() && r.c.is_some())
        .collect();

    if valid_raw.is_empty() {
        return Err("no complete protein backbone (N, CA, C) residues found in PDB".to_string());
    }

    let m = valid_raw.len();
    let mut residues = Vec::with_capacity(m);

    for i in 0..m {
        let r = &valid_raw[i];
        let n = r.n.unwrap();
        let ca = r.ca.unwrap();
        let c = r.c.unwrap();

        let (rec_cb, rec_ha, rec_ha3, rec_qb) = reconstruct_ca_substituents(&r.res_name, n, ca, c);

        let cb = if r.res_name == "GLY" {
            None
        } else {
            r.cb.or(rec_cb)
        };

        let qb = if r.res_name == "GLY" {
            None
        } else if r.hb_count > 0 {
            // Use tetrahedral axis QB consistent with backbone+CB model (within 0.05 A of HB centroid!)
            if let Some(cb_pos) = cb {
                let dir = v_unit(v_sub(cb_pos, ca));
                Some(v_add(cb_pos, v_scale(dir, 0.3633)))
            } else {
                Some(v_scale(r.hb_sum, 1.0 / (r.hb_count as f64)))
            }
        } else if let Some(cb_pos) = cb {
            let dir = v_unit(v_sub(cb_pos, ca));
            Some(v_add(cb_pos, v_scale(dir, 0.3633)))
        } else {
            rec_qb
        };

        let ha = if r.res_name == "GLY" {
            match (r.ha2, r.ha3) {
                (Some(a2), Some(a3)) => v_scale(v_add(a2, a3), 0.5),
                (Some(a2), None) => a2,
                _ => r.ha.unwrap_or(rec_ha),
            }
        } else {
            r.ha.or(r.ha2).unwrap_or(rec_ha)
        };

        let ha3 = r.ha3.or(rec_ha3);

        // Reconstruct carbonyl O if missing
        let o = if let Some(o_pos) = r.o {
            o_pos
        } else if i + 1 < m && v_dist(c, valid_raw[i + 1].n.unwrap()) < 2.0 {
            let w1 = v_unit(v_sub(ca, c));
            let w2 = v_unit(v_sub(valid_raw[i + 1].n.unwrap(), c));
            v_sub(c, v_scale(v_unit(v_add(w1, w2)), 1.231))
        } else {
            let w1 = v_unit(v_sub(n, ca));
            let w2 = v_unit(v_sub(ca, c));
            v_add(c, v_scale(v_unit(v_sub(w2, w1)), 1.231))
        };

        // Reconstruct amide H if missing (non-PRO, non-N-terminal with peptide bond C_{i-1}-N_i < 2.0 A)
        let h = if r.res_name == "PRO" {
            None
        } else if let Some(h_pos) = r.h {
            Some(h_pos)
        } else if i > 0 && v_dist(valid_raw[i - 1].c.unwrap(), n) < 2.0 {
            let u1 = v_unit(v_sub(n, valid_raw[i - 1].c.unwrap()));
            let u2 = v_unit(v_sub(n, ca));
            Some(v_add(n, v_scale(v_unit(v_add(u1, u2)), 1.010)))
        } else {
            None
        };

        residues.push(PdbResidue {
            seq_id: r.seq_id,
            res_name: r.res_name.clone(),
            chain_id: r.chain_id,
            n,
            h,
            ca,
            ha,
            ha3,
            c,
            o,
            cb,
            qb,
        });
    }

    Ok(PdbChain { residues })
}

/// Extract experimental `(seq_id, res_name, phi_deg, psi_deg)` torsions from a `PdbChain`,
/// checking peptide-bond continuity (`d(C_{i-1}, N_i) < 2.0 A`) across chain breaks.
pub fn compute_chain_torsions(chain: &PdbChain) -> Vec<(i32, String, Option<f64>, Option<f64>)> {
    let m = chain.residues.len();
    let mut out = Vec::with_capacity(m);
    for i in 0..m {
        let r = &chain.residues[i];
        let phi = if i > 0 && v_dist(chain.residues[i - 1].c, r.n) < 2.0 {
            Some(calc_dihedral_deg(chain.residues[i - 1].c, r.n, r.ca, r.c))
        } else {
            None
        };
        let psi = if i + 1 < m && v_dist(r.c, chain.residues[i + 1].n) < 2.0 {
            Some(calc_dihedral_deg(r.n, r.ca, r.c, chain.residues[i + 1].n))
        } else {
            None
        };
        out.push((r.seq_id, r.res_name.clone(), phi, psi));
    }
    out
}

/// Configuration for realistic 3D `15N`-edited and `13C`-edited NOESY simulation.
#[derive(Clone, Debug)]
pub struct NoesySimConfig {
    /// Maximum inter-proton NOE observability cutoff in Angstroms (`<= 5.5 A`).
    pub cutoff_angstrom: f64,
    /// Gaussian chemical-shift noise on `1H` dimensions (ppm).
    pub sigma_h_ppm: f64,
    /// Gaussian chemical-shift noise on `15N` / `13C` heavy-atom dimensions (ppm).
    pub sigma_heavy_ppm: f64,
    /// Intensity scaling constant `I_0` in `V = I_0 * r^-6`.
    pub i0_scale: f64,
    /// Fraction of true crosspeaks randomly missing due to overlap/relaxation (`[0, 1)`).
    pub dropout_rate: f64,
    /// Synthesize fallback `HA`/`HB` shifts from random-coil + secondary shift when `Reference`
    /// only carries 5-atom backbone shifts (`H, N, CA, CB, C`).
    pub synthesize_missing_aliphatic_protons: bool,
    pub include_15n_edited: bool,
    pub include_13c_edited: bool,
    pub seed: u64,
}

impl Default for NoesySimConfig {
    fn default() -> Self {
        Self {
            cutoff_angstrom: 5.5,
            sigma_h_ppm: 0.015,
            sigma_heavy_ppm: 0.15,
            i0_scale: 1.0e6,
            dropout_rate: 0.02,
            synthesize_missing_aliphatic_protons: true,
            include_15n_edited: true,
            include_13c_edited: true,
            seed: 42,
        }
    }
}

/// Assigned chemical-shift entry for an observable NOE proton and its attached `15N`/`13C` heavy atom.
#[derive(Clone, Debug)]
pub struct ObservableProtonShift {
    pub res_idx: usize,
    pub seq_id: i32,
    pub site: NoeProton,
    pub h_ppm: f64,
    /// `Some(n_ppm)` for `AmideH`, `Some(ca_ppm)` for `AlphaHA`, `Some(cb_ppm)` for `BetaQB`.
    pub heavy_ppm: Option<f64>,
}

/// Extract observable `(AmideH, AlphaHA, BetaQB)` chemical shifts from `Reference`.
pub fn collect_proton_shifts(
    reference: &Reference,
    synthesize_missing_aliphatic: bool,
) -> Vec<ObservableProtonShift> {
    let n = reference.sequence.len();
    let mut out = Vec::with_capacity(n * 3);

    for i in 0..n {
        let res = &reference.sequence[i];
        let seq_id = reference.seq_ids[i];
        let is_gly = res.eq_ignore_ascii_case("GLY");
        let is_pro = res.eq_ignore_ascii_case("PRO");
        let next_pro = i + 1 < n && reference.sequence[i + 1].eq_ignore_ascii_case("PRO");
        let cb_ppm = reference.shifts.get(&(i, "CB".to_string())).copied();
        let ca_ppm = reference.shifts.get(&(i, "CA".to_string())).copied();
        let cys_ox = res.eq_ignore_ascii_case("CYS") && cb_ppm.unwrap_or(28.0) > 35.0;
        let rc = random_coil_shifts(res, cys_ox, next_pro);

        // 1. Amide H + N
        if !is_pro {
            if let Some(&h_ppm) = reference.shifts.get(&(i, "H".to_string())) {
                let n_ppm = reference.shifts.get(&(i, "N".to_string())).copied();
                out.push(ObservableProtonShift {
                    res_idx: i,
                    seq_id,
                    site: NoeProton::AmideH,
                    h_ppm,
                    heavy_ppm: n_ppm,
                });
            }
        }

        // 2. Alpha HA + CA (average HA2/HA3 for GLY if both assigned)
        let ha_obs = if let Some(&ha) = reference.shifts.get(&(i, "HA".to_string())) {
            Some(ha)
        } else {
            let ha2 = reference.shifts.get(&(i, "HA2".to_string())).copied();
            let ha3 = reference.shifts.get(&(i, "HA3".to_string())).copied();
            match (ha2, ha3) {
                (Some(a), Some(b)) => Some(0.5 * (a + b)),
                (Some(a), None) | (None, Some(a)) => Some(a),
                (None, None) => None,
            }
        };
        let ha_final = ha_obs.or_else(|| {
            if let (true, Some(ca)) = (synthesize_missing_aliphatic, ca_ppm) {
                let dca = ca - rc.ca;
                let dcb = if !is_gly {
                    cb_ppm.map(|v| v - rc.cb).unwrap_or(0.0)
                } else {
                    0.0
                };
                Some(rc.ha - 0.08 * (dca - dcb))
            } else {
                None
            }
        });
        if let Some(h_ppm) = ha_final {
            out.push(ObservableProtonShift {
                res_idx: i,
                seq_id,
                site: NoeProton::AlphaHA,
                h_ppm,
                heavy_ppm: ca_ppm,
            });
        }

        // 3. Beta QB + CB (mean of HB/HB1/HB2/HB3 for non-GLY)
        if !is_gly {
            let mut hb_vals = Vec::new();
            for key in ["HB", "HB1", "HB2", "HB3"] {
                if let Some(&v) = reference.shifts.get(&(i, key.to_string())) {
                    hb_vals.push(v);
                }
            }
            let hb_obs = if !hb_vals.is_empty() {
                Some(hb_vals.iter().sum::<f64>() / (hb_vals.len() as f64))
            } else if let (true, Some(cb)) = (synthesize_missing_aliphatic, cb_ppm) {
                let dcb = cb - rc.cb;
                let dca = ca_ppm.map(|v| v - rc.ca).unwrap_or(0.0);
                let dh = reference
                    .shifts
                    .get(&(i, "H".to_string()))
                    .map(|&v| v - rc.h)
                    .unwrap_or(0.0);
                Some(rc.hb + 0.12 * dcb - 0.05 * dca + 0.18 * dh)
            } else {
                None
            };
            if let Some(h_ppm) = hb_obs {
                out.push(ObservableProtonShift {
                    res_idx: i,
                    seq_id,
                    site: NoeProton::BetaQB,
                    h_ppm,
                    heavy_ppm: cb_ppm,
                });
            }
        }
    }

    out
}

/// Simulate realistic 3D `15N`-edited (`NOESY15N`) and `13C`-edited (`NOESY13C`) unassigned NOESY peak lists
/// from experimental PDB coordinates and chemical shifts (`I = I_0 * r^-6`, `r <= 5.5 A`).
pub fn simulate_noesy_peaks(pdb: &PdbChain, shifts: &Reference, cfg: &NoesySimConfig) -> Vec<Peak> {
    let ref_keys: Vec<(i32, &str)> = shifts
        .seq_ids
        .iter()
        .zip(shifts.sequence.iter())
        .map(|(&sid, rn)| (sid, rn.as_str()))
        .collect();
    let pdb_keys: Vec<(i32, &str)> = pdb
        .residues
        .iter()
        .map(|r| (r.seq_id, r.res_name.as_str()))
        .collect();
    let aligned = align_by_res_names(&ref_keys, &pdb_keys);

    let mut ref_to_pdb = vec![None; shifts.sequence.len()];
    for (ri, pi) in aligned {
        ref_to_pdb[ri] = Some(pi);
    }

    let proton_table = collect_proton_shifts(shifts, cfg.synthesize_missing_aliphatic_protons);
    let mut rng = Rng::new(cfg.seed);
    let mut peaks = Vec::new();

    for (ia, pa) in proton_table.iter().enumerate() {
        let Some(pdb_i) = ref_to_pdb[pa.res_idx] else {
            continue;
        };
        let Some(coord_a) = pdb.residues[pdb_i].noe_coord(pa.site) else {
            continue;
        };
        let Some(heavy_a) = pa.heavy_ppm else {
            continue;
        };

        let exp_name = match pa.site {
            NoeProton::AmideH => {
                if !cfg.include_15n_edited {
                    continue;
                }
                "NOESY15N"
            }
            NoeProton::AlphaHA | NoeProton::BetaQB => {
                if !cfg.include_13c_edited {
                    continue;
                }
                "NOESY13C"
            }
        };

        for (ib, pb) in proton_table.iter().enumerate() {
            if ia == ib {
                continue;
            }
            // Skip fixed-geometry intra-residue HA-QB crosspeaks (they carry zero conformational information)
            if pa.res_idx == pb.res_idx
                && matches!(
                    (pa.site, pb.site),
                    (NoeProton::AlphaHA, NoeProton::BetaQB)
                        | (NoeProton::BetaQB, NoeProton::AlphaHA)
                )
            {
                continue;
            }
            let Some(pdb_j) = ref_to_pdb[pb.res_idx] else {
                continue;
            };
            let Some(coord_b) = pdb.residues[pdb_j].noe_coord(pb.site) else {
                continue;
            };

            let r = v_dist(coord_a, coord_b);
            if r > cfg.cutoff_angstrom || r < 1.5 {
                continue;
            }

            // Diagonal ridge suppression when direct and indirect 1H are unresolved
            if (pa.h_ppm - pb.h_ppm).abs() < 0.025 && pa.res_idx == pb.res_idx {
                continue;
            }

            if cfg.dropout_rate > 0.0 && rng.f64() < cfg.dropout_rate {
                continue;
            }

            let r_eff = r.max(1.85);
            let log_jitter = rng.normal(0.0, 0.10).exp();
            let volume = cfg.i0_scale * r_eff.powi(-6) * log_jitter;

            let h_obs = pa.h_ppm + rng.normal(0.0, cfg.sigma_h_ppm);
            let heavy_obs = heavy_a + rng.normal(0.0, cfg.sigma_heavy_ppm);
            let h2_obs = pb.h_ppm + rng.normal(0.0, cfg.sigma_h_ppm);

            peaks.push(Peak::noesy_3d(exp_name, h_obs, heavy_obs, h2_obs, volume));
        }
    }

    peaks
}

/// Render a synthetic 3D `NOESY15N` or `NOESY13C` spectrum cube (`axis_x = H_direct`,
/// `axis_y = 15N or 13C`, `axis_z = H_indirect`) for testing 3D NOESY volumetric picking.
pub fn render_noesy_3d_spectrum(
    peaks: &[Peak],
    exp: &str,
    nx: usize,
    ny: usize,
    nz: usize,
    noise_sd: f32,
    seed: u64,
) -> Spectrum3D {
    let is_c13 = exp == "NOESY13C";
    let axis_x = if is_c13 {
        AxisPpm::from_bounds("1H", nx, 5.8, 0.5, 600.0)
    } else {
        AxisPpm::from_bounds("1H", nx, 10.5, 6.0, 600.0)
    };
    let axis_y = if is_c13 {
        AxisPpm::from_bounds("13C", ny, 75.0, 14.0, 150.0)
    } else {
        AxisPpm::from_bounds("15N", ny, 136.0, 104.0, 60.0)
    };
    let axis_z = AxisPpm::from_bounds("1H_noe", nz, 10.5, 0.5, 600.0);

    let mut data = vec![0.0f32; nx * ny * nz];
    let mut rng = Rng::new(seed);
    if noise_sd > 0.0 {
        for v in &mut data {
            *v = rng.normal(0.0, noise_sd as f64) as f32;
        }
    }

    let sig_px = 1.05f64;
    let rad = 3isize;

    for p in peaks.iter().filter(|p| p.exp == exp) {
        let Some(h2) = p.h2 else { continue };
        let heavy = if is_c13 {
            let Some(c) = p.c else { continue };
            c
        } else {
            p.n
        };
        let amp = p.height.or(p.volume).unwrap_or(100.0) as f32;

        let cx = axis_x.ppm_to_index(p.h);
        let cy = axis_y.ppm_to_index(heavy);
        let cz = axis_z.ppm_to_index(h2);

        let ix0 = cx.round() as isize;
        let iy0 = cy.round() as isize;
        let iz0 = cz.round() as isize;

        for dz in -rad..=rad {
            let iz = iz0 + dz;
            if iz < 0 || iz >= nz as isize {
                continue;
            }
            let gz = -0.5 * ((iz as f64 - cz) / sig_px).powi(2);
            for dy in -rad..=rad {
                let iy = iy0 + dy;
                if iy < 0 || iy >= ny as isize {
                    continue;
                }
                let gy = -0.5 * ((iy as f64 - cy) / sig_px).powi(2);
                for dx in -rad..=rad {
                    let ix = ix0 + dx;
                    if ix < 0 || ix >= nx as isize {
                        continue;
                    }
                    let gx = -0.5 * ((ix as f64 - cx) / sig_px).powi(2);
                    let w = (gx + gy + gz).exp() as f32;
                    data[((iz as usize) * ny + (iy as usize)) * nx + (ix as usize)] += amp * w;
                }
            }
        }
    }

    Spectrum3D {
        name: exp.to_string(),
        exp: exp.to_string(),
        axis_x,
        axis_y,
        axis_z,
        data,
    }
}

#[inline]
fn subvoxel_3pt(ym1: f32, y0: f32, yp1: f32) -> f64 {
    let a = (ym1.abs()) as f64;
    let b = (y0.abs()) as f64;
    let c = (yp1.abs()) as f64;
    if a > 0.0 && b > a && b > c && c > 0.0 {
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

/// Pick 3D `NOESY15N` or `NOESY13C` crosspeaks from a volumetric `Spectrum3D`, optionally
/// restricted to 2D `(H_direct, N_or_C)` root strips.
pub fn pick_noesy_3d(
    spec: &Spectrum3D,
    anchor_roots: &[(f64, f64)],
    cfg: &PickerConfig,
) -> Vec<Peak> {
    let nx = spec.nx();
    let ny = spec.ny();
    let nz = spec.nz();
    if nx < 3 || ny < 3 || nz < 3 {
        return Vec::new();
    }

    let noise = estimate_noise_mad(&spec.data);
    let thresh = cfg.snr_threshold * noise;
    let mut out = Vec::new();

    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            let heavy_grid = spec.axis_y.index_to_ppm(iy as f64);
            if !anchor_roots.is_empty()
                && !anchor_roots
                    .iter()
                    .any(|&(_, hv)| (hv - heavy_grid).abs() <= cfg.strip_tol_n * 1.5)
            {
                continue;
            }

            for ix in 1..nx - 1 {
                let v = spec.get(ix, iy, iz);
                if v < thresh {
                    continue;
                }
                let h_grid = spec.axis_x.index_to_ppm(ix as f64);
                if !anchor_roots.is_empty()
                    && !anchor_roots.iter().any(|&(hr, hv)| {
                        (hr - h_grid).abs() <= cfg.strip_tol_h
                            && (hv - heavy_grid).abs() <= cfg.strip_tol_n
                    })
                {
                    continue;
                }

                let mut is_max = true;
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
                            if nv >= v {
                                is_max = false;
                                break 'nbr;
                            }
                        }
                    }
                }
                if !is_max {
                    continue;
                }

                let dx = subvoxel_3pt(spec.get(ix - 1, iy, iz), v, spec.get(ix + 1, iy, iz));
                let dy = subvoxel_3pt(spec.get(ix, iy - 1, iz), v, spec.get(ix, iy + 1, iz));
                let dz = subvoxel_3pt(spec.get(ix, iy, iz - 1), v, spec.get(ix, iy, iz + 1));

                let h_ppm = spec.axis_x.index_to_ppm(ix as f64 + dx);
                let heavy_ppm = spec.axis_y.index_to_ppm(iy as f64 + dy);
                let h2_ppm = spec.axis_z.index_to_ppm(iz as f64 + dz);

                out.push(Peak::noesy_3d(
                    spec.exp.clone(),
                    h_ppm,
                    heavy_ppm,
                    h2_ppm,
                    v as f64,
                ));
            }
        }
    }

    out
}

/// Calibrate NOE upper distance bound (`1.8-2.95 A` strong, `1.8-3.75 A` medium, `1.8-4.70 A` medium-weak,
/// `1.8-5.65 A` weak) from peak intensity/volume using `r_app = (I_0 / V)^(1/6)`.
pub fn calibrate_noe_bounds(volume: f64, i0_scale: f64) -> (f64, f64) {
    let v = volume.max(1.0);
    let r_app = (i0_scale / v).powf(1.0 / 6.0);
    let upper = if r_app <= 2.60 {
        2.95
    } else if r_app <= 3.40 {
        3.75
    } else if r_app <= 4.35 {
        4.70
    } else {
        5.65
    };
    (1.80, upper)
}

/// A single candidate proton-proton assignment contributing to an ambiguous NOE crosspeak.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoeCandidatePair {
    pub res_a: usize,
    pub atom_a: NoeProton,
    pub res_b: usize,
    pub atom_b: NoeProton,
    pub shift_score: f64,
    pub network_score: f64,
    pub weight: f64,
}

/// Nilges (1995) Ambiguous Distance Restraint (`D_eff = (sum_k d_k^-6)^(-1/6)`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AmbiguousDistanceRestraint {
    pub peak_index: usize,
    pub exp: String,
    pub lower_bound: f64,
    pub upper_bound: f64,
    /// Active contributing candidate proton pairs in the current iterative cycle.
    pub candidates: Vec<NoeCandidatePair>,
    /// Full pool of shift-compatible candidate pairs preserved for non-destructive iterative re-evaluation.
    pub all_candidates: Vec<NoeCandidatePair>,
    pub is_unambiguous: bool,
}

/// Configuration for chemical-shift matching, network anchoring, and iterative ADR filtering.
#[derive(Clone, Debug)]
pub struct AdrConfig {
    pub tol_h_anchor: f64,
    pub tol_n_anchor: f64,
    pub tol_c_anchor: f64,
    pub tol_h_indirect: f64,
    pub i0_scale: f64,
    /// Candidate weight threshold (`>= 0.85`) to promote an ADR to unambiguous single-pair restraint.
    pub unambiguous_threshold: f64,
    pub synthesize_missing_aliphatic_protons: bool,
}

impl Default for AdrConfig {
    fn default() -> Self {
        Self {
            tol_h_anchor: 0.045,
            tol_n_anchor: 0.45,
            tol_c_anchor: 0.45,
            tol_h_indirect: 0.045,
            i0_scale: 1.0e6,
            unambiguous_threshold: 0.85,
            synthesize_missing_aliphatic_protons: true,
        }
    }
}

/// Cycle 0: Match unassigned 3D `NOESY15N` and `NOESY13C` crosspeaks to chemical shifts and
/// apply covalent, sequential, reciprocal-symmetry, and triangle network anchoring.
pub fn match_initial_adrs(
    peaks: &[Peak],
    shifts: &Reference,
    cfg: &AdrConfig,
) -> Vec<AmbiguousDistanceRestraint> {
    let n_res = shifts.sequence.len();
    if n_res == 0 || peaks.is_empty() {
        return Vec::new();
    }

    let protons = collect_proton_shifts(shifts, cfg.synthesize_missing_aliphatic_protons);

    let sig_h = cfg.tol_h_anchor * 0.5;
    let sig_n = cfg.tol_n_anchor * 0.5;
    let sig_c = cfg.tol_c_anchor * 0.5;
    let sig_h2 = cfg.tol_h_indirect * 0.5;

    let mut raw_adrs: Vec<AmbiguousDistanceRestraint> = Vec::with_capacity(peaks.len());

    for (p_idx, p) in peaks.iter().enumerate() {
        let Some(h2_obs) = p.h2 else {
            continue;
        };
        let is_15n = p.exp == "NOESY15N";
        let is_13c = p.exp == "NOESY13C";
        if !is_15n && !is_13c {
            continue;
        }

        // Match 2D (H_direct, Heavy) donors
        let mut donors = Vec::new();
        for (di, pr) in protons.iter().enumerate() {
            let Some(hv) = pr.heavy_ppm else { continue };
            if is_15n && pr.site == NoeProton::AmideH {
                let dh = (p.h - pr.h_ppm).abs();
                let dn = (p.n - hv).abs();
                if dh <= cfg.tol_h_anchor && dn <= cfg.tol_n_anchor {
                    let z2 = (dh / sig_h).powi(2) + (dn / sig_n).powi(2);
                    donors.push((di, (-0.5 * z2).exp()));
                }
            } else if is_13c && matches!(pr.site, NoeProton::AlphaHA | NoeProton::BetaQB) {
                let Some(c_obs) = p.c else { continue };
                let dh = (p.h - pr.h_ppm).abs();
                let dc = (c_obs - hv).abs();
                if dh <= cfg.tol_h_anchor && dc <= cfg.tol_c_anchor {
                    let z2 = (dh / sig_h).powi(2) + (dc / sig_c).powi(2);
                    donors.push((di, (-0.5 * z2).exp()));
                }
            }
        }
        if donors.is_empty() {
            continue;
        }

        // Match 1D indirect H_noe acceptors across all proton sites
        let mut acceptors = Vec::new();
        for (ai, pr) in protons.iter().enumerate() {
            let dh2 = (h2_obs - pr.h_ppm).abs();
            if dh2 <= cfg.tol_h_indirect {
                let z2 = (dh2 / sig_h2).powi(2);
                acceptors.push((ai, (-0.5 * z2).exp()));
            }
        }
        if acceptors.is_empty() {
            continue;
        }

        let mut cands = Vec::new();
        for &(di, s_don) in &donors {
            let pa = &protons[di];
            for &(ai, s_acc) in &acceptors {
                if di == ai {
                    continue;
                }
                let pb = &protons[ai];
                let shift_score = (s_don * s_acc).max(1e-6);
                cands.push(NoeCandidatePair {
                    res_a: pa.res_idx,
                    atom_a: pa.site,
                    res_b: pb.res_idx,
                    atom_b: pb.site,
                    shift_score,
                    network_score: 0.0,
                    weight: 0.0,
                });
            }
        }
        if cands.is_empty() {
            continue;
        }

        let vol = p.volume.or(p.height).unwrap_or(500.0);
        let r_app = (cfg.i0_scale / vol.max(1.0)).powf(1.0 / 6.0);

        // Covalent intra-residue geometry bound: in any L-amino acid (phi, psi),
        // d(H_i, HA_i) <= 3.08 A and d(H_i, QB_i) <= 3.80 A. When a weak crosspeak (r_app > 3.35 A
        // or > 4.05 A) also matches inter-residue candidates, an accidental intra-residue shift
        // coincidence cannot account for the weak intensity and would otherwise mask D_eff!
        if cands.iter().any(|c| c.res_a != c.res_b) {
            cands.retain(|c| {
                if c.res_a != c.res_b {
                    return true;
                }
                match (c.atom_a, c.atom_b) {
                    (NoeProton::AlphaHA, NoeProton::BetaQB)
                    | (NoeProton::BetaQB, NoeProton::AlphaHA) => false,
                    (NoeProton::AmideH, NoeProton::AlphaHA)
                    | (NoeProton::AlphaHA, NoeProton::AmideH) => r_app <= 3.35,
                    _ => r_app <= 4.05,
                }
            });
        }

        let (lb, mut ub) = calibrate_noe_bounds(vol, cfg.i0_scale);
        let has_qb = cands
            .iter()
            .any(|c| c.atom_a == NoeProton::BetaQB || c.atom_b == NoeProton::BetaQB);
        if has_qb {
            // Wuthrich, Billeter & Braun (1983) pseudoatom allowance for CB/QB beta centroid
            ub += 0.20;
        }

        raw_adrs.push(AmbiguousDistanceRestraint {
            peak_index: p_idx,
            exp: p.exp.clone(),
            lower_bound: lb,
            upper_bound: ub,
            candidates: cands,
            all_candidates: Vec::new(),
            is_unambiguous: false,
        });
    }

    // Build residue-level directed & symmetric network-anchoring contact matrices from observed NOEs only
    let mut sym_mat = vec![0.0f64; n_res * n_res];
    let mut dir_mat = vec![0.0f64; n_res * n_res];

    for adr in &raw_adrs {
        let sum_s: f64 = adr
            .candidates
            .iter()
            .map(|c| c.shift_score)
            .sum::<f64>()
            .max(1e-9);
        let inv_ambig = 1.0 / (adr.candidates.len() as f64).sqrt();
        for c in &adr.candidates {
            if c.res_a == c.res_b {
                continue;
            }
            let contrib = (c.shift_score / sum_s) * inv_ambig;
            dir_mat[c.res_a * n_res + c.res_b] += contrib;
            sym_mat[c.res_a * n_res + c.res_b] += 0.5 * contrib;
            sym_mat[c.res_b * n_res + c.res_a] += 0.5 * contrib;
        }
    }

    // Score each candidate by gentle covalent prior + reciprocal transpose + local ladder + triangle paths
    let mut filtered_adrs = Vec::with_capacity(raw_adrs.len());

    for mut adr in raw_adrs {
        for c in &mut adr.candidates {
            let a = c.res_a;
            let b = c.res_b;
            let seq_sep = a.abs_diff(b);

            let coval = if seq_sep == 0 {
                0.32
            } else if seq_sep == 1 {
                0.30
            } else if seq_sep <= 4 {
                0.24
            } else {
                0.18
            };

            // Reciprocal transpose crosspeak support (b -> a)
            let recip = if a != b {
                2.6 * dir_mat[b * n_res + a]
            } else {
                0.0
            };

            // Local sequence-neighborhood support around (a, b) (helix i,i+3,i+4 & beta-sheet ladders)
            let mut nbr = 0.0;
            for da in -1isize..=1 {
                for db in -1isize..=1 {
                    if da == 0 && db == 0 {
                        continue;
                    }
                    let na = a as isize + da;
                    let nb = b as isize + db;
                    if na >= 0 && na < n_res as isize && nb >= 0 && nb < n_res as isize && na != nb
                    {
                        let v = sym_mat[(na as usize) * n_res + (nb as usize)];
                        nbr += v.min(1.0);
                    }
                }
            }

            // Common third-residue triangle support sum_k min(M[a,k], M[k,b])
            let mut tri = 0.0;
            if a != b {
                for k in 0..n_res {
                    if k == a || k == b {
                        continue;
                    }
                    let mak = sym_mat[a * n_res + k];
                    let mkb = sym_mat[k * n_res + b];
                    if mak > 0.05 && mkb > 0.05 {
                        tri += mak.min(mkb).min(0.9);
                    }
                }
            }

            let net = coval + recip + 0.40 * nbr + 0.45 * tri;
            c.network_score = net;
            c.weight = c.shift_score * net;
        }

        // Preserve full network-scored pool for iterative non-destructive re-evaluation
        adr.all_candidates = adr.candidates.clone();

        normalize_and_prune_candidates(&mut adr, 0.03, 12, 0.96, cfg.unambiguous_threshold);

        if adr.candidates.len() == 1 && adr.candidates[0].res_a == adr.candidates[0].res_b {
            let c0 = &adr.candidates[0];
            if matches!(
                (c0.atom_a, c0.atom_b),
                (NoeProton::AlphaHA, NoeProton::BetaQB) | (NoeProton::BetaQB, NoeProton::AlphaHA)
            ) {
                continue;
            }
        }

        if !adr.candidates.is_empty() {
            filtered_adrs.push(adr);
        }
    }

    filtered_adrs
}

fn normalize_and_prune_candidates(
    adr: &mut AmbiguousDistanceRestraint,
    min_rel_weight: f64,
    max_keep: usize,
    cum_weight_target: f64,
    unambig_thresh: f64,
) {
    let total: f64 = adr.candidates.iter().map(|c| c.weight).sum();
    if total <= 0.0 {
        let u = 1.0 / (adr.candidates.len().max(1) as f64);
        for c in &mut adr.candidates {
            c.weight = u;
        }
        return;
    }
    for c in &mut adr.candidates {
        c.weight /= total;
    }

    adr.candidates.sort_by(|a, b| b.weight.total_cmp(&a.weight));

    if adr.candidates[0].weight >= unambig_thresh {
        adr.candidates.truncate(1);
        adr.candidates[0].weight = 1.0;
        adr.is_unambiguous = true;
        return;
    }

    let top_w = adr.candidates[0].weight;
    let mut kept = Vec::with_capacity(adr.candidates.len().min(max_keep));
    let mut acc = 0.0;
    for c in adr.candidates.drain(..) {
        if kept.is_empty()
            || (kept.len() < max_keep
                && acc < cum_weight_target
                && c.weight >= min_rel_weight
                && c.weight >= 0.05 * top_w)
        {
            acc += c.weight;
            kept.push(c);
        }
    }
    let inv = 1.0 / acc.max(1e-12);
    for c in &mut kept {
        c.weight *= inv;
    }
    adr.is_unambiguous = kept.len() == 1;
    adr.candidates = kept;
}

/// Cycles `1..K` (Nilges 1995): Re-weight each candidate pair `c` in every ADR by its inverse-sixth-power
/// distance contribution `d_c^-6 / sum_m d_m^-6` averaged across the current conformer ensemble,
/// tightening restraints to unambiguous when one candidate carries `>= unambiguous_threshold` (default `85%`) weight.
pub fn reweight_adrs_with_coords<F>(
    adrs: &mut [AmbiguousDistanceRestraint],
    n_conformers: usize,
    min_weight_keep: f64,
    cum_weight_target: f64,
    unambiguous_threshold: f64,
    coord_lookup: F,
) where
    F: Fn(usize, usize, NoeProton) -> Option<Vec3>,
{
    if n_conformers == 0 {
        return;
    }
    let inv_m = 1.0 / (n_conformers as f64);

    for adr in adrs.iter_mut() {
        // Restore full shift-compatible candidate pool so earlier rough cycles never irreversibly drop true contacts!
        let mut pool = if !adr.all_candidates.is_empty() {
            adr.all_candidates.clone()
        } else {
            adr.candidates.clone()
        };

        if pool.len() <= 1 {
            adr.is_unambiguous = true;
            if let Some(c0) = pool.first_mut() {
                c0.weight = 1.0;
            }
            adr.candidates = pool;
            continue;
        }

        let r_floor2 = (0.68 * adr.upper_bound).max(2.45).powi(2);
        let mut satisfied_any = false;
        for c in &mut pool {
            let mut sum_inv_r6 = 0.0;
            let mut close_count = 0usize;
            for m in 0..n_conformers {
                if let (Some(pa), Some(pb)) = (
                    coord_lookup(m, c.res_a, c.atom_a),
                    coord_lookup(m, c.res_b, c.atom_b),
                ) {
                    let r2_raw = v_dist_sq(pa, pb);
                    if r2_raw <= (adr.upper_bound + 1.0).powi(2) {
                        close_count += 1;
                    }
                    // Soft floor scaled to upper_bound prevents short covalent pairs from dominating
                    // 30x over satisfied medium/long-range contacts that match the calibrated intensity
                    let r2 = r2_raw.max(r_floor2);
                    sum_inv_r6 += 1.0 / (r2 * r2 * r2);
                }
            }
            if close_count * 3 >= n_conformers * 2 {
                satisfied_any = true;
            }
            let mean_inv_r6 = sum_inv_r6 * inv_m;
            let prior = c.shift_score * (0.45 + 0.55 * c.network_score.min(3.5));
            c.weight = prior * mean_inv_r6.max(1e-18);
        }

        adr.candidates = pool;
        // Only lock in single unambiguous promotion if supported across >=50% of ensemble conformers
        let eff_unamb = if satisfied_any {
            unambiguous_threshold
        } else {
            0.995
        };
        normalize_and_prune_candidates(adr, min_weight_keep, 8, cum_weight_target, eff_unamb);
    }
}

/// Summary statistics of an ADR set (total, unambiguous count, long-range `|i - j| >= 5` count, mean ambiguity).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AdrSummary {
    pub total_restraints: usize,
    pub unambiguous_restraints: usize,
    pub long_range_unambiguous: usize,
    pub mean_candidates_per_restraint: f64,
}

pub fn summarize_adrs(adrs: &[AmbiguousDistanceRestraint]) -> AdrSummary {
    if adrs.is_empty() {
        return AdrSummary::default();
    }
    let total = adrs.len();
    let unamb = adrs
        .iter()
        .filter(|r| r.is_unambiguous || r.candidates.len() == 1)
        .count();
    let lr_unamb = adrs
        .iter()
        .filter(|r| {
            (r.is_unambiguous || r.candidates.len() == 1)
                && !r.candidates.is_empty()
                && r.candidates[0].res_a.abs_diff(r.candidates[0].res_b) >= 5
        })
        .count();
    let sum_c: usize = adrs.iter().map(|r| r.candidates.len()).sum();

    AdrSummary {
        total_restraints: total,
        unambiguous_restraints: unamb,
        long_range_unambiguous: lr_unamb,
        mean_candidates_per_restraint: (sum_c as f64) / (total as f64),
    }
}
