//! 2D Structure Coordinate Generation (`clean_2d`), Ring Fusion, Spiro & Bridged Systems,
//! Organometallic η-Hapticity Layout, Wedge/Hash Stereobond Synthesis, and VSEPR / Distance-Geometry
//! 3D Builder (`clean_3d`).
//!
//! Ported/adapted from Octet-Draw by Sam Ellis (https://github.com/SamChemMan/Octet-Draw, Apache-2.0:
//! `clean2d.ts`, `fusion.ts`, `complexes.ts`, `metals.ts`, `clean3d.ts`) and ACS Document 1996
//! 2D structure layout conventions.

use super::graph::{
    covalent_radius, is_metal, BondKind, DoubleBondGeometry, Molecule, TetrahedralParity,
};
use super::stereo::{perceive_aromaticity, perceive_cip, perceive_sssr, permutation_is_odd};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::f64::consts::PI;

/// Standard C–C single bond length in Ångströms used across `Molecule` 2D coordinates (MOL/SDF native scale).
pub const ACS_BOND_LENGTH_ANGSTROM: f64 = 1.54;

/// Standard ACS Document 1996 bond length in typographic points (`14.4 pt`), used in CDXML/CDX serialization.
pub const ACS_BOND_LENGTH_PT: f64 = 14.4;

/// Conversion factor from Ångströms to ChemDraw / ACS points (`14.4 / 1.54`).
pub const ANGSTROM_TO_PT: f64 = ACS_BOND_LENGTH_PT / ACS_BOND_LENGTH_ANGSTROM;

/// Generate clean 2D coordinates (`1.54` Å standard bond length, 120° zig-zag chains, regular polygons,
/// fused/spiro/bridged rings, organometallic complexes) while preserving `(R)/(S)` and `(E)/(Z)` stereochemistry.
pub fn clean_2d(mol: &mut Molecule) {
    clean_2d_with_bond_length(mol, ACS_BOND_LENGTH_ANGSTROM);
}

/// Generate clean 2D coordinates with an explicit target `bond_length` (e.g. `1.54` Å or `14.4` pt).
pub fn clean_2d_with_bond_length(mol: &mut Molecule, bond_length: f64) {
    let n = mol.atoms.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        mol.atoms[0].x = 0.0;
        mol.atoms[0].y = 0.0;
        mol.atoms[0].z_coord = None;
        return;
    }

    // 1. Capture existing stereochemical configuration BEFORE touching coordinates!
    // Capture CIP (E)/(Z) onto DoubleBondGeometry::Cis/Trans if currently Auto and coordinates exist
    let had_2d = !mol.needs_2d_coords();
    if had_2d {
        let cip_before = perceive_cip(mol);
        // Save any wedge-derived tetrahedral pyramids into `atom.parity` so re-layout preserves them!
        save_current_pyramids_to_parity(mol);
        for (&bi, &desc) in &cip_before.bonds {
            if mol.bonds[bi].double_geometry == DoubleBondGeometry::Auto {
                let a = mol.bonds[bi].a;
                let b = mol.bonds[bi].b;
                let sub_a = mol
                    .covalent_neighbors(a)
                    .into_iter()
                    .map(|(nb, _)| nb)
                    .find(|&nb| nb != b);
                let sub_b = mol
                    .covalent_neighbors(b)
                    .into_iter()
                    .map(|(nb, _)| nb)
                    .find(|&nb| nb != a);
                if let (Some(ra), Some(rb)) = (sub_a, sub_b) {
                    // Check if ra & rb were on same side or opposite side in original drawing
                    let ax = mol.atoms[a].x;
                    let ay = mol.atoms[a].y;
                    let bx = mol.atoms[b].x;
                    let by = mol.atoms[b].y;
                    let nx = -(by - ay);
                    let ny = bx - ax;
                    let pa = (mol.atoms[ra].x - ax) * nx + (mol.atoms[ra].y - ay) * ny;
                    let pb = (mol.atoms[rb].x - bx) * nx + (mol.atoms[rb].y - by) * ny;
                    let _ = desc;
                    mol.bonds[bi].double_geometry = if pa * pb > 0.0 {
                        DoubleBondGeometry::Cis {
                            ref_a: ra,
                            ref_b: rb,
                        }
                    } else {
                        DoubleBondGeometry::Trans {
                            ref_a: ra,
                            ref_b: rb,
                        }
                    };
                }
            }
        }
    }

    // Clear z_coord for pure 2D layout
    for a in &mut mol.atoms {
        a.z_coord = None;
    }

    // 2. Lay out each connected component independently, then tile left-to-right
    let comps = mol.connected_components();
    let mut cursor_x = 0.0;

    for comp in &comps {
        layout_component_2d(mol, comp, bond_length);
        // Shift component bounding box to `[cursor_x, ...]` centered vertically around y = 0
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for &u in comp {
            min_x = min_x.min(mol.atoms[u].x);
            max_x = max_x.max(mol.atoms[u].x);
            min_y = min_y.min(mol.atoms[u].y);
            max_y = max_y.max(mol.atoms[u].y);
        }
        let mid_y = 0.5 * (min_y + max_y);
        let shift_x = cursor_x - min_x;
        for &u in comp {
            mol.atoms[u].x += shift_x;
            mol.atoms[u].y -= mid_y;
        }
        cursor_x += (max_x - min_x) + 2.5 * bond_length;
    }

    // 3. Reconstruct clean 2D wedge/hash stereo bonds from stored `TetrahedralParity`!
    assign_wedges_from_parity(mol);
}

fn save_current_pyramids_to_parity(mol: &mut Molecule) {
    // Re-use perceive_cip's internal pyramid perception indirectly by checking atoms with stereo bonds or 3D coords
    let n = mol.atoms.len();
    for i in 0..n {
        if mol.atoms[i].parity.is_some() {
            continue;
        }
        // Only record if atom currently has a stereo bond or 3D coordinates
        let has_stereo_bond = mol
            .bonds
            .iter()
            .any(|b| (b.a == i || b.b == i) && b.kind.is_stereo_display());
        let has_3d = mol.atoms[i].z_coord.is_some();
        if !has_stereo_bond && !has_3d {
            continue;
        }
        // Build temporary CipGraph through perceive_cip? Even simpler: call our helper via a minimal check
        let cip = perceive_cip(mol);
        if cip.atoms.contains_key(&i) {
            // Reconstruct R-sense pyramid around `i` by testing candidate parity clockwise=true vs false!
            let nbs: Vec<usize> = mol
                .covalent_neighbors(i)
                .into_iter()
                .map(|(nb, _)| nb)
                .collect();
            if nbs.len() == 3 || nbs.len() == 4 {
                let target = cip.atoms[&i];
                let arr = if nbs.len() == 4 {
                    [nbs[0], nbs[1], nbs[2], nbs[3]]
                } else {
                    [nbs[0], nbs[1], nbs[2], usize::MAX]
                };
                // Test clockwise = true on clone without wedges
                let mut probe = mol.clone();
                for b in &mut probe.bonds {
                    if (b.a == i || b.b == i) && b.kind.is_stereo_display() {
                        b.kind = BondKind::Single;
                    }
                }
                for a in &mut probe.atoms {
                    a.z_coord = None;
                }
                probe.atoms[i].parity = Some(TetrahedralParity {
                    neighbors: arr,
                    clockwise: true,
                });
                let test_cip = perceive_cip(&probe);
                let cw = test_cip.atoms.get(&i) == Some(&target);
                mol.atoms[i].parity = Some(TetrahedralParity {
                    neighbors: arr,
                    clockwise: cw,
                });
            }
        }
    }
}

fn layout_component_2d(mol: &mut Molecule, comp: &[usize], bl: f64) {
    if comp.len() == 1 {
        let u = comp[0];
        mol.atoms[u].x = 0.0;
        mol.atoms[u].y = 0.0;
        return;
    }

    let n_total = mol.atoms.len();
    let mut placed = vec![false; n_total];

    // Perceived SSSR rings within this component
    let comp_set: BTreeSet<usize> = comp.iter().copied().collect();
    let all_rings: Vec<Vec<usize>> = perceive_sssr(mol)
        .into_iter()
        .filter(|r| r.iter().all(|u| comp_set.contains(u)))
        .collect();

    // Group SSSR rings into connected ring systems (sharing >= 1 atom)
    let mut ring_used = vec![false; all_rings.len()];
    let mut ring_systems: Vec<Vec<Vec<usize>>> = Vec::new();
    for i in 0..all_rings.len() {
        if ring_used[i] {
            continue;
        }
        let mut sys = vec![all_rings[i].clone()];
        let mut sys_atoms: BTreeSet<usize> = all_rings[i].iter().copied().collect();
        ring_used[i] = true;
        let mut changed = true;
        while changed {
            changed = false;
            for j in 0..all_rings.len() {
                if !ring_used[j] && all_rings[j].iter().any(|u| sys_atoms.contains(u)) {
                    for &u in &all_rings[j] {
                        sys_atoms.insert(u);
                    }
                    sys.push(all_rings[j].clone());
                    ring_used[j] = true;
                    changed = true;
                }
            }
        }
        ring_systems.push(sys);
    }

    // Check for organometallic sandwich / η-hapticity center first (e.g. ferrocene)
    if let Some(metal_idx) = comp.iter().copied().find(|&u| is_metal(mol.atoms[u].z)) {
        if layout_organometallic_seed(mol, comp, metal_idx, &ring_systems, &mut placed, bl) {
            grow_remaining_bfs(mol, comp, &all_rings, &ring_systems, &mut placed, bl);
            polish_2d_coords(mol, comp, &all_rings, bl);
            return;
        }
    }

    // Seed placement with the largest ring system if one exists; otherwise seed with the longest chain's start!
    if let Some((best_sys_idx, _)) = ring_systems
        .iter()
        .enumerate()
        .max_by_key(|(_, sys)| sys.iter().map(|r| r.len()).sum::<usize>())
    {
        place_ring_system(mol, &ring_systems[best_sys_idx], None, &mut placed, bl);
    } else {
        // Purely acyclic component: find longest chain via two BFS passes so main backbone runs cleanly left-to-right!
        let backbone = find_longest_chain(mol, comp);
        place_acyclic_backbone(mol, &backbone, &mut placed, bl);
    }

    // Grow all remaining branches and secondary ring systems outward via BFS
    grow_remaining_bfs(mol, comp, &all_rings, &ring_systems, &mut placed, bl);

    // Resolve steric clashes on rotatable branches & polish strained ring/bridged bonds
    polish_2d_coords(mol, comp, &all_rings, bl);
}

fn layout_organometallic_seed(
    mol: &mut Molecule,
    comp: &[usize],
    metal: usize,
    ring_systems: &[Vec<Vec<usize>>],
    placed: &mut [bool],
    bl: f64,
) -> bool {
    let _ = comp;
    // Check if metal connects to multi-center dummy centroids (`z == 0 && !multi_center_atoms.is_empty()`)
    let mut mc_dummies = Vec::new();
    for (nb, _) in mol.neighbors(metal) {
        if !mol.atoms[nb].multi_center_atoms.is_empty() {
            mc_dummies.push(nb);
        }
    }
    if !mc_dummies.is_empty() {
        mol.atoms[metal].x = 0.0;
        mol.atoms[metal].y = 0.0;
        placed[metal] = true;
        for (idx, &dummy) in mc_dummies.iter().enumerate() {
            let sign = if idx % 2 == 0 { 1.0 } else { -1.0 };
            let cy = sign * 1.15 * bl;
            mol.atoms[dummy].x = 0.0;
            mol.atoms[dummy].y = cy;
            placed[dummy] = true;
            let ring_atoms = mol.atoms[dummy].multi_center_atoms.clone();
            let k = ring_atoms.len();
            if k >= 3 {
                let rx = bl * 0.85;
                let ry = bl * 0.42;
                for (j, &ru) in ring_atoms.iter().enumerate() {
                    let ang = 2.0 * PI * (j as f64) / (k as f64) - PI / 2.0;
                    mol.atoms[ru].x = rx * ang.cos();
                    mol.atoms[ru].y = cy + ry * ang.sin();
                    placed[ru] = true;
                }
            }
        }
        return true;
    }

    // Standard coordination complex around metal (e.g. cisplatin, M(CO)6, M(L)4) if metal has >= 3 ligands and is not inside a small organic ring
    let metal_in_ring = ring_systems
        .iter()
        .any(|sys| sys.iter().any(|r| r.contains(&metal)));
    let nbs: Vec<usize> = mol.neighbors(metal).into_iter().map(|(nb, _)| nb).collect();
    if !metal_in_ring && nbs.len() >= 3 {
        mol.atoms[metal].x = 0.0;
        mol.atoms[metal].y = 0.0;
        placed[metal] = true;
        let deg = nbs.len();
        for (k, &nb) in nbs.iter().enumerate() {
            // Square planar (4 -> 45, 135, 225, 315 deg), octahedral (6 -> 0, 60, 120, 180, 240, 300 or 90/270 + X)
            let ang = if deg == 4 {
                PI / 4.0 + (k as f64) * (PI / 2.0)
            } else {
                (k as f64) * (2.0 * PI / (deg as f64))
            };
            mol.atoms[nb].x = bl * 1.15 * ang.cos();
            mol.atoms[nb].y = bl * 1.15 * ang.sin();
            placed[nb] = true;
        }
        return true;
    }

    false
}

/// Place a connected ring system (monocycle, ortho/peri-fused polycycle, spiro, or bridged bicycle).
fn place_ring_system(
    mol: &mut Molecule,
    sys: &[Vec<usize>],
    anchor: Option<(usize, [f64; 2], [f64; 2])>, // (anchor_ring_atom, pos, outward_dir_from_parent)
    placed: &mut [bool],
    bl: f64,
) {
    if sys.is_empty() {
        return;
    }

    // Check if this is a bridged bicyclic system (two SSSR rings sharing >= 2 edges!)
    if sys.len() == 2
        && anchor.is_none()
        && try_place_bridged_bicycle(mol, &sys[0], &sys[1], placed, bl)
    {
        return;
    }

    // Pick initial ring: if anchored, pick ring containing anchor atom; else prefer 6-ring or largest ring
    let first_idx = if let Some((anc_u, _, _)) = anchor {
        sys.iter().position(|r| r.contains(&anc_u)).unwrap_or(0)
    } else {
        sys.iter()
            .enumerate()
            .max_by_key(|(_, r)| if r.len() == 6 { 100 } else { r.len() })
            .map(|(i, _)| i)
            .unwrap_or(0)
    };

    let mut ring_done = vec![false; sys.len()];
    let r0 = &sys[first_idx];
    let k0 = r0.len();
    let radius0 = bl / (2.0 * (PI / (k0 as f64)).sin());

    if let Some((anc_u, anc_pos, dir)) = anchor {
        let pos_in_r = r0.iter().position(|&u| u == anc_u).unwrap_or(0);
        // Center of regular polygon lies at `anc_pos + radius0 * dir`
        let cx = anc_pos[0] + radius0 * dir[0];
        let cy = anc_pos[1] + radius0 * dir[1];
        // Angle from center to `anc_u` is `atan2(-dir[1], -dir[0])`
        let base_ang = (-dir[1]).atan2(-dir[0]);
        for step in 0..k0 {
            let u = r0[(pos_in_r + step) % k0];
            let ang = base_ang + 2.0 * PI * (step as f64) / (k0 as f64);
            mol.atoms[u].x = cx + radius0 * ang.cos();
            mol.atoms[u].y = cy + radius0 * ang.sin();
            placed[u] = true;
        }
    } else {
        // Place first ring centered at origin, oriented cleanly (vertical sides for 6-ring: first vertex at 90° top)
        let start_ang = if k0 == 6 || k0 == 5 {
            PI / 2.0
        } else {
            PI / 2.0 - PI / (k0 as f64)
        };
        for (step, &u) in r0.iter().enumerate() {
            let ang = start_ang - 2.0 * PI * (step as f64) / (k0 as f64);
            mol.atoms[u].x = radius0 * ang.cos();
            mol.atoms[u].y = radius0 * ang.sin();
            placed[u] = true;
        }
    }
    ring_done[first_idx] = true;

    // Iteratively place remaining fused / spiro / bridged rings in `sys`
    for _pass in 0..sys.len() {
        // Prefer rings sharing >= 2 placed atoms (fused/bridged) before spiro (1 placed atom)
        let mut best_cand: Option<(usize, usize)> = None;
        for (ri, r) in sys.iter().enumerate() {
            if ring_done[ri] {
                continue;
            }
            let shared_cnt = r.iter().filter(|&&u| placed[u]).count();
            if shared_cnt > 0 && best_cand.is_none_or(|(_, best_s)| shared_cnt > best_s) {
                best_cand = Some((ri, shared_cnt));
            }
        }
        let Some((ri, shared_cnt)) = best_cand else {
            break;
        };
        let r = &sys[ri];
        let kr = r.len();

        if shared_cnt == 1 {
            // Spiro ring! Shared single atom `s_atom`
            let s_pos = r.iter().position(|&u| placed[u]).unwrap();
            let s_atom = r[s_pos];
            // Compute average vector of already-placed neighbors of `s_atom`, then point opposite!
            let mut vx = 0.0;
            let mut vy = 0.0;
            for (nb, _) in mol.covalent_neighbors(s_atom) {
                if placed[nb] {
                    vx += mol.atoms[nb].x - mol.atoms[s_atom].x;
                    vy += mol.atoms[nb].y - mol.atoms[s_atom].y;
                }
            }
            let vlen = vx.hypot(vy);
            let out_dir = if vlen > 1e-5 {
                [-vx / vlen, -vy / vlen]
            } else {
                [1.0, 0.0]
            };
            let rad = bl / (2.0 * (PI / (kr as f64)).sin());
            let cx = mol.atoms[s_atom].x + rad * out_dir[0];
            let cy = mol.atoms[s_atom].y + rad * out_dir[1];
            let base_ang = (-out_dir[1]).atan2(-out_dir[0]);
            for step in 1..kr {
                let u = r[(s_pos + step) % kr];
                if !placed[u] {
                    let ang = base_ang + 2.0 * PI * (step as f64) / (kr as f64);
                    mol.atoms[u].x = cx + rad * ang.cos();
                    mol.atoms[u].y = cy + rad * ang.sin();
                    placed[u] = true;
                }
            }
        } else {
            // Ortho-fused (2 adjacent placed atoms) or peri-fused / bridged (>= 2 placed atoms):
            // Find a contiguous run of unplaced atoms in `r` bounded by placed atoms `u_start` and `u_end`!
            let mut start_idx = None;
            for idx in 0..kr {
                if placed[r[idx]] && !placed[r[(idx + 1) % kr]] {
                    start_idx = Some(idx);
                    break;
                }
            }
            if let Some(s_idx) = start_idx {
                // Collect unplaced chain from `s_idx + 1` up to next placed atom `e_idx`
                let mut unplaced_chain = Vec::new();
                let mut cur = (s_idx + 1) % kr;
                while !placed[r[cur]] {
                    unplaced_chain.push(r[cur]);
                    cur = (cur + 1) % kr;
                }
                let u_start = r[s_idx];
                let u_end = r[cur];
                let p_start = [mol.atoms[u_start].x, mol.atoms[u_start].y];
                let p_end = [mol.atoms[u_end].x, mol.atoms[u_end].y];

                // Centroid of already-placed atoms in this ring system to point the new fused ring AWAY from existing rings
                let mut old_cx = 0.0;
                let mut old_cy = 0.0;
                let mut old_cnt = 0.0;
                for prev_r_idx in 0..sys.len() {
                    if ring_done[prev_r_idx] {
                        for &pu in &sys[prev_r_idx] {
                            old_cx += mol.atoms[pu].x;
                            old_cy += mol.atoms[pu].y;
                            old_cnt += 1.0;
                        }
                    }
                }
                if old_cnt > 0.0 {
                    old_cx /= old_cnt;
                    old_cy /= old_cnt;
                }

                let dx = p_end[0] - p_start[0];
                let dy = p_end[1] - p_start[1];
                let chord = dx.hypot(dy).max(1e-5);
                let mid = [0.5 * (p_start[0] + p_end[0]), 0.5 * (p_start[1] + p_end[1])];
                // Two candidate normals to `p_start -> p_end`
                let n1 = [-dy / chord, dx / chord];
                let n2 = [dy / chord, -dx / chord];
                // Pick the normal pointing AWAY from `(old_cx, old_cy)`
                let away = [mid[0] - old_cx, mid[1] - old_cy];
                let out_n = if n1[0] * away[0] + n1[1] * away[1] >= 0.0 {
                    n1
                } else {
                    n2
                };

                if shared_cnt == 2 && unplaced_chain.len() == kr - 2 {
                    // Exact ortho-fusion across shared bond `(u_start, u_end)`!
                    // Apothem `h = (chord / 2) / tan(PI / kr)`
                    let apothem = (0.5 * chord) / (PI / (kr as f64)).tan();
                    let cx = mid[0] + apothem * out_n[0];
                    let cy = mid[1] + apothem * out_n[1];
                    let rad = (0.5 * chord) / (PI / (kr as f64)).sin();
                    let a_start = (p_start[1] - cy).atan2(p_start[0] - cx);
                    let a_end = (p_end[1] - cy).atan2(p_end[0] - cx);
                    // Determine step sign from `u_start` -> `unplaced_chain` -> `u_end` going the LONG way `(kr - 1)` steps around circle!
                    // Short step from `u_start` to `u_end` is `± 2*PI/kr`; long arc goes the OPPOSITE angular direction!
                    let mut diff = a_end - a_start;
                    while diff > PI {
                        diff -= 2.0 * PI;
                    }
                    while diff < -PI {
                        diff += 2.0 * PI;
                    }
                    let step_sign = if diff > 0.0 { -1.0 } else { 1.0 };
                    for (idx_c, &u_new) in unplaced_chain.iter().enumerate() {
                        let ang =
                            a_start + step_sign * 2.0 * PI * ((idx_c + 1) as f64) / (kr as f64);
                        mol.atoms[u_new].x = cx + rad * ang.cos();
                        mol.atoms[u_new].y = cy + rad * ang.sin();
                        placed[u_new] = true;
                    }
                } else {
                    // Peri-fusion or multi-atom bridge: place `unplaced_chain` on a clean circular arc from `p_start` to `p_end` bulging along `out_n`!
                    let m_segs = (unplaced_chain.len() + 1) as f64;
                    let arch_height = bl * 0.75 * (unplaced_chain.len() as f64).sqrt();
                    for (idx_c, &u_new) in unplaced_chain.iter().enumerate() {
                        let t = ((idx_c + 1) as f64) / m_segs;
                        let bulge = (PI * t).sin() * arch_height;
                        mol.atoms[u_new].x = p_start[0] + t * dx + bulge * out_n[0];
                        mol.atoms[u_new].y = p_start[1] + t * dy + bulge * out_n[1];
                        placed[u_new] = true;
                    }
                }
            }
        }

        ring_done[ri] = true;
    }
}

fn try_place_bridged_bicycle(
    mol: &mut Molecule,
    r1: &[usize],
    r2: &[usize],
    placed: &mut [bool],
    bl: f64,
) -> bool {
    let mut atoms_set: BTreeSet<usize> = r1.iter().copied().collect();
    for &u in r2 {
        atoms_set.insert(u);
    }
    let mut bonds_in = 0usize;
    for b in &mol.bonds {
        if !matches!(
            b.kind,
            BondKind::Dative | BondKind::Hydrogen | BondKind::Ionic
        ) && atoms_set.contains(&b.a)
            && atoms_set.contains(&b.b)
        {
            bonds_in += 1;
        }
    }
    if bonds_in != atoms_set.len() + 1 {
        return false;
    }
    let heads: Vec<usize> = atoms_set
        .iter()
        .copied()
        .filter(|&u| {
            mol.covalent_neighbors(u)
                .into_iter()
                .filter(|&(v, _)| atoms_set.contains(&v))
                .count()
                == 3
        })
        .collect();
    if heads.len() != 2 {
        return false;
    }
    let h1 = heads[0];
    let h2 = heads[1];
    let starts: Vec<usize> = mol
        .covalent_neighbors(h1)
        .into_iter()
        .map(|(v, _)| v)
        .filter(|v| atoms_set.contains(v))
        .collect();
    if starts.len() != 3 {
        return false;
    }
    let mut bridges = Vec::with_capacity(3);
    for s in starts {
        let mut path = Vec::new();
        let mut prev = h1;
        let mut cur = s;
        while cur != h2 {
            path.push(cur);
            let Some((nxt, _)) = mol
                .covalent_neighbors(cur)
                .into_iter()
                .find(|&(v, _)| atoms_set.contains(&v) && v != prev)
            else {
                return false;
            };
            prev = cur;
            cur = nxt;
        }
        bridges.push(path);
    }
    bridges.sort_by_key(|b| std::cmp::Reverse(b.len()));
    if bridges[2].is_empty() {
        return false; // simple ortho-fused, not bridged
    }

    // Classic bridged bicyclic 2D projection: bridgeheads left/right, x-bridge top arch, y-bridge bottom arch, z-bridge offset middle!
    let half_span = bl * 1.10;
    mol.atoms[h1].x = -half_span;
    mol.atoms[h1].y = 0.0;
    placed[h1] = true;
    mol.atoms[h2].x = half_span;
    mol.atoms[h2].y = 0.0;
    placed[h2] = true;

    let arches = [bl * 0.92, -bl * 0.92, bl * 0.28];
    for (bi, path) in bridges.iter().enumerate() {
        let m = (path.len() + 1) as f64;
        for (step, &u) in path.iter().enumerate() {
            let t = ((step + 1) as f64) / m;
            let x = -half_span + 2.0 * half_span * t + if bi == 2 { bl * 0.15 } else { 0.0 };
            let y = arches[bi] * (PI * t).sin();
            mol.atoms[u].x = x;
            mol.atoms[u].y = y;
            placed[u] = true;
        }
    }
    true
}

fn find_longest_chain(mol: &Molecule, comp: &[usize]) -> Vec<usize> {
    let comp_set: BTreeSet<usize> = comp.iter().copied().collect();
    // Prefer heavy atoms for the main zig-zag chain
    let heavy: Vec<usize> = comp
        .iter()
        .copied()
        .filter(|&u| mol.atoms[u].z != 1)
        .collect();
    let pool = if heavy.len() >= 2 { &heavy } else { comp };
    let pool_set: BTreeSet<usize> = pool.iter().copied().collect();

    let bfs_farthest = |start: usize| -> (usize, Vec<usize>) {
        let mut dist = BTreeMap::new();
        let mut parent = BTreeMap::new();
        let mut q = VecDeque::new();
        dist.insert(start, 0usize);
        q.push_back(start);
        let mut best = start;
        while let Some(u) = q.pop_front() {
            if dist[&u] > dist[&best] {
                best = u;
            }
            for (v, _) in mol.neighbors(u) {
                if comp_set.contains(&v) && pool_set.contains(&v) && !dist.contains_key(&v) {
                    dist.insert(v, dist[&u] + 1);
                    parent.insert(v, u);
                    q.push_back(v);
                }
            }
        }
        let mut path = vec![best];
        let mut cur = best;
        while let Some(&p) = parent.get(&cur) {
            path.push(p);
            cur = p;
        }
        path.reverse();
        (best, path)
    };

    let (end_a, _) = bfs_farthest(pool[0]);
    let (_, longest) = bfs_farthest(end_a);
    longest
}

fn is_linear_atom(mol: &Molecule, u: usize) -> bool {
    let nbs = mol.covalent_neighbors(u);
    if nbs.len() != 2 {
        return false;
    }
    let b0 = mol.bonds[nbs[0].1].kind;
    let b1 = mol.bonds[nbs[1].1].kind;
    matches!(b0, BondKind::Triple)
        || matches!(b1, BondKind::Triple)
        || (b0 == BondKind::Double && b1 == BondKind::Double)
}

fn place_acyclic_backbone(mol: &mut Molecule, chain: &[usize], placed: &mut [bool], bl: f64) {
    if chain.is_empty() {
        return;
    }
    mol.atoms[chain[0]].x = 0.0;
    mol.atoms[chain[0]].y = 0.0;
    placed[chain[0]] = true;
    if chain.len() == 1 {
        return;
    }

    // Standard ACS 120° zig-zag: first bond goes +30° above horizontal!
    let mut cur_angle = PI / 6.0;
    mol.atoms[chain[1]].x = bl * cur_angle.cos();
    mol.atoms[chain[1]].y = bl * cur_angle.sin();
    placed[chain[1]] = true;

    let mut turn_sign = -1.0f64; // alternates -60°, +60°
    for k in 2..chain.len() {
        let prev2 = chain[k - 2];
        let prev = chain[k - 1];
        let curr = chain[k];

        let next_angle = if is_linear_atom(mol, prev) {
            cur_angle // 180° linear through sp center!
        } else if let Some(bi) = mol.find_bond(prev2, prev) {
            // Check if `prev2 = prev` is a double bond with explicit Cis/Trans geometry!
            if mol.bonds[bi].kind == BondKind::Double && k >= 3 {
                let prev3 = chain[k - 3];
                let desired_same_side = match mol.bonds[bi].double_geometry {
                    DoubleBondGeometry::Cis { ref_a, ref_b } => {
                        let match_prev3 = ref_a == prev3 || ref_b == prev3;
                        let match_curr = ref_a == curr || ref_b == curr;
                        let flip = (!match_prev3) ^ (!match_curr);
                        Some(!flip)
                    }
                    DoubleBondGeometry::Trans { ref_a, ref_b } => {
                        let match_prev3 = ref_a == prev3 || ref_b == prev3;
                        let match_curr = ref_a == curr || ref_b == curr;
                        let flip = (!match_prev3) ^ (!match_curr);
                        Some(flip)
                    }
                    _ => None,
                };
                if let Some(same_side) = desired_same_side {
                    // Normal to prev2 -> prev
                    let dx = mol.atoms[prev].x - mol.atoms[prev2].x;
                    let dy = mol.atoms[prev].y - mol.atoms[prev2].y;
                    let nx = -dy;
                    let ny = dx;
                    let side_prev3 = (mol.atoms[prev3].x - mol.atoms[prev2].x) * nx
                        + (mol.atoms[prev3].y - mol.atoms[prev2].y) * ny;
                    let cand1 = cur_angle + PI / 3.0;
                    let cand2 = cur_angle - PI / 3.0;
                    let side1 = cand1.cos() * nx + cand1.sin() * ny;
                    let is_cand1_same = side1 * side_prev3 > 0.0;
                    let chosen = if is_cand1_same == same_side {
                        cand1
                    } else {
                        cand2
                    };
                    turn_sign = if (chosen - cur_angle) > 0.0 {
                        -1.0
                    } else {
                        1.0
                    };
                    chosen
                } else {
                    let a = cur_angle + turn_sign * (PI / 3.0);
                    turn_sign = -turn_sign;
                    a
                }
            } else {
                let a = cur_angle + turn_sign * (PI / 3.0);
                turn_sign = -turn_sign;
                a
            }
        } else {
            let a = cur_angle + turn_sign * (PI / 3.0);
            turn_sign = -turn_sign;
            a
        };

        cur_angle = next_angle;
        mol.atoms[curr].x = mol.atoms[prev].x + bl * cur_angle.cos();
        mol.atoms[curr].y = mol.atoms[prev].y + bl * cur_angle.sin();
        placed[curr] = true;
    }
}

fn grow_remaining_bfs(
    mol: &mut Molecule,
    comp: &[usize],
    _all_rings: &[Vec<usize>],
    ring_systems: &[Vec<Vec<usize>>],
    placed: &mut [bool],
    bl: f64,
) {
    let comp_set: BTreeSet<usize> = comp.iter().copied().collect();

    for _guard in 0..comp.len() + 4 {
        if comp.iter().all(|&u| placed[u]) {
            break;
        }
        // Find a placed atom `u` that has at least one unplaced neighbor in `comp`
        let mut frontier_atom = None;
        for &u in comp {
            if placed[u]
                && mol
                    .neighbors(u)
                    .iter()
                    .any(|&(v, _)| comp_set.contains(&v) && !placed[v])
            {
                frontier_atom = Some(u);
                break;
            }
        }
        let Some(u) = frontier_atom else {
            // Disconnected via non-neighbor link: place first unplaced atom
            if let Some(&unp) = comp.iter().find(|&&x| !placed[x]) {
                mol.atoms[unp].x = 0.0;
                mol.atoms[unp].y = 0.0;
                placed[unp] = true;
            }
            continue;
        };

        let placed_nbs: Vec<usize> = mol
            .neighbors(u)
            .into_iter()
            .map(|(v, _)| v)
            .filter(|&v| placed[v])
            .collect();
        let unplaced_nbs: Vec<usize> = mol
            .neighbors(u)
            .into_iter()
            .map(|(v, _)| v)
            .filter(|&v| comp_set.contains(&v) && !placed[v])
            .collect();

        let ux = mol.atoms[u].x;
        let uy = mol.atoms[u].y;

        // Compute optimal sprout angles for `unplaced_nbs` from `u`
        let angles = compute_sprout_angles(mol, u, &placed_nbs, &unplaced_nbs, placed);

        for (idx, &v) in unplaced_nbs.iter().enumerate() {
            if placed[v] {
                continue;
            }
            let ang = angles[idx];
            let vx = ux + bl * ang.cos();
            let vy = uy + bl * ang.sin();

            // Check if `v` belongs to an unplaced ring system!
            if let Some(sys) = ring_systems
                .iter()
                .find(|sys| sys.iter().any(|r| r.contains(&v)))
            {
                place_ring_system(
                    mol,
                    sys,
                    Some((v, [vx, vy], [ang.cos(), ang.sin()])),
                    placed,
                    bl,
                );
            } else {
                mol.atoms[v].x = vx;
                mol.atoms[v].y = vy;
                placed[v] = true;
            }
        }
    }
}

fn compute_sprout_angles(
    mol: &Molecule,
    u: usize,
    placed_nbs: &[usize],
    unplaced_nbs: &[usize],
    placed: &[bool],
) -> Vec<f64> {
    let ux = mol.atoms[u].x;
    let uy = mol.atoms[u].y;
    let k = unplaced_nbs.len();
    if k == 0 {
        return Vec::new();
    }

    if placed_nbs.is_empty() {
        return (0..k)
            .map(|i| PI / 6.0 + 2.0 * PI * (i as f64) / (k as f64))
            .collect();
    }

    if placed_nbs.len() == 1 {
        let p = placed_nbs[0];
        let in_ang = (uy - mol.atoms[p].y).atan2(ux - mol.atoms[p].x);
        if k == 1 {
            let v = unplaced_nbs[0];
            if is_linear_atom(mol, u) {
                return vec![in_ang];
            }
            // Check double bond Cis/Trans on `p = u`!
            if let Some(bi) = mol.find_bond(p, u) {
                if mol.bonds[bi].kind == BondKind::Double {
                    if let Some(chosen) =
                        resolve_double_bond_sprout_angle(mol, p, u, v, in_ang, placed)
                    {
                        return vec![chosen];
                    }
                }
            }
            // Choose `+60°` vs `-60°` to point away from grandparent (`trans` zig-zag continuation!)
            let p_placed_nbs: Vec<usize> = mol
                .neighbors(p)
                .into_iter()
                .map(|(w, _)| w)
                .filter(|&w| w != u && placed[w])
                .collect();
            let cand1 = in_ang + PI / 3.0;
            let cand2 = in_ang - PI / 3.0;
            if p_placed_nbs.len() == 1 {
                let gp = p_placed_nbs[0];
                let nx = -(uy - mol.atoms[p].y);
                let ny = ux - mol.atoms[p].x;
                let gp_side = (mol.atoms[gp].x - mol.atoms[p].x) * nx
                    + (mol.atoms[gp].y - mol.atoms[p].y) * ny;
                let c1_side = cand1.cos() * nx + cand1.sin() * ny;
                // Trans zig-zag puts `v` on OPPOSITE side of `p-u` from `gp`!
                return vec![if c1_side * gp_side < 0.0 {
                    cand1
                } else {
                    cand2
                }];
            }
            // Otherwise pick candidate farther from all currently placed atoms
            let score = |ang: f64| -> f64 {
                let tx = ux + 1.54 * ang.cos();
                let ty = uy + 1.54 * ang.sin();
                mol.atoms
                    .iter()
                    .enumerate()
                    .filter(|&(idx, _)| idx != u && idx != p && placed[idx])
                    .map(|(_, a)| (a.x - tx).powi(2) + (a.y - ty).powi(2))
                    .fold(f64::INFINITY, f64::min)
            };
            return vec![if score(cand1) >= score(cand2) {
                cand1
            } else {
                cand2
            }];
        }
        if k == 2 {
            // Check if `p = u` is a double bond with Cis/Trans specifying which of `unplaced_nbs[0], [1]` goes where!
            if let Some(bi) = mol.find_bond(p, u) {
                if mol.bonds[bi].kind == BondKind::Double {
                    if let Some(a0) =
                        resolve_double_bond_sprout_angle(mol, p, u, unplaced_nbs[0], in_ang, placed)
                    {
                        let a1 = if (a0 - (in_ang + PI / 3.0)).abs() < 0.1 {
                            in_ang - PI / 3.0
                        } else {
                            in_ang + PI / 3.0
                        };
                        return vec![a0, a1];
                    }
                }
            }
            return vec![in_ang + PI / 3.0, in_ang - PI / 3.0];
        }
        // k == 3 (e.g. -tBu, -CF3, -SO3H): +60°, 0° (180° continuation), -60° or ±90° + 0°!
        if k == 3 {
            return vec![in_ang + PI / 2.0, in_ang, in_ang - PI / 2.0];
        }
    }

    // 2 or more placed neighbors (e.g. ring atom or chain carbon sprouting 1 or 2 substituents):
    // Point outward along the negative sum of unit vectors to placed neighbors!
    let mut sx = 0.0;
    let mut sy = 0.0;
    for &p in placed_nbs {
        let dx = mol.atoms[p].x - ux;
        let dy = mol.atoms[p].y - uy;
        let len = dx.hypot(dy).max(1e-5);
        sx += dx / len;
        sy += dy / len;
    }
    let out_ang = if sx.hypot(sy) > 1e-4 {
        (-sy).atan2(-sx)
    } else {
        // Collinear placed neighbors (180° apart): perpendicular normal!
        let p = placed_nbs[0];
        (mol.atoms[p].y - uy).atan2(mol.atoms[p].x - ux) + PI / 2.0
    };

    if k == 1 {
        vec![out_ang]
    } else if k == 2 {
        // Gem-disubstituted (e.g. gem-dimethyl on ring or chain): ±35° around outward bisector!
        vec![out_ang + 0.62, out_ang - 0.62]
    } else {
        (0..k)
            .map(|i| out_ang - 0.8 + 1.6 * (i as f64) / ((k - 1) as f64))
            .collect()
    }
}

fn resolve_double_bond_sprout_angle(
    mol: &Molecule,
    p: usize,
    u: usize,
    v: usize,
    in_ang: f64,
    placed: &[bool],
) -> Option<f64> {
    let bi = mol.find_bond(p, u)?;
    let (same_side, ref_a, ref_b) = match mol.bonds[bi].double_geometry {
        DoubleBondGeometry::Cis { ref_a, ref_b } => (true, ref_a, ref_b),
        DoubleBondGeometry::Trans { ref_a, ref_b } => (false, ref_a, ref_b),
        _ => return None,
    };
    let placed_p_subs: Vec<usize> = mol
        .covalent_neighbors(p)
        .into_iter()
        .map(|(w, _)| w)
        .filter(|&w| w != u && placed[w])
        .collect();
    if placed_p_subs.is_empty() {
        return None;
    }
    let ref_on_p = if placed_p_subs.contains(&ref_a) {
        ref_a
    } else if placed_p_subs.contains(&ref_b) {
        ref_b
    } else {
        placed_p_subs[0]
    };
    let flip_p = ref_on_p != ref_a && ref_on_p != ref_b;
    let flip_u = v != ref_a && v != ref_b;
    let want_same = same_side ^ flip_p ^ flip_u;

    let nx = -(mol.atoms[u].y - mol.atoms[p].y);
    let ny = mol.atoms[u].x - mol.atoms[p].x;
    let p_side = (mol.atoms[ref_on_p].x - mol.atoms[p].x) * nx
        + (mol.atoms[ref_on_p].y - mol.atoms[p].y) * ny;
    let cand1 = in_ang + PI / 3.0;
    let cand2 = in_ang - PI / 3.0;
    let c1_side = cand1.cos() * nx + cand1.sin() * ny;
    Some(if (c1_side * p_side > 0.0) == want_same {
        cand1
    } else {
        cand2
    })
}

fn polish_2d_coords(mol: &mut Molecule, comp: &[usize], rings: &[Vec<usize>], bl: f64) {
    if comp.len() <= 2 {
        return;
    }
    // Check if there are strained bond lengths (e.g. bridged rings / peri-fusions) or severe steric clashes (< 0.55 * bl)
    let comp_set: BTreeSet<usize> = comp.iter().copied().collect();
    let has_strain_or_clash = {
        let mut bad = false;
        for b in &mol.bonds {
            if matches!(b.kind, BondKind::Hydrogen | BondKind::Ionic) {
                continue;
            }
            if comp_set.contains(&b.a) && comp_set.contains(&b.b) {
                if !mol.atoms[b.a].multi_center_atoms.is_empty()
                    || !mol.atoms[b.b].multi_center_atoms.is_empty()
                {
                    continue;
                }
                let d = (mol.atoms[b.a].x - mol.atoms[b.b].x)
                    .hypot(mol.atoms[b.a].y - mol.atoms[b.b].y);
                if (d - bl).abs() > 0.12 * bl {
                    bad = true;
                    break;
                }
            }
        }
        if !bad {
            'outer: for i in 0..comp.len() {
                for j in (i + 1)..comp.len() {
                    let u = comp[i];
                    let v = comp[j];
                    if mol.find_bond(u, v).is_some() {
                        continue;
                    }
                    let d =
                        (mol.atoms[u].x - mol.atoms[v].x).hypot(mol.atoms[u].y - mol.atoms[v].y);
                    if d < 0.55 * bl {
                        bad = true;
                        break 'outer;
                    }
                }
            }
        }
        bad
    };

    if !has_strain_or_clash {
        return;
    }

    let _ = rings;
    // Gentle spring + 1,3-angle + steric clash relaxation preserving bond lengths at `bl`
    let mut adj_pairs = Vec::new();
    let mut bonded_set = BTreeSet::new();
    for b in &mol.bonds {
        if matches!(b.kind, BondKind::Hydrogen | BondKind::Ionic) {
            continue;
        }
        if comp_set.contains(&b.a)
            && comp_set.contains(&b.b)
            && mol.atoms[b.a].multi_center_atoms.is_empty()
            && mol.atoms[b.b].multi_center_atoms.is_empty()
        {
            adj_pairs.push((b.a, b.b));
            bonded_set.insert((b.a.min(b.b), b.a.max(b.b)));
        }
    }

    // 1,3 geminal pairs
    let mut gem_pairs = Vec::new();
    for &u in comp {
        let nbs: Vec<usize> = mol
            .covalent_neighbors(u)
            .into_iter()
            .map(|(v, _)| v)
            .filter(|v| comp_set.contains(v))
            .collect();
        let target_13 = if is_linear_atom(mol, u) {
            2.0 * bl
        } else {
            bl * 3.0f64.sqrt()
        };
        for i in 0..nbs.len() {
            for j in (i + 1)..nbs.len() {
                let a = nbs[i].min(nbs[j]);
                let b = nbs[i].max(nbs[j]);
                if !bonded_set.contains(&(a, b)) {
                    gem_pairs.push((a, b, target_13));
                }
            }
        }
    }

    let n_total = mol.atoms.len();
    let mut gx = vec![0.0; n_total];
    let mut gy = vec![0.0; n_total];

    for iter in 0..80 {
        for &u in comp {
            gx[u] = 0.0;
            gy[u] = 0.0;
        }
        // Bond springs (strong)
        for &(u, v) in &adj_pairs {
            let dx = mol.atoms[v].x - mol.atoms[u].x;
            let dy = mol.atoms[v].y - mol.atoms[u].y;
            let d = dx.hypot(dy).max(1e-4);
            let f = 4.0 * (d - bl) / d;
            gx[u] += f * dx;
            gy[u] += f * dy;
            gx[v] -= f * dx;
            gy[v] -= f * dy;
        }
        // 1,3 angle springs (medium)
        for &(u, v, t13) in &gem_pairs {
            let dx = mol.atoms[v].x - mol.atoms[u].x;
            let dy = mol.atoms[v].y - mol.atoms[u].y;
            let d = dx.hypot(dy).max(1e-4);
            let f = 0.8 * (d - t13) / d;
            gx[u] += f * dx;
            gy[u] += f * dy;
            gx[v] -= f * dx;
            gy[v] -= f * dy;
        }
        // Non-bonded clash repulsion
        let min_sep = 0.78 * bl;
        for i in 0..comp.len() {
            for j in (i + 1)..comp.len() {
                let u = comp[i];
                let v = comp[j];
                let key = (u.min(v), u.max(v));
                if bonded_set.contains(&key) {
                    continue;
                }
                let dx = mol.atoms[v].x - mol.atoms[u].x;
                let dy = mol.atoms[v].y - mol.atoms[u].y;
                let d = dx.hypot(dy);
                if d < min_sep {
                    let (ux, uy) = if d < 1e-4 {
                        let ang = ((i * 7 + j * 13) as f64) * 0.7;
                        (ang.cos(), ang.sin())
                    } else {
                        (dx / d, dy / d)
                    };
                    let push = 1.5 * (min_sep - d);
                    gx[u] -= push * ux;
                    gy[u] -= push * uy;
                    gx[v] += push * ux;
                    gy[v] += push * uy;
                }
            }
        }

        let step = 0.08 * (1.0 - (iter as f64) / 100.0);
        for &u in comp {
            let mag = gx[u].hypot(gy[u]);
            let cap = (0.15 * bl) / mag.max(0.15 * bl / step.max(1e-4));
            let s = step.min(cap);
            mol.atoms[u].x += s * gx[u];
            mol.atoms[u].y += s * gy[u];
        }
    }
}

/// Synthesize clean 2D `WedgeUp` / `HashDown` stereo bonds on `mol` from each stereocenter's `TetrahedralParity`
/// so that 2D stereochemical perception (`perceive_cip`) matches the exact 3D/SMILES configuration!
pub fn assign_wedges_from_parity(mol: &mut Molecule) {
    let ring_bonds = super::stereo::perceive_ring_bonds(mol);
    let n = mol.atoms.len();

    for center in 0..n {
        let Some(parity) = mol.atoms[center].parity else {
            continue;
        };
        // Skip if center already has a valid outgoing WedgeUp / HashDown
        let already_has_wedge = mol
            .bonds
            .iter()
            .any(|b| b.a == center && matches!(b.kind, BondKind::WedgeUp | BondKind::HashDown));
        if already_has_wedge {
            continue;
        }

        // Target R-sense pyramid from `parity`:
        let target_pyr = if parity.clockwise {
            parity.neighbors
        } else {
            [
                parity.neighbors[0],
                parity.neighbors[2],
                parity.neighbors[1],
                parity.neighbors[3],
            ]
        };

        // Rank candidate single bonds incident to `center` for placing the wedge/hash:
        // Prefer: (1) explicit H, (2) non-ring terminal substituent, (3) non-ring single bond, (4) ring single bond
        // Avoid bonds whose other endpoint is ALSO a stereocenter if possible!
        let mut candidates: Vec<(usize, usize, i32)> = Vec::new();
        for (nb, bi) in mol.covalent_neighbors(center) {
            if mol.bonds[bi].kind != BondKind::Single {
                continue;
            }
            let mut score = 0i32;
            if !ring_bonds[bi] {
                score += 100;
            }
            if mol.atoms[nb].z == 1 {
                score += 50;
            } else if mol.covalent_neighbors(nb).len() == 1 {
                score += 30;
            }
            if mol.atoms[nb].parity.is_none() {
                score += 20;
            }
            candidates.push((nb, bi, score));
        }
        candidates.sort_by_key(|b| std::cmp::Reverse(b.2));

        // Test candidates and WedgeUp vs HashDown until `perceive_cip` / pyramid matches `target_pyr`!
        let mut assigned = false;
        for &(_nb, bi, _sc) in &candidates {
            // Orient bond so narrow tip is at `center` (`b.a == center`)
            let orig_a = mol.bonds[bi].a;
            let orig_b = mol.bonds[bi].b;
            if mol.bonds[bi].a != center {
                mol.bonds[bi].a = center;
                mol.bonds[bi].b = if orig_a == center { orig_b } else { orig_a };
            }

            // Temporarily clear `parity` so perception MUST come from the 2D wedge/hash!
            let saved_parity = mol.atoms[center].parity.take();

            for candidate_kind in [BondKind::WedgeUp, BondKind::HashDown] {
                mol.bonds[bi].kind = candidate_kind;
                // Check if 2D wedge pyramid has EVEN permutation parity with `target_pyr`!
                if wedge_matches_target_pyramid(mol, center, &target_pyr) {
                    assigned = true;
                    break;
                }
            }

            mol.atoms[center].parity = saved_parity;
            if assigned {
                break;
            } else {
                mol.bonds[bi].kind = BondKind::Single;
                mol.bonds[bi].a = orig_a;
                mol.bonds[bi].b = orig_b;
            }
        }
    }
}

fn wedge_matches_target_pyramid(mol: &Molecule, center: usize, target_pyr: &[usize; 4]) -> bool {
    // Build minimal CipGraph via stereo helper? Wait: `perceive_center_pyramid` is `pub(crate)` in `stereo.rs`,
    // or we can test directly via `perceive_cip` when chiral, AND for general centers (even prochiral/ring centers
    // like meso or multi-center isoleucine) let's check both `perceive_cip` or direct 3D-lifted wedge pyramid!
    // Let's check using `perceive_cip` on an isotopically disambiguated 1-hop probe if needed, OR direct wedge geometry!
    // Wait: let's inspect the exact 2D wedge geometry at `center`:
    let nbs: Vec<usize> = mol
        .covalent_neighbors(center)
        .into_iter()
        .map(|(nb, _)| nb)
        .collect();
    if nbs.len() < 3 || nbs.len() > 4 {
        return false;
    }
    // Create a tiny local probe molecule with center + its 3 or 4 neighbors given distinct Z = [9, 17, 35, 53]
    // so `perceive_cip` ALWAYS produces an exact `R` or `S` label on ANY tetrahedral geometry!
    let mut probe = Molecule::new();
    let c_idx = probe.add_atom(mol.atoms[center].clone());
    probe.atoms[c_idx].z = 6;
    probe.atoms[c_idx].symbol = "C".to_string();
    probe.atoms[c_idx].formal_charge = 0;
    probe.atoms[c_idx].parity = None;
    probe.atoms[c_idx].explicit_hydrogens = if nbs.len() == 3 { Some(1) } else { Some(0) };

    let halogens = [(53u8, "I"), (35u8, "Br"), (17u8, "Cl"), (9u8, "F")];
    let mut map_nb = BTreeMap::new();
    for (k, &nb) in nbs.iter().enumerate() {
        let mut a = mol.atoms[nb].clone();
        a.z = halogens[k].0;
        a.symbol = halogens[k].1.to_string();
        a.isotope = None;
        a.formal_charge = 0;
        a.explicit_hydrogens = Some(0);
        a.parity = None;
        let p_nb = probe.add_atom(a);
        map_nb.insert(nb, p_nb);
        let bi = mol.find_bond(center, nb).unwrap();
        let kind = if mol.bonds[bi].a == center {
            mol.bonds[bi].kind
        } else {
            BondKind::Single
        };
        probe.add_bond(c_idx, p_nb, kind);
    }

    let cip_wedge = perceive_cip(&probe);
    let Some(&wedge_desc) = cip_wedge.atoms.get(&c_idx) else {
        return false;
    };

    // Now compute what `target_pyr` gives on the exact same halogen-labeled probe!
    for b in &mut probe.bonds {
        b.kind = BondKind::Single;
    }
    let mut mapped_pyr = [usize::MAX; 4];
    for k in 0..4 {
        let orig = target_pyr[k];
        mapped_pyr[k] = if orig >= mol.atoms.len() {
            usize::MAX
        } else if let Some(&m) = map_nb.get(&orig) {
            m
        } else {
            return false;
        };
    }
    probe.atoms[c_idx].parity = Some(TetrahedralParity {
        neighbors: mapped_pyr,
        clockwise: true, // R-sense pyramid directly corresponds to clockwise=true!
    });
    let cip_target = perceive_cip(&probe);
    let _ = permutation_is_odd;
    cip_target.atoms.get(&c_idx) == Some(&wedge_desc)
}

// ============================================================================
// 4. VSEPR + Distance-Geometry 3D Coordinate Builder (`clean_3d`)
// ============================================================================

/// Generate chemically realistic 3D coordinates `(x, y, z)` for `mol` using VSEPR hybridization
/// (`sp` 180°, `sp2` 120°, `sp3` 109.47°), covalent-radius bond lengths, aromatic planarity,
/// saturated 6-ring chair conformations, `(E)/(Z)` alkene geometry, and tetrahedral `(R)/(S)` handedness!
pub fn clean_3d(mol: &mut Molecule) {
    let n = mol.atoms.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        mol.atoms[0].x = 0.0;
        mol.atoms[0].y = 0.0;
        mol.atoms[0].z_coord = Some(0.0);
        return;
    }

    // Ensure 2D layout & stored stereochemistry are populated first if all atoms are at (0, 0)
    if mol.needs_2d_coords() {
        clean_2d(mol);
    } else {
        save_current_pyramids_to_parity(mol);
    }

    let arom = perceive_aromaticity(mol);
    let rings = perceive_sssr(mol);

    // 1. Determine VSEPR ideal bond angle (radians) at each atom
    let mut ideal_angle = vec![109.4712206f64.to_radians(); n];
    for (u, ang_ref) in ideal_angle.iter_mut().enumerate() {
        if is_linear_atom(mol, u) {
            *ang_ref = PI;
            continue;
        }
        let has_double = mol
            .covalent_neighbors(u)
            .iter()
            .any(|&(_, bi)| matches!(mol.bonds[bi].kind, BondKind::Double | BondKind::Aromatic));
        // Amide / aniline sp2 nitrogen conjugation check
        let is_conjugated_n_or_o = matches!(mol.atoms[u].z, 7 | 8)
            && mol.covalent_neighbors(u).iter().any(|&(nb, _)| {
                arom.aromatic_atoms[nb]
                    || mol
                        .covalent_neighbors(nb)
                        .iter()
                        .any(|&(_, b2)| mol.bonds[b2].kind == BondKind::Double)
            });
        if arom.aromatic_atoms[u] || has_double || is_conjugated_n_or_o {
            *ang_ref = 2.0 * PI / 3.0; // 120°
        }
    }

    // 2. Collect distance restraints `(u, v, target_dist, weight)`
    let mut restraints: Vec<(usize, usize, f64, f64)> = Vec::new();
    let mut bonded_or_13: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut target_bond_len: BTreeMap<(usize, usize), f64> = BTreeMap::new();

    for (bi, b) in mol.bonds.iter().enumerate() {
        if matches!(b.kind, BondKind::Hydrogen | BondKind::Ionic) {
            continue;
        }
        if !mol.atoms[b.a].multi_center_atoms.is_empty()
            || !mol.atoms[b.b].multi_center_atoms.is_empty()
        {
            continue;
        }
        let mut r = covalent_radius(mol.atoms[b.a].z) + covalent_radius(mol.atoms[b.b].z);
        if arom.aromatic_bonds[bi] || b.kind == BondKind::Aromatic {
            r -= 0.12;
        } else if b.kind == BondKind::Double {
            r -= 0.18;
        } else if b.kind == BondKind::Triple {
            r -= 0.32;
        }
        let r = r.clamp(0.95, 2.60);
        let key = (b.a.min(b.b), b.a.max(b.b));
        bonded_or_13.insert(key);
        target_bond_len.insert(key, r);
        restraints.push((b.a, b.b, r, 8.0));
    }

    // 1,3-geminal angle distance restraints via Law of Cosines
    for (u, &theta) in ideal_angle.iter().enumerate() {
        let nbs: Vec<usize> = mol
            .covalent_neighbors(u)
            .into_iter()
            .map(|(v, _)| v)
            .collect();
        if nbs.len() < 2 {
            continue;
        }
        // Inside small rings (3, 4, 5), ring closure already dictates interior angles
        for i in 0..nbs.len() {
            for j in (i + 1)..nbs.len() {
                let a = nbs[i];
                let b = nbs[j];
                let key = (a.min(b), a.max(b));
                if bonded_or_13.contains(&key) {
                    continue;
                }
                // Check if both `a` and `b` share a small ring (<= 5) with `u`
                let in_small_ring = rings
                    .iter()
                    .any(|r| r.len() <= 5 && r.contains(&u) && r.contains(&a) && r.contains(&b));
                if in_small_ring {
                    bonded_or_13.insert(key);
                    continue;
                }
                let r1 = target_bond_len
                    .get(&(u.min(a), u.max(a)))
                    .copied()
                    .unwrap_or(1.52);
                let r2 = target_bond_len
                    .get(&(u.min(b), u.max(b)))
                    .copied()
                    .unwrap_or(1.52);
                let d13 = (r1 * r1 + r2 * r2 - 2.0 * r1 * r2 * theta.cos()).sqrt();
                bonded_or_13.insert(key);
                restraints.push((a, b, d13, 3.0));
            }
        }
    }

    // 1,4-double-bond cis/trans & aromatic coplanar distance restraints
    let cip_2d = perceive_cip(mol);
    for (bi, b) in mol.bonds.iter().enumerate() {
        if b.kind == BondKind::Double {
            let subs_a: Vec<usize> = mol
                .covalent_neighbors(b.a)
                .into_iter()
                .map(|(v, _)| v)
                .filter(|&v| v != b.b)
                .collect();
            let subs_b: Vec<usize> = mol
                .covalent_neighbors(b.b)
                .into_iter()
                .map(|(v, _)| v)
                .filter(|&v| v != b.a)
                .collect();
            let r_ab = target_bond_len
                .get(&(b.a.min(b.b), b.a.max(b.b)))
                .copied()
                .unwrap_or(1.34);
            for &sa in &subs_a {
                for &sb in &subs_b {
                    let key = (sa.min(sb), sa.max(sb));
                    if bonded_or_13.contains(&key) {
                        continue;
                    }
                    // Determine cis vs trans from current 2D coordinates (which clean_2d already aligned to Cis/Trans!)
                    let nx = -(mol.atoms[b.b].y - mol.atoms[b.a].y);
                    let ny = mol.atoms[b.b].x - mol.atoms[b.a].x;
                    let pa = (mol.atoms[sa].x - mol.atoms[b.a].x) * nx
                        + (mol.atoms[sa].y - mol.atoms[b.a].y) * ny;
                    let pb = (mol.atoms[sb].x - mol.atoms[b.b].x) * nx
                        + (mol.atoms[sb].y - mol.atoms[b.b].y) * ny;
                    let is_cis = pa * pb > 0.0;
                    let r1 = target_bond_len
                        .get(&(b.a.min(sa), b.a.max(sa)))
                        .copied()
                        .unwrap_or(1.50);
                    let r2 = target_bond_len
                        .get(&(b.b.min(sb), b.b.max(sb)))
                        .copied()
                        .unwrap_or(1.50);
                    // Exact planar 120° cis vs trans 1,4-distance!
                    let dx = r_ab + 0.5 * (r1 + r2);
                    let dy = if is_cis {
                        0.8660254 * (r1 - r2)
                    } else {
                        0.8660254 * (r1 + r2)
                    };
                    let d14 = dx.hypot(dy);
                    let _ = (&cip_2d, bi);
                    bonded_or_13.insert(key);
                    restraints.push((sa, sb, d14, 2.5));
                }
            }
        }
    }

    // Aromatic ring cross-ring planar distance restraints (keeps benzene/indole/naphthalene flat!)
    for r in &rings {
        if r.len() == 6 && r.iter().all(|&u| arom.aromatic_atoms[u]) {
            for k in 0..3 {
                let u = r[k];
                let v = r[k + 3];
                let r_avg = 1.395;
                restraints.push((u, v, 2.0 * r_avg, 3.0));
            }
        }
    }

    // 3. Collect tetrahedral chiral volume restraints for all stereocenters with `parity` (plus non-chiral sp3 centers for non-planarity!)
    // Each chiral restraint `(center, [n0, n1, n2, n3])` enforces `signed_tetra_volume < -target_vol`!
    let mut chiral_targets: Vec<(usize, [usize; 4])> = Vec::new();
    for u in 0..n {
        if let Some(p) = mol.atoms[u].parity {
            let pyr = if p.clockwise {
                p.neighbors
            } else {
                [
                    p.neighbors[0],
                    p.neighbors[2],
                    p.neighbors[1],
                    p.neighbors[3],
                ]
            };
            chiral_targets.push((u, pyr));
        }
    }

    // 4. Initialize 3D coordinates `(x, y, z)`:
    // Start from 2D `(x, y)` scaled to covalent bond scale, with chair Z-puckering on saturated 6-rings,
    // wedge/hash Z-lifts on stereocenters, and small deterministic tetrahedral Z-offsets on sp3 atoms!
    let mut pos: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let z_init = if arom.aromatic_atoms[i] || (ideal_angle[i] - 2.0 * PI / 3.0).abs() < 0.05
            {
                0.0
            } else {
                // Deterministic non-zero Z seed to break 2D planarity for sp3 chains
                ((i as i32 % 2) * 2 - 1) as f64 * 0.28
            };
            [mol.atoms[i].x, mol.atoms[i].y, z_init]
        })
        .collect();

    // Initialize saturated 6-membered rings into an alternating chair (`±0.26 Å`)!
    for r in &rings {
        if r.len() == 6
            && r.iter()
                .all(|&u| !arom.aromatic_atoms[u] && (ideal_angle[u] - 1.9106).abs() < 0.1)
        {
            for (k, &u) in r.iter().enumerate() {
                pos[u][2] = if k % 2 == 0 { 0.26 } else { -0.26 };
            }
        }
    }

    // Apply wedge/hash Z hints so stereocenters start in the right hemisphere!
    for b in &mol.bonds {
        match b.kind {
            BondKind::WedgeUp | BondKind::HollowWedge => {
                pos[b.b][2] = pos[b.a][2] + 0.85;
            }
            BondKind::HashDown => {
                pos[b.b][2] = pos[b.a][2] - 0.85;
            }
            _ => {}
        }
    }

    // 5. Iterative 3D optimization (distance restraints + steric repulsion + aromatic planarity + chiral volume projection)
    let mut grad = vec![[0.0f64; 3]; n];
    for iter in 0..220 {
        grad.fill([0.0, 0.0, 0.0]);

        // Distance restraints (bonds, 1,3-angles, 1,4-alkenes, aromatic diagonals)
        for &(u, v, target, w) in &restraints {
            let dx = pos[v][0] - pos[u][0];
            let dy = pos[v][1] - pos[u][1];
            let dz = pos[v][2] - pos[u][2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-5);
            let f = w * (d - target) / d;
            grad[u][0] += f * dx;
            grad[u][1] += f * dy;
            grad[u][2] += f * dz;
            grad[v][0] -= f * dx;
            grad[v][1] -= f * dy;
            grad[v][2] -= f * dz;
        }

        // Aromatic ring planarity restraint (gently pull aromatic ring + direct substituents toward common plane)
        for r in &rings {
            if r.iter().all(|&u| arom.aromatic_atoms[u]) {
                let mean_z = r.iter().map(|&u| pos[u][2]).sum::<f64>() / (r.len() as f64);
                for &u in r {
                    grad[u][2] += 2.5 * (mean_z - pos[u][2]);
                    for (nb, _) in mol.covalent_neighbors(u) {
                        if !arom.aromatic_atoms[nb] {
                            grad[nb][2] += 1.2 * (mean_z - pos[nb][2]);
                        }
                    }
                }
            }
        }

        // Non-bonded steric repulsion (`d_min = 2.2 Å` for heavy-heavy, `1.7 Å` involving H)
        for u in 0..n {
            for v in (u + 1)..n {
                if bonded_or_13.contains(&(u, v)) {
                    continue;
                }
                let min_d = if mol.atoms[u].z == 1 || mol.atoms[v].z == 1 {
                    1.75
                } else {
                    2.35
                };
                let dx = pos[v][0] - pos[u][0];
                let dy = pos[v][1] - pos[u][1];
                let dz = pos[v][2] - pos[u][2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < min_d * min_d {
                    let d = d2.sqrt().max(1e-4);
                    let push = 1.5 * (min_d - d) / d;
                    grad[u][0] -= push * dx;
                    grad[u][1] -= push * dy;
                    grad[u][2] -= push * dz;
                    grad[v][0] += push * dx;
                    grad[v][1] += push * dy;
                    grad[v][2] += push * dz;
                }
            }
        }

        // Chiral tetrahedral handedness enforcement!
        // Remember R-sense `[a0, a1, a2, a3]` requires `vol(u0 - u3, u1 - u3, u2 - u3) < -0.35`!
        for &(center, pyr) in &chiral_targets {
            let pc = pos[center];
            let vec_of = |idx: usize, others: &[usize]| -> [f64; 3] {
                if idx < n {
                    [
                        pos[idx][0] - pc[0],
                        pos[idx][1] - pc[1],
                        pos[idx][2] - pc[2],
                    ]
                } else {
                    // Virtual implicit H / lone pair opposite the 3 real ligands!
                    let mut sx = 0.0;
                    let mut sy = 0.0;
                    let mut sz = 0.0;
                    for &o in others {
                        if o < n {
                            sx -= pos[o][0] - pc[0];
                            sy -= pos[o][1] - pc[1];
                            sz -= pos[o][2] - pc[2];
                        }
                    }
                    let l = (sx * sx + sy * sy + sz * sz).sqrt().max(1e-4);
                    [1.09 * sx / l, 1.09 * sy / l, 1.09 * sz / l]
                }
            };
            let v0 = vec_of(pyr[0], &pyr);
            let v1 = vec_of(pyr[1], &pyr);
            let v2 = vec_of(pyr[2], &pyr);
            let v3 = vec_of(pyr[3], &pyr);

            let a = [v0[0] - v3[0], v0[1] - v3[1], v0[2] - v3[2]];
            let b = [v1[0] - v3[0], v1[1] - v3[1], v1[2] - v3[2]];
            let c = [v2[0] - v3[0], v2[1] - v3[1], v2[2] - v3[2]];
            let vol = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);

            // Target volume for ~1.5 Å tetrahedral vectors is around `-1.8 .. -2.5`; enforce `vol <= -1.2`!
            if vol > -1.2 {
                // Analytical gradient of `vol` w.r.t. `v0, v1, v2` to drive `vol` more negative!
                // d(vol)/d(a) = b x c; d(vol)/d(b) = c x a; d(vol)/d(c) = a x b
                let bxc = [
                    b[1] * c[2] - b[2] * c[1],
                    b[2] * c[0] - b[0] * c[2],
                    b[0] * c[1] - b[1] * c[0],
                ];
                let cxa = [
                    c[1] * a[2] - c[2] * a[1],
                    c[2] * a[0] - c[0] * a[2],
                    c[0] * a[1] - c[1] * a[0],
                ];
                let axb = [
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                ];
                // If completely flat (`vol == 0` and cross products have Z == 0), nudge `pyr[0]` in +Z and `pyr[1]` in -Z!
                if bxc[0].abs() + bxc[1].abs() + bxc[2].abs() < 1e-4 {
                    if pyr[0] < n {
                        pos[pyr[0]][2] += 0.5;
                    }
                    if pyr[1] < n {
                        pos[pyr[1]][2] -= 0.5;
                    }
                } else {
                    let k_chiral = 2.2 * (vol + 1.4);
                    if pyr[0] < n {
                        for d in 0..3 {
                            grad[pyr[0]][d] -= k_chiral * bxc[d];
                        }
                    }
                    if pyr[1] < n {
                        for d in 0..3 {
                            grad[pyr[1]][d] -= k_chiral * cxa[d];
                        }
                    }
                    if pyr[2] < n {
                        for d in 0..3 {
                            grad[pyr[2]][d] -= k_chiral * axb[d];
                        }
                    }
                    if pyr[3] < n {
                        for d in 0..3 {
                            grad[pyr[3]][d] += k_chiral * (bxc[d] + cxa[d] + axb[d]);
                        }
                    } else {
                        // Push center in the opposite direction when 4th ligand is implicit H!
                        for d in 0..3 {
                            grad[center][d] += 0.5 * k_chiral * (bxc[d] + cxa[d] + axb[d]);
                        }
                    }
                }
            }
        }

        let lr = 0.045 * (1.0 - 0.65 * (iter as f64) / 220.0);
        for u in 0..n {
            let gmag =
                (grad[u][0] * grad[u][0] + grad[u][1] * grad[u][1] + grad[u][2] * grad[u][2])
                    .sqrt();
            let step = if lr * gmag > 0.20 { 0.20 / gmag } else { lr };
            pos[u][0] += step * grad[u][0];
            pos[u][1] += step * grad[u][1];
            pos[u][2] += step * grad[u][2];
        }
    }

    // Center 3D coordinates around origin and clear 2D-only wedge display? Keep bond kinds intact while populating x, y, z_coord!
    let mean = [
        pos.iter().map(|p| p[0]).sum::<f64>() / (n as f64),
        pos.iter().map(|p| p[1]).sum::<f64>() / (n as f64),
        pos.iter().map(|p| p[2]).sum::<f64>() / (n as f64),
    ];
    for (i, a) in mol.atoms.iter_mut().enumerate() {
        a.x = pos[i][0] - mean[0];
        a.y = pos[i][1] - mean[1];
        a.z_coord = Some(pos[i][2] - mean[2]);
    }
}
