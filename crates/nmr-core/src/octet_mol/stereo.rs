//! Ring Perception (SSSR), Hückel $4n+2$ Aromaticity, Kekulization, and Stereochemistry
//! (IUPAC 2013 CIP `(R)/(S)` & `(E)/(Z)`, Axial `(M)/(P)`, Bridged `endo`/`exo`/`syn`/`anti`,
//! and Chair Cyclohexane `axial`/`equatorial`).
//!
//! Ported/adapted from Octet-Draw by Sam Ellis (https://github.com/SamChemMan/Octet-Draw, Apache-2.0)
//! and open cheminformatics specifications (IUPAC 2013 Blue Book P-92 / P-93;
//! Hanson et al., *J. Chem. Inf. Model.* 2018, 58, 1755–1765).

use super::graph::{BondKind, ChemError, DoubleBondGeometry, Molecule, TetrahedralParity};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

// ============================================================================
// 1. SSSR / Smallest Ring Perception & Bridge Detection
// ============================================================================

/// Perceive the Smallest Set of Smallest Rings (SSSR) of `mol` as ordered lists of atom indices around each cycle.
/// Uses shortest-cycle enumeration per covalent edge followed by GF(2) Gaussian elimination over bond incidence vectors.
pub fn perceive_sssr(mol: &Molecule) -> Vec<Vec<usize>> {
    let n_atoms = mol.atoms.len();
    // Collect covalent bonds (excluding Hydrogen, Ionic, Dative, and multi-center dummies).
    let mut cov_bonds: Vec<(usize, usize, usize)> = Vec::new();
    for (bi, b) in mol.bonds.iter().enumerate() {
        if matches!(
            b.kind,
            BondKind::Hydrogen | BondKind::Ionic | BondKind::Dative
        ) {
            continue;
        }
        if mol.atoms[b.a].z == 0 && !mol.atoms[b.a].multi_center_atoms.is_empty() {
            continue;
        }
        if mol.atoms[b.b].z == 0 && !mol.atoms[b.b].multi_center_atoms.is_empty() {
            continue;
        }
        cov_bonds.push((bi, b.a, b.b));
    }
    if cov_bonds.is_empty() || n_atoms < 3 {
        return Vec::new();
    }

    // Compute cyclomatic number E - V + C on covalent graph
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_atoms];
    for (local_e, &(_bi, a, b)) in cov_bonds.iter().enumerate() {
        adj[a].push((b, local_e));
        adj[b].push((a, local_e));
    }
    let mut visited = vec![false; n_atoms];
    let mut active_vertices = 0usize;
    let mut comps = 0usize;
    for i in 0..n_atoms {
        if adj[i].is_empty() {
            continue;
        }
        active_vertices += 1;
        if !visited[i] {
            comps += 1;
            let mut q = VecDeque::new();
            q.push_back(i);
            visited[i] = true;
            while let Some(u) = q.pop_front() {
                for &(v, _) in &adj[u] {
                    if !visited[v] {
                        visited[v] = true;
                        q.push_back(v);
                    }
                }
            }
        }
    }
    let n_edges = cov_bonds.len();
    let rank = (n_edges + comps).saturating_sub(active_vertices);
    if rank == 0 {
        return Vec::new();
    }

    // Enumerate shortest simple cycles through each edge (plus vertex-pair shortest paths for polycycles).
    let mut candidates: Vec<(Vec<usize>, Vec<u64>)> = Vec::new();
    let mut seen_masks: BTreeSet<Vec<u64>> = BTreeSet::new();
    let n_words = n_edges.div_ceil(64);

    let add_cycle = |atoms_cycle: Vec<usize>,
                     candidates: &mut Vec<(Vec<usize>, Vec<u64>)>,
                     seen_masks: &mut BTreeSet<Vec<u64>>| {
        let len = atoms_cycle.len();
        if len < 3 {
            return;
        }
        let mut mask = vec![0u64; n_words];
        for k in 0..len {
            let u = atoms_cycle[k];
            let v = atoms_cycle[(k + 1) % len];
            if let Some(&(_, le)) = adj[u].iter().find(|&&(nb, _)| nb == v) {
                mask[le / 64] |= 1u64 << (le % 64);
            } else {
                return;
            }
        }
        if seen_masks.insert(mask.clone()) {
            candidates.push((atoms_cycle, mask));
        }
    };

    // Shortest cycle for every removed edge
    for (skip_e, &(_bi, u0, v0)) in cov_bonds.iter().enumerate() {
        let mut parent: Vec<Option<usize>> = vec![None; n_atoms];
        let mut dist: Vec<usize> = vec![usize::MAX; n_atoms];
        let mut q = VecDeque::new();
        dist[u0] = 0;
        q.push_back(u0);
        while let Some(curr) = q.pop_front() {
            if curr == v0 {
                break;
            }
            if dist[curr] >= 24 {
                continue;
            }
            for &(nxt, le) in &adj[curr] {
                if le == skip_e {
                    continue;
                }
                if dist[nxt] == usize::MAX {
                    dist[nxt] = dist[curr] + 1;
                    parent[nxt] = Some(curr);
                    q.push_back(nxt);
                }
            }
        }
        if dist[v0] != usize::MAX {
            let mut path = Vec::with_capacity(dist[v0] + 1);
            let mut cur = v0;
            path.push(cur);
            while let Some(p) = parent[cur] {
                path.push(p);
                cur = p;
            }
            add_cycle(path, &mut candidates, &mut seen_masks);
        }
    }

    // Also run root-BFS shortest cycles to catch interior cycles in fused polycyclics (e.g. coronene / cubane)
    for root in 0..n_atoms {
        if adj[root].len() < 2 {
            continue;
        }
        let mut dist = vec![usize::MAX; n_atoms];
        let mut parent = vec![None; n_atoms];
        let mut q = VecDeque::new();
        dist[root] = 0;
        q.push_back(root);
        while let Some(u) = q.pop_front() {
            if dist[u] > 12 {
                continue;
            }
            for &(v, _le) in &adj[u] {
                if dist[v] == usize::MAX {
                    dist[v] = dist[u] + 1;
                    parent[v] = Some(u);
                    q.push_back(v);
                } else if parent[u] != Some(v) && u < v {
                    // Reconstruct paths u..root and v..root; verify lowest common ancestor == root
                    let mut pu = Vec::new();
                    let mut cu = u;
                    while cu != root {
                        pu.push(cu);
                        cu = parent[cu].unwrap();
                    }
                    let mut pv = Vec::new();
                    let mut cv = v;
                    let mut overlap = false;
                    while cv != root {
                        if pu.contains(&cv) {
                            overlap = true;
                            break;
                        }
                        pv.push(cv);
                        cv = parent[cv].unwrap();
                    }
                    if !overlap {
                        pu.reverse();
                        let mut cyc = vec![root];
                        cyc.extend(pu);
                        cyc.extend(pv);
                        add_cycle(cyc, &mut candidates, &mut seen_masks);
                    }
                }
            }
        }
    }

    candidates.sort_by_key(|(c, _)| c.len());

    // GF(2) Gaussian elimination to extract `rank` linearly independent smallest rings
    let mut basis: Vec<(usize, Vec<u64>)> = Vec::with_capacity(rank);
    let mut sssr = Vec::with_capacity(rank);

    for (cyc, mut row) in candidates {
        for &(pivot_bit, ref brow) in &basis {
            if (row[pivot_bit / 64] >> (pivot_bit % 64)) & 1 == 1 {
                for w in 0..n_words {
                    row[w] ^= brow[w];
                }
            }
        }
        let mut pivot = None;
        for (w, &word) in row.iter().enumerate().take(n_words) {
            if word != 0 {
                pivot = Some(w * 64 + word.trailing_zeros() as usize);
                break;
            }
        }
        if let Some(p) = pivot {
            basis.push((p, row));
            sssr.push(cyc);
            if sssr.len() == rank {
                break;
            }
        }
    }

    sssr
}

/// Return a boolean vector marking whether each bond in `mol.bonds` lies in at least one ring.
pub fn perceive_ring_bonds(mol: &Molecule) -> Vec<bool> {
    let mut in_ring = vec![false; mol.bonds.len()];
    let rings = perceive_sssr(mol);
    for r in &rings {
        for k in 0..r.len() {
            let u = r[k];
            let v = r[(k + 1) % r.len()];
            if let Some(bi) = mol.find_bond(u, v) {
                in_ring[bi] = true;
            }
        }
    }
    in_ring
}

// ============================================================================
// 2. Hückel 4n+2 Aromaticity Perception & Kekulization
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AromaticityResult {
    pub aromatic_atoms: Vec<bool>,
    pub aromatic_bonds: Vec<bool>,
}

/// Perceive Hückel $4n+2$ aromatic atoms and bonds across isolated and fused ring systems.
pub fn perceive_aromaticity(mol: &Molecule) -> AromaticityResult {
    let n = mol.atoms.len();
    let mut aromatic_atoms = vec![false; n];
    let mut aromatic_bonds = vec![false; mol.bonds.len()];

    let rings = perceive_sssr(mol);
    if rings.is_empty() {
        return AromaticityResult {
            aromatic_atoms,
            aromatic_bonds,
        };
    }

    let mut ring_member = vec![false; n];
    for r in &rings {
        for &u in r {
            ring_member[u] = true;
        }
    }

    // Also include pairwise fused perimeter cycles (up to 3 fused SSSR rings, e.g. indole, naphthalene, purine, anthracene)
    let mut test_cycles: Vec<Vec<usize>> = rings.clone();
    // Build edge masks of SSSR rings to find fused combinations that form a single simple cycle
    let m = mol.bonds.len();
    if m > 0 && rings.len() <= 24 {
        let ring_edges: Vec<BTreeSet<usize>> = rings
            .iter()
            .map(|r| {
                let mut s = BTreeSet::new();
                for k in 0..r.len() {
                    if let Some(bi) = mol.find_bond(r[k], r[(k + 1) % r.len()]) {
                        s.insert(bi);
                    }
                }
                s
            })
            .collect();

        // 2-ring fusions
        for i in 0..rings.len() {
            for j in (i + 1)..rings.len() {
                if ring_edges[i].intersection(&ring_edges[j]).next().is_some() {
                    let sym_diff: BTreeSet<usize> = ring_edges[i]
                        .symmetric_difference(&ring_edges[j])
                        .copied()
                        .collect();
                    if let Some(cyc) = edges_to_simple_cycle(mol, &sym_diff) {
                        if cyc.len() <= 18 {
                            test_cycles.push(cyc);
                        }
                    }
                    // 3-ring fusions (e.g. anthracene, phenanthrene, carbazole)
                    for rk in ring_edges.iter().skip(j + 1) {
                        if rk.intersection(&sym_diff).next().is_some() {
                            let sym3: BTreeSet<usize> =
                                sym_diff.symmetric_difference(rk).copied().collect();
                            if let Some(cyc3) = edges_to_simple_cycle(mol, &sym3) {
                                if cyc3.len() <= 22 {
                                    test_cycles.push(cyc3);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for cyc in &test_cycles {
        if is_huckel_cycle(mol, cyc) {
            for k in 0..cyc.len() {
                let u = cyc[k];
                let v = cyc[(k + 1) % cyc.len()];
                aromatic_atoms[u] = true;
                if let Some(bi) = mol.find_bond(u, v) {
                    aromatic_bonds[bi] = true;
                }
            }
        }
    }

    // Any bond between two aromatic atoms within an SSSR ring is also aromatic (e.g. central bond of naphthalene/indole)
    for r in &rings {
        if r.iter().all(|&u| aromatic_atoms[u]) {
            for k in 0..r.len() {
                if let Some(bi) = mol.find_bond(r[k], r[(k + 1) % r.len()]) {
                    aromatic_bonds[bi] = true;
                }
            }
        }
    }

    let _ = ring_member;
    AromaticityResult {
        aromatic_atoms,
        aromatic_bonds,
    }
}

fn edges_to_simple_cycle(mol: &Molecule, edges: &BTreeSet<usize>) -> Option<Vec<usize>> {
    if edges.len() < 3 {
        return None;
    }
    let mut adj: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &bi in edges {
        let b = &mol.bonds[bi];
        adj.entry(b.a).or_default().push(b.b);
        adj.entry(b.b).or_default().push(b.a);
    }
    if adj.values().any(|nbs| nbs.len() != 2) {
        return None;
    }
    let start = *adj.keys().next()?;
    let mut cycle = vec![start];
    let mut prev = start;
    let mut cur = adj[&start][0];
    while cur != start {
        if cycle.len() > edges.len() {
            return None;
        }
        cycle.push(cur);
        let nbs = &adj[&cur];
        let nxt = if nbs[0] == prev { nbs[1] } else { nbs[0] };
        prev = cur;
        cur = nxt;
    }
    if cycle.len() == edges.len() {
        Some(cycle)
    } else {
        None
    }
}

/// Evaluate whether `cycle` satisfies Hückel $4n+2$ $\pi$-electron conjugation.
fn is_huckel_cycle(mol: &Molecule, cycle: &[usize]) -> bool {
    let in_cyc: BTreeSet<usize> = cycle.iter().copied().collect();
    let mut pi_electrons: usize = 0;

    for &u in cycle {
        let a = &mol.atoms[u];
        // Only B, C, N, O, P, S, As, Se, Te can participate in organic Hückel aromaticity
        if !matches!(a.z, 5 | 6 | 7 | 8 | 15 | 16 | 33 | 34 | 52) {
            return false;
        }
        let nbs = mol.covalent_neighbors(u);
        let mut endo_double = 0usize;
        let mut endo_arom = 0usize;
        let mut exo_double_hetero = false;
        let mut exo_double_other = false;
        let mut triple = false;
        let mut single_bonds = 0usize;

        for &(v, bi) in &nbs {
            let b = &mol.bonds[bi];
            match b.kind {
                BondKind::Triple => triple = true,
                BondKind::Aromatic => {
                    if in_cyc.contains(&v) {
                        endo_arom += 1;
                    } else {
                        single_bonds += 1;
                    }
                }
                BondKind::Double => {
                    if in_cyc.contains(&v) {
                        endo_double += 1;
                    } else {
                        let zv = mol.atoms[v].z;
                        if matches!(zv, 7 | 8 | 16 | 34) {
                            exo_double_hetero = true;
                        } else {
                            exo_double_other = true;
                        }
                    }
                }
                _ => {
                    single_bonds += 1;
                }
            }
        }
        if triple || exo_double_other || endo_double > 1 {
            return false;
        }
        let total_deg = nbs.len() + mol.implicit_hydrogens(u) as usize;
        // Quaternary sp3 ring atom cannot be aromatic
        if total_deg >= 4 {
            return false;
        }

        // Already marked aromatic (e.g. from lowercase SMILES)
        if endo_arom > 0 || a.aromatic {
            let contrib = match (a.z, a.formal_charge) {
                (6, 0) => {
                    if exo_double_hetero {
                        0 // e.g. pyridone / uracil / caffeine c(=O)
                    } else {
                        1
                    }
                }
                (6, -1) => 2, // [cH-] cyclopentadienyl
                (6, 1) => 0,  // [cH+] tropylium
                (7, 0) => {
                    // Pyridine-like `n` (total_deg == 2) gives 1; pyrrole-like `[nH]` / `n(R)` (total_deg == 3) gives 2!
                    if endo_double == 1 {
                        1
                    } else if total_deg == 3 {
                        2
                    } else {
                        1
                    }
                }
                (7, 1) => 1,
                (7, -1) => 2,
                (8, 0) | (16, 0) | (34, 0) | (52, 0) => 2,
                (8, 1) | (16, 1) => 1,
                (5, 0) => 0,
                (15, 0) | (33, 0) => {
                    if total_deg == 3 && endo_double == 0 {
                        2
                    } else {
                        1
                    }
                }
                _ => return false,
            };
            pi_electrons += contrib;
            continue;
        }

        // Kekule bond representation
        let _ = single_bonds;
        let contrib = match (a.z, a.formal_charge) {
            (6, 0) => {
                if endo_double == 1 && !exo_double_hetero {
                    1
                } else if endo_double == 0 && exo_double_hetero && total_deg == 3 {
                    // Carbonyl carbon in pyridone / uracil / coumarin
                    0
                } else {
                    return false;
                }
            }
            (6, -1) => {
                if endo_double == 0 && !exo_double_hetero && total_deg <= 3 {
                    2
                } else {
                    return false;
                }
            }
            (6, 1) => {
                if endo_double == 0 && !exo_double_hetero && total_deg <= 3 {
                    0
                } else {
                    return false;
                }
            }
            (7, 0) => {
                if endo_double == 1 && total_deg == 2 {
                    1
                } else if endo_double == 0 && !exo_double_hetero && total_deg == 3 {
                    2
                } else {
                    return false;
                }
            }
            (7, 1) => {
                if endo_double == 1 && total_deg == 3 {
                    1
                } else {
                    return false;
                }
            }
            (7, -1) => {
                if endo_double == 0 && total_deg == 2 {
                    2
                } else {
                    return false;
                }
            }
            (8, 0) | (16, 0) | (34, 0) | (52, 0) => {
                if endo_double == 0 && !exo_double_hetero && total_deg == 2 {
                    2
                } else {
                    return false;
                }
            }
            (8, 1) | (16, 1) => {
                if endo_double == 1 && total_deg == 2 {
                    1
                } else {
                    return false;
                }
            }
            (5, 0) => {
                if endo_double == 0 && total_deg == 3 {
                    0
                } else {
                    return false;
                }
            }
            (15, 0) | (33, 0) => {
                if endo_double == 1 && total_deg == 2 {
                    1
                } else if endo_double == 0 && total_deg == 3 {
                    2
                } else {
                    return false;
                }
            }
            _ => return false,
        };
        pi_electrons += contrib;
    }

    pi_electrons >= 2 && pi_electrons % 4 == 2
}

/// Convert `BondKind::Aromatic` bonds in `mol` into a valid alternating Kekulé `Single`/`Double` assignment
/// while keeping `atom.aromatic` updated via Hückel perception.
pub fn kekulize(mol: &mut Molecule) -> Result<(), ChemError> {
    let arom_bond_indices: Vec<usize> = mol
        .bonds
        .iter()
        .enumerate()
        .filter_map(|(bi, b)| (b.kind == BondKind::Aromatic).then_some(bi))
        .collect();

    if arom_bond_indices.is_empty() {
        let perc = perceive_aromaticity(mol);
        for (i, is_a) in perc.aromatic_atoms.into_iter().enumerate() {
            mol.atoms[i].aromatic = is_a;
        }
        return Ok(());
    }

    let n = mol.atoms.len();
    let mut in_arom_sub = vec![false; n];
    for &bi in &arom_bond_indices {
        in_arom_sub[mol.bonds[bi].a] = true;
        in_arom_sub[mol.bonds[bi].b] = true;
    }

    // Determine for each atom in the aromatic bond subgraph whether it needs ONE double bond from `arom_bond_indices`
    // (`Some(true)` = must have 1 double bond, `Some(false)` = must have 0 double bonds, `None` = ambiguous, try false/true).
    let mut req: Vec<Option<bool>> = vec![Some(false); n];
    for u in 0..n {
        if !in_arom_sub[u] {
            continue;
        }
        let a = &mol.atoms[u];
        // Count non-aromatic covalent bond orders + explicit H
        let mut non_arom_order = 0usize;
        let mut has_exo_double = false;
        let mut arom_deg = 0usize;
        for &(_v, bi) in &mol.covalent_neighbors(u) {
            match mol.bonds[bi].kind {
                BondKind::Aromatic => arom_deg += 1,
                BondKind::Double => {
                    non_arom_order += 2;
                    has_exo_double = true;
                }
                BondKind::Triple => non_arom_order += 3,
                other => non_arom_order += other.valence_order() as usize,
            }
        }
        let eh = a.explicit_hydrogens.map(|h| h as usize);

        req[u] = match (a.z, a.formal_charge) {
            (6, 0) => {
                if has_exo_double {
                    Some(false) // e.g. c(=O) in pyridone/caffeine: 2 arom singles + 1 exo double = 4
                } else {
                    Some(true) // standard aromatic carbon needs 1 double bond
                }
            }
            (6, -1) | (6, 1) => {
                if eh == Some(0) && arom_deg + non_arom_order == 2 {
                    Some(true)
                } else {
                    Some(false)
                }
            }
            (7, 0) => {
                if has_exo_double {
                    Some(false)
                } else if let Some(h) = eh {
                    // e.g. [nH] in pyrrole: arom_deg(2) + h(1) == 3 -> 0 double bonds!
                    if arom_deg + non_arom_order + h >= 3 {
                        Some(false)
                    } else {
                        Some(true)
                    }
                } else if arom_deg + non_arom_order >= 3 {
                    // e.g. n(C) in caffeine/N-methylpyrrole/indole junction: 3 bonds -> lone pair, 0 double bonds!
                    Some(false)
                } else {
                    // e.g. pyridine `n` with 2 aromatic bonds and no explicit H -> needs 1 double bond!
                    Some(true)
                }
            }
            (7, 1) => {
                // [n+] in pyridinium (3 bonds -> needs 1 double bond = 4) vs N-oxide [n+]([O-]) (3 bonds -> needs 1 double = 4)
                let h = eh.unwrap_or(0);
                if arom_deg + non_arom_order + h <= 3 {
                    Some(true)
                } else {
                    Some(false)
                }
            }
            (7, -1) => Some(false),
            (8, 0) | (16, 0) | (34, 0) | (52, 0) => {
                if has_exo_double {
                    None // e.g. thiophene S-oxide / dioxide
                } else {
                    Some(false)
                }
            }
            (8, 1) | (16, 1) => Some(true),
            (5, 0) => Some(false),
            (15, 0) | (33, 0) => {
                let h = eh.unwrap_or(0);
                if arom_deg + non_arom_order + h >= 3 {
                    Some(false)
                } else {
                    Some(true)
                }
            }
            _ => None,
        };
    }

    // Solve component-by-component on the aromatic bond subgraph via perfect-matching backtracking
    let mut arom_adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for (local_idx, &bi) in arom_bond_indices.iter().enumerate() {
        let b = &mol.bonds[bi];
        arom_adj[b.a].push((b.b, local_idx));
        arom_adj[b.b].push((b.a, local_idx));
    }

    let mut chosen_double = vec![false; arom_bond_indices.len()];
    let mut visited = vec![false; n];

    for start in 0..n {
        if !in_arom_sub[start] || visited[start] {
            continue;
        }
        let mut comp = Vec::new();
        let mut q = VecDeque::new();
        visited[start] = true;
        q.push_back(start);
        while let Some(u) = q.pop_front() {
            comp.push(u);
            for &(v, _) in &arom_adj[u] {
                if !visited[v] {
                    visited[v] = true;
                    q.push_back(v);
                }
            }
        }

        let mut matched = vec![false; n];
        if !solve_kekule_component(0, &comp, &arom_adj, &req, &mut matched, &mut chosen_double) {
            return Err(ChemError::new(
                "Kekulization failed: no valid alternating pi-bond assignment for aromatic system",
            ));
        }
    }

    // Before changing bond kinds, preserve implicit H counts on unbracketed atoms so valence doesn't drift
    // Actually, once kekulized, standard implicit_hydrogens(u) matches exact Kekule bond orders!
    for (local_idx, &bi) in arom_bond_indices.iter().enumerate() {
        mol.bonds[bi].kind = if chosen_double[local_idx] {
            BondKind::Double
        } else {
            BondKind::Single
        };
        mol.atoms[mol.bonds[bi].a].aromatic = true;
        mol.atoms[mol.bonds[bi].b].aromatic = true;
    }

    Ok(())
}

fn solve_kekule_component(
    pos: usize,
    comp: &[usize],
    arom_adj: &[Vec<(usize, usize)>],
    req: &[Option<bool>],
    matched: &mut [bool],
    chosen_double: &mut [bool],
) -> bool {
    if pos == comp.len() {
        return true;
    }
    let u = comp[pos];
    if matched[u] {
        return solve_kekule_component(pos + 1, comp, arom_adj, req, matched, chosen_double);
    }
    match req[u] {
        Some(false) => solve_kekule_component(pos + 1, comp, arom_adj, req, matched, chosen_double),
        Some(true) => {
            for &(v, edge_idx) in &arom_adj[u] {
                if matched[v] || req[v] == Some(false) {
                    continue;
                }
                matched[u] = true;
                matched[v] = true;
                chosen_double[edge_idx] = true;
                if solve_kekule_component(pos + 1, comp, arom_adj, req, matched, chosen_double) {
                    return true;
                }
                matched[u] = false;
                matched[v] = false;
                chosen_double[edge_idx] = false;
            }
            false
        }
        None => {
            // Try unmatched first, then matched
            if solve_kekule_component(pos + 1, comp, arom_adj, req, matched, chosen_double) {
                return true;
            }
            for &(v, edge_idx) in &arom_adj[u] {
                if matched[v] || req[v] == Some(false) {
                    continue;
                }
                matched[u] = true;
                matched[v] = true;
                chosen_double[edge_idx] = true;
                if solve_kekule_component(pos + 1, comp, arom_adj, req, matched, chosen_double) {
                    return true;
                }
                matched[u] = false;
                matched[v] = false;
                chosen_double[edge_idx] = false;
            }
            false
        }
    }
}

// ============================================================================
// 3. Cahn–Ingold–Prelog (CIP) Stereochemistry: (R)/(S) & (E)/(Z)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CipAtomDescriptor {
    R,
    S,
    PseudoR,
    PseudoS,
}

impl CipAtomDescriptor {
    pub fn as_str(self) -> &'static str {
        match self {
            CipAtomDescriptor::R => "R",
            CipAtomDescriptor::S => "S",
            CipAtomDescriptor::PseudoR => "r",
            CipAtomDescriptor::PseudoS => "s",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CipBondDescriptor {
    E,
    Z,
}

impl CipBondDescriptor {
    pub fn as_str(self) -> &'static str {
        match self {
            CipBondDescriptor::E => "E",
            CipBondDescriptor::Z => "Z",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CipResult {
    pub atoms: BTreeMap<usize, CipAtomDescriptor>,
    pub bonds: BTreeMap<usize, CipBondDescriptor>,
}

/// Internal CIP graph with mancude fractional atomic numbers so Kekulé resonance forms rank identically.
pub(crate) struct CipGraph {
    near: Vec<Vec<(usize, u8)>>,
    implicit_h: Vec<u8>,
    z: Vec<f64>,
    mass: Vec<f64>,
    mancude_dup: Vec<Option<(f64, f64)>>,
    ring_bonds: Vec<bool>,
}

impl CipGraph {
    fn from_mol(mol: &Molecule) -> Self {
        let n = mol.atoms.len();
        let arom = perceive_aromaticity(mol);
        let ring_bonds = perceive_ring_bonds(mol);
        let implicit_h: Vec<u8> = (0..n).map(|i| mol.implicit_hydrogens(i)).collect();
        let z: Vec<f64> = mol.atoms.iter().map(|a| a.z as f64).collect();
        let mass: Vec<f64> = mol
            .atoms
            .iter()
            .map(|a| {
                if let Some(iso) = a.isotope {
                    iso as f64
                } else {
                    // Slight offset from integer mass so explicit natural isotope (e.g. [12C]) differs if needed,
                    // while higher isotopes ([13C], [2H]) strictly beat natural abundance.
                    a.mass_number() as f64 + 0.01
                }
            })
            .collect();

        let mut near: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n];
        for (bi, b) in mol.bonds.iter().enumerate() {
            if matches!(
                b.kind,
                BondKind::Dative | BondKind::Hydrogen | BondKind::Ionic
            ) {
                continue;
            }
            if mol.atoms[b.a].z == 0 || mol.atoms[b.b].z == 0 {
                continue;
            }
            // In a mancude aromatic ring, treat aromatic ring bonds as order 1 + fractional mancude duplicate
            let is_arom_ring_bond =
                ring_bonds[bi] && arom.aromatic_atoms[b.a] && arom.aromatic_atoms[b.b];
            let order = if is_arom_ring_bond {
                1
            } else {
                b.kind.order_u8()
            };
            near[b.a].push((b.b, order));
            near[b.b].push((b.a, order));
        }

        let mut mancude_dup = vec![None; n];
        for (i, m_slot) in mancude_dup.iter_mut().enumerate() {
            if !arom.aromatic_atoms[i] {
                continue;
            }
            let mut arom_ring_nbs = Vec::new();
            for (bi, b) in mol.bonds.iter().enumerate() {
                if !ring_bonds[bi] {
                    continue;
                }
                if let Some(nb) = b.other(i) {
                    if arom.aromatic_atoms[nb] {
                        arom_ring_nbs.push(nb);
                    }
                }
            }
            if !arom_ring_nbs.is_empty() {
                let len = arom_ring_nbs.len() as f64;
                let mean_z: f64 = arom_ring_nbs.iter().map(|&j| z[j]).sum::<f64>() / len;
                let mean_m: f64 = arom_ring_nbs.iter().map(|&j| mass[j]).sum::<f64>() / len;
                *m_slot = Some((mean_z, mean_m));
            }
        }

        Self {
            near,
            implicit_h,
            z,
            mass,
            mancude_dup,
            ring_bonds,
        }
    }
}

#[derive(Clone)]
struct DigraphNode {
    atom: usize,
    is_dup: bool,
    is_h: bool,
    is_lone_pair: bool,
    ring_root_dist: usize, // for Rule 1b: distance from root of duplicated ring-closure atom (smaller = higher priority)
    mancude_key: Option<(f64, f64)>,
    path: Vec<usize>,
}

impl DigraphNode {
    fn key(&self, g: &CipGraph, rule: u8, aux_map: &BTreeMap<usize, CipAtomDescriptor>) -> f64 {
        if self.is_lone_pair {
            return 0.0;
        }
        match rule {
            0 => {
                // Rule 1a: atomic number
                if self.is_h {
                    1.0
                } else if let Some((mz, _)) = self.mancude_key {
                    mz
                } else {
                    g.z[self.atom]
                }
            }
            1 => {
                // Rule 1b: duplicate ring-closure root distance (closer to root => higher priority => larger key)
                if self.ring_root_dist > 0 {
                    1000.0 - (self.ring_root_dist as f64)
                } else {
                    0.0
                }
            }
            2 => {
                // Rule 2: atomic mass / isotope
                if self.is_h {
                    1.01
                } else if self.is_dup {
                    0.0
                } else {
                    g.mass[self.atom]
                }
            }
            3 => {
                // Rule 4/5: auxiliary stereodescriptors (R > S for Rule 5)
                if self.is_dup || self.is_h {
                    return 0.0;
                }
                match aux_map.get(&self.atom) {
                    Some(CipAtomDescriptor::R) => 2.0,
                    Some(CipAtomDescriptor::S) => 1.0,
                    _ => 0.0,
                }
            }
            _ => 0.0,
        }
    }
}

fn expand_children(
    g: &CipGraph,
    node: &DigraphNode,
    rule: u8,
    aux_map: &BTreeMap<usize, CipAtomDescriptor>,
) -> Vec<DigraphNode> {
    if node.is_dup || node.is_h || node.is_lone_pair {
        return Vec::new();
    }
    let from = if node.path.len() >= 2 {
        Some(node.path[node.path.len() - 2])
    } else {
        None
    };
    let mut out = Vec::new();
    for &(to, order) in &g.near[node.atom] {
        let make_dup = |target: usize, rdist: usize| DigraphNode {
            atom: target,
            is_dup: true,
            is_h: false,
            is_lone_pair: false,
            ring_root_dist: rdist,
            mancude_key: None,
            path: {
                let mut p = node.path.clone();
                p.push(target);
                p
            },
        };
        if Some(to) == from {
            // Multiple-bond back-duplicates (except back to origin root)
            if node.path.first().copied() != Some(to) {
                for _ in 1..order {
                    out.push(make_dup(to, 0));
                }
            }
            continue;
        }
        if let Some(pos) = node.path.iter().position(|&x| x == to) {
            // Ring closure duplicate! pos + 1 is 1-based depth from root for Rule 1b
            out.push(make_dup(to, pos + 1));
        } else {
            let mut p = node.path.clone();
            p.push(to);
            out.push(DigraphNode {
                atom: to,
                is_dup: false,
                is_h: false,
                is_lone_pair: false,
                ring_root_dist: 0,
                mancude_key: None,
                path: p,
            });
        }
        for _ in 1..order {
            out.push(make_dup(to, 0));
        }
    }
    if let Some(mc) = g.mancude_dup[node.atom] {
        out.push(DigraphNode {
            atom: node.atom,
            is_dup: true,
            is_h: false,
            is_lone_pair: false,
            ring_root_dist: 0,
            mancude_key: Some(mc),
            path: node.path.clone(),
        });
    }
    for _ in 0..g.implicit_h[node.atom] {
        out.push(DigraphNode {
            atom: node.atom,
            is_dup: false,
            is_h: true,
            is_lone_pair: false,
            ring_root_dist: 0,
            mancude_key: None,
            path: node.path.clone(),
        });
    }
    out.sort_by(|a, b| {
        b.key(g, rule, aux_map)
            .partial_cmp(&a.key(g, rule, aux_map))
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                b.key(g, 0, aux_map)
                    .partial_cmp(&a.key(g, 0, aux_map))
                    .unwrap_or(Ordering::Equal)
            })
    });
    out
}

/// Compare two ligands `lig_a` and `lig_b` attached to `root` (`usize::MAX` = implicit H, `usize::MAX - 1` = lone pair).
/// Returns `Ordering::Greater` when `lig_a` has HIGHER CIP priority than `lig_b`.
fn compare_cip_ligands(
    g: &CipGraph,
    root: usize,
    lig_a: usize,
    lig_b: usize,
    aux_map: &BTreeMap<usize, CipAtomDescriptor>,
    max_rule: u8,
) -> Ordering {
    if lig_a == lig_b {
        return Ordering::Equal;
    }
    let make_start = |lig: usize| -> DigraphNode {
        if lig == usize::MAX - 1 {
            DigraphNode {
                atom: root,
                is_dup: false,
                is_h: false,
                is_lone_pair: true,
                ring_root_dist: 0,
                mancude_key: None,
                path: vec![root],
            }
        } else if lig == usize::MAX {
            DigraphNode {
                atom: root,
                is_dup: false,
                is_h: true,
                is_lone_pair: false,
                ring_root_dist: 0,
                mancude_key: None,
                path: vec![root],
            }
        } else {
            DigraphNode {
                atom: lig,
                is_dup: false,
                is_h: false,
                is_lone_pair: false,
                ring_root_dist: 0,
                mancude_key: None,
                path: vec![root, lig],
            }
        }
    };

    for rule in 0..=max_rule {
        let mut sphere_a: Vec<Vec<DigraphNode>> = vec![vec![make_start(lig_a)]];
        let mut sphere_b: Vec<Vec<DigraphNode>> = vec![vec![make_start(lig_b)]];
        let mut node_budget = 0usize;

        for _depth in 0..24 {
            if sphere_a.is_empty() && sphere_b.is_empty() {
                break;
            }
            let keys_a: Vec<f64> = sphere_a
                .iter()
                .flat_map(|s| s.iter().map(|n| n.key(g, rule, aux_map)))
                .collect();
            let keys_b: Vec<f64> = sphere_b
                .iter()
                .flat_map(|s| s.iter().map(|n| n.key(g, rule, aux_map)))
                .collect();
            let max_len = keys_a.len().max(keys_b.len());
            for k in 0..max_len {
                let ka = keys_a.get(k).copied().unwrap_or(-1.0);
                let kb = keys_b.get(k).copied().unwrap_or(-1.0);
                if (ka - kb).abs() > 1e-7 {
                    return ka.partial_cmp(&kb).unwrap_or(Ordering::Equal);
                }
            }

            let step_sphere = |sphere: Vec<Vec<DigraphNode>>| -> Vec<Vec<DigraphNode>> {
                let mut next_sets = Vec::new();
                for group in sphere {
                    let mut with_kids: Vec<(DigraphNode, Vec<DigraphNode>)> = group
                        .into_iter()
                        .map(|n| {
                            let kids = expand_children(g, &n, rule, aux_map);
                            (n, kids)
                        })
                        .collect();
                    with_kids.sort_by(|(na, ca), (nb, cb)| {
                        nb.key(g, rule, aux_map)
                            .partial_cmp(&na.key(g, rule, aux_map))
                            .unwrap_or(Ordering::Equal)
                            .then_with(|| {
                                let ml = ca.len().max(cb.len());
                                for idx in 0..ml {
                                    let va = ca
                                        .get(idx)
                                        .map(|x| x.key(g, rule, aux_map))
                                        .unwrap_or(-1.0);
                                    let vb = cb
                                        .get(idx)
                                        .map(|x| x.key(g, rule, aux_map))
                                        .unwrap_or(-1.0);
                                    if (vb - va).abs() > 1e-7 {
                                        return vb.partial_cmp(&va).unwrap_or(Ordering::Equal);
                                    }
                                }
                                Ordering::Equal
                            })
                    });
                    for (_n, kids) in with_kids {
                        if !kids.is_empty() {
                            next_sets.push(kids);
                        }
                    }
                }
                next_sets
            };

            sphere_a = step_sphere(sphere_a);
            sphere_b = step_sphere(sphere_b);
            node_budget += sphere_a.iter().map(|s| s.len()).sum::<usize>()
                + sphere_b.iter().map(|s| s.len()).sum::<usize>();
            if node_budget > 4096 {
                break;
            }
        }
    }
    Ordering::Equal
}

/// Perceive all CIP `(R)/(S)`/`(r)/(s)` tetrahedral stereodescriptors and `(E)/(Z)` alkene stereodescriptors on `mol`.
pub fn perceive_cip(mol: &Molecule) -> CipResult {
    let mut result = CipResult::default();
    if mol.atoms.is_empty() {
        return result;
    }
    let g = CipGraph::from_mol(mol);

    // Pass 1: Constitutional + isotopic CIP (Rules 1a, 1b, 2) for tetrahedral centers
    let empty_aux = BTreeMap::new();
    for i in 0..mol.atoms.len() {
        if let Some(desc) = evaluate_tetrahedral_center(mol, &g, i, &empty_aux, 2) {
            result.atoms.insert(i, desc);
        }
    }

    // Pass 2: Re-evaluate unresolved centers that depend on auxiliary (R)/(S) descriptors (Rule 4c / Rule 5: pseudoasymmetric r/s or R/S)
    let pass1_aux = result.atoms.clone();
    if !pass1_aux.is_empty() {
        for i in 0..mol.atoms.len() {
            if result.atoms.contains_key(&i) {
                continue;
            }
            if let Some(desc) = evaluate_tetrahedral_center(mol, &g, i, &pass1_aux, 3) {
                result.atoms.insert(i, desc);
            }
        }
    }

    // Pass 3: (E)/(Z) double bonds & odd cumulenes
    evaluate_double_bonds_cip(mol, &g, &result.atoms, &mut result.bonds);

    result
}

/// Determine R-sense pyramid `[a0, a1, a2, a3]` around atom `center` where `a0 -> a1 -> a2` winds CLOCKWISE
/// when viewed with `a3` pointing away from the viewer:
/// 1. From 3D coordinates if non-planar (`z_coord.is_some()` and depth spread >= 0.15 Å).
/// 2. From 2D stereo wedges/hashes (`WedgeUp`, `HashDown`, `HollowWedge`, `Bold`, `Dashed`).
/// 3. From stored `Atom::parity` (`TetrahedralParity` from SMILES `@`/`@@`, InChI `/t`, or MOL parity).
pub(crate) fn perceive_center_pyramid(
    mol: &Molecule,
    g: &CipGraph,
    center: usize,
) -> Option<[usize; 4]> {
    let a = &mol.atoms[center];
    // Any wavy bond incident on `center` marks unknown/racemic stereochemistry
    for b in &mol.bonds {
        if (b.a == center || b.b == center) && b.kind == BondKind::Wavy {
            return None;
        }
    }

    let nbs: Vec<usize> = mol
        .covalent_neighbors(center)
        .into_iter()
        .map(|(nb, _)| nb)
        .collect();
    if nbs.len() < 3 || nbs.len() > 4 {
        return None;
    }
    let double_count = mol
        .covalent_neighbors(center)
        .into_iter()
        .filter(|&(_, bi)| mol.bonds[bi].kind == BondKind::Double)
        .count();
    let triple_or_arom = mol
        .covalent_neighbors(center)
        .into_iter()
        .any(|(_, bi)| matches!(mol.bonds[bi].kind, BondKind::Triple | BondKind::Aromatic));
    if triple_or_arom {
        return None;
    }

    // Check allowed tetrahedral / sulfinyl / phosphine stereocenter types
    let ih = g.implicit_h[center];
    let has_lone_pair_fourth = match (a.z, a.formal_charge, nbs.len(), double_count, ih) {
        (6, 0, 4, 0, 0) | (6, 0, 3, 0, 1) => false,
        (14, 0, 4, 0, 0) | (14, 0, 3, 0, 1) => false,
        (7, 1, 4, 0, 0) | (7, 1, 3, 0, 1) => false,
        (7, 0, 4, 1, 0) => false,
        (5, -1, 4, 0, 0) => false,
        (15, 1, 4, 0, 0) | (15, 0, 4, 1, 0) => false,
        (15, 0, 3, 0, 0) => true, // chiral phosphine
        (16, 0, 4, 2, 0) | (16, 1, 4, 1, 0) => false,
        (16, 0, 3, 1, 0) | (16, 1, 3, 0, 0) => true, // chiral sulfoxide / sulfonium!
        _ => return None,
    };

    let virtual_fourth = if has_lone_pair_fourth {
        usize::MAX - 1
    } else {
        usize::MAX
    };

    // 1. Check 2D stereo wedges/hashes first if drawn on bonds incident to `center`
    if let Some(pyr) = pyramid_from_2d_wedges(mol, g, center, &nbs, virtual_fourth) {
        return Some(pyr);
    }

    // 2. Check 3D coordinates if non-coplanar
    if let Some(zc) = a.z_coord {
        if nbs.iter().all(|&j| mol.atoms[j].z_coord.is_some())
            && nbs
                .iter()
                .any(|&j| (mol.atoms[j].z_coord.unwrap() - zc).abs() > 0.15)
        {
            if let Some(pyr) = pyramid_from_3d_points(mol, center, &nbs, virtual_fourth) {
                return Some(pyr);
            }
        }
    }

    // 3. Fall back to stored `TetrahedralParity` (e.g. parsed from SMILES `@`/`@@` or InChI `/t` before clean_2d)
    if let Some(TetrahedralParity {
        neighbors,
        clockwise,
    }) = a.parity
    {
        // Verify stored parity neighbor set matches current neighbors
        let expected_set: BTreeSet<usize> = if nbs.len() == 4 {
            nbs.iter().copied().collect()
        } else {
            let mut s: BTreeSet<usize> = nbs.iter().copied().collect();
            s.insert(virtual_fourth);
            s
        };
        let mut actual_mapped = neighbors;
        for slot in &mut actual_mapped {
            if *slot == usize::MAX && has_lone_pair_fourth {
                *slot = virtual_fourth;
            }
        }
        let actual_set: BTreeSet<usize> = actual_mapped.iter().copied().collect();
        if actual_set == expected_set {
            // As proved: in OpenSMILES, `@@` (clockwise == true) already has R-sense `[n0, n1, n2, n3]`;
            // `@` (clockwise == false) swaps `n1` and `n2` to convert to R-sense!
            return Some(if clockwise {
                actual_mapped
            } else {
                [
                    actual_mapped[0],
                    actual_mapped[2],
                    actual_mapped[1],
                    actual_mapped[3],
                ]
            });
        }
    }

    None
}

fn evaluate_tetrahedral_center(
    mol: &Molecule,
    g: &CipGraph,
    center: usize,
    aux_map: &BTreeMap<usize, CipAtomDescriptor>,
    max_rule: u8,
) -> Option<CipAtomDescriptor> {
    let around = perceive_center_pyramid(mol, g, center)?;
    let mut ranked = around;

    // Sort `ranked` highest CIP priority first
    ranked.sort_by(|&a, &b| compare_cip_ligands(g, center, b, a, aux_map, max_rule));

    // Check that all 4 ligands are strictly distinguishable
    for k in 0..3 {
        if compare_cip_ligands(g, center, ranked[k], ranked[k + 1], aux_map, max_rule)
            == Ordering::Equal
        {
            return None;
        }
    }

    // Check if distinction ONLY happened via Rule 5 (auxiliary R vs S on constitutionally identical enantiomorphic branches => pseudoasymmetric r/s!)
    let mut pseudo = false;
    if max_rule >= 3 {
        for k in 0..3 {
            if compare_cip_ligands(g, center, ranked[k], ranked[k + 1], aux_map, 2)
                == Ordering::Equal
            {
                pseudo = true;
            }
        }
    }

    let odd = permutation_is_odd(&around, &ranked)?;
    Some(match (pseudo, odd) {
        (false, false) => CipAtomDescriptor::R,
        (false, true) => CipAtomDescriptor::S,
        (true, false) => CipAtomDescriptor::PseudoR,
        (true, true) => CipAtomDescriptor::PseudoS,
    })
}

/// Compute parity of permutation mapping `a` to `b` (`Some(true)` = odd number of swaps, `Some(false)` = even).
pub(crate) fn permutation_is_odd(a: &[usize; 4], b: &[usize; 4]) -> Option<bool> {
    let mut perm = [0usize; 4];
    for i in 0..4 {
        perm[i] = b.iter().position(|&x| x == a[i])?;
    }
    let mut inversions = 0usize;
    for i in 0..4 {
        for j in (i + 1)..4 {
            if perm[i] > perm[j] {
                inversions += 1;
            }
        }
    }
    Some(inversions % 2 == 1)
}

fn pyramid_from_3d_points(
    mol: &Molecule,
    center: usize,
    nbs: &[usize],
    virtual_fourth: usize,
) -> Option<[usize; 4]> {
    let c = &mol.atoms[center];
    let cz = c.z_coord.unwrap_or(0.0);
    let mut u: Vec<[f64; 3]> = Vec::with_capacity(4);
    for &j in nbs {
        let aj = &mol.atoms[j];
        let dx = aj.x - c.x;
        let dy = aj.y - c.y;
        let dz = aj.z_coord.unwrap_or(0.0) - cz;
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len < 1e-6 {
            return None;
        }
        u.push([dx / len, dy / len, dz / len]);
    }
    let ids = if nbs.len() == 4 {
        [nbs[0], nbs[1], nbs[2], nbs[3]]
    } else {
        let sx = -(u[0][0] + u[1][0] + u[2][0]);
        let sy = -(u[0][1] + u[1][1] + u[2][1]);
        let sz = -(u[0][2] + u[1][2] + u[2][2]);
        let len = (sx * sx + sy * sy + sz * sz).sqrt().max(1e-6);
        u.push([sx / len, sy / len, sz / len]);
        [nbs[0], nbs[1], nbs[2], virtual_fourth]
    };

    let vol = signed_tetra_volume(u[0], u[1], u[2], u[3]);
    if vol.abs() < 1e-3 {
        return None;
    }
    Some(if vol < 0.0 {
        ids
    } else {
        [ids[0], ids[2], ids[1], ids[3]]
    })
}

fn signed_tetra_volume(u0: [f64; 3], u1: [f64; 3], u2: [f64; 3], u3: [f64; 3]) -> f64 {
    let a = [u0[0] - u3[0], u0[1] - u3[1], u0[2] - u3[2]];
    let b = [u1[0] - u3[0], u1[1] - u3[1], u1[2] - u3[2]];
    let c = [u2[0] - u3[0], u2[1] - u3[1], u2[2] - u3[2]];
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

/// Reconstruct 3D R-sense pyramid from 2D coordinates + wedge/hash stereo bonds (adapted from Octet-Draw `cip.ts`).
fn pyramid_from_2d_wedges(
    mol: &Molecule,
    g: &CipGraph,
    center: usize,
    nbs: &[usize],
    virtual_fourth: usize,
) -> Option<[usize; 4]> {
    // Direction (+1 = toward viewer, -1 = away from viewer, 0 = in plane) from `center` to `j`
    let dir_from_center = |j: usize, allow_reverse: bool| -> i8 {
        let Some(bi) = mol.find_bond(center, j) else {
            return 0;
        };
        let b = &mol.bonds[bi];
        if b.a == center {
            match b.kind {
                BondKind::WedgeUp | BondKind::HollowWedge => 1,
                BondKind::HashDown => -1,
                BondKind::Dashed if allow_reverse => -1,
                BondKind::Bold if allow_reverse && !g.ring_bonds[bi] => 1,
                _ => 0,
            }
        } else if allow_reverse {
            match b.kind {
                BondKind::WedgeUp | BondKind::HollowWedge => -1,
                BondKind::HashDown => 1,
                BondKind::Dashed => -1,
                BondKind::Bold if !g.ring_bonds[bi] => 1,
                _ => 0,
            }
        } else {
            0
        }
    };

    let has_outgoing = nbs.iter().any(|&j| dir_from_center(j, false) != 0);
    let allow_rev = !has_outgoing && virtual_fourth == usize::MAX;
    let d_raw: Vec<i8> = nbs.iter().map(|&j| dir_from_center(j, allow_rev)).collect();
    if d_raw.iter().all(|&x| x == 0) {
        return None;
    }

    let is_pure_h = |j: usize| mol.atoms[j].z == 1 && mol.atoms[j].isotope.is_none();
    let mut order: Vec<usize> = nbs.to_vec();
    order.sort_by_key(|&j| if is_pure_h(j) { 1_000_000 + j } else { j });

    let cx = mol.atoms[center].x;
    let cy = mol.atoms[center].y;
    let mut w: Vec<[f64; 2]> = Vec::with_capacity(order.len());
    for &j in &order {
        let dx = mol.atoms[j].x - cx;
        let dy = mol.atoms[j].y - cy;
        let len = dx.hypot(dy);
        if len < 1e-5 {
            return None;
        }
        w.push([dx / len, dy / len]);
    }
    let d: Vec<i8> = order
        .iter()
        .map(|&j| dir_from_center(j, allow_rev))
        .collect();

    // Helper functions matching Indigo / Octet-Draw 2D stereocenter geometry
    let cross2 = |a: [f64; 2], b: [f64; 2]| a[0] * b[1] - a[1] * b[0];
    let dot2 = |a: [f64; 2], b: [f64; 2]| a[0] * b[0] + a[1] * b[1];
    let sector = |v1: [f64; 2], v2: [f64; 2], s: [f64; 2]| -> Option<u8> {
        let c1 = cross2(v1, s);
        let c2 = cross2(v2, s);
        let c12 = cross2(v1, v2);
        if c1.abs() < 1e-4 && dot2(v1, s) > 0.0 {
            return None;
        }
        if c2.abs() < 1e-4 && dot2(v2, s) > 0.0 {
            return None;
        }
        if c12 > 1e-4 {
            if c1 > 0.0 && c2 < 0.0 {
                Some(1)
            } else if c1 < 0.0 && c2 > 0.0 {
                Some(2)
            } else {
                Some(4)
            }
        } else if c12 < -1e-4 {
            if c1 < 0.0 && c2 > 0.0 {
                Some(1)
            } else if c1 > 0.0 && c2 < 0.0 {
                Some(2)
            } else {
                Some(8)
            }
        } else {
            // v1 and v2 are 180 deg apart
            if c1 > 0.0 {
                Some(4)
            } else {
                Some(8)
            }
        }
    };
    let turn = |v1: [f64; 2], v2: [f64; 2], v3: [f64; 2]| -> Option<i8> {
        if dot2(v1, v2) > 0.997 || dot2(v1, v3) > 0.997 || dot2(v2, v3) > 0.997 {
            return None;
        }
        let r = (v1[0] - v3[0]) * (v2[1] - v3[1]) - (v1[1] - v3[1]) * (v2[0] - v3[0]);
        if r > 1e-4 {
            Some(1)
        } else if r < -1e-4 {
            Some(-1)
        } else {
            None
        }
    };

    if order.len() == 4 {
        // Explicit 4 drawn substituents: construct 3D vectors using z = d[k] (or opposite for trans-tetrahedral pairs)
        let pure_h_cnt = order.iter().filter(|&&j| is_pure_h(j)).count();
        if pure_h_cnt > 1 {
            return None;
        }
        let m1 = d.iter().position(|&x| x != 0)?;
        let mut m2 = None;
        let mut s1 = 0;
        let mut s2 = 0;
        for (o, a, b) in [(1usize, 2usize, 3usize), (2, 1, 3), (3, 1, 2)] {
            let x = sector(w[m1], w[(m1 + o) % 4], w[(m1 + a) % 4])?;
            let y = sector(w[m1], w[(m1 + o) % 4], w[(m1 + b) % 4])?;
            if x + y == 3 || x + y == 12 {
                m2 = Some((m1 + o) % 4);
                s1 = (m1 + a) % 4;
                s2 = (m1 + b) % 4;
                break;
            }
        }
        if let Some(m2_idx) = m2 {
            if d[m2_idx] == -d[m1] || d[s1] == d[m1] || d[s2] == d[m1] {
                // Fallback to direct 3D lift if e.g. explicit H has hash and side chain has wedge
                let mut pts = [[0.0; 3]; 4];
                for k in 0..4 {
                    pts[k] = [w[k][0], w[k][1], d[k] as f64 * 0.8];
                }
                let vol = signed_tetra_volume(pts[0], pts[1], pts[2], pts[3]);
                if vol.abs() < 1e-3 {
                    return None;
                }
                let arr = [order[0], order[1], order[2], order[3]];
                return Some(if vol < 0.0 {
                    arr
                } else {
                    [arr[0], arr[2], arr[1], arr[3]]
                });
            }
            let last = if m1 == 3 || m2_idx == 3 {
                d[m1]
            } else {
                -d[m1]
            };
            let t = turn(w[0], w[1], w[2])?;
            let arr = [order[0], order[1], order[2], order[3]];
            return Some(if last == t {
                arr
            } else {
                [arr[0], arr[2], arr[1], arr[3]]
            });
        }
        return None;
    }

    // 3 drawn bonds + 1 implicit H or lone pair
    let up = d.iter().filter(|&&x| x > 0).count();
    let down = d.iter().filter(|&&x| x < 0).count();
    let mut mid: Option<usize> = None;
    let mut flat = true;
    let mut sense: i8 = 1;
    for k in 0..3 {
        let x = sector(w[(k + 1) % 3], w[(k + 2) % 3], w[k])?;
        if x == 1 {
            mid = Some(k);
        }
        if x == 2 {
            flat = false;
        }
    }
    if let Some(m_idx) = mid {
        if d[m_idx] != 0 {
            if d[(m_idx + 1) % 3] == d[m_idx] || d[(m_idx + 2) % 3] == d[m_idx] {
                return None;
            }
            if d[m_idx] > 0 {
                sense = -1;
            }
        } else {
            let d1 = if d[(m_idx + 1) % 3] != 0 {
                d[(m_idx + 1) % 3]
            } else {
                d[(m_idx + 2) % 3]
            };
            let d2 = d[(m_idx + 2) % 3];
            if d2 != 0 && d1 != d2 {
                return None;
            }
            if d1 < 0 {
                sense = -1;
            }
        }
    } else if !flat {
        if up > 0 && down > 0 {
            let p0 = [w[0][0], w[0][1], d[0] as f64];
            let p1 = [w[1][0], w[1][1], d[1] as f64];
            let p2 = [w[2][0], w[2][1], d[2] as f64];
            let p3 = [
                -(p0[0] + p1[0] + p2[0]),
                -(p0[1] + p1[1] + p2[1]),
                -(p0[2] + p1[2] + p2[2]),
            ];
            let vol = signed_tetra_volume(p0, p1, p2, p3);
            if vol.abs() < 1e-3 {
                return None;
            }
            return Some(if vol < 0.0 {
                [order[0], order[1], order[2], virtual_fourth]
            } else {
                [order[0], order[2], order[1], virtual_fourth]
            });
        }
        if virtual_fourth == usize::MAX && (up == 3 || down == 3) {
            return None;
        }
        if down > 0 {
            sense = -1;
        }
    } else {
        return None;
    }

    let t = turn(w[0], w[1], w[2])?;
    Some(if t == sense {
        [order[0], order[2], order[1], virtual_fourth]
    } else {
        [order[0], order[1], order[2], virtual_fourth]
    })
}

fn evaluate_double_bonds_cip(
    mol: &Molecule,
    g: &CipGraph,
    aux_atoms: &BTreeMap<usize, CipAtomDescriptor>,
    out_bonds: &mut BTreeMap<usize, CipBondDescriptor>,
) {
    let rings = perceive_sssr(mol);
    for (bi, b) in mol.bonds.iter().enumerate() {
        if b.kind != BondKind::Double || b.double_geometry == DoubleBondGeometry::Either {
            continue;
        }
        // Double bonds in rings of size < 8 do not receive (E)/(Z) descriptors per IUPAC P-93.5.1.2
        if rings.iter().any(|r| {
            r.len() < 8
                && (0..r.len()).any(|k| {
                    let u = r[k];
                    let v = r[(k + 1) % r.len()];
                    (u == b.a && v == b.b) || (u == b.b && v == b.a)
                })
        }) {
            continue;
        }
        let za = mol.atoms[b.a].z;
        let zb = mol.atoms[b.b].z;
        if !matches!(za, 6 | 7 | 14 | 32) || !matches!(zb, 6 | 7 | 14 | 32) {
            continue;
        }
        // Exclude any atom with wavy bond
        if mol.bonds.iter().any(|wb| {
            wb.kind == BondKind::Wavy && (wb.a == b.a || wb.b == b.a || wb.a == b.b || wb.b == b.b)
        }) {
            continue;
        }

        let subs_a: Vec<usize> = mol
            .covalent_neighbors(b.a)
            .into_iter()
            .map(|(nb, _)| nb)
            .filter(|&nb| nb != b.b)
            .collect();
        let subs_b: Vec<usize> = mol
            .covalent_neighbors(b.b)
            .into_iter()
            .map(|(nb, _)| nb)
            .filter(|&nb| nb != b.a)
            .collect();
        if subs_a.is_empty() || subs_a.len() > 2 || subs_b.is_empty() || subs_b.len() > 2 {
            continue;
        }

        // Determine highest-priority substituent on `b.a` and `b.b`
        let top_on = |end: usize, subs: &[usize]| -> Option<usize> {
            if subs.len() == 2 {
                match compare_cip_ligands(g, end, subs[0], subs[1], aux_atoms, 3) {
                    Ordering::Greater => Some(subs[0]),
                    Ordering::Less => Some(subs[1]),
                    Ordering::Equal => None,
                }
            } else {
                // Single drawn substituent + implicit H (or lone pair on imine =N-): drawn substituent vs H/LP
                let virtual_other = if mol.atoms[end].z == 7 && g.implicit_h[end] == 0 {
                    usize::MAX - 1
                } else if g.implicit_h[end] >= 1 {
                    usize::MAX
                } else {
                    return None;
                };
                match compare_cip_ligands(g, end, subs[0], virtual_other, aux_atoms, 3) {
                    Ordering::Greater => Some(subs[0]),
                    Ordering::Less => None,
                    Ordering::Equal => None,
                }
            }
        };

        let Some(hi_a) = top_on(b.a, &subs_a) else {
            continue;
        };
        let Some(hi_b) = top_on(b.b, &subs_b) else {
            continue;
        };

        // Determine whether `hi_a` and `hi_b` lie on the SAME side (Z) or OPPOSITE sides (E)
        let is_same_side: Option<bool> = match b.double_geometry {
            DoubleBondGeometry::Cis { ref_a, ref_b } => {
                // Determine if (ref_a, ref_b) match (b.a side, b.b side)
                let (ra, rb) = if subs_a.contains(&ref_a) && subs_b.contains(&ref_b) {
                    (ref_a, ref_b)
                } else if subs_a.contains(&ref_b) && subs_b.contains(&ref_a) {
                    (ref_b, ref_a)
                } else {
                    continue;
                };
                let flip_a = ra != hi_a;
                let flip_b = rb != hi_b;
                Some(!(flip_a ^ flip_b))
            }
            DoubleBondGeometry::Trans { ref_a, ref_b } => {
                let (ra, rb) = if subs_a.contains(&ref_a) && subs_b.contains(&ref_b) {
                    (ref_a, ref_b)
                } else if subs_a.contains(&ref_b) && subs_b.contains(&ref_a) {
                    (ref_b, ref_a)
                } else {
                    continue;
                };
                let flip_a = ra != hi_a;
                let flip_b = rb != hi_b;
                Some(flip_a ^ flip_b)
            }
            DoubleBondGeometry::Either => None,
            DoubleBondGeometry::Auto => alkene_same_side_2d(mol, b.a, b.b, hi_a, hi_b),
        };

        if let Some(same) = is_same_side {
            out_bonds.insert(
                bi,
                if same {
                    CipBondDescriptor::Z
                } else {
                    CipBondDescriptor::E
                },
            );
        }
    }
}

fn alkene_same_side_2d(
    mol: &Molecule,
    a: usize,
    b: usize,
    sub_a: usize,
    sub_b: usize,
) -> Option<bool> {
    let ax = mol.atoms[a].x;
    let ay = mol.atoms[a].y;
    let bx = mol.atoms[b].x;
    let by = mol.atoms[b].y;
    let dx = bx - ax;
    let dy = by - ay;
    let blen = dx.hypot(dy);
    if blen < 1e-5 {
        return None;
    }
    // Normal vector (-dy, dx) / blen
    let nx = -dy / blen;
    let ny = dx / blen;

    let vax = mol.atoms[sub_a].x - ax;
    let vay = mol.atoms[sub_a].y - ay;
    let la = vax.hypot(vay);
    let vbx = mol.atoms[sub_b].x - bx;
    let vby = mol.atoms[sub_b].y - by;
    let lb = vbx.hypot(vby);
    if la < 1e-5 || lb < 1e-5 {
        return None;
    }
    let proj_a = (vax * nx + vay * ny) / la;
    let proj_b = (vbx * nx + vby * ny) / lb;
    if proj_a.abs() < 0.1 || proj_b.abs() < 0.1 {
        return None;
    }
    Some(proj_a * proj_b > 0.0)
}

// ============================================================================
// 4. Bridged Bicyclic Endo / Exo / Syn / Anti Perception (IUPAC P-93.5.2.4)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EndoExoDescriptor {
    Endo,
    Exo,
    Syn,
    Anti,
}

impl EndoExoDescriptor {
    pub fn as_str(self) -> &'static str {
        match self {
            EndoExoDescriptor::Endo => "endo",
            EndoExoDescriptor::Exo => "exo",
            EndoExoDescriptor::Syn => "syn",
            EndoExoDescriptor::Anti => "anti",
        }
    }
}

/// Perceive `endo`/`exo`/`syn`/`anti` stereodescriptors on substituted ring atoms of bridged
/// von Baeyer bicyclo`[x.y.z]` systems (such as norbornane / bicyclo`[2.2.1]`heptane, camphor,
/// bicyclo`[3.2.1]`octane), keyed by the ring carbon index!
///
/// Follows IUPAC P-93.5.2.4 (`endoexo.ts`):
/// - In bicyclo`[x.y.z]` ($x \ge y > z > 0$), a substituent on the $x$ or $y$ bridge is **`Exo`**
///   when it points toward the $z$ (shortest) bridge and **`Endo`** when it points away from the $z$ bridge.
/// - A substituent on the $z$ bridge (when $x > y$) is **`Syn`** when it points toward the $x$ (longest) bridge
///   and **`Anti`** when it points away.
pub fn perceive_endo_exo(mol: &Molecule) -> BTreeMap<usize, EndoExoDescriptor> {
    let mut out = BTreeMap::new();
    let rings = perceive_sssr(mol);
    if rings.len() < 2 {
        return out;
    }
    let g = CipGraph::from_mol(mol);

    // Group SSSR rings that share >= 2 edges (bridged bicyclic systems)
    for r1_idx in 0..rings.len() {
        for r2_idx in (r1_idx + 1)..rings.len() {
            let mut bicyc_atoms: BTreeSet<usize> = rings[r1_idx].iter().copied().collect();
            for &u in &rings[r2_idx] {
                bicyc_atoms.insert(u);
            }
            // Count induced covalent bonds inside `bicyc_atoms`
            let mut bicyc_bonds = 0usize;
            for b in &mol.bonds {
                if matches!(
                    b.kind,
                    BondKind::Dative | BondKind::Hydrogen | BondKind::Ionic
                ) {
                    continue;
                }
                if bicyc_atoms.contains(&b.a) && bicyc_atoms.contains(&b.b) {
                    bicyc_bonds += 1;
                }
            }
            if bicyc_bonds != bicyc_atoms.len() + 1 {
                continue;
            }
            // Identify the 2 bridgehead atoms (degree 3 inside `bicyc_atoms`, all others degree 2)
            let mut heads = Vec::new();
            let mut valid = true;
            for &u in &bicyc_atoms {
                let deg = mol
                    .covalent_neighbors(u)
                    .into_iter()
                    .filter(|&(v, _)| bicyc_atoms.contains(&v))
                    .count();
                if deg == 3 {
                    heads.push(u);
                } else if deg != 2 {
                    valid = false;
                    break;
                }
            }
            if !valid || heads.len() != 2 {
                continue;
            }
            let h1 = heads[0];
            let h2 = heads[1];

            // Walk the 3 bridges from h1 to h2
            let starts: Vec<usize> = mol
                .covalent_neighbors(h1)
                .into_iter()
                .map(|(v, _)| v)
                .filter(|v| bicyc_atoms.contains(v))
                .collect();
            if starts.len() != 3 {
                continue;
            }
            let mut bridges: Vec<Vec<usize>> = Vec::with_capacity(3);
            for s in starts {
                let mut path = Vec::new();
                let mut prev = h1;
                let mut cur = s;
                while cur != h2 {
                    path.push(cur);
                    let nxt = mol
                        .covalent_neighbors(cur)
                        .into_iter()
                        .map(|(v, _)| v)
                        .find(|&v| bicyc_atoms.contains(&v) && v != prev);
                    let Some(n_atom) = nxt else {
                        valid = false;
                        break;
                    };
                    prev = cur;
                    cur = n_atom;
                }
                bridges.push(path);
            }
            if !valid {
                continue;
            }
            // Sort bridges descending by length: x >= y >= z
            bridges.sort_by_key(|b| std::cmp::Reverse(b.len()));
            let x_len = bridges[0].len();
            let y_len = bridges[1].len();
            let z_len = bridges[2].len();
            // Bridged bicycle requires z_len >= 1 and y_len > z_len (so z bridge is uniquely shortest!)
            if z_len == 0 || y_len <= z_len {
                continue;
            }

            // Build canonical reference 3D frame (same as `endoexo.ts` lines 68-71):
            // h1 = (-1.1, 0, 0), h2 = (+1.1, 0, 0)
            // bridge 0 (x): y = +1.2, z = -0.5, outward = (0, +1, 0)
            // bridge 1 (y): y = -1.2, z = -0.5, outward = (0, -1, 0)
            // bridge 2 (z): y = 0.0, z = +1.2, outward = (0, 0, +1)
            let mut frame_pos: BTreeMap<usize, [f64; 3]> = BTreeMap::new();
            frame_pos.insert(h1, [-1.1, 0.0, 0.0]);
            frame_pos.insert(h2, [1.1, 0.0, 0.0]);
            let bridge_specs: [(f64, f64, [f64; 3]); 3] = [
                (1.2, -0.5, [0.0, 1.0, 0.0]),
                (-1.2, -0.5, [0.0, -1.0, 0.0]),
                (0.0, 1.2, [0.0, 0.0, 1.0]),
            ];
            for (b_idx, path) in bridges.iter().enumerate() {
                let (fy, fz, _) = bridge_specs[b_idx];
                for (step, &atom_u) in path.iter().enumerate() {
                    let fx = -1.1 + 2.2 * ((step + 1) as f64) / ((path.len() + 1) as f64);
                    frame_pos.insert(atom_u, [fx, fy, fz]);
                }
            }

            // Note: the 2D drawing could show either enantiomer of the bicycle frame!
            // Check bridgehead h1's 3D/2D pyramid if specified, OR determine relative face directly from the
            // pyramid `[prev_ring, next_ring, subst, other_subst_or_H]` on each bridge atom!
            // Specifically: in our canonical 3D frame (where h1->h2 runs along +X, x-bridge at +Y/-Z, y-bridge at -Y/-Z,
            // z-bridge at +Z), let's check whether the drawn molecule's bridgehead h1 matches this frame or its mirror!
            // Even better: if bridgehead h1 has wedges/3D, check its handedness; if bridgehead h1 has NO wedges (standard
            // flat norbornane projection where z-bridge is drawn in the middle/top and wedges/hashes are on C2!),
            // inspect the 2D projection or 3D coordinates directly!
            let has_3d = mol.has_3d_coords()
                && bicyc_atoms
                    .iter()
                    .any(|&u| mol.atoms[u].z_coord.unwrap_or(0.0).abs() > 0.15);

            for (b_idx, path) in bridges.iter().enumerate() {
                if b_idx == 2 && x_len == y_len {
                    // When x == y (e.g. norbornane [2.2.1]), syn/anti on C7 requires distinguishing bridges by CIP priority
                    continue;
                }
                for (step, &u) in path.iter().enumerate() {
                    let prev_ring = if step == 0 { h1 } else { path[step - 1] };
                    let next_ring = if step + 1 == path.len() {
                        h2
                    } else {
                        path[step + 1]
                    };
                    let exo_nbs: Vec<usize> = mol
                        .covalent_neighbors(u)
                        .into_iter()
                        .map(|(v, _)| v)
                        .filter(|v| !bicyc_atoms.contains(v))
                        .collect();
                    // Pick the single non-H substituent (or higher-CIP substituent if two drawn)
                    let heavy_subs: Vec<usize> = exo_nbs
                        .iter()
                        .copied()
                        .filter(|&v| mol.atoms[v].z != 1)
                        .collect();
                    if heavy_subs.len() != 1 {
                        continue;
                    }
                    let sub_atom = heavy_subs[0];
                    let other_lig = exo_nbs
                        .iter()
                        .copied()
                        .find(|&v| v != sub_atom)
                        .unwrap_or(usize::MAX);

                    if has_3d {
                        // Direct 3D vector comparison!
                        let pu = [
                            mol.atoms[u].x,
                            mol.atoms[u].y,
                            mol.atoms[u].z_coord.unwrap_or(0.0),
                        ];
                        let ps = [
                            mol.atoms[sub_atom].x,
                            mol.atoms[sub_atom].y,
                            mol.atoms[sub_atom].z_coord.unwrap_or(0.0),
                        ];
                        let sub_vec = [ps[0] - pu[0], ps[1] - pu[1], ps[2] - pu[2]];
                        if b_idx < 2 {
                            // Centroid of z-bridge relative to u
                            let z_atom = bridges[2][bridges[2].len() / 2];
                            let pz = [
                                mol.atoms[z_atom].x,
                                mol.atoms[z_atom].y,
                                mol.atoms[z_atom].z_coord.unwrap_or(0.0),
                            ];
                            // Compare sub_vec vs other_lig_vec dot (pz - pu)
                            let to_z = [pz[0] - pu[0], pz[1] - pu[1], pz[2] - pu[2]];
                            let other_vec = if other_lig != usize::MAX {
                                let po = [
                                    mol.atoms[other_lig].x,
                                    mol.atoms[other_lig].y,
                                    mol.atoms[other_lig].z_coord.unwrap_or(0.0),
                                ];
                                [po[0] - pu[0], po[1] - pu[1], po[2] - pu[2]]
                            } else {
                                [-sub_vec[0], -sub_vec[1], -sub_vec[2]]
                            };
                            let dot_sub =
                                sub_vec[0] * to_z[0] + sub_vec[1] * to_z[1] + sub_vec[2] * to_z[2];
                            let dot_oth = other_vec[0] * to_z[0]
                                + other_vec[1] * to_z[1]
                                + other_vec[2] * to_z[2];
                            out.insert(
                                u,
                                if dot_sub > dot_oth {
                                    EndoExoDescriptor::Exo
                                } else {
                                    EndoExoDescriptor::Endo
                                },
                            );
                        } else {
                            let x_atom = bridges[0][bridges[0].len() / 2];
                            let px = [
                                mol.atoms[x_atom].x,
                                mol.atoms[x_atom].y,
                                mol.atoms[x_atom].z_coord.unwrap_or(0.0),
                            ];
                            let to_x = [px[0] - pu[0], px[1] - pu[1], px[2] - pu[2]];
                            let dot_sub =
                                sub_vec[0] * to_x[0] + sub_vec[1] * to_x[1] + sub_vec[2] * to_x[2];
                            out.insert(
                                u,
                                if dot_sub > 0.0 {
                                    EndoExoDescriptor::Syn
                                } else {
                                    EndoExoDescriptor::Anti
                                },
                            );
                        }
                        continue;
                    }

                    // 2D drawing with wedge/hash (or parity):
                    // Determine whether the canonical frame matches the 2D drawing's bridge orientation!
                    // In a top-down 2D drawing of bicyclo[x.y.z] where z-bridge points toward the viewer (+Z),
                    // `cross2D(h2 - h1, x_or_y_atom - h1)` tells which side of h1->h2 this bridge lies on!
                    // In our 3D model frame (z-bridge at +Z toward viewer):
                    // - A bridge with `cross2D(h2 - h1, u - h1) > 0` (left of h1->h2, i.e. +Y) has `exo` (+Z) giving
                    //   R-sense `[prev_ring, next_ring, exo_pos, endo_pos]` determined by `signed_tetra_volume`!
                    let Some(drawn_pyr) = perceive_center_pyramid(mol, &g, u) else {
                        continue;
                    };
                    let dhx = mol.atoms[h2].x - mol.atoms[h1].x;
                    let dhy = mol.atoms[h2].y - mol.atoms[h1].y;
                    let dux = mol.atoms[u].x - mol.atoms[h1].x;
                    let duy = mol.atoms[u].y - mol.atoms[h1].y;
                    let cross_2d = dhx * duy - dhy * dux;

                    if b_idx < 2 {
                        // Construct local 3D frame matching 2D side of h1->h2 with z-bridge at +Z (toward viewer, standard projection):
                        let side_y = if cross_2d >= 0.0 { 1.0 } else { -1.0 };
                        let p_prev = if prev_ring == h1 {
                            [-1.1, 0.0, 0.0]
                        } else {
                            [-0.4, 1.2 * side_y, -0.5]
                        };
                        let p_next = if next_ring == h2 {
                            [1.1, 0.0, 0.0]
                        } else {
                            [0.4, 1.2 * side_y, -0.5]
                        };
                        let p_u = [0.0, 1.2 * side_y, -0.5];
                        let p_exo = [p_u[0], p_u[1] + 0.6 * side_y, p_u[2] + 0.8];
                        let p_endo = [p_u[0], p_u[1] + 0.6 * side_y, p_u[2] - 0.8];
                        let vol_exo = signed_tetra_volume(
                            [p_prev[0] - p_u[0], p_prev[1] - p_u[1], p_prev[2] - p_u[2]],
                            [p_next[0] - p_u[0], p_next[1] - p_u[1], p_next[2] - p_u[2]],
                            [p_exo[0] - p_u[0], p_exo[1] - p_u[1], p_exo[2] - p_u[2]],
                            [p_endo[0] - p_u[0], p_endo[1] - p_u[1], p_endo[2] - p_u[2]],
                        );
                        let exo_ref_pyr = if vol_exo < 0.0 {
                            [prev_ring, next_ring, sub_atom, other_lig]
                        } else {
                            [prev_ring, sub_atom, next_ring, other_lig]
                        };
                        if let Some(odd) = permutation_is_odd(&drawn_pyr, &exo_ref_pyr) {
                            out.insert(
                                u,
                                if !odd {
                                    EndoExoDescriptor::Exo
                                } else {
                                    EndoExoDescriptor::Endo
                                },
                            );
                        }
                    }
                }
            }
        }
    }
    out
}

// ============================================================================
// 5. Chair Cyclohexane Axial / Equatorial Perception & Axial Chirality (M)/(P)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChairOrientation {
    Axial,
    Equatorial,
}

impl ChairOrientation {
    pub fn as_str(self) -> &'static str {
        match self {
            ChairOrientation::Axial => "axial",
            ChairOrientation::Equatorial => "equatorial",
        }
    }
}

/// Compute ideal 2D `(axial_unit_vec, equatorial_unit_vec)` for a ring vertex `ring_atom` in a drawn
/// chair 6-membered ring `ring` (ported from Octet-Draw `edit.ts` `chairWay`).
/// Returns `None` if `ring` is a regular flat hexagon rather than a chair perspective drawing!
pub fn chair_sprout_vectors(
    mol: &Molecule,
    ring: &[usize],
    ring_atom: usize,
) -> Option<([f64; 2], [f64; 2])> {
    if ring.len() != 6 {
        return None;
    }
    let pos_in_ring = ring.iter().position(|&u| u == ring_atom)?;
    let mut dirs = [[0.0f64; 2]; 6];
    for k in 0..6 {
        let a = &mol.atoms[ring[k]];
        let b = &mol.atoms[ring[(k + 1) % 6]];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let len = dx.hypot(dy);
        if len < 1e-4 {
            return None;
        }
        dirs[k] = [dx / len, dy / len];
    }
    // In a 2D chair cyclohexane, opposite ring bonds (k and k+3) are parallel in opposite directions: dot(dirs[k], dirs[k+3]) < -0.92
    for k in 0..3 {
        let dp = dirs[k][0] * dirs[k + 3][0] + dirs[k][1] * dirs[k + 3][1];
        if dp > -0.90 {
            return None;
        }
    }
    // Distinguish a chair from a regular hexagon: in a regular hexagon adjacent bond cross products all have the SAME sign;
    // in a chair perspective drawing, two opposite flaps bend one way and the middle edges bend so cross-product signs alternate or flatten!
    let mut crosses = [0.0f64; 6];
    for k in 0..6 {
        let prev = dirs[(k + 5) % 6];
        let next = dirs[k];
        crosses[k] = prev[0] * next[1] - prev[1] * next[0];
    }
    let pos_signs = crosses.iter().filter(|&&c| c > 0.08).count();
    let neg_signs = crosses.iter().filter(|&&c| c < -0.08).count();
    // Regular convex hexagon has all 6 turns with the same sign (`pos_signs == 6` or `neg_signs == 6`)!
    if pos_signs == 6 || neg_signs == 6 {
        return None;
    }

    // Bisector of the two ring bonds pointing AWAY from ring vertex `pos_in_ring` gives the vertical axial direction:
    let b_prev = [
        -dirs[(pos_in_ring + 5) % 6][0],
        -dirs[(pos_in_ring + 5) % 6][1],
    ];
    let b_next = dirs[pos_in_ring];
    let bx = b_prev[0] + b_next[0];
    let by = b_prev[1] + b_next[1];
    let blen = bx.hypot(by);
    if blen < 1e-4 {
        return None;
    }
    // Outward from the vertex is `-bisector` when the vertex points outward (Wait: `b_prev` and `b_next` point FROM `ring_atom` TO its two ring neighbors,
    // so `-(b_prev + b_next)` points OUTWARD from the V-corner — that IS the axial direction at the flap tips, and alternates up/down around the chair!)
    // Let's verify: in a standard chair, C1 (rightmost flap pointing down-right) has neighbors up-left; `-(b_prev + b_next)` points down!
    // Actually in standard chair cyclohexane, axial bonds are vertical (`±Y`), alternating `+Y, -Y, +Y, -Y, +Y, -Y` around the 6-ring,
    // and each equatorial bond is parallel to the ring bonds 2 steps away (`dirs[(pos_in_ring + 1) % 6]` or `-dirs[(pos_in_ring + 4) % 6]`)!
    // Let's compute exact chair orientation from the overall chair axis so every one of the 6 positions gets exact axial & equatorial vectors!
    let mut best_up = [0.0f64, 1.0f64];
    // Find the flap vertex with sharpest interior angle (largest `|b_prev + b_next|`) to anchor the axial axis!
    let mut best_flap = 0usize;
    let mut best_mag = -1.0f64;
    for k in 0..6 {
        let vp = [-dirs[(k + 5) % 6][0], -dirs[(k + 5) % 6][1]];
        let vn = dirs[k];
        let mx = -(vp[0] + vn[0]);
        let my = -(vp[1] + vn[1]);
        // Axial axis is perpendicular to the mean plane / aligned with the flap V-tip bisector's normal:
        // Wait: in standard ChemDraw chair:
        // Ring vertices: 0=(top-left), 1=(mid-top), 2=(right-top flap), 3=(bottom-right), 4=(mid-bottom), 5=(left-bottom flap)
        // Notice at the middle vertices (1 and 4), `-(vp + vn)` is strictly VERTICAL (`+Y` at top-middle, `-Y` at bottom-middle)!
        // Let's check! At middle vertex `1` (where one bond goes down-left and one goes down-right at obtuse ~120°),
        // `vp` and `vn` both have negative Y and opposite X, so `-(vp + vn)` points straight `+Y` (vertical)!
        // Whereas at the acute flap vertex `2`, `vp` and `vn` both point left, so `-(vp + vn)` has smaller magnitude or points right!
        //Wait: obtuse angle (120°) has `|vp + vn| = 1.0`, acute angle (60°) has `|vp + vn| = 1.732`!
        // Even better: the chair's axial axis `A` is perpendicular to the chair's central axis, OR in 2D it is perpendicular to the average horizontal span,
        // and specifically at the two vertices whose interior cross product has the minority sign (the two puckered "middle" vertices of a chair!),
        // `-(vp + vn)` points EXACTLY along the chair's axial direction!
        let mag = mx.hypot(my);
        if mag > best_mag {
            best_mag = mag;
            best_flap = k;
        }
    }
    let _ = best_flap;
    // Find the two minority-turn vertices (in any chair 2D projection, 4 turns have one sign and 2 opposite vertices have the opposite turn sign!):
    let minority_positive = pos_signs < neg_signs;
    for k in 0..6 {
        if (minority_positive && crosses[k] > 0.05) || (!minority_positive && crosses[k] < -0.05) {
            let vp = [-dirs[(k + 5) % 6][0], -dirs[(k + 5) % 6][1]];
            let vn = dirs[k];
            // At the re-entrant vertex of a chair projection, `(vp + vn)` points INTO the V (which is OUTWARD vertically from the chair's waist? Wait:
            // let's trace standard chair:
            // v0=(-1.3, 0.4), v1=(0.0, 0.0), v2=(1.3, 0.4), v3=(0.8, -0.5), v4=(-0.5, -0.1), v5=(-1.8, -0.5)
            // Look at v1=(0.0, 0.0): neighbors are v0=(-1.3, 0.4) and v2=(1.3, 0.4).
            // `vp + vn = (0, 0.8)` pointing UP (+Y)! Is C1's axial bond UP or DOWN?
            // Let's check 3D chair! v0 is UP (+z), v1 is DOWN (-z) — wait: v1=(0.0, 0.0) is lower on screen than v0=(-1.3, +0.4) and v2=(+1.3, +0.4)!
            // Since v1 is LOWER than both neighbors v0 and v2, v1 is a DOWN carbon, so its AXIAL bond points DOWN `(0, -1)` (`-(vp + vn)`)!
            // Look! `-(vp + vn)` = `(0, -0.8)` = DOWN! And its EQUATORIAL bond points UP-back `(vp + vn)` / parallel to C2-C3!
            let ax_x = -(vp[0] + vn[0]);
            let ax_y = -(vp[1] + vn[1]);
            let al = ax_x.hypot(ax_y);
            if al > 1e-4 {
                let sign = if (pos_in_ring + 6 - k) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                best_up = [sign * ax_x / al, sign * ax_y / al];
                break;
            }
        }
    }

    // Equatorial vector at `pos_in_ring` is parallel to the C(k+1)-C(k+2) and C(k-1)-C(k-2) ring bonds, pointing outward from the ring centroid!
    let eq_raw = dirs[(pos_in_ring + 1) % 6];
    let cx: f64 = ring.iter().map(|&u| mol.atoms[u].x).sum::<f64>() / 6.0;
    let cy: f64 = ring.iter().map(|&u| mol.atoms[u].y).sum::<f64>() / 6.0;
    let out_x = mol.atoms[ring_atom].x - cx;
    let out_y = mol.atoms[ring_atom].y - cy;
    let eq_sign = if eq_raw[0] * out_x + eq_raw[1] * out_y >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let eq_vec = [eq_sign * eq_raw[0], eq_sign * eq_raw[1]];

    Some((best_up, eq_vec))
}

/// Perceive `Axial` vs `Equatorial` orientation for exocyclic substituent atoms attached to saturated
/// 6-membered chair rings in `mol` (supporting both 3D chair coordinates AND 2D chair perspective drawings!).
/// Returns a map from **exocyclic substituent atom index** to `ChairOrientation`.
pub fn perceive_chair_axial_equatorial(mol: &Molecule) -> BTreeMap<usize, ChairOrientation> {
    let mut out = BTreeMap::new();
    let rings = perceive_sssr(mol);
    let has_3d = mol.has_3d_coords();

    for ring in &rings {
        if ring.len() != 6 {
            continue;
        }
        let ring_set: BTreeSet<usize> = ring.iter().copied().collect();
        // Ring bonds must all be saturated single bonds
        let all_single = (0..6).all(|k| {
            mol.find_bond(ring[k], ring[(k + 1) % 6])
                .map(|bi| mol.bonds[bi].kind.order_u8() == 1 && !mol.bonds[bi].is_aromatic())
                .unwrap_or(false)
        });
        if !all_single {
            continue;
        }

        // Case A: 3D chair coordinates (non-planar 6-ring with alternating puckering around mean plane normal)
        if has_3d {
            let pts: Vec<[f64; 3]> = ring
                .iter()
                .map(|&u| {
                    [
                        mol.atoms[u].x,
                        mol.atoms[u].y,
                        mol.atoms[u].z_coord.unwrap_or(0.0),
                    ]
                })
                .collect();
            let centroid = [
                pts.iter().map(|p| p[0]).sum::<f64>() / 6.0,
                pts.iter().map(|p| p[1]).sum::<f64>() / 6.0,
                pts.iter().map(|p| p[2]).sum::<f64>() / 6.0,
            ];
            // Cremer-Pople mean ring plane normal via R1 x R2:
            let mut r1 = [0.0; 3];
            let mut r2 = [0.0; 3];
            for (j, p) in pts.iter().enumerate() {
                let ang = std::f64::consts::PI * (j as f64) / 3.0;
                let s = ang.sin();
                let c = ang.cos();
                for d in 0..3 {
                    r1[d] += (p[d] - centroid[d]) * s;
                    r2[d] += (p[d] - centroid[d]) * c;
                }
            }
            let nx = r1[1] * r2[2] - r1[2] * r2[1];
            let ny = r1[2] * r2[0] - r1[0] * r2[2];
            let nz = r1[0] * r2[1] - r1[1] * r2[0];
            let nlen = (nx * nx + ny * ny + nz * nz).sqrt();
            if nlen > 1e-4 {
                let normal = [nx / nlen, ny / nlen, nz / nlen];
                // Check chair puckering amplitude q2/q3: alternating z_disp around ring
                let disp: Vec<f64> = pts
                    .iter()
                    .map(|p| {
                        (p[0] - centroid[0]) * normal[0]
                            + (p[1] - centroid[1]) * normal[1]
                            + (p[2] - centroid[2]) * normal[2]
                    })
                    .collect();
                let is_3d_chair = disp.iter().all(|d| d.abs() > 0.12)
                    && (0..6).all(|k| disp[k] * disp[(k + 1) % 6] < 0.0);
                if is_3d_chair {
                    for &u in ring {
                        let pu = [
                            mol.atoms[u].x,
                            mol.atoms[u].y,
                            mol.atoms[u].z_coord.unwrap_or(0.0),
                        ];
                        for (nb, _) in mol.covalent_neighbors(u) {
                            if ring_set.contains(&nb) {
                                continue;
                            }
                            let pv = [
                                mol.atoms[nb].x,
                                mol.atoms[nb].y,
                                mol.atoms[nb].z_coord.unwrap_or(0.0),
                            ];
                            let dx = pv[0] - pu[0];
                            let dy = pv[1] - pu[1];
                            let dz = pv[2] - pu[2];
                            let bl = (dx * dx + dy * dy + dz * dz).sqrt();
                            if bl < 1e-4 {
                                continue;
                            }
                            let cos_axis =
                                ((dx * normal[0] + dy * normal[1] + dz * normal[2]) / bl).abs();
                            // Ideal chair axial |cos| ~ 0.94..1.0; ideal equatorial |cos| ~ 0.33
                            out.insert(
                                nb,
                                if cos_axis > 0.68 {
                                    ChairOrientation::Axial
                                } else {
                                    ChairOrientation::Equatorial
                                },
                            );
                        }
                    }
                    continue;
                }
            }
        }

        // Case B: 2D perspective chair drawing
        for &u in ring {
            if let Some((ax_vec, eq_vec)) = chair_sprout_vectors(mol, ring, u) {
                for (nb, _) in mol.covalent_neighbors(u) {
                    if ring_set.contains(&nb) {
                        continue;
                    }
                    let dx = mol.atoms[nb].x - mol.atoms[u].x;
                    let dy = mol.atoms[nb].y - mol.atoms[u].y;
                    let bl = dx.hypot(dy);
                    if bl < 1e-4 {
                        continue;
                    }
                    let ux = dx / bl;
                    let uy = dy / bl;
                    let align_ax = ux * ax_vec[0] + uy * ax_vec[1];
                    let align_eq = ux * eq_vec[0] + uy * eq_vec[1];
                    out.insert(
                        nb,
                        if align_ax >= align_eq {
                            ChairOrientation::Axial
                        } else {
                            ChairOrientation::Equatorial
                        },
                    );
                }
            }
        }
    }

    out
}

/// Axial chirality `(M)` / `(P)` descriptor for chiral allenes/even cumulenes and hindered biaryls (`axial.ts`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AxialChirality {
    M,
    P,
}

impl AxialChirality {
    pub fn as_str(self) -> &'static str {
        match self {
            AxialChirality::M => "M",
            AxialChirality::P => "P",
        }
    }
}

/// Perceive `(M)`/`(P)` axial chirality on central atoms of chiral allenes (`C=C=C` with wedge/hash end bonds).
pub fn perceive_axial_chirality(mol: &Molecule) -> BTreeMap<usize, AxialChirality> {
    let mut out = BTreeMap::new();
    let g = CipGraph::from_mol(mol);
    let empty_aux = BTreeMap::new();

    for i in 0..mol.atoms.len() {
        // Check central sp carbon of an allene (2 double bonds)
        if g.near[i].len() != 2 || !g.near[i].iter().all(|&(_, ord)| ord == 2) {
            continue;
        }
        let t1 = g.near[i][0].0;
        let t2 = g.near[i][1].0;
        if t1 == t2 {
            continue;
        }
        let end_coords = |t: usize, across: usize| -> Option<([f64; 3], [f64; 3])> {
            let others: Vec<usize> = g.near[t]
                .iter()
                .map(|&(nb, _)| nb)
                .filter(|&nb| nb != across)
                .collect();
            let depth_of = |nb: usize| -> f64 {
                mol.find_bond(t, nb)
                    .map(|bi| {
                        let b = &mol.bonds[bi];
                        if b.a == t {
                            match b.kind {
                                BondKind::WedgeUp | BondKind::HollowWedge => 1.0,
                                BondKind::HashDown => -1.0,
                                _ => 0.0,
                            }
                        } else {
                            0.0
                        }
                    })
                    .unwrap_or(0.0)
            };
            let pos3 = |k: usize| -> [f64; 3] {
                [
                    mol.atoms[k].x,
                    mol.atoms[k].y,
                    mol.atoms[k].z_coord.unwrap_or_else(|| depth_of(k)),
                ]
            };
            if others.len() == 2 {
                let cmp = compare_cip_ligands(&g, t, others[0], others[1], &empty_aux, 2);
                if cmp == Ordering::Equal {
                    return None;
                }
                let (hi, lo) = if cmp == Ordering::Greater {
                    (others[0], others[1])
                } else {
                    (others[1], others[0])
                };
                Some((pos3(hi), pos3(lo)))
            } else if others.len() == 1 && g.implicit_h[t] == 1 {
                let o = others[0];
                if mol.atoms[o].z == 1 && mol.atoms[o].isotope.is_none() {
                    return None;
                }
                let p = pos3(o);
                let c = [
                    mol.atoms[t].x,
                    mol.atoms[t].y,
                    mol.atoms[t].z_coord.unwrap_or(0.0),
                ];
                let a = [
                    mol.atoms[across].x,
                    mol.atoms[across].y,
                    mol.atoms[across].z_coord.unwrap_or(0.0),
                ];
                let ux = c[0] - a[0];
                let uy = c[1] - a[1];
                let uz = c[2] - a[2];
                let ul = (ux * ux + uy * uy + uz * uz).sqrt().max(1e-6);
                let u = [ux / ul, uy / ul, uz / ul];
                let v = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
                let along = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
                let h_pos = [
                    c[0] + 2.0 * along * u[0] - v[0],
                    c[1] + 2.0 * along * u[1] - v[1],
                    c[2] + 2.0 * along * u[2] - v[2],
                ];
                Some((p, h_pos))
            } else {
                None
            }
        };

        if let (Some((p1, p2)), Some((p3, p4))) = (end_coords(t1, i), end_coords(t2, i)) {
            if p1[2].abs() < 1e-4 && p2[2].abs() < 1e-4 && p3[2].abs() < 1e-4 && p4[2].abs() < 1e-4
            {
                continue;
            }
            let vol = signed_tetra_volume(p1, p2, p3, p4);
            if vol.abs() > 1e-5 {
                // Ra -> M (vol < 0), Sa -> P (vol > 0) per IUPAC P-92.1.4 / `axial.ts`
                out.insert(
                    i,
                    if vol < 0.0 {
                        AxialChirality::M
                    } else {
                        AxialChirality::P
                    },
                );
            }
        }
    }

    out
}
