#!/usr/bin/env python3
"""Clean-room deterministic builder and verifier for `data/torsion_db/torsion_corpus.tsv`.

Provenance & Literature Citations:
- Atomic coordinates: Worldwide Protein Data Bank (wwPDB / RCSB PDB, https://www.rcsb.org, CC0 1.0).
- Experimental chemical shifts: BioMagResBank (BMRB, https://bmrb.io, CC0 1.0).
- Fold & family identifiers: CATH / PDBe SIFTS & UniProt open scientific metadata.
- Random-coil reference shifts & nearest-neighbor Pro / oxidized-Cys corrections:
  Wishart, D. S., Bigam, C. G., Holm, A., Hodges, R. S. & Sykes, B. D. (1995).
  1H, 13C and 15N random coil NMR chemical shifts of the common amino acids.
  J. Biomol. NMR 5(1), 67-81.
- Coil-origin (`|dCA - dCB| <= 1.5 ppm`) chemical-shift reference offset screening:
  Wang, L., Eghbalnia, H. R., Bahrami, A. & Markley, J. L. (2005).
  Linear analysis of carbon-13 chemical shift differences and its application to the
  detection and correction of errors in referencing and spin system identifications.
  J. Biomol. NMR 31(2), 143-148.
- Tripeptide chemical-shift / backbone torsion fragment methodology:
  Shen, Y., Vernon, R., Baker, D. & Bax, A. (2008). Proc. Natl. Acad. Sci. U.S.A. 105, 4685-4690;
  Shen, Y. & Bax, A. (2013). J. Biomol. NMR 56, 227-241.

Python 3 standard library only (zero third-party packages).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import math
import pathlib
import re
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

EXPECTED_CORPUS_SHA256 = (
    "e34d3001d4c6989564d6bdfacff9dc401f92566aa77b499a23933b202045d88b"
)
EXPECTED_CHAIN_COUNT = 129
EXPECTED_RESIDUE_COUNT = 13647
EXPECTED_LINE_COUNT = 13781

USER_AGENT = (
    "nmr-assign-corpus-builder/0.1 (clean-room open-source NMR benchmark; CC0 wwPDB/BMRB)"
)

CORPUS_HEADER_LINES = [
    "# Provenance: Public-domain (CC0 1.0) non-redundant BMRB + wwPDB + CATH protein backbone torsion & secondary-shift corpus",
    "# Sources: BMRB (https://bmrb.io, CC0), wwPDB / RCSB PDB (https://www.rcsb.org, CC0), CATH / PDBe SIFTS (CC-BY / open scientific metadata)",
    "# Secondary shifts are computed relative to Wishart et al. (J. Biomol. NMR 1995, 5, 67-81) random-coil values with oxidized-Cys and pre-Pro corrections.",
    "# @CHAIN\tchain_key\tpdb_id\tchain_id\tbmrb_id\tuniprot_id\tfamily_id\ttopology_id\tlength",
    "# RES\tchain_key\tseq_id\tres_name\tdh\tdn\tdca\tdcb\tdc\tdha\tphi_deg\tpsi_deg\tchi1_deg\tca_x\tca_y\tca_z",
]

# Wishart et al. (J. Biomol. NMR 1995, 5, 67-81): (CA, CB, C, N, H, HA) in ppm (DSS, pH 5, 25 C)
WISHART_1995: Dict[str, Tuple[float, float, float, float, float, float]] = {
    "ALA": (52.5, 19.1, 177.8, 123.8, 8.24, 4.32),
    "ARG": (56.0, 30.9, 176.3, 121.3, 8.23, 4.34),
    "ASN": (53.1, 38.9, 175.2, 118.7, 8.40, 4.74),
    "ASP": (54.2, 41.1, 176.3, 120.4, 8.34, 4.64),
    "CYS_RED": (58.2, 28.0, 174.6, 118.8, 8.32, 4.55),
    "CYS_OX": (55.6, 41.1, 174.6, 118.8, 8.32, 4.71),
    "GLN": (55.7, 29.4, 176.0, 119.8, 8.32, 4.34),
    "GLU": (56.6, 29.9, 176.6, 120.2, 8.42, 4.35),
    "GLY": (45.1, 0.0, 173.9, 108.8, 8.33, 3.96),
    "HIS": (55.0, 29.0, 174.1, 118.2, 8.42, 4.73),
    "ILE": (61.1, 38.8, 176.4, 119.9, 8.00, 4.17),
    "LEU": (55.1, 42.4, 177.6, 121.8, 8.16, 4.34),
    "LYS": (56.2, 33.1, 176.6, 120.4, 8.29, 4.32),
    "MET": (55.4, 32.9, 176.3, 119.6, 8.28, 4.48),
    "PHE": (57.7, 39.6, 175.8, 120.3, 8.30, 4.62),
    "PRO": (63.3, 32.1, 177.3, 135.0, 0.00, 4.42),
    "SER": (58.3, 63.8, 174.6, 115.7, 8.31, 4.47),
    "THR": (61.8, 69.8, 174.7, 113.6, 8.15, 4.35),
    "TRP": (57.3, 29.6, 176.1, 121.3, 8.25, 4.66),
    "TYR": (57.9, 38.8, 175.9, 120.3, 8.12, 4.55),
    "VAL": (62.2, 32.9, 176.3, 119.2, 8.03, 4.12),
}

# IUPAC heavy-atom gamma substituent defining sidechain chi1 = dihedral(N, CA, CB, X_gamma)
CHI1_GAMMA_ATOM: Dict[str, str] = {
    "ARG": "CG",
    "ASN": "CG",
    "ASP": "CG",
    "CYS": "SG",
    "GLN": "CG",
    "GLU": "CG",
    "HIS": "CG",
    "ILE": "CG1",
    "LEU": "CG",
    "LYS": "CG",
    "MET": "CG",
    "PHE": "CG",
    "PRO": "CG",
    "SER": "OG",
    "THR": "OG1",
    "TRP": "CG",
    "TYR": "CG",
    "VAL": "CG1",
}

# Deadzone thresholds (ppm) for LACS coil-origin (`|dCA - dCB| <= 1.5 ppm`) median offset correction
# Ordered as: (dh, dn, dca, dcb, dc, dha)
COIL_OFFSET_THRESHOLDS: Tuple[float, float, float, float, float, float] = (
    0.20,
    1.20,
    1.00,
    1.00,
    1.00,
    0.20,
)

Vec3 = Tuple[float, float, float]


def calc_dihedral_deg(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> float:
    """Compute IUPAC torsion angle in degrees on (-180, 180] via atan2."""
    b1 = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    b2 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    b3 = (p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2])

    n1 = (
        b1[1] * b2[2] - b1[2] * b2[1],
        b1[2] * b2[0] - b1[0] * b2[2],
        b1[0] * b2[1] - b1[1] * b2[0],
    )
    n2 = (
        b2[1] * b3[2] - b2[2] * b3[1],
        b2[2] * b3[0] - b2[0] * b3[2],
        b2[0] * b3[1] - b2[1] * b3[0],
    )
    b2_norm = math.sqrt(b2[0] * b2[0] + b2[1] * b2[1] + b2[2] * b2[2])
    if b2_norm < 1e-12:
        return 0.0
    b2_u = (b2[0] / b2_norm, b2[1] / b2_norm, b2[2] / b2_norm)
    m1 = (
        b2_u[1] * n1[2] - b2_u[2] * n1[1],
        b2_u[2] * n1[0] - b2_u[0] * n1[2],
        b2_u[0] * n1[1] - b2_u[1] * n1[0],
    )
    x = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]
    y = m1[0] * n2[0] + m1[1] * n2[1] + m1[2] * n2[2]
    return math.degrees(math.atan2(y, x))


def random_coil_ref(
    res_name: str, cys_oxidized: bool, next_is_pro: bool
) -> Tuple[float, float, float, float, float, float]:
    """Return Wishart et al. (1995) random-coil (CA, CB, C, N, H, HA) shifts in ppm."""
    up = res_name.strip().upper()
    if up == "CYS":
        key = "CYS_OX" if cys_oxidized else "CYS_RED"
    else:
        key = up
    ca, cb, c, n, h, ha = WISHART_1995.get(
        key, (56.0, 33.0, 176.0, 120.0, 8.25, 4.35)
    )
    if next_is_pro and up != "GLY":
        ca -= 1.95
        c -= 2.00
        cb += 0.25
    elif next_is_pro and up == "GLY":
        ca -= 1.00
        c -= 1.50
    return ca, cb, c, n, h, ha


def parse_pdb_chain_model1(
    pdb_text: str, target_chain: str
) -> List[Tuple[int, str, Dict[str, Vec3]]]:
    """Extract Model 1 standard amino-acid residues `(seq_id, res_name, atoms)` for `target_chain`.

    Uses `MODEL 1` (stopping at the first `ENDMDL`), `altLoc` in `(' ', 'A')`, and skips insertion
    codes (`iCode != ' '`), retaining only residues with complete `(N, CA, C)` backbone coordinates.
    """
    residues: Dict[int, Tuple[str, Dict[str, Vec3]]] = {}
    order: List[int] = []
    in_first_model = True

    for line in pdb_text.splitlines():
        if line.startswith("ENDMDL"):
            if residues:
                break
            in_first_model = False
            continue
        if line.startswith("MODEL "):
            parts = line.split()
            in_first_model = len(parts) < 2 or parts[1] == "1"
            continue
        if not in_first_model or not line.startswith("ATOM  "):
            continue
        if len(line) < 54:
            continue
        alt_loc = line[16]
        if alt_loc not in (" ", "A"):
            continue
        icode = line[26].strip()
        if icode:
            continue
        chain_id = line[21].strip() or "A"
        if chain_id != target_chain:
            continue
        res_name = line[17:20].strip().upper()
        if res_name not in WISHART_1995 and res_name != "CYS":
            continue
        seq_id = int(line[22:26])
        atom_name = line[12:16].strip().upper()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        if seq_id not in residues:
            residues[seq_id] = (res_name, {})
            order.append(seq_id)
        atom_map = residues[seq_id][1]
        if atom_name not in atom_map:
            atom_map[atom_name] = (x, y, z)

    out: List[Tuple[int, str, Dict[str, Vec3]]] = []
    for sq in order:
        res_name, atom_map = residues[sq]
        if "N" in atom_map and "CA" in atom_map and "C" in atom_map:
            out.append((sq, res_name, atom_map))
    return out


def _tokenize_star_row(line: str) -> List[str]:
    """Tokenize a whitespace-delimited NMR-STAR loop data row respecting single/double quotes."""
    tokens: List[str] = []
    i = 0
    n = len(line)
    while i < n:
        while i < n and line[i].isspace():
            i += 1
        if i >= n:
            break
        if line[i] in ("'", '"'):
            quote = line[i]
            i += 1
            start = i
            while i < n and line[i] != quote:
                i += 1
            tokens.append(line[start:i])
            i += 1
        else:
            start = i
            while i < n and not line[i].isspace():
                i += 1
            tokens.append(line[start:i])
    return tokens


def parse_bmrb_star_primary_shifts(
    star_text: str,
) -> List[Tuple[int, str, Dict[str, float]]]:
    """Parse the primary `_Atom_chem_shift` loop (`list_id == 1`) from an NMR-STAR v3 file."""
    lines = star_text.splitlines()
    i = 0
    m = len(lines)
    residues: Dict[int, Tuple[str, Dict[str, float]]] = {}

    while i < m:
        s = lines[i].strip()
        if s == "loop_":
            i += 1
            tags: List[str] = []
            while i < m and lines[i].strip().startswith("_"):
                tags.append(lines[i].strip())
                i += 1
            if any(t.startswith("_Atom_chem_shift.") for t in tags):
                col = {t.split(".", 1)[1]: idx for idx, t in enumerate(tags)}
                idx_seq = col.get("Seq_ID", col.get("Comp_index_ID"))
                idx_comp = col.get("Comp_ID")
                idx_atom = col.get("Atom_ID")
                idx_val = col.get("Val")
                idx_list = col.get("Assigned_chem_shift_list_ID")
                idx_ent = col.get("Entity_ID")
                first_ent: Optional[str] = None
                while i < m:
                    row_s = lines[i].strip()
                    if row_s in ("stop_", "save_"):
                        break
                    if row_s and not row_s.startswith("#"):
                        toks = _tokenize_star_row(row_s)
                        if (
                            idx_seq is not None
                            and idx_comp is not None
                            and idx_atom is not None
                            and idx_val is not None
                            and len(toks) > max(idx_seq, idx_comp, idx_atom, idx_val)
                        ):
                            if idx_list is not None and len(toks) > idx_list:
                                if toks[idx_list] not in (".", "1"):
                                    i += 1
                                    continue
                            if idx_ent is not None and len(toks) > idx_ent:
                                ent_v = toks[idx_ent]
                                if ent_v != ".":
                                    if first_ent is None:
                                        first_ent = ent_v
                                    elif ent_v != first_ent:
                                        i += 1
                                        continue
                            try:
                                sq = int(toks[idx_seq])
                                val = float(toks[idx_val])
                            except ValueError:
                                i += 1
                                continue
                            comp = toks[idx_comp].upper()
                            atom = toks[idx_atom].upper()
                            if atom == "HN":
                                atom = "H"
                            elif atom == "CO":
                                atom = "C"
                            if sq not in residues:
                                residues[sq] = (comp, {})
                            if atom not in residues[sq][1]:
                                residues[sq][1][atom] = val
                    i += 1
                break
        i += 1

    return [(sq, residues[sq][0], residues[sq][1]) for sq in sorted(residues)]


def align_bmrb_to_pdb(
    pdb_res: Sequence[Tuple[int, str, Dict[str, Vec3]]],
    bmrb_res: Sequence[Tuple[int, str, Dict[str, float]]],
) -> List[Optional[int]]:
    """Semi-global Needleman-Wunsch alignment mapping each PDB residue index to its BMRB index."""
    lp = len(pdb_res)
    lb = len(bmrb_res)
    if lp == 0 or lb == 0:
        return [None] * lp

    # Fast exact seq_id match when residue names agree everywhere
    bmrb_by_seq = {sq: idx for idx, (sq, _, _) in enumerate(bmrb_res)}
    if all(
        (sq in bmrb_by_seq and bmrb_res[bmrb_by_seq[sq]][1] == rn)
        for sq, rn, _ in pdb_res
    ):
        return [bmrb_by_seq[sq] for sq, _, _ in pdb_res]

    match_sc = 4
    mismatch_sc = -3
    gap_sc = -2
    dp = [[0] * (lb + 1) for _ in range(lp + 1)]
    tb = [[0] * (lb + 1) for _ in range(lp + 1)]

    for i in range(1, lp + 1):
        p_rn = pdb_res[i - 1][1]
        for j in range(1, lb + 1):
            b_rn = bmrb_res[j - 1][1]
            s_diag = dp[i - 1][j - 1] + (match_sc if p_rn == b_rn else mismatch_sc)
            s_up = dp[i - 1][j] + (0 if j == lb else gap_sc)
            s_left = dp[i][j - 1] + (0 if i == lp else gap_sc)
            if s_diag >= s_up and s_diag >= s_left:
                dp[i][j] = s_diag
                tb[i][j] = 1
            elif s_up >= s_left:
                dp[i][j] = s_up
                tb[i][j] = 2
            else:
                dp[i][j] = s_left
                tb[i][j] = 3

    mapping: List[Optional[int]] = [None] * lp
    i, j = lp, lb
    while i > 0 and j > 0:
        step = tb[i][j]
        if step == 1:
            if pdb_res[i - 1][1] == bmrb_res[j - 1][1]:
                mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif step == 2:
            i -= 1
        else:
            j -= 1
    return mapping


def compute_chain_corpus_rows(
    chain_key: str,
    pdb_res: Sequence[Tuple[int, str, Dict[str, Vec3]]],
    bmrb_res: Sequence[Tuple[int, str, Dict[str, float]]],
) -> List[str]:
    """Compute exact `RES` TSV lines for a single `(chain_key, pdb_res, bmrb_res)` entry."""
    mapping = align_bmrb_to_pdb(pdb_res, bmrb_res)
    n_res = len(pdb_res)

    raw_sec: List[List[Optional[float]]] = []
    for idx, (_sq, res_name, _atoms) in enumerate(pdb_res):
        next_pro = (idx + 1 < n_res) and (pdb_res[idx + 1][1] == "PRO")
        b_idx = mapping[idx]
        sh = bmrb_res[b_idx][2] if b_idx is not None else {}
        cys_ox = (res_name == "CYS") and (sh.get("CB", 28.0) > 35.0)
        rc_ca, rc_cb, rc_c, rc_n, rc_h, rc_ha = random_coil_ref(
            res_name, cys_ox, next_pro
        )

        ha_val = sh.get("HA")
        if ha_val is None and res_name == "GLY":
            gly_ha = [sh[k] for k in ("HA2", "HA3") if k in sh]
            if gly_ha:
                ha_val = sum(gly_ha) / len(gly_ha)

        raw_sec.append(
            [
                (sh["H"] - rc_h) if (res_name != "PRO" and "H" in sh) else None,
                (sh["N"] - rc_n) if (res_name != "PRO" and "N" in sh) else None,
                (sh["CA"] - rc_ca) if "CA" in sh else None,
                (sh["CB"] - rc_cb) if (res_name != "GLY" and "CB" in sh) else None,
                (sh["C"] - rc_c) if "C" in sh else None,
                (ha_val - rc_ha) if ha_val is not None else None,
            ]
        )

    # LACS coil-origin (`|dCA - dCB| <= 1.5 ppm`) median offset correction
    coil_sub = [
        rv
        for rv in raw_sec
        if rv[2] is not None and rv[3] is not None and abs(rv[2] - rv[3]) <= 1.5
    ]
    offs = [0.0] * 6
    if len(coil_sub) >= 5:
        for k in range(6):
            vals = sorted(rv[k] for rv in coil_sub if rv[k] is not None)
            if len(vals) >= 5:
                med = vals[len(vals) // 2]
                if abs(med) > COIL_OFFSET_THRESHOLDS[k]:
                    offs[k] = med

    def fmt_opt(val: Optional[float], decimals: int) -> str:
        if val is None:
            return "."
        s = f"{val:.{decimals}f}"
        return f"{0.0:.{decimals}f}" if float(s) == 0.0 else s

    out_rows: List[str] = []
    for idx, (sq, res_name, atoms) in enumerate(pdb_res):
        ca = atoms["CA"]
        corr = [
            (raw_sec[idx][k] - offs[k]) if raw_sec[idx][k] is not None else None
            for k in range(6)
        ]

        phi_s = "."
        if idx > 0:
            prev_atoms = pdb_res[idx - 1][2]
            if math.dist(prev_atoms["C"], atoms["N"]) < 2.0:
                phi_v = calc_dihedral_deg(
                    prev_atoms["C"], atoms["N"], atoms["CA"], atoms["C"]
                )
                phi_s = fmt_opt(phi_v, 1)

        psi_s = "."
        if idx + 1 < n_res:
            next_atoms = pdb_res[idx + 1][2]
            if math.dist(atoms["C"], next_atoms["N"]) < 2.0:
                psi_v = calc_dihedral_deg(
                    atoms["N"], atoms["CA"], atoms["C"], next_atoms["N"]
                )
                psi_s = fmt_opt(psi_v, 1)

        chi1_s = "."
        gamma_name = CHI1_GAMMA_ATOM.get(res_name)
        if gamma_name and "CB" in atoms and gamma_name in atoms:
            chi1_v = calc_dihedral_deg(
                atoms["N"], atoms["CA"], atoms["CB"], atoms[gamma_name]
            )
            chi1_s = fmt_opt(chi1_v, 1)

        out_rows.append(
            "\t".join(
                [
                    "RES",
                    chain_key,
                    str(sq),
                    res_name,
                    fmt_opt(corr[0], 3),
                    fmt_opt(corr[1], 2),
                    fmt_opt(corr[2], 2),
                    fmt_opt(corr[3], 2),
                    fmt_opt(corr[4], 2),
                    fmt_opt(corr[5], 3),
                    phi_s,
                    psi_s,
                    chi1_s,
                    fmt_opt(ca[0], 2),
                    fmt_opt(ca[1], 2),
                    fmt_opt(ca[2], 2),
                ]
            )
        )
    return out_rows


def parse_provenance_inventory(
    provenance_path: pathlib.Path,
) -> List[Tuple[str, str, str, str, str, str, str, int]]:
    """Parse the 129-chain inventory markdown table from `data/torsion_db/PROVENANCE.md`."""
    entries: List[Tuple[str, str, str, str, str, str, str, int]] = []
    pat = re.compile(
        r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|"
    )
    for line in provenance_path.read_text(encoding="utf-8").splitlines():
        m = pat.match(line.strip())
        if m:
            entries.append(
                (
                    m.group(1),
                    m.group(2),
                    m.group(3),
                    m.group(4),
                    m.group(5),
                    m.group(6),
                    m.group(7),
                    int(m.group(8)),
                )
            )
    return entries


def http_get_text(url: str) -> str:
    """Fetch a public UTF-8/gzip text document from wwPDB or BMRB with polite User-Agent."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    if url.endswith(".gz"):
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def load_cached_or_fetch_pdb(
    pdb_id: str,
    pdb_dir: Optional[pathlib.Path],
    download_missing: bool,
    rate_limit_sec: float,
    need_delay: bool,
) -> Tuple[str, bool]:
    """Load a wwPDB entry from local `pdb_dir` cache or fetch CC0 gzip PDB from RCSB."""
    if pdb_dir is not None:
        for cand_name in (
            f"{pdb_id.lower()}.pdb",
            f"{pdb_id.upper()}.pdb",
            f"pdb{pdb_id.lower()}.ent",
            f"{pdb_id.lower()}.pdb.gz",
        ):
            cand = pdb_dir / cand_name
            if cand.is_file():
                raw = cand.read_bytes()
                if cand_name.endswith(".gz"):
                    raw = gzip.decompress(raw)
                return raw.decode("utf-8", errors="replace"), False
        if not download_missing:
            raise FileNotFoundError(
                f"PDB {pdb_id} not found in {pdb_dir} (pass --download-missing to fetch)"
            )

    if need_delay and rate_limit_sec > 0.0:
        time.sleep(rate_limit_sec)
    pdb_url = f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb.gz"
    pdb_text = http_get_text(pdb_url)
    if pdb_dir is not None:
        pdb_dir.mkdir(parents=True, exist_ok=True)
        (pdb_dir / f"{pdb_id.lower()}.pdb").write_text(pdb_text, encoding="utf-8")
    return pdb_text, True


def load_cached_or_fetch_bmrb(
    bmrb_id: str,
    bmrb_dir: Optional[pathlib.Path],
    download_missing: bool,
    rate_limit_sec: float,
    need_delay: bool,
) -> Tuple[str, bool]:
    """Load a BMRB NMR-STAR v3 entry from `bmrb_dir` cache or fetch CC0 STAR from BMRB API."""
    if bmrb_dir is not None:
        for cand_name in (
            f"bmr{bmrb_id}.str",
            f"bmr{bmrb_id}_3.str",
            f"{bmrb_id}.str",
        ):
            cand = bmrb_dir / cand_name
            if cand.is_file():
                return cand.read_text(encoding="utf-8", errors="replace"), False
        if not download_missing:
            raise FileNotFoundError(
                f"BMRB {bmrb_id} not found in {bmrb_dir} (pass --download-missing to fetch)"
            )

    if need_delay and rate_limit_sec > 0.0:
        time.sleep(rate_limit_sec)
    bmrb_url = f"https://api.bmrb.io/v2/entry/{bmrb_id}?format=rawnmrstar"
    star_text = http_get_text(bmrb_url)
    if bmrb_dir is not None:
        bmrb_dir.mkdir(parents=True, exist_ok=True)
        (bmrb_dir / f"bmr{bmrb_id}_3.str").write_text(star_text, encoding="utf-8")
    return star_text, True


def build_corpus_from_public_archives(
    provenance_path: pathlib.Path,
    out_path: pathlib.Path,
    pdb_dir: Optional[pathlib.Path] = None,
    bmrb_dir: Optional[pathlib.Path] = None,
    download_missing: bool = False,
    rate_limit_sec: float = 1.0,
) -> None:
    """Rebuild `torsion_corpus.tsv` from local caches and/or public CC0 wwPDB + BMRB sources."""
    inventory = parse_provenance_inventory(provenance_path)
    if len(inventory) != EXPECTED_CHAIN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CHAIN_COUNT} chains in {provenance_path}, found {len(inventory)}"
        )

    lines: List[str] = list(CORPUS_HEADER_LINES)
    did_http = False
    for _idx, (ck, pdb_id, chain_id, bmrb_id, unp, fam, top, exp_len) in enumerate(
        inventory
    ):
        pdb_text, fetched_pdb = load_cached_or_fetch_pdb(
            pdb_id,
            pdb_dir,
            download_missing or (pdb_dir is None),
            rate_limit_sec,
            did_http,
        )
        did_http = did_http or fetched_pdb
        star_text, fetched_star = load_cached_or_fetch_bmrb(
            bmrb_id,
            bmrb_dir,
            download_missing or (bmrb_dir is None),
            rate_limit_sec,
            did_http,
        )
        did_http = did_http or fetched_star

        pdb_res = parse_pdb_chain_model1(pdb_text, chain_id)
        if len(pdb_res) != exp_len:
            raise ValueError(
                f"{ck}: parsed {len(pdb_res)} residues from PDB {pdb_id} chain {chain_id}, expected {exp_len}"
            )
        bmrb_res = parse_bmrb_star_primary_shifts(star_text)
        res_rows = compute_chain_corpus_rows(ck, pdb_res, bmrb_res)
        lines.append(
            "\t".join(
                [
                    "@CHAIN",
                    ck,
                    pdb_id,
                    chain_id,
                    bmrb_id,
                    unp,
                    fam,
                    top,
                    str(len(res_rows)),
                ]
            )
        )
        lines.extend(res_rows)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_corpus(
    corpus_path: pathlib.Path,
    spot_pdb: Optional[pathlib.Path] = None,
    spot_star: Optional[pathlib.Path] = None,
    spot_chain: str = "1UBQ_A",
) -> None:
    """Deterministically verify schema, bounds, SHA-256, and optional local PDB+STAR spot-check."""
    raw = corpus_path.read_bytes()
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{corpus_path} contains non-ASCII bytes: {exc}") from exc

    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_CORPUS_SHA256:
        raise ValueError(
            f"SHA-256 mismatch on {corpus_path}: got {digest}, expected {EXPECTED_CORPUS_SHA256}"
        )

    lines = text.splitlines()
    if len(lines) != EXPECTED_LINE_COUNT:
        raise ValueError(
            f"Line count mismatch: got {len(lines)}, expected {EXPECTED_LINE_COUNT}"
        )

    chain_headers: Dict[str, List[str]] = {}
    chain_rows: Dict[str, List[List[str]]] = {}

    for line_no, line in enumerate(lines, 1):
        if line.startswith("#") or not line.strip():
            continue
        cols = line.split("\t")
        if cols[0] == "@CHAIN":
            if len(cols) != 9:
                raise ValueError(f"Line {line_no}: @CHAIN expects 9 columns, got {len(cols)}")
            ck = cols[1]
            if ck in chain_headers:
                raise ValueError(f"Line {line_no}: duplicate @CHAIN {ck}")
            chain_headers[ck] = cols
            chain_rows[ck] = []
        elif cols[0] == "RES":
            if len(cols) != 16:
                raise ValueError(f"Line {line_no}: RES expects 16 columns, got {len(cols)}")
            ck = cols[1]
            if ck not in chain_headers:
                raise ValueError(f"Line {line_no}: RES for undeclared chain {ck}")
            _ = int(cols[2])
            res_name = cols[3]
            if res_name not in WISHART_1995 and res_name != "CYS":
                raise ValueError(f"Line {line_no}: unknown residue {res_name}")
            # Validate secondary-shift bounds
            shift_caps = (15.0, 45.0, 25.0, 30.0, 25.0, 10.0)
            for k, cap in enumerate(shift_caps):
                tok = cols[4 + k]
                if tok != ".":
                    v = float(tok)
                    if not math.isfinite(v) or abs(v) > cap:
                        raise ValueError(
                            f"Line {line_no}: secondary shift col {k} out of bounds ({v})"
                        )
            # Validate dihedral bounds [-180.0, 180.0]
            for k in range(3):
                tok = cols[10 + k]
                if tok != ".":
                    ang = float(tok)
                    if not math.isfinite(ang) or ang < -180.0 or ang > 180.0:
                        raise ValueError(
                            f"Line {line_no}: torsion angle out of [-180, 180] ({ang})"
                        )
            # Validate finite C-alpha coordinates
            for k in range(3):
                coord = float(cols[13 + k])
                if not math.isfinite(coord):
                    raise ValueError(f"Line {line_no}: non-finite CA coordinate")
            chain_rows[ck].append(cols)
        else:
            raise ValueError(f"Line {line_no}: unexpected record tag {cols[0]!r}")

    total_res = sum(len(v) for v in chain_rows.values())
    if len(chain_headers) != EXPECTED_CHAIN_COUNT:
        raise ValueError(
            f"Chain count mismatch: got {len(chain_headers)}, expected {EXPECTED_CHAIN_COUNT}"
        )
    if total_res != EXPECTED_RESIDUE_COUNT:
        raise ValueError(
            f"Residue count mismatch: got {total_res}, expected {EXPECTED_RESIDUE_COUNT}"
        )
    for ck, hdr in chain_headers.items():
        decl_len = int(hdr[8])
        if len(chain_rows[ck]) != decl_len:
            raise ValueError(
                f"Chain {ck}: header length {decl_len} != RES count {len(chain_rows[ck])}"
            )

    if spot_star is not None and spot_pdb is None:
        raise ValueError("--spot-check-star requires --spot-check-pdb")

    if spot_pdb is not None:
        if spot_chain not in chain_headers:
            raise ValueError(f"Spot-check chain {spot_chain} not in corpus")
        target_chain_id = chain_headers[spot_chain][3]
        pdb_res = parse_pdb_chain_model1(
            spot_pdb.read_text(encoding="utf-8"), target_chain_id
        )
        bmrb_res = (
            parse_bmrb_star_primary_shifts(spot_star.read_text(encoding="utf-8"))
            if spot_star is not None
            else []
        )
        recomputed = [
            r.split("\t")
            for r in compute_chain_corpus_rows(spot_chain, pdb_res, bmrb_res)
        ]
        expected = chain_rows[spot_chain]
        if len(recomputed) != len(expected):
            raise ValueError(
                f"Spot-check {spot_chain}: length {len(recomputed)} != {len(expected)}"
            )
        checked_shifts = 0
        for r_got, r_exp in zip(recomputed, expected):
            # Exact match on seq_id, res_name, phi, psi, chi1, ca_x, ca_y, ca_z
            if r_got[0:4] != r_exp[0:4] or r_got[10:16] != r_exp[10:16]:
                raise ValueError(
                    f"Spot-check geometry mismatch on {spot_chain} seq {r_exp[2]}: got {r_got[10:16]} vs {r_exp[10:16]}"
                )
            if spot_star is not None:
                # Exact match on every shift column present in the local STAR fixture
                for col_i in range(4, 10):
                    if r_got[col_i] != ".":
                        checked_shifts += 1
                        if r_got[col_i] != r_exp[col_i]:
                            raise ValueError(
                                f"Spot-check shift mismatch on {spot_chain} seq {r_exp[2]} col {col_i}: got {r_got[col_i]} vs {r_exp[col_i]}"
                            )
        if spot_star is not None and checked_shifts == 0:
            raise ValueError(f"Spot-check {spot_chain} matched zero assigned shifts")
        print(
            f"SPOT_CHECK_OK chain={spot_chain} residues={len(recomputed)} verified_shifts={checked_shifts}"
        )

    print(
        f"VERIFY_OK chains={len(chain_headers)} residues={total_res} lines={len(lines)} sha256={digest}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean-room builder and offline deterministic verifier for torsion_corpus.tsv."
    )
    parser.add_argument(
        "--verify",
        nargs="?",
        const=pathlib.Path("data/torsion_db/torsion_corpus.tsv"),
        type=pathlib.Path,
        help="Verify schema, bounds, counts, and exact SHA-256 of torsion_corpus.tsv (default: data/torsion_db/torsion_corpus.tsv).",
    )
    parser.add_argument(
        "--spot-check-pdb",
        type=pathlib.Path,
        help="Optional local wwPDB file for live coordinate & dihedral recomputation check.",
    )
    parser.add_argument(
        "--spot-check-star",
        type=pathlib.Path,
        help="Optional local BMRB NMR-STAR file for live secondary-shift recomputation check.",
    )
    parser.add_argument(
        "--chain",
        default="1UBQ_A",
        help="Chain key in torsion_corpus.tsv for --spot-check-pdb/--spot-check-star (default: 1UBQ_A).",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Fetch public CC0 wwPDB + BMRB entries listed in PROVENANCE.md and build torsion_corpus.tsv.",
    )
    parser.add_argument(
        "--provenance",
        "--manifest",
        dest="provenance",
        type=pathlib.Path,
        default=pathlib.Path("data/torsion_db/PROVENANCE.md"),
        help="Path to PROVENANCE.md containing the 129-chain inventory table.",
    )
    parser.add_argument(
        "--pdb-dir",
        type=pathlib.Path,
        default=None,
        help="Optional local cache directory for wwPDB .pdb/.pdb.gz files during --build.",
    )
    parser.add_argument(
        "--bmrb-dir",
        type=pathlib.Path,
        default=None,
        help="Optional local cache directory for BMRB NMR-STAR .str files during --build.",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="When --pdb-dir or --bmrb-dir is given, fetch missing entries from public archives into the cache.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("data/torsion_db/torsion_corpus.tsv"),
        help="Output path when --build is specified.",
    )
    parser.add_argument(
        "--rate-limit-sec",
        type=float,
        default=1.0,
        help="Polite delay in seconds between archive HTTP requests during --build (default: 1.0).",
    )
    args = parser.parse_args(argv)

    if not args.verify and not args.build:
        parser.error("Specify --verify <torsion_corpus.tsv> and/or --build.")

    try:
        if args.build:
            build_corpus_from_public_archives(
                args.provenance,
                args.out,
                pdb_dir=args.pdb_dir,
                bmrb_dir=args.bmrb_dir,
                download_missing=args.download_missing,
                rate_limit_sec=args.rate_limit_sec,
            )
            verify_corpus(args.out)
        if args.verify:
            verify_corpus(
                args.verify,
                spot_pdb=args.spot_check_pdb,
                spot_star=args.spot_check_star,
                spot_chain=args.chain,
            )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
