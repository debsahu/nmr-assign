"""Minimal NMR-STAR reader and writer for backbone chemical shifts.

Original implementation of the public NMR-STAR v3 and v2.1 syntax
(loop_/stop_ blocks, quoted and semicolon-delimited values).
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path

from .priors import ATOMS, ONE

_ATOM_ALIASES = {"HN": "H", "CO": "C"}


@dataclass
class Reference:
    """A sequence plus assigned backbone shifts, keyed by (index, atom)."""
    sequence: list[str]                     # three-letter codes, index 0..K-1
    seq_ids: list[int]                      # original numbering
    shifts: dict[tuple[int, str], float] = field(default_factory=dict)

    @property
    def one_letter(self) -> str:
        return "".join(ONE.get(r, "X") for r in self.sequence)


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(";"):
            buf = [line[1:]]
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                buf.append(lines[i])
                i += 1
            out.append("\n".join(buf).strip())
            i += 1
            continue
        try:
            out.extend(shlex.split(line, comments=True, posix=True))
        except ValueError:
            out.extend(line.split("#", 1)[0].split())
        i += 1
    return out


def _loops(tokens: list[str]) -> list[tuple[list[str], list[list[str]]]]:
    loops = []
    i, n = 0, len(tokens)
    while i < n:
        if tokens[i].lower() == "loop_":
            i += 1
            tags = []
            while i < n and tokens[i].startswith("_"):
                tags.append(tokens[i])
                i += 1
            vals = []
            while i < n and tokens[i].lower() != "stop_" and not tokens[i].startswith(
                    ("_", "save_")) and tokens[i].lower() != "loop_":
                vals.append(tokens[i])
                i += 1
            if tags:
                rows = [vals[j:j + len(tags)] for j in range(0, len(vals) - len(tags) + 1, len(tags))]
                loops.append((tags, rows))
        else:
            i += 1
    return loops


def _col(tags: list[str], *names: str) -> int | None:
    low = [t.lower() for t in tags]
    for name in names:
        name = name.lower()
        for j, t in enumerate(low):
            if t == name or t.endswith("." + name.lstrip("_")):
                return j
    return None


def read_star(path: str | Path, entity: str | None = None) -> Reference:
    tokens = _tokens(Path(path).read_text(errors="replace"))
    loops = _loops(tokens)

    shift_rows: list[tuple[str, int, str, str, float]] = []   # entity, seq, comp, atom, val
    seq_rows: list[tuple[str, int, str]] = []
    for tags, rows in loops:
        val = _col(tags, "_Atom_chem_shift.Val", "_Chem_shift_value")
        if val is not None:
            sid = _col(tags, "_Atom_chem_shift.Seq_ID", "_Atom_chem_shift.Comp_index_ID",
                       "_Residue_seq_code")
            comp = _col(tags, "_Atom_chem_shift.Comp_ID", "_Residue_label")
            atom = _col(tags, "_Atom_chem_shift.Atom_ID", "_Atom_name")
            ent = _col(tags, "_Atom_chem_shift.Entity_ID")
            if None in (sid, comp, atom):
                continue
            for r in rows:
                try:
                    shift_rows.append((r[ent] if ent is not None else "1", int(r[sid]),
                                       r[comp].upper(), r[atom].upper(), float(r[val])))
                except (ValueError, IndexError):
                    continue
            continue
        num = _col(tags, "_Entity_poly_seq.Num", "_Residue_seq_code")
        mon = _col(tags, "_Entity_poly_seq.Mon_ID", "_Residue_label")
        if num is not None and mon is not None and _col(tags, "_Atom_name") is None:
            ent = _col(tags, "_Entity_poly_seq.Entity_ID")
            for r in rows:
                try:
                    seq_rows.append((r[ent] if ent is not None else "1", int(r[num]), r[mon].upper()))
                except (ValueError, IndexError):
                    continue

    if not shift_rows:
        raise ValueError(f"no chemical shift loop found in {path}")
    if entity is None:                                  # entity with most shifts
        counts: dict[str, int] = {}
        for e, *_ in shift_rows:
            counts[e] = counts.get(e, 0) + 1
        entity = max(counts, key=counts.get)

    comps = {s: c for e, s, c, _, _ in shift_rows if e == entity}
    for e, s, c in seq_rows:
        if e == entity or not any(x[0] == entity for x in seq_rows):
            comps.setdefault(s, c)
    lo, hi = min(comps), max(comps)
    seq_ids = list(range(lo, hi + 1))
    sequence = [comps.get(s, "UNK") for s in seq_ids]
    ref = Reference(sequence, seq_ids)
    for e, s, _, atom, v in shift_rows:
        atom = _ATOM_ALIASES.get(atom, atom)
        if e == entity and atom in ATOMS:
            ref.shifts[(s - lo, atom)] = v
    return ref


def write_star(path: str | Path, ref: Reference,
               confidence: dict[int, float] | None = None) -> None:
    lines = ["data_nmr_assign_output", "", "save_assigned_chem_shift_list_1",
             "   _Assigned_chem_shift_list.Sf_category  assigned_chemical_shifts", "",
             "   loop_"]
    tags = ["ID", "Entity_ID", "Seq_ID", "Comp_ID", "Atom_ID", "Atom_type", "Val",
            "Ambiguity_code", "Details"]
    lines += [f"      _Atom_chem_shift.{t}" for t in tags]
    lines.append("")
    n = 0
    for (idx, atom), v in sorted(ref.shifts.items(), key=lambda x: (x[0][0], ATOMS.index(x[0][1]))):
        n += 1
        conf = confidence.get(idx) if confidence else None
        det = f"conf={conf:.2f}" if conf is not None else "."
        lines.append(f"      {n:5d} 1 {ref.seq_ids[idx]:5d} {ref.sequence[idx]} {atom:3s} "
                     f"{atom[0]} {v:9.3f} 1 {det}")
    lines += ["   stop_", "save_", ""]
    Path(path).write_text("\n".join(lines))
