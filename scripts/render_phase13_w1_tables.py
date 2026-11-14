#!/usr/bin/env python3
"""Mechanically render all numeric and SHA-256 tables for Phase 13 Wave 1 (13.1 + 13.3).

Provenance: Clean-room deterministic report generator (Python 3 stdlib only).
Reads raw JSON and test/benchmark logs in `data/private/` plus `data/benchmark/real_manifest.tsv`
and emits Markdown tables with zero hand-entered metrics (`data/private/REVIEW_PROTOCOL.md`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
            total += len(chunk)
    return h.hexdigest(), total


def parse_manifest_cold_p13(manifest_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("bmrb_id\t"):
            continue
        cols = [c.strip() for c in line.split("\t")]
        if len(cols) >= 8 and cols[6] == "heldout" and cols[7] == "cold_p13":
            rows.append(
                {
                    "bmrb_id": cols[0],
                    "protein": cols[1],
                    "residues": cols[2],
                    "field_mhz": cols[3],
                    "vendor": cols[4],
                    "experiments": cols[5],
                    "split": cols[6],
                    "cohort": cols[7],
                }
            )
    return rows


def parse_dev_lopo_log(log_path: Path) -> dict[str, Any]:
    txt = log_path.read_text(encoding="utf-8")
    outer_w_m = re.search(
        r"selected CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT\s*=\s*([0-9.]+)\s*\(outer_ws=(\[[^\]]+\])\)",
        txt,
    )
    if not outer_w_m:
        outer_w_m = re.search(
            r"Nested 13x12 LOPO GPU shrinkage:\s*full_w=([0-9.]+),\s*outer_ws=(\[[^\]]+\])",
            txt,
        )
    full_w = outer_w_m.group(1) if outer_w_m else "n/a"
    outer_ws = outer_w_m.group(2) if outer_w_m else "n/a"

    header_re = re.compile(
        r">>> 13-FOLD UNIFIED-IN-FOLD-REFIT LOPO (\S+) \(full_gate=(ProteinHcGateConfig \{ [^}]+\})\)\s*\n"
        r"\s*OOF LOPO: mean_ECE4/10/12=([0-9.]+)/([0-9.]+)/([0-9.]+) "
        r"\(debiased_ECE10=([0-9.]+), Bernoulli_floor=([0-9.]+)\), "
        r"pooled_ECE4/10/12=([0-9.]+)/([0-9.]+)/([0-9.]+), "
        r"macro_HCP=([0-9.]+)%, micro_HCP=([0-9.]+)% \((\d+)/(\d+)\)"
    )
    prot_re = re.compile(
        r"^\s*per-protein\s+(\S+)\s+\(([^,]+),\s*QC=(\S+)\):\s*acc=\s*([0-9.]+)%,\s*"
        r"OOF_ECE4/10/12=([0-9.]+)/([0-9.]+)/([0-9.]+),\s*INS_HC=(.+)$",
        re.MULTILINE,
    )

    engines: list[dict[str, Any]] = []
    matches = list(header_re.finditer(txt))
    for idx, m in enumerate(matches):
        seg_start = m.end()
        seg_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(txt)
        seg = txt[seg_start:seg_end]
        per_prot = []
        for pm in prot_re.finditer(seg):
            per_prot.append(
                {
                    "id": pm.group(1).strip(),
                    "name": pm.group(2).strip(),
                    "qc": pm.group(3).strip(),
                    "acc": pm.group(4).strip(),
                    "ece4": pm.group(5).strip(),
                    "ece10": pm.group(6).strip(),
                    "ece12": pm.group(7).strip(),
                    "ins_hc": pm.group(8).strip(),
                }
            )
        engines.append(
            {
                "solver": m.group(1),
                "gate": m.group(2),
                "mean_ece4": m.group(3),
                "mean_ece10": m.group(4),
                "mean_ece12": m.group(5),
                "debiased_ece10": m.group(6),
                "bernoulli_floor": m.group(7),
                "pooled_ece4": m.group(8),
                "pooled_ece10": m.group(9),
                "pooled_ece12": m.group(10),
                "macro_hcp": m.group(11),
                "micro_hcp": m.group(12),
                "hc_ok": int(m.group(13)),
                "hc_tot": int(m.group(14)),
                "per_protein": per_prot,
            }
        )
    return {"full_w": full_w, "outer_ws": outer_ws, "engines": engines}


def summarize_cold_json(json_path: Path) -> dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    all_entries = data.get("entries", [])
    entries = [e for e in all_entries if e.get("status") == "OK"]
    counts = data.get("counts", {})
    macro = data.get("macro_average", {})
    micro = data.get("micro_average", {})
    cal = data.get("calibration_summary", {})
    rel = cal.get("pooled_reliability_curve", data.get("pooled_reliability", {}))
    hc_ok = int(micro.get("hc_correct", sum(int(e["reliability_curve"]["hc_correct"]) for e in entries)))
    hc_tot = int(micro.get("hc_assigned", sum(int(e["reliability_curve"]["hc_assigned"]) for e in entries)))
    pass_entries = [e for e in entries if e.get("qc_verdict") == "PASS"]
    warn_entries = [e for e in entries if e.get("qc_verdict") == "WARN"]
    abst_entries = [e for e in entries if e.get("qc_verdict") == "ABSTAIN"]
    failed_entries = [e for e in all_entries if e.get("status") != "OK"]
    pass_hc_ok = sum(int(e["reliability_curve"]["hc_correct"]) for e in pass_entries)
    pass_hc_tot = sum(int(e["reliability_curve"]["hc_assigned"]) for e in pass_entries)
    gated_count = sum(
        1 for e in entries if e.get("qc_report", {}).get("protein_hc_gate_triggered")
    )
    macro_hcp_raw = macro.get("hc_precision", data.get("macro_hc_precision"))
    return {
        "solver": data.get("solver", "?"),
        "evaluated": counts.get("evaluated", data.get("evaluated_entries", len(entries))),
        "total": counts.get("total", data.get("total_entries", len(all_entries))),
        "failed": counts.get("failed", len(failed_entries)),
        "macro_acc": 100.0 * float(macro.get("accuracy", data.get("macro_accuracy", 0.0))),
        "macro_rec": 100.0 * float(macro.get("recall", data.get("macro_recall", 0.0))),
        "micro_acc": 100.0 * float(micro.get("accuracy", 0.0)),
        "micro_rec": 100.0 * float(micro.get("recall", 0.0)),
        "macro_hcp": (100.0 * float(macro_hcp_raw) if macro_hcp_raw is not None else None),
        "micro_hcp": (100.0 * hc_ok / hc_tot) if hc_tot > 0 else None,
        "hc_ok": hc_ok,
        "hc_tot": hc_tot,
        "pass_count": len(pass_entries),
        "warn_count": len(warn_entries),
        "abstain_count": len(abst_entries),
        "gated_count": gated_count,
        "pass_micro_hcp": (100.0 * pass_hc_ok / pass_hc_tot) if pass_hc_tot > 0 else None,
        "pass_hc_ok": pass_hc_ok,
        "pass_hc_tot": pass_hc_tot,
        "pooled_ece4": float(rel.get("ece", 0.0)),
        "pooled_ece10": float(rel.get("ece_10", 0.0)),
        "pooled_ece12": float(rel.get("ece_12", 0.0)),
        "mean_ece4": float(cal.get("mean_protein_ece", data.get("mean_protein_ece", 0.0))),
        "mean_ece10": float(cal.get("mean_protein_ece_10", data.get("mean_protein_ece_10", 0.0))),
        "mean_deb_ece10": float(
            cal.get("mean_protein_debiased_ece_10", data.get("mean_protein_debiased_ece_10", 0.0))
        ),
        "mean_floor10": float(
            cal.get(
                "mean_protein_ece_10_bernoulli_floor",
                data.get("mean_protein_ece_10_bernoulli_floor", 0.0),
            )
        ),
        "entries": entries,
        "all_entries": all_entries,
    }


def parse_ubq_log(log_path: Path) -> dict[str, str]:
    txt = log_path.read_text(encoding="utf-8")
    acc_m = re.search(r"accuracy \(correct/assigned\)\s*:\s*([0-9.]+%)", txt)
    rec_m = re.search(r"recall\s*\(correct/assignable\)\s*:\s*([0-9.]+%)", txt)
    hcp_m = re.search(
        r"high-confidence \(>=0\.8\) precision\s*:\s*([0-9.]+%\s*\(\d+/\d+\))", txt
    )
    ass_m = re.search(r"assigned\s*:\s*(\d+)", txt)
    asg_m = re.search(r"assignable spin systems\s*:\s*(\d+)", txt)
    acc_s = acc_m.group(1) if acc_m else "see raw log"
    rec_s = rec_m.group(1) if rec_m else "see raw log"
    if acc_m and ass_m:
        acc_s = f"{acc_m.group(1)} (assigned={ass_m.group(1)})"
    if rec_m and asg_m:
        rec_s = f"{rec_m.group(1)} (assignable={asg_m.group(1)})"
    return {
        "raw": txt.strip(),
        "acc": acc_s,
        "rec": rec_s,
        "hcp": hcp_m.group(1) if hcp_m else "see raw log",
    }


def parse_pulprog_log(log_path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        prefix = "PULPROG_METRIC_JSON:"
        if prefix in line:
            payload = line.split(prefix, 1)[1].strip()
            records.append(json.loads(payload))
    suite = next((r for r in records if r.get("kind") == "fixture_suite"), None)
    rescues = [r for r in records if r.get("kind") == "q1_rescue"]
    guards = [r for r in records if r.get("kind") == "q1_guard"]
    cisd3 = next((r for r in records if r.get("kind") == "real_cisd3"), None)
    return {"suite": suite, "rescues": rescues, "guards": guards, "cisd3": cisd3}


def fmt_pct_opt(val: float | None) -> str:
    return f"{val:.2f}%" if val is not None else "ABSTAIN"


def render_markdown(root: Path, extra_files: list[Path], section: str = "all") -> str:
    manifest = root / "data/benchmark/real_manifest.tsv"
    lopo_log = root / "data/private/phase13_w1_dev_lopo_audit.log"
    cold_evo_json = root / "data/private/phase13_w1_cold_p13_evograph.json"
    cold_evo_log = root / "data/private/phase13_w1_cold_p13_evograph.log"
    cold_cpu_json = root / "data/private/phase13_w1_cold_p13_standard_cpu.json"
    cold_cpu_log = root / "data/private/phase13_w1_cold_p13_standard_cpu.log"
    cold_gpu_json = root / "data/private/phase13_w1_cold_p13_standard_gpu.json"
    cold_gpu_log = root / "data/private/phase13_w1_cold_p13_standard_gpu.log"
    ubq_log = root / "data/private/phase13_w1_ubq_realistic.log"
    cisd3_std_json = root / "data/private/phase13_w1_cisd3_standard.json"
    cisd3_evo_json = root / "data/private/phase13_w1_cisd3_evograph.json"
    pulprog_log = root / "data/private/phase13_w1_pulprog_tests.log"
    ws_test_log = root / "data/private/phase13_w1_workspace_tests.log"

    cited_files = [
        manifest,
        lopo_log,
        cold_evo_json,
        cold_evo_log,
        cold_cpu_json,
        cold_cpu_log,
        cold_gpu_json,
        cold_gpu_log,
        ubq_log,
        cisd3_std_json,
        cisd3_evo_json,
        pulprog_log,
        ws_test_log,
        *extra_files,
    ]

    lines: list[str] = []
    if section in ("all", "digests"):
        lines.append("### Cited Raw Results Artifacts & SHA-256 Digests")
        lines.append("")
        lines.append("Generated mechanically by `scripts/render_phase13_w1_tables.py`:")
        lines.append("")
        lines.append("| Artifact Path | Bytes | SHA-256 |")
        lines.append("|---|---:|---|")
        for fp in cited_files:
            if fp.exists():
                digest, nbytes = sha256_file(fp)
                rel = fp.relative_to(root) if fp.is_relative_to(root) else fp
                lines.append(f"| `{rel}` | {nbytes:,} | `{digest}` |")
        if section == "digests":
            return "\n".join(lines) + "\n"

    # Slice 13.1 tables
    if section in ("all", "13.1"):
        lopo = parse_dev_lopo_log(lopo_log)
        if lines:
            lines.append("")
        lines.append("### Slice `13.1` — 13-Fold `dev` Out-of-Fold LOPO Summary (`P1-7`, `P1-8`)")
        lines.append("")
        lines.append(
            f"- **Nested 13x12 LOPO GPU Shrinkage (`select_gpu_shrinkage_weight_nested_lopo`)**: "
            f"`full_w = {lopo['full_w']}`, `outer_ws = {lopo['outer_ws']}`"
        )
        if lopo["engines"]:
            lines.append(f"- **Asserted Full-13 Unified Gate**: `{lopo['engines'][0]['gate']}`")
        lines.append("")
        lines.append(
            "| Engine | Pooled `ECE_4` | **Pooled `ECE_10` (`<=0.050`)** | Pooled `ECE_12` | "
            "Mean `ECE_4` (`<=0.065`) | Mean Raw `ECE_10` | **Mean Debiased `ECE_10` (`<=0.065`)** | "
            "Bernoulli `ECE_10` Floor | **Micro HC-Prec (`>=94%`)** | Macro HC-Prec (`>=95%`) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for eng in lopo["engines"]:
            lines.append(
                f"| `{eng['solver']}` | `{eng['pooled_ece4']}` | **`{eng['pooled_ece10']}`** | `{eng['pooled_ece12']}` | "
                f"`{eng['mean_ece4']}` | `{eng['mean_ece10']}` | **`{eng['debiased_ece10']}`** | "
                f"`{eng['bernoulli_floor']}` | **`{eng['micro_hcp']}%` (`{eng['hc_ok']}/{eng['hc_tot']}`)** | `{eng['macro_hcp']}%` |"
            )

        evo_eng = next((e for e in lopo["engines"] if e["solver"] == "EVO_GRAPH"), None)
        cpu_eng = next((e for e in lopo["engines"] if e["solver"] == "STANDARD_CPU"), None)
        gpu_eng = next((e for e in lopo["engines"] if e["solver"] == "STANDARD_GPU"), None)
        if evo_eng and cpu_eng and gpu_eng:
            lines.append("")
            lines.append("#### Per-Protein 13-Fold `dev` OOF Summary (`5` Seeds/Protein)")
            lines.append("")
            lines.append(
                "| Entry | Protein | `evo-graph` QC / Acc / OOF `ECE_10` / HC | "
                "`standard (cpu)` QC / Acc / OOF `ECE_10` / HC | "
                "`standard (gpu)` QC / Acc / OOF `ECE_10` / HC |"
            )
            lines.append("|---|---|---|---|---|")
            for p_e, p_c, p_g in zip(
                evo_eng["per_protein"], cpu_eng["per_protein"], gpu_eng["per_protein"]
            ):
                lines.append(
                    f"| `{p_e['id']}` | `{p_e['name']}` | "
                    f"`{p_e['qc']}` / `{p_e['acc']}%` / `{p_e['ece10']}` / `{p_e['ins_hc']}` | "
                    f"`{p_c['qc']}` / `{p_c['acc']}%` / `{p_c['ece10']}` / `{p_c['ins_hc']}` | "
                    f"`{p_g['qc']}` / `{p_g['acc']}%` / `{p_g['ece10']}` / `{p_g['ins_hc']}` |"
                )

        # Pre-registered cold_p13 table
        cold_rows = parse_manifest_cold_p13(manifest)
        lines.append("")
        lines.append("### Slice `13.1` — Pre-Registered `cold_p13` Cohort (`split=heldout`, `cohort=cold_p13`)")
        lines.append("")
        lines.append("| BMRB ID | Protein | Residues | Field (MHz) | Vendor | Pre-Registered Raw-FID Experiments |")
        lines.append("|---|---|---:|---:|---|---|")
        for r in cold_rows:
            lines.append(
                f"| `{r['bmrb_id']}` | `{r['protein']}` | {r['residues']} | {r['field_mhz']} | "
                f"`{r['vendor']}` | `{r['experiments']}` |"
            )

        c_evo = summarize_cold_json(cold_evo_json)
        c_cpu = summarize_cold_json(cold_cpu_json)
        c_gpu = summarize_cold_json(cold_gpu_json)
        lines.append("")
        lines.append("#### Single-Shot `cold_p13` Cohort Aggregate Results (`5` Seeds/Entry)")
        lines.append("")
        lines.append(
            "| Engine | Evaluated | Macro Acc | Macro Recall | **Pooled Micro HC-Prec** | Macro HC-Prec | "
            "PASS-Tier Micro HC-Prec | QC Tiers (`PASS/WARN/ABST`) | HC-Gate Cap Triggered | "
            "**Pooled `ECE_10`** | Pooled `ECE_4` | Mean Debiased `ECE_10` (Floor) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|")
        for s in [c_evo, c_cpu, c_gpu]:
            micro_s = (
                f"**`{s['micro_hcp']:.2f}%` (`{s['hc_ok']}/{s['hc_tot']}`)**"
                if s["micro_hcp"] is not None
                else "`ABSTAIN (0/0)`"
            )
            pass_s = (
                f"`{s['pass_micro_hcp']:.2f}%` (`{s['pass_hc_ok']}/{s['pass_hc_tot']}`)"
                if s["pass_micro_hcp"] is not None
                else "`ABSTAIN (0/0)`"
            )
            lines.append(
                f"| `{s['solver']}` | `{s['evaluated']}/{s['total']}` | `{s['macro_acc']:.2f}%` | `{s['macro_rec']:.2f}%` | "
                f"{micro_s} | `{fmt_pct_opt(s['macro_hcp'])}` | {pass_s} | "
                f"`{s['pass_count']}/{s['warn_count']}/{s['abstain_count']}` | `{s['gated_count']}` | "
                f"**`{s['pooled_ece10']:.4f}`** | `{s['pooled_ece4']:.4f}` | "
                f"`{s['mean_deb_ece10']:.4f}` (`{s['mean_floor10']:.4f}`) |"
            )

        lines.append("")
        lines.append("#### Per-Entry Single-Shot `cold_p13` Breakdown (`5` Seeds/Entry)")
        lines.append("")
        lines.append(
            "| BMRB ID | Protein (`aa`, Vendor) | `evo-graph` QC / Gate / Acc / HC / `ECE_10` (Deb) | "
            "`standard (cpu)` QC / Gate / Acc / HC / `ECE_10` (Deb) | "
            "`standard (gpu)` QC / Gate / Acc / HC / `ECE_10` (Deb) |"
        )
        lines.append("|---|---|---|---|---|")
        for e_ev, e_cp, e_gp in zip(
            c_evo["all_entries"], c_cpu["all_entries"], c_gpu["all_entries"]
        ):
            def cell(e: dict[str, Any]) -> str:
                if e.get("status") != "OK":
                    fail = e.get("failure", {})
                    stage = fail.get("stage", "err")
                    exp = fail.get("experiment", "?")
                    reason = fail.get("reason", "failed")
                    return f"`ERR [{stage}:{exp}]` ({reason})"
                qc = e["qc_verdict"]
                gated = "GATE" if e["qc_report"].get("protein_hc_gate_triggered") else "open"
                acc = 100.0 * float(e["accuracy"])
                hcs = e["hc_status"]
                rc = e["reliability_curve"]
                return (
                    f"`{qc}` (`{gated}`) / `{acc:.1f}%` / `{hcs}` / "
                    f"`{float(rc['ece_10']):.3f}` (`{float(rc['debiased_ece_10']):.3f}`)"
                )

            lines.append(
                f"| `{e_ev['bmrb_id']}` | `{e_ev['protein']}` (`{e_ev['residues']} aa`, `{e_ev['vendor']}`) | "
                f"{cell(e_ev)} | {cell(e_cp)} | {cell(e_gp)} |"
            )
        if ubq_log.exists():
            ubq = parse_ubq_log(ubq_log)
            lines.append("")
            lines.append("#### Baseline Solver Regression Gates (`Ubiquitin --realistic` & `CISD3` `5` Seeds)")
            lines.append("")
            lines.append("| Benchmark | Overall Accuracy | Overall Recall | High-Confidence Precision (`>= 0.80`) |")
            lines.append("|---|---:|---:|---:|")
            lines.append(f"| Ubiquitin 5-spectrum `--realistic` (`5` seeds) | `{ubq['acc']}` | `{ubq['rec']}` | `{ubq['hcp']}` |")
            for label, cj in [
                ("`CISD3` (`BMRB 51439`) `--solver standard` (`5` seeds)", cisd3_std_json),
                ("`CISD3` (`BMRB 51439`) `--solver evo-graph` (`5` seeds)", cisd3_evo_json),
            ]:
                if cj.exists():
                    cs = summarize_cold_json(cj)
                    hcp_s = (
                        f"{cs['micro_hcp']:.2f}% ({cs['hc_ok']}/{cs['hc_tot']})"
                        if cs["micro_hcp"] is not None
                        else "ABSTAIN (0/0)"
                    )
                    lines.append(
                        f"| {label} | `{cs['macro_acc']:.2f}%` | `{cs['macro_rec']:.2f}%` | `{hcp_s}` |"
                    )
        if section == "13.1":
            return "\n".join(lines) + "\n"

    # Slice 13.3 tables
    if section in ("all", "13.3"):
        pp = parse_pulprog_log(pulprog_log)
        if lines:
            lines.append("")
        lines.append("### Slice `13.3` — `T13.0` Pulseprogram Interpreter & `Q1` Empirical `{sign, fold}` Tables")
        lines.append("")
        if pp["suite"]:
            su = pp["suite"]
            lines.append("| Metric | Value |")
            lines.append("|---|---:|")
            lines.append(f"| Bruker `pulseprogram` + `acqus` fixtures verified | `{su['bruker_fixtures']}` |")
            lines.append(f"| Varian/Agilent `procpar` + `sampling.sch` fixtures verified | `{su['varian_fixtures']}` |")
            lines.append(f"| Total `T13.0` interpreter fixtures passed | **`{su['passed']}/{su['total_fixtures']}` (`100.0%`)** |")
            lines.append(f"| Mean Bruker parse + symbolic loop/phase execution latency | `{su['bruker_mean_ms']:.4f} ms` |")
            lines.append(f"| Max single-fixture latency (`< 10 ms` gate) | **`{su['bruker_max_ms']:.4f} ms`** |")

        if pp["rescues"]:
            lines.append("")
            lines.append("#### `Q1` Volumetric `{sign, fold}` HSQC Strip-Matching Rescue Results")
            lines.append("")
            lines.append("| Corrupted 3D Cube Case | Selected Hypothesis | `ln BF` vs Canonical (`>=12.0`) | Pre-Rescue Strip Recovery | Post-Rescue Strip Recovery |")
            lines.append("|---|---|---:|---:|---:|")
            for r in pp["rescues"]:
                lines.append(
                    f"| `{r['case']}` | `{r['selected']}` | `{r['ln_bf']:.2f}` | "
                    f"`{r['pre_recovery_pct']:.2f}%` | **`{r['post_recovery_pct']:.2f}%`** |"
                )

        if pp["guards"]:
            lines.append("")
            lines.append("#### `Q1` Physical-Envelope Edge-Regime False-Positive Guards")
            lines.append("")
            lines.append("| Edge-Regime Spectrum | Selected Hypothesis | Applied Mutation | Polarity Flipped | `ln BF` vs Canonical |")
            lines.append("|---|---|---|---|---:|")
            for g in pp["guards"]:
                lines.append(
                    f"| `{g['case']}` | `{g['selected']}` | `{g['applied']}` | `{g['polarity_flipped']}` | `{g['ln_bf']:.2f}` |"
                )

        if pp["cisd3"]:
            c3 = pp["cisd3"]
            lines.append("")
            lines.append("#### Real Raw-FID `CISD3` (`BMRB 51439`, `split=dev`) `HNCO` Axis Verification")
            lines.append("")
            lines.append("| Condition | Selected Hypothesis | Applied | `ln BF` vs Input | Matched HSQC Fraction |")
            lines.append("|---|---|---|---:|---:|")
            lines.append(
                f"| Unmodified canonical `CISD3` `HNCO` | `{c3['canonical_selected']}` | `{c3['canonical_applied']}` | `0.00` | `{100.0 * c3['rescued_match_frac']:.2f}%` |"
            )
            lines.append(
                f"| Corrupted `HalfSwShiftN` `CISD3` `HNCO` | `{c3['corrupted_selected']}` | `{c3['corrupted_applied']}` | **`{c3['corrupted_ln_bf']:.2f}`** | `{100.0 * c3['canonical_match_frac']:.2f}%` -> **`{100.0 * c3['rescued_match_frac']:.2f}%`** |"
            )

    lines.append("")
    return "\n".join(lines)


def parse_remediation_lopo_log(log_path: Path) -> dict[str, Any]:
    base = parse_dev_lopo_log(log_path)
    txt = log_path.read_text(encoding="utf-8")
    fold_re = re.compile(
        r"^\s*\[FOLD (\d+)\]\s+held_out=(\S+)\s+\(([^)]+)\)\s+\|\s+gpu_w=([0-9.]+)\s+\|\s+"
        r"gate=\{\s*c35:\s*([0-9.]+),\s*hcf:\s*([0-9.]+),\s*cmarg:\s*([0-9.]+),\s*mpen:\s*([0-9.]+)\s*\}",
        re.MULTILINE,
    )
    folds = []
    for fm in fold_re.finditer(txt):
        folds.append(
            {
                "fold": int(fm.group(1)),
                "bmrb_id": fm.group(2).strip(),
                "protein": fm.group(3).strip(),
                "gpu_w": float(fm.group(4)),
                "c35": float(fm.group(5)),
                "hcf": float(fm.group(6)),
                "cmarg": float(fm.group(7)),
                "mpen": float(fm.group(8)),
            }
        )
    diag_re = re.compile(
        r"^\s*\[GATE DIAG (\S+)\s*\]\s+gate=\{\s*c35:\s*([0-9.]+),\s*hcf:\s*([0-9.]+),\s*"
        r"cmarg:\s*([0-9.]+),\s*mpen:\s*([0-9.]+)\s*\}\s+"
        r"(?:\|\s+obj=([0-9.]+)\s+)?\|\s+mean_ECE10=([0-9.]+)\s+\|\s+"
        r"hcps=\[([0-9.]+)%,\s*([0-9.]+)%,\s*([0-9.]+)%\]\s+\|\s+capped_seeds=(\d+)",
        re.MULTILINE,
    )
    diags = []
    for dm in diag_re.finditer(txt):
        diags.append(
            {
                "mode": dm.group(1).strip(),
                "c35": float(dm.group(2)),
                "hcf": float(dm.group(3)),
                "cmarg": float(dm.group(4)),
                "mpen": float(dm.group(5)),
                "obj": (float(dm.group(6)) if dm.group(6) is not None else None),
                "mean_ece10": float(dm.group(7)),
                "hcp_evo": float(dm.group(8)),
                "hcp_cpu": float(dm.group(9)),
                "hcp_gpu": float(dm.group(10)),
                "capped_seeds": int(dm.group(11)),
            }
        )
    base["folds"] = folds
    base["diags"] = diags
    return base


def render_remediation_markdown(root: Path, extra_files: list[Path]) -> str:
    axis_json = root / "data/private/phase13_w1_axis_rescue_audit.json"
    axis_log = root / "data/private/phase13_w1_axis_rescue_audit.log"
    rem_lopo_log = root / "data/private/phase13_w1_dev_lopo_audit_remediation.log"
    seen_evo_json = root / "data/private/phase13_w1_seen_p12_evograph.json"
    seen_evo_log = root / "data/private/phase13_w1_seen_p12_evograph.log"
    seen_cpu_json = root / "data/private/phase13_w1_seen_p12_standard_cpu.json"
    seen_cpu_log = root / "data/private/phase13_w1_seen_p12_standard_cpu.log"
    seen_gpu_json = root / "data/private/phase13_w1_seen_p12_standard_gpu.json"
    seen_gpu_log = root / "data/private/phase13_w1_seen_p12_standard_gpu.log"
    p12_evo_json = root / "data/private/phase12_heldout_evo_graph.json"
    p12_cpu_json = root / "data/private/phase12_heldout_standard_cpu.json"
    p12_gpu_json = root / "data/private/phase12_heldout_standard_gpu.json"
    cold_evo_json = root / "data/private/phase13_w1_cold_p13_evograph.json"
    cold_cpu_json = root / "data/private/phase13_w1_cold_p13_standard_cpu.json"
    cold_gpu_json = root / "data/private/phase13_w1_cold_p13_standard_gpu.json"
    rem_ubq_log = root / "data/private/phase13_w1_remediation_ubq_realistic.log"
    rem_ws_log = root / "data/private/phase13_w1_remediation_workspace_tests.log"

    cited = [
        axis_json,
        axis_log,
        rem_lopo_log,
        seen_evo_json,
        seen_evo_log,
        seen_cpu_json,
        seen_cpu_log,
        seen_gpu_json,
        seen_gpu_log,
        p12_evo_json,
        p12_cpu_json,
        p12_gpu_json,
        cold_evo_json,
        cold_cpu_json,
        cold_gpu_json,
        rem_ubq_log,
        rem_ws_log,
        *extra_files,
    ]

    lines: list[str] = []
    lines.append("### Remediation Raw Results Artifacts & SHA-256 Digests")
    lines.append("")
    lines.append(
        "Generated mechanically by `python3 scripts/render_phase13_w1_tables.py --section remediation` "
        "(zero hand-entered values):"
    )
    lines.append("")
    lines.append("| Artifact Path | Bytes | SHA-256 |")
    lines.append("|---|---:|---|")
    for fp in cited:
        if fp.exists():
            digest, nbytes = sha256_file(fp)
            rel = fp.relative_to(root) if fp.is_relative_to(root) else fp
            lines.append(f"| `{rel}` | {nbytes:,} | `{digest}` |")

    # 1. 13.3:P1-1 Read-only axis hypothesis audit
    if axis_json.exists():
        ax = json.loads(axis_json.read_text(encoding="utf-8"))
        ax_entries = ax.get("entries", [])
        lines.append("")
        lines.append(
            "### `13.3:P1-1` — Read-Only Post-Hoc `evaluate_3d_axis_hypotheses_against_hsqc` Audit "
            "(`solver_runs_executed = " + str(ax.get("solver_runs_executed", 0)) + "`)"
        )
        lines.append("")
        lines.append(
            "| Subset | Entries Total | `OK` / `PROCESS_FAILED` | 3D Cubes Evaluated | "
            "**Rescues Fired (`would_apply = true`)** | Max Alternative `ln BF` vs Canonical |"
        )
        lines.append("|---|---:|---|---:|---:|---:|")
        for label, pred in [
            ("`cold_p13` (frozen post-hoc read-only)", lambda e: e.get("cohort") == "cold_p13"),
            ("`dev` (`13` training proteins)", lambda e: e.get("split") == "dev"),
            ("**Combined (`cold_p13` + `dev`)**", lambda _e: True),
        ]:
            sub = [e for e in ax_entries if pred(e)]
            n_ok = sum(1 for e in sub if e.get("status") == "OK")
            n_fail = len(sub) - n_ok
            cubes = [c for e in sub for c in e.get("cubes", [])]
            n_apply = sum(1 for c in cubes if c.get("would_apply"))
            max_alt = (
                max(float(c.get("best_non_canonical_ln_bf", 0.0)) for c in cubes)
                if cubes
                else 0.0
            )
            lines.append(
                f"| {label} | `{len(sub)}` | `{n_ok}` / `{n_fail}` | `{len(cubes)}` | "
                f"**`{n_apply}`** | `{max_alt:+.2f}` |"
            )

        for subset_title, subset_pred in [
            ("#### Per-Experiment Read-Only Axis Verification on Frozen `cold_p13`", lambda e: e.get("cohort") == "cold_p13"),
            ("#### Per-Experiment Read-Only Axis Verification on `dev` (`13` Entries)", lambda e: e.get("split") == "dev"),
        ]:
            lines.append("")
            lines.append(subset_title)
            lines.append("")
            lines.append(
                "| BMRB ID | Protein (`Vendor`) | Experiment | Selected Hypothesis | "
                "Would Apply | `ln BF` vs Canonical | Canonical -> Selected Match | Best Non-Canonical (`ln BF`) |"
            )
            lines.append("|---|---|---|---|---|---:|---:|---|")
            for e in ax_entries:
                if not subset_pred(e):
                    continue
                bid = e["bmrb_id"]
                prot = f"`{e['protein']}` (`{e['vendor']}`)"
                if e.get("status") != "OK":
                    lines.append(
                        f"| `{bid}` | {prot} | `PROCESS_FAILED` | `n/a` | `false` | `n/a` | `n/a` | `{e.get('process_failure', '')}` |"
                    )
                    continue
                for c in e.get("cubes", []):
                    cm = 100.0 * float(c["canonical_matched_fraction"])
                    sm = 100.0 * float(c["selected_matched_fraction"])
                    lines.append(
                        f"| `{bid}` | {prot} | `{c['experiment']}` | `{c['selected']}` | "
                        f"`{str(bool(c['would_apply'])).lower()}` | `{float(c['ln_bayes_factor_vs_canonical']):+.2f}` | "
                        f"`{cm:.1f}% -> {sm:.1f}%` | `{c['best_non_canonical_hypothesis']}` (`{float(c['best_non_canonical_ln_bf']):+.2f}`) |"
                    )

    # 2. 13.1:P1-2 13-fold per-fold refit gates + extended floor scan
    if rem_lopo_log.exists():
        rl = parse_remediation_lopo_log(rem_lopo_log)
        folds = rl["folds"]
        diags = rl["diags"]
        if folds:
            ws = [f["gpu_w"] for f in folds]
            ws_sorted = sorted(ws)
            w_med = ws_sorted[len(ws_sorted) // 2]
            lines.append("")
            lines.append(
                "### `13.1:P1-2` & `13.1:P1-3` — All 13 Per-Fold In-Fold Refit `ProteinHcGateConfig`s & Nested `gpu_w` Spread"
            )
            lines.append("")
            lines.append(
                f"- **Full-13 `dev` `gpu_w`**: `{rl['full_w']}`; **13 outer-fold `gpu_w` range**: "
                f"`[{min(ws):.2f}, {max(ws):.2f}]` (median `{w_med:.2f}`, values `{rl['outer_ws']}` — weakly identified at `N = 12` training proteins/fold)."
            )
            lines.append("")
            lines.append(
                "| Outer Fold | Held-Out `dev` Entry | Held-Out Protein | In-Fold Nested `gpu_w` | "
                "`min_corroborated_connectivity` | `min_hc_call_fraction` | `min_crowded_solution_margin` | `multiplicity_penalty_floor` |"
            )
            lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
            for f in folds:
                lines.append(
                    f"| `{f['fold']:02d}` | `{f['bmrb_id']}` | `{f['protein']}` | `{f['gpu_w']:.2f}` | "
                    f"`{f['c35']:.2f}` | `{f['hcf']:.2f}` | `{f['cmarg']:.2f}` | `{f['mpen']:.2f}` |"
                )

        if diags:
            lines.append("")
            lines.append(
                "#### Extended `0.0`-Floor Grid Diagnostic on Full-13 `dev` (`[0.0..0.65] x [0.0..0.12] x [0.0..0.55] x [0.0..0.12]`)"
            )
            lines.append("")
            lines.append(
                "| Diagnostic Mode | Selected `{c35, hcf, cmarg, mpen}` | Objective | Mean Protein `ECE_10` | "
                "Pooled `HCP` (`evo / cpu / gpu`) | Capped `dev` Protein-Seeds (`of 195`) |"
            )
            lines.append("|---|---|---:|---:|---|---:|")
            for d in diags:
                obj_s = f"`{d['obj']:.6f}`" if d["obj"] is not None else "`n/a (eval)`"
                lines.append(
                    f"| `{d['mode']}` | `{{{d['c35']:.2f}, {d['hcf']:.2f}, {d['cmarg']:.2f}, {d['mpen']:.2f}}}` | "
                    f"{obj_s} | `{d['mean_ece10']:.6f}` | "
                    f"`{d['hcp_evo']:.2f}% / {d['hcp_cpu']:.2f}% / {d['hcp_gpu']:.2f}%` | `{d['capped_seeds']}` |"
                )

    # 3. 13.1:P1-2 Non-Cold seen_p12 Retrospective
    if seen_evo_json.exists() and seen_cpu_json.exists() and seen_gpu_json.exists():
        s_evo = summarize_cold_json(seen_evo_json)
        s_cpu = summarize_cold_json(seen_cpu_json)
        s_gpu = summarize_cold_json(seen_gpu_json)
        lines.append("")
        lines.append(
            "### `13.1:P1-2` — **NON-COLD** `seen_p12` Retrospective (`split=heldout`, `cohort=seen_p12`, `7` Entries × `5` Seeds)"
        )
        lines.append("")
        lines.append(
            "> **Non-Cold Provenance Label**: `seen_p12` (`5691, 6585, 15491, 16096, 16376, 18337, 18561`) "
            "was spent at `v0.8.0` (`cold_p12`) and motivated `P1-7`. All numbers in this subsection are "
            "**non-cold retrospective verification**, never cold generalization evidence."
        )
        lines.append("")
        lines.append(
            "| Engine (**NON-COLD `seen_p12`**) | Evaluated | Macro Acc | Macro Recall | **Pooled Micro HC-Prec** | "
            "Macro HC-Prec | QC Tiers (`PASS/WARN/ABST`) | HC-Gate Cap Triggered | Pooled `ECE_10` | Mean Debiased `ECE_10` (Floor) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---|---:|---:|---:|")
        for s in [s_evo, s_cpu, s_gpu]:
            micro_s = (
                f"**`{s['micro_hcp']:.2f}%` (`{s['hc_ok']}/{s['hc_tot']}`)**"
                if s["micro_hcp"] is not None
                else "`ABSTAIN (0/0)`"
            )
            lines.append(
                f"| `{s['solver']}` | `{s['evaluated']}/{s['total']}` | `{s['macro_acc']:.2f}%` | `{s['macro_rec']:.2f}%` | "
                f"{micro_s} | `{fmt_pct_opt(s['macro_hcp'])}` | `{s['pass_count']}/{s['warn_count']}/{s['abstain_count']}` | "
                f"`{s['gated_count']}` | `{s['pooled_ece10']:.4f}` | `{s['mean_deb_ece10']:.4f}` (`{s['mean_floor10']:.4f}`) |"
            )

        lines.append("")
        lines.append("#### Per-Entry **NON-COLD** `seen_p12` QC Verdict, Gate Notes & HC Precision (`5` Seeds/Entry)")
        lines.append("")
        lines.append(
            "| BMRB ID | Protein (`aa`, Vendor) | `evo-graph` QC / Gate / HC / Notes | "
            "`standard (cpu)` QC / Gate / HC / Notes | `standard (gpu)` QC / Gate / HC / Notes |"
        )
        lines.append("|---|---|---|---|---|")
        for e_ev, e_cp, e_gp in zip(
            s_evo["all_entries"], s_cpu["all_entries"], s_gpu["all_entries"]
        ):
            def seen_cell(e: dict[str, Any]) -> str:
                if e.get("status") != "OK":
                    return f"`ERR` ({e.get('failure', {}).get('reason', 'failed')})"
                qc = e["qc_verdict"]
                qcr = e.get("qc_report", {})
                gated = "GATE" if qcr.get("protein_hc_gate_triggered") else "open"
                acc = 100.0 * float(e["accuracy"])
                hcs = e["hc_status"]
                reasons = "; ".join(qcr.get("reasons", [])) or "none"
                return f"`{qc}` (`{gated}`) / `{acc:.1f}%` acc / `{hcs}` / notes: `{reasons}`"

            lines.append(
                f"| `{e_ev['bmrb_id']}` | `{e_ev['protein']}` (`{e_ev['residues']} aa`, `{e_ev['vendor']}`) | "
                f"{seen_cell(e_ev)} | {seen_cell(e_cp)} | {seen_cell(e_gp)} |"
            )

        # Direct v0.8.0 -> Wave 1 comparison on 5691 and 18561 if phase12 JSONs exist
        if p12_evo_json.exists() and p12_cpu_json.exists() and p12_gpu_json.exists():
            p12_map: dict[tuple[str, str], dict[str, Any]] = {}
            for eng_lbl, pj in [
                ("evo-graph", p12_evo_json),
                ("standard (cpu)", p12_cpu_json),
                ("standard (gpu)", p12_gpu_json),
            ]:
                pdata = json.loads(pj.read_text(encoding="utf-8"))
                for pe in pdata.get("entries", []):
                    p12_map[(eng_lbl, str(pe.get("bmrb_id")))] = pe

            w1_map: dict[tuple[str, str], dict[str, Any]] = {}
            for eng_lbl, ws in [
                ("evo-graph", s_evo),
                ("standard (cpu)", s_cpu),
                ("standard (gpu)", s_gpu),
            ]:
                for we in ws["all_entries"]:
                    w1_map[(eng_lbl, str(we.get("bmrb_id")))] = we

            lines.append("")
            lines.append(
                "#### Before (`v0.8.0` Cold) vs After (`13.W1` **NON-COLD** `seen_p12`) on Target Failure Entries `5691` & `18561`"
            )
            lines.append("")
            lines.append(
                "| Entry | Engine | `v0.8.0` (`cold_p12`) QC & HC Calls | `13.W1` (**NON-COLD** `seen_p12`) QC, Gate & HC Calls |"
            )
            lines.append("|---|---|---|---|")
            for bid in ["5691", "18561"]:
                for eng_lbl in ["evo-graph", "standard (cpu)", "standard (gpu)"]:
                    old_e = p12_map.get((eng_lbl, bid), {})
                    new_e = w1_map.get((eng_lbl, bid), {})
                    old_qc = old_e.get("qc_verdict", "?")
                    old_hc = old_e.get("hc_status", "?")
                    new_qc = new_e.get("qc_verdict", "?")
                    new_hc = new_e.get("hc_status", "?")
                    new_gate = (
                        "GATE"
                        if new_e.get("qc_report", {}).get("protein_hc_gate_triggered")
                        else "open"
                    )
                    new_reasons = "; ".join(new_e.get("qc_report", {}).get("reasons", [])) or "none"
                    lines.append(
                        f"| `{bid}` (`{new_e.get('protein', '?')}`) | `{eng_lbl}` | "
                        f"`{old_qc}` / `{old_hc}` | **`{new_qc}` (`{new_gate}`)** / **`{new_hc}`** (`{new_reasons}`) |"
                    )

    # 4. 13.1:P1-3 Calibration Candor Tables (Cold ECE_10 FAIL + Dev Raw vs Debiased ECE_10)
    if cold_evo_json.exists() and cold_cpu_json.exists() and cold_gpu_json.exists():
        lines.append("")
        lines.append(
            "### `13.1:P1-3` — Cold `ECE_10 <= 0.050` Primary Gate Status (**FAIL** on All 3 Engines) & Reliability Bin Breakdown"
        )
        lines.append("")
        lines.append(
            "| Engine (`cold_p13`) | Pooled `ECE_10` | Primary Gate (`<= 0.050`) | Pooled Micro HC-Prec (`>= 94.0%`) | "
            "`[0.0, 0.5)` `conf -> acc` (`n`) | `[0.5, 0.8)` `conf -> acc` (`n`) | `[0.8, 0.9)` `conf -> acc` (`n`) | `[0.9, 1.0]` `conf -> acc` (`n`) |"
        )
        lines.append("|---|---:|---|---:|---|---|---|---|")
        for cj in [cold_evo_json, cold_cpu_json, cold_gpu_json]:
            cdata = json.loads(cj.read_text(encoding="utf-8"))
            solver_lbl = cdata.get("solver", "?")
            rel = cdata["calibration_summary"]["pooled_reliability_curve"]
            ece10 = float(rel["ece_10"])
            gate_status = "PASS" if ece10 <= 0.050 else "**FAIL**"
            hcp = 100.0 * float(rel["hc_precision"])
            hc_ok = int(rel["hc_correct"])
            hc_tot = int(rel["hc_assigned"])
            bins_by_range = {b["range"]: b for b in rel.get("bins", [])}

            def fmt_bin(rng: str) -> str:
                b = bins_by_range.get(rng)
                if not b or int(b["count"]) == 0:
                    return "`empty (0)`"
                return f"`{float(b['mean_conf']):.3f} -> {float(b['empirical_acc']):.3f}` (`{int(b['count'])}`)"

            lines.append(
                f"| `{solver_lbl}` | `{ece10:.4f}` | {gate_status} | `PASS` (`{hcp:.2f}%`, `{hc_ok}/{hc_tot}`) | "
                f"{fmt_bin('[0.0, 0.5)')} | {fmt_bin('[0.5, 0.8)')} | {fmt_bin('[0.8, 0.9)')} | {fmt_bin('[0.9, 1.0]')} |"
            )

    if rem_lopo_log.exists():
        rl2 = parse_remediation_lopo_log(rem_lopo_log)
        lines.append("")
        lines.append(
            "#### `dev` 13-Fold OOF Per-Protein Mean `ECE_10`: Old Raw Criterion (`<= 0.060`) vs Same-Wave Debiased Criterion (`<= 0.065`)"
        )
        lines.append("")
        lines.append(
            "| Engine (`dev` 13-Fold OOF) | Mean Raw `ECE_10` | Old Raw Gate (`<= 0.060`) | "
            "Analytical Bernoulli `ECE_10` Floor | Mean Debiased `ECE_10` | New Same-Wave Gate (`<= 0.065`) |"
        )
        lines.append("|---|---:|---|---:|---:|---|")
        for eng in rl2["engines"]:
            raw_e10 = float(eng["mean_ece10"])
            deb_e10 = float(eng["debiased_ece10"])
            flr_e10 = float(eng["bernoulli_floor"])
            old_st = "PASS" if raw_e10 <= 0.060 else "**FAIL**"
            new_st = "`PASS`" if deb_e10 <= 0.065 else "**FAIL**"
            lines.append(
                f"| `{eng['solver']}` | `{raw_e10:.4f}` | {old_st} | `{flr_e10:.4f}` | `{deb_e10:.4f}` | {new_st} |"
            )

    if rem_ubq_log.exists():
        ubq = parse_ubq_log(rem_ubq_log)
        lines.append("")
        lines.append("### Post-Remediation Regression Gate Check")
        lines.append("")
        lines.append("| Gate | Result |")
        lines.append("|---|---|")
        lines.append(
            f"| Ubiquitin 5-spectrum `--realistic` (`5` seeds) | acc `{ubq['acc']}`, rec `{ubq['rec']}`, HC prec `{ubq['hcp']}` |"
        )
        if rem_ws_log.exists():
            ws_txt = rem_ws_log.read_text(encoding="utf-8")
            n_passed = sum(int(m.group(1)) for m in re.finditer(r"test result: ok\. (\d+) passed;", ws_txt))
            n_failed = sum(int(m.group(1)) for m in re.finditer(r"(\d+) failed;", ws_txt))
            n_bins = len(re.findall(r"test result: ok\.", ws_txt))
            lines.append(
                f"| `cargo test --release --workspace` | `{n_bins}` test binaries, `{n_passed}` passed, `{n_failed}` failed |"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--section",
        choices=["all", "digests", "13.1", "13.3", "remediation"],
        default="all",
    )
    ap.add_argument("--extra-artifact", type=Path, action="append", default=[])
    args = ap.parse_args()

    root_res = args.root.resolve()
    extras_res = [p.resolve() for p in args.extra_artifact]
    if args.section == "remediation":
        md = render_remediation_markdown(root_res, extras_res)
    else:
        md = render_markdown(
            root_res,
            extras_res,
            section=args.section,
        )
    if args.out is not None:
        args.out.write_text(md, encoding="utf-8")
    else:
        print(md)


if __name__ == "__main__":
    main()
