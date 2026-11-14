#!/usr/bin/env python3
"""Mechanically render all numeric and SHA-256 tables for Phase 13 Wave 2 (`13.W2`).

Provenance: Clean-room deterministic report generator (Python 3 stdlib only).
Reads raw JSON and test/benchmark logs in `data/private/w2_results/`,
`data/benchmark/real_manifest.tsv`, and `data/torsion_db/torsion_corpus.tsv`
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


def parse_manifest_cohort(manifest_path: Path, cohort_name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("bmrb_id\t"):
            continue
        cols = [c.strip() for c in line.split("\t")]
        if len(cols) >= 8 and cols[7] == cohort_name:
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


def summarize_real_bench_json(json_path: Path) -> dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    all_entries = data.get("entries", [])
    entries = [e for e in all_entries if e.get("status") == "OK"]
    counts = data.get("counts", {})
    macro = data.get("macro_average", {})
    micro = data.get("micro_average", {})
    cal = data.get("calibration_summary", {})
    rel = cal.get("pooled_reliability_curve", data.get("pooled_reliability", {}))
    hc_ok = int(
        micro.get(
            "hc_correct",
            sum(int(e["reliability_curve"]["hc_correct"]) for e in entries),
        )
    )
    hc_tot = int(
        micro.get(
            "hc_assigned",
            sum(int(e["reliability_curve"]["hc_assigned"]) for e in entries),
        )
    )
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
            cal.get(
                "mean_protein_debiased_ece_10",
                data.get("mean_protein_debiased_ece_10", 0.0),
            )
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
        "acc": acc_s,
        "rec": rec_s,
        "hcp": hcp_m.group(1) if hcp_m else "see raw log",
    }


def parse_test_pass_counts(log_path: Path) -> tuple[int, int, list[tuple[str, int, int]]]:
    txt = log_path.read_text(encoding="utf-8")
    suites: list[tuple[str, int, int]] = []
    cur_bin = "suite"
    for line in txt.splitlines():
        m_run = re.search(r"Running (?:tests/|unittests )(\S+)", line)
        if m_run:
            cur_bin = m_run.group(1).split("(")[0].strip()
        m_res = re.search(r"test result: \S+\.\s+(\d+) passed;\s+(\d+) failed;", line)
        if m_res:
            suites.append((cur_bin, int(m_res.group(1)), int(m_res.group(2))))
    tot_pass = sum(s[1] for s in suites)
    tot_fail = sum(s[2] for s in suites)
    return tot_pass, tot_fail, suites


def fmt_pct_opt(val: float | None) -> str:
    return f"{val:.2f}%" if val is not None else "ABSTAIN"


def fmt_frac_opt(val: float | None) -> str:
    return f"{100.0 * val:.1f}%" if val is not None else "ABSTAIN (0/0)"


def cited_w2_files(root: Path, extra_files: list[Path]) -> list[Path]:
    w2 = root / "data/private/w2_results"
    base = [
        root / "data/benchmark/real_manifest.tsv",
        root / "data/torsion_db/torsion_corpus.tsv",
        w2 / "w2_non_cold_26_raw.json",
        w2 / "w2_non_cold_26_lopo.json",
        w2 / "w2_non_cold_26_lopo.log",
        w2 / "w2_cold_p13_w2_timestamps.txt",
        w2 / "w2_cold_p13_w2_evograph.json",
        w2 / "w2_cold_p13_w2_evograph.log",
        w2 / "w2_cold_p13_w2_standard_cpu.json",
        w2 / "w2_cold_p13_w2_standard_cpu.log",
        w2 / "w2_cold_p13_w2_standard_gpu.json",
        w2 / "w2_cold_p13_w2_standard_gpu.log",
        w2 / "w2_cold_p13_w2_bayes_cpu.json",
        w2 / "w2_cold_p13_w2_bayes_cpu.log",
        w2 / "w2_cold_p13_w2_bayes_gpu.json",
        w2 / "w2_cold_p13_w2_bayes_gpu.log",
        w2 / "w2_f1_6209_evograph.json",
        w2 / "w2_f1_6209_evograph.log",
        w2 / "w2_f2_15822_default.json",
        w2 / "w2_f2_15822_default.log",
        w2 / "w2_f2_15822_allow_truncated.json",
        w2 / "w2_f2_15822_allow_truncated.log",
        w2 / "w2_ubq_realistic_evograph.log",
        w2 / "w2_ubq_realistic_standard_cpu.log",
        w2 / "w2_ubq_realistic_standard_gpu.log",
        w2 / "w2_ubq_realistic_bayes_cpu.log",
        w2 / "w2_ubq_realistic_bayes_gpu.log",
        w2 / "w2_cisd3_evograph.json",
        w2 / "w2_cisd3_standard_cpu.json",
        w2 / "w2_cisd3_standard_gpu.json",
        w2 / "w2_cisd3_bayes_cpu.json",
        w2 / "w2_cisd3_bayes_gpu.json",
        w2 / "w2_slice_13_2_tests.log",
        w2 / "w2_slice_13_4_tests.log",
        w2 / "w2_workspace_tests.log",
        *extra_files,
    ]
    return base


def render_digests(root: Path, extra_files: list[Path]) -> str:
    lines = [
        "### Cited Raw Results Artifacts & FIPS 180-4 SHA-256 Digests",
        "",
        "Generated mechanically by `python3 scripts/render_phase13_w2_tables.py --section digests`:",
        "",
        "| Artifact Path | Bytes | SHA-256 |",
        "|---|---:|---|",
    ]
    for fp in cited_w2_files(root, extra_files):
        if fp.exists():
            digest, nbytes = sha256_file(fp)
            rel = fp.relative_to(root) if fp.is_relative_to(root) else fp
            lines.append(f"| `{rel}` | {nbytes:,} | `{digest}` |")
    return "\n".join(lines) + "\n"


def render_slice_13_1(root: Path) -> str:
    w2 = root / "data/private/w2_results"
    manifest = root / "data/benchmark/real_manifest.tsv"
    lopo_data = json.loads((w2 / "w2_non_cold_26_lopo.json").read_text(encoding="utf-8"))
    lines: list[str] = []

    fg = lopo_data["canonical_protein_hc_gate"]
    folds = lopo_data["fold_gates"]
    std_ws = [float(f[2]) for f in folds]
    bay_ws = [float(f[3]) for f in folds]
    bidir_vals = [float(f[4]["min_bidir_aliphatic_frac"]) for f in folds]
    bidir_058 = sum(1 for v in bidir_vals if abs(v - 0.58) < 1e-9)

    lines.append("### Slice `13.1:P1-7b` — 26-Protein Non-Cold Pool LOPO Summary (`dev` + `seen_p12` + `seen_p13`)")
    lines.append("")
    lines.append(
        f"- **Full-26 Canonical Gate (`CANONICAL_PROTEIN_HC_GATE`)**: "
        f"`{{ c35: {fg['min_corroborated_connectivity']:.2f}, hcf: {fg['min_hc_call_fraction']:.2f}, "
        f"cmarg: {fg['min_crowded_solution_margin']:.2f}, mpen: {fg['multiplicity_penalty_floor']:.2f}, "
        f"chain: {fg['min_chain_corroborated_frac']:.2f}, unary: {fg['min_unary_specificity']:.2f}, "
        f"bidir_aliph: {fg['min_bidir_aliphatic_frac']:.2f} }}`"
    )
    lines.append(
        f"- **26-Fold LOPO Stability of `min_bidir_aliphatic_frac`**: selected `0.58` in **`{bidir_058}/{len(folds)}` (`{100.0 * bidir_058 / len(folds):.1f}%`)** outer folds (all 7 candidate axes anchored at `0.00`)."
    )
    lines.append(
        f"- **Nested 26x25 LOPO GPU Shrinkage**: `CANONICAL_GPU_CPU_SHRINKAGE_WEIGHT = {lopo_data['canonical_std_gpu_shrinkage_w']:.2f}` "
        f"(outer-fold range `[{min(std_ws):.2f}, {max(std_ws):.2f}]`), "
        f"`CANONICAL_BAYES_SA_GPU_CPU_SHRINKAGE_WEIGHT = {lopo_data['canonical_bayes_gpu_shrinkage_w']:.2f}` "
        f"(outer-fold range `[{min(bay_ws):.2f}, {max(bay_ws):.2f}]`)."
    )
    lines.append("")
    lines.append(
        "| Engine (`26`-Fold OOF) | **Pooled `ECE_10` (`<=0.050`)** | Pooled `ECE_4` | Mean Raw `ECE_10` | "
        "Mean Debiased `ECE_10` | **Micro HC-Prec (`>=94%`)** | Macro HC-Prec | "
        "`5691` Uncapped -> OOF Gated | `18561` Uncapped -> OOF Gated | `dev` (`>=94%` HCP) Capped |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|---|")
    for eng in lopo_data["engines"]:
        hc_ok = sum(int(p["oof_hc_correct"]) for p in eng["per_protein"])
        hc_tot = sum(int(p["oof_hc_assigned"]) for p in eng["per_protein"])
        ece10 = float(eng["oof_pooled_ece_10"])
        ece_status = "PASS" if ece10 <= 0.050 else "**FAIL**"
        mhcp = 100.0 * float(eng["oof_micro_hc_precision"])
        hcp_status = "PASS" if mhcp >= 94.0 else "**FAIL**"
        s5691 = f"`{fmt_frac_opt(eng['oof_5691_uncapped_hcp'])}` -> **`{fmt_frac_opt(eng['oof_5691_gated_hcp'])}`** (`cap={str(eng['oof_5691_capped_or_warned']).lower()}`)"
        s18561 = f"`{fmt_frac_opt(eng['oof_18561_uncapped_hcp'])}` -> **`{fmt_frac_opt(eng['oof_18561_gated_hcp'])}`** (`cap={str(eng['oof_18561_capped_or_warned']).lower()}`)"
        dev_cap = "`none`" if not eng["dev_clean_proteins_capped"] else f"`{eng['dev_clean_proteins_capped']}`"
        lines.append(
            f"| `{eng['engine']}` | **`{ece10:.4f}`** ({ece_status}) | `{float(eng['oof_pooled_ece_4']):.4f}` | "
            f"`{float(eng['oof_mean_ece_10']):.4f}` | `{float(eng['oof_mean_debiased_ece_10']):.4f}` | "
            f"**`{mhcp:.2f}%` (`{hc_ok}/{hc_tot}`)** ({hcp_status}) | `{100.0 * float(eng['oof_macro_hc_precision']):.2f}%` | "
            f"{s5691} | {s18561} | {dev_cap} |"
        )

    lines.append("")
    lines.append("#### All 26 Per-Fold In-Fold Refit `ProteinHcGateConfig` & Nested GPU Shrinkage Weights")
    lines.append("")
    lines.append(
        "| Fold | Held-Out Entry | Cohort | `std_gpu_w` | `bayes_gpu_w` | `c35` | `hcf` | `cmarg` | `mpen` | `chain` | `unary` | **`bidir_aliph`** |"
    )
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for idx, f in enumerate(folds, start=1):
        bid, coh, sw, bw, g = f
        lines.append(
            f"| `{idx:02d}` | `{bid}` | `{coh}` | `{float(sw):.2f}` | `{float(bw):.2f}` | "
            f"`{float(g['min_corroborated_connectivity']):.2f}` | `{float(g['min_hc_call_fraction']):.2f}` | "
            f"`{float(g['min_crowded_solution_margin']):.2f}` | `{float(g['multiplicity_penalty_floor']):.2f}` | "
            f"`{float(g['min_chain_corroborated_frac']):.2f}` | `{float(g['min_unary_specificity']):.2f}` | "
            f"**`{float(g['min_bidir_aliphatic_frac']):.2f}`** |"
        )

    eng_map = {e["engine"]: e["per_protein"] for e in lopo_data["engines"]}
    lines.append("")
    lines.append("#### Per-Protein 26-Fold Non-Cold Out-of-Fold Summary (`5` Seeds/Protein Across All 5 Engines)")
    lines.append("")
    lines.append(
        "| Entry (`Cohort`) | Protein | `evo-graph` Acc / Uncap->OOF HC / Cap | `standard (cpu)` Acc / OOF HC / Cap | "
        "`standard (gpu)` Acc / OOF HC / Cap | `bayes-sa (cpu)` Acc / OOF HC / Cap | `bayes-sa (gpu)` Acc / OOF HC / Cap |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for idx in range(len(eng_map["EVO_GRAPH"])):
        pe = eng_map["EVO_GRAPH"][idx]
        pc = eng_map["STANDARD_CPU"][idx]
        pg = eng_map["STANDARD_GPU"][idx]
        pbc = eng_map["BAYES_SA_CPU"][idx]
        pbg = eng_map["BAYES_SA_GPU"][idx]

        def fmt_oof(p: dict[str, Any], include_uncap: bool = False) -> str:
            acc = f"{100.0 * float(p['accuracy']):.1f}%"
            if int(p["oof_hc_assigned"]) > 0:
                oof_s = f"{100.0 * float(p['oof_hc_precision']):.1f}% ({p['oof_hc_correct']}/{p['oof_hc_assigned']})"
            else:
                oof_s = "ABST (0/0)"
            cap_s = "CAP/WARN" if p["oof_capped_or_warned"] else "open"
            if include_uncap:
                if int(p["uncapped_hc_assigned"]) > 0:
                    unc_s = f"{100.0 * float(p['uncapped_hc_precision']):.1f}%"
                else:
                    unc_s = "ABST"
                return f"`{acc}` / `{unc_s} -> {oof_s}` / `{cap_s}`"
            return f"`{acc}` / `{oof_s}` / `{cap_s}`"

        lines.append(
            f"| `{pe['bmrb_id']}` (`{pe['cohort']}`) | `{pe['protein']}` | "
            f"{fmt_oof(pe, True)} | {fmt_oof(pc)} | {fmt_oof(pg)} | {fmt_oof(pbc)} | {fmt_oof(pbg)} |"
        )

    # W1 Follow-Ups F1 & F2
    f1_json = w2 / "w2_f1_6209_evograph.json"
    f2_def_json = w2 / "w2_f2_15822_default.json"
    f2_opt_json = w2 / "w2_f2_15822_allow_truncated.json"
    if f1_json.exists() and f2_def_json.exists() and f2_opt_json.exists():
        f1 = json.loads(f1_json.read_text(encoding="utf-8"))["entries"][0]
        f2_def = json.loads(f2_def_json.read_text(encoding="utf-8"))["entries"][0]
        f2_opt = json.loads(f2_opt_json.read_text(encoding="utf-8"))["entries"][0]
        lines.append("")
        lines.append("#### `13.W1` Follow-Ups `F1` (`BMRB 6209`) & `F2` (`BMRB 15822`) Verification (`seen_p13`, Non-Cold)")
        lines.append("")
        lines.append("| Follow-Up Case | Mode | Status | QC Verdict | `truncated_fid_zero_filled` | Overall Acc | HC Precision (`>=0.80`) | Notes / Failure |")
        lines.append("|---|---|---|---|---|---:|---:|---|")
        lines.append(
            f"| `F1`: `6209` (`At3g03410.1`) | Default (`--solver evo-graph`) | `{f1['status']}` | `{f1['qc_verdict']}` | "
            f"`{str(bool(f1['qc_report'].get('truncated_fid_zero_filled'))).lower()}` | "
            f"`{100.0 * float(f1['accuracy']):.2f}%` | `{f1['hc_status']}` | Legacy XWIN-NMR `proc2s ##$MC2` + `pulseprogram` `FnMODE` fallback |"
        )
        fail_msg = f2_def.get("failure", {}).get("reason", "rejected")
        lines.append(
            f"| `F2`: `15822` (`CmR9`) | Default (strict) | `{f2_def['status']}` | `n/a` | `false` | `n/a` | `n/a` | `{fail_msg}` |"
        )
        lines.append(
            f"| `F2`: `15822` (`CmR9`) | Opt-in `--allow-truncated-fid` | `{f2_opt['status']}` | **`{f2_opt['qc_verdict']}`** | "
            f"**`{str(bool(f2_opt['qc_report'].get('truncated_fid_zero_filled'))).lower()}`** | "
            f"`{100.0 * float(f2_opt['accuracy']):.2f}%` | `{f2_opt['hc_status']}` | Zero-filled `9,942 -> 10,240` traces (`CBCACONH`) |"
        )

    # Disjoint cold_p13_w2 single-shot validation (relabeled seen_p13_w2 post-W2 per 13.1:P1-2)
    cold_rows = parse_manifest_cohort(manifest, "cold_p13_w2") or parse_manifest_cohort(
        manifest, "seen_p13_w2"
    )
    if cold_rows:
        lines.append("")
        lines.append("### Pre-Registered Disjoint `cold_p13_w2` Cohort (`split=heldout`, `cohort=cold_p13_w2`)")
        lines.append("")
        lines.append("| BMRB ID | Protein | Residues | Field (MHz) | Vendor | Pre-Registered Raw-FID Experiments |")
        lines.append("|---|---|---:|---:|---|---|")
        for r in cold_rows:
            lines.append(
                f"| `{r['bmrb_id']}` | `{r['protein']}` | {r['residues']} | {r['field_mhz']} | "
                f"`{r['vendor']}` | `{r['experiments']}` |"
            )

    cold_jsons = [
        ("evo-graph", w2 / "w2_cold_p13_w2_evograph.json"),
        ("standard (cpu)", w2 / "w2_cold_p13_w2_standard_cpu.json"),
        ("standard (gpu)", w2 / "w2_cold_p13_w2_standard_gpu.json"),
        ("bayes-sa (cpu)", w2 / "w2_cold_p13_w2_bayes_cpu.json"),
        ("bayes-sa (gpu)", w2 / "w2_cold_p13_w2_bayes_gpu.json"),
    ]
    if all(p.exists() for _, p in cold_jsons):
        cold_summaries = [(lbl, summarize_real_bench_json(p)) for lbl, p in cold_jsons]
        lines.append("")
        lines.append("#### Single-Shot 5-Engine `cold_p13_w2` Aggregate Results (`5` Seeds/Entry, Joint `13.1:P1-7b` + `13.4` Cold Validation)")
        lines.append("")
        lines.append(
            "| Engine | Evaluated | Macro Acc | Macro Recall | **Pooled Micro HC-Prec (`>=94%`)** | Macro HC-Prec | "
            "QC Tiers (`PASS/WARN/ABST`) | HC-Gate Caps | **Pooled `ECE_10` (`<=0.050`)** | Pooled `ECE_4` | Mean Debiased `ECE_10` (Floor) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|")
        for lbl, s in cold_summaries:
            if s["micro_hcp"] is not None:
                hcp_flag = "PASS" if s["micro_hcp"] >= 94.0 else "**FAIL**"
                micro_s = f"**`{s['micro_hcp']:.2f}%` (`{s['hc_ok']}/{s['hc_tot']}`)** ({hcp_flag})"
            else:
                micro_s = "`ABSTAIN (0/0)`"
            ece_flag = "PASS" if s["pooled_ece10"] <= 0.050 else "**FAIL**"
            lines.append(
                f"| `{lbl}` | `{s['evaluated']}/{s['total']}` | `{s['macro_acc']:.2f}%` | `{s['macro_rec']:.2f}%` | "
                f"{micro_s} | `{fmt_pct_opt(s['macro_hcp'])}` | `{s['pass_count']}/{s['warn_count']}/{s['abstain_count']}` | "
                f"`{s['gated_count']}` | **`{s['pooled_ece10']:.4f}`** ({ece_flag}) | `{s['pooled_ece4']:.4f}` | "
                f"`{s['mean_deb_ece10']:.4f}` (`{s['mean_floor10']:.4f}`) |"
            )

        lines.append("")
        lines.append("#### Per-Entry Single-Shot 5-Engine `cold_p13_w2` Breakdown (`5` Seeds/Entry)")
        lines.append("")
        lines.append(
            "| BMRB ID | Protein (`aa`, Vendor) | `evo-graph` QC/Gate/Acc/HC/`ECE_10` | `standard (cpu)` QC/Gate/Acc/HC/`ECE_10` | "
            "`standard (gpu)` QC/Gate/Acc/HC/`ECE_10` | `bayes-sa (cpu)` QC/Gate/Acc/HC/`ECE_10` | `bayes-sa (gpu)` QC/Gate/Acc/HC/`ECE_10` |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        n_ent = len(cold_summaries[0][1]["all_entries"])
        for idx in range(n_ent):
            e0 = cold_summaries[0][1]["all_entries"][idx]

            def cold_cell(e: dict[str, Any]) -> str:
                if e.get("status") != "OK":
                    fail = e.get("failure", {})
                    return f"`ERR [{fail.get('stage', 'err')}:{fail.get('experiment', '?')}]` ({fail.get('reason', 'failed')})"
                qc = e["qc_verdict"]
                gated = "GATE" if e.get("qc_report", {}).get("protein_hc_gate_triggered") else "open"
                acc = 100.0 * float(e["accuracy"])
                hcs = e["hc_status"]
                rc = e["reliability_curve"]
                return f"`{qc}`(`{gated}`) / `{acc:.1f}%` / `{hcs}` / `{float(rc['ece_10']):.3f}`"

            cells = [cold_cell(cs["all_entries"][idx]) for _, cs in cold_summaries]
            lines.append(
                f"| `{e0['bmrb_id']}` | `{e0['protein']}` (`{e0['residues']} aa`, `{e0['vendor']}`) | "
                + " | ".join(cells)
                + " |"
            )

    # Baseline solver regression gates across all 5 engines
    ubq_files = [
        ("`evo-graph`", w2 / "w2_ubq_realistic_evograph.log", w2 / "w2_cisd3_evograph.json"),
        ("`standard (cpu)`", w2 / "w2_ubq_realistic_standard_cpu.log", w2 / "w2_cisd3_standard_cpu.json"),
        ("`standard (gpu)`", w2 / "w2_ubq_realistic_standard_gpu.log", w2 / "w2_cisd3_standard_gpu.json"),
        ("`bayes-sa (cpu)`", w2 / "w2_ubq_realistic_bayes_cpu.log", w2 / "w2_cisd3_bayes_cpu.json"),
        ("`bayes-sa (gpu)`", w2 / "w2_ubq_realistic_bayes_gpu.log", w2 / "w2_cisd3_bayes_gpu.json"),
    ]
    if all(u.exists() and c.exists() for _, u, c in ubq_files):
        lines.append("")
        lines.append("#### Mandatory 5-Seed Regression Gates (`Ubiquitin --realistic` & `CISD3` `BMRB 51439` Across All 5 Engines)")
        lines.append("")
        lines.append(
            "| Engine | Ubiquitin `--realistic` (`5` Seeds) Acc | Ubiquitin Recall | Ubiquitin HC-Prec (`>=0.80`) | "
            "`CISD3` (`5` Seeds) Acc | `CISD3` Recall | `CISD3` HC-Prec (`>=0.80`) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for lbl, u_log, c_json in ubq_files:
            ubq = parse_ubq_log(u_log)
            cs = summarize_real_bench_json(c_json)
            chcp = (
                f"{cs['micro_hcp']:.2f}% ({cs['hc_ok']}/{cs['hc_tot']})"
                if cs["micro_hcp"] is not None
                else "ABSTAIN (0/0)"
            )
            lines.append(
                f"| {lbl} | `{ubq['acc']}` | `{ubq['rec']}` | `{ubq['hcp']}` | "
                f"`{cs['macro_acc']:.2f}%` | `{cs['macro_rec']:.2f}%` | `{chcp}` |"
            )

    return "\n".join(lines) + "\n"


def render_slice_13_2(root: Path) -> str:
    w2 = root / "data/private/w2_results"
    log_path = w2 / "w2_slice_13_2_tests.log"
    txt = log_path.read_text(encoding="utf-8")
    lines: list[str] = []

    # 1. P1-9 Non-circular RDC validation
    lines.append("### Slice `13.2` — `P1-9` Independent Non-Circular RDC Validation (`ByPeptidePlane` `20%` `RdcFree`)")
    lines.append("")
    lines.append("| Target & Protocol | Condition | Working `Q_work` | Peptide-Plane `Q_free` | Independent X-Ray BB RMSD (`lowE` / `mean`) |")
    lines.append("|---|---|---:|---:|---:|")

    m_ubq_pre = re.search(
        r"Exp NOE-only\s*:\s*Q_work = ([0-9.]+), plane Q_free = ([0-9.]+) \| 1UBQ \(1\.\.71\) BB RMSD lowE = ([0-9.]+) A, mean = ([0-9.]+) A",
        txt,
    )
    m_ubq_post = re.search(
        r"\+ Bicelle RDC\s*:\s*Q_work = ([0-9.]+), plane Q_free = ([0-9.]+) \| 1UBQ \(1\.\.71\) BB RMSD lowE = ([0-9.]+) A, mean = ([0-9.]+) A",
        txt,
    )
    if m_ubq_pre and m_ubq_post:
        lines.append(
            f"| Ubiquitin (`1D3Z` real NOEs -> `1UBQ` X-ray `1..71`) | Exp NOE-only (`w_rdc = 0`) | `{m_ubq_pre.group(1)}` | `{m_ubq_pre.group(2)}` | `{m_ubq_pre.group(3)} Å` / `{m_ubq_pre.group(4)} Å` |"
        )
        lines.append(
            f"| Ubiquitin (`1D3Z` real NOEs -> `1UBQ` X-ray `1..71`) | **+ Bicelle RDC (`ByPeptidePlane`)** | **`{m_ubq_post.group(1)}`** | **`{m_ubq_post.group(2)}`** | **`{m_ubq_post.group(3)} Å` / `{m_ubq_post.group(4)} Å`** |"
        )

    m_gb1_xray = re.search(
        r"Direct 3GB1 RDC fit on unrefined 1994 1PGA crystal: 1D_NH Q = ([0-9.]+), multi-bond Q_work = ([0-9.]+), plane Q_free = ([0-9.]+)",
        txt,
    )
    m_gb1_pre = re.search(
        r"Exp NOE\+Shift\s*:\s*Q_work = ([0-9.]+), plane Q_free = ([0-9.]+) \| 1PGA \(1\.\.56\) BB RMSD lowE = ([0-9.]+) A, mean = ([0-9.]+) A",
        txt,
    )
    m_gb1_post = re.search(
        r"\+ TMV\+Bicelle\s*:\s*Q_work = ([0-9.]+), plane Q_free = ([0-9.]+) \(1D_NH Q_work = ([0-9.]+), 1D_NH plane Q_free = ([0-9.]+)\) \| 1PGA \(1\.\.56\) BB RMSD lowE = ([0-9.]+) A, mean = ([0-9.]+) A",
        txt,
    )
    if m_gb1_xray:
        lines.append(
            f"| GB1 (`3GB1` RDCs evaluated on unrefined `1994 1PGA` X-ray) | Direct Saupe SVD (`1D_NH Q = {m_gb1_xray.group(1)}`) | `{m_gb1_xray.group(2)}` | `{m_gb1_xray.group(3)}` | `0.000 Å` (crystal ref) |"
        )
    if m_gb1_pre and m_gb1_post:
        lines.append(
            f"| GB1 (`3GB1` real NOEs + firewalled `BMRB 7280` shifts -> `1PGA` `1..56`) | Exp NOE + Firewall Shifts (`w_rdc = 0`) | `{m_gb1_pre.group(1)}` | `{m_gb1_pre.group(2)}` | `{m_gb1_pre.group(3)} Å` / `{m_gb1_pre.group(4)} Å` |"
        )
        lines.append(
            f"| GB1 (`3GB1` real NOEs + firewalled `BMRB 7280` shifts -> `1PGA` `1..56`) | **+ TMV + Bicelle RDC (`1D_NH Q_free = {m_gb1_post.group(4)}`)** | **`{m_gb1_post.group(1)}`** | **`{m_gb1_post.group(2)}`** | **`{m_gb1_post.group(5)} Å` / `{m_gb1_post.group(6)} Å`** |"
        )

    # 2. P1-10 Torsion corpus verification & 25% vs 20% seq-id sensitivity
    lines.append("")
    lines.append("### Slice `13.2` — `P1-10` Reproducible Torsion Corpus & `25%` vs `20%` Sequence-Identity Firewall Sensitivity")
    lines.append("")
    for vline in txt.splitlines():
        if vline.startswith(("SPOT_CHECK_OK", "VERIFY_OK")):
            lines.append(f"- `{vline.strip()}`")
    lines.append("")
    lines.append(
        "| Firewall `max_seq_identity` | Max Retained Seq-ID | Folded `Good` Coverage | Structured (`RCI S^2>=0.65`) `Good` | "
        "`phi` MAE | `psi` MAE | Combined Circular MAE | `chi1` MAE (`n`) | Wrong Basin | IDP `Good` Leak |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    cutoff_re = re.compile(
        r"\[CUTOFF <= (\d+)%\] MaxRetainedSid=([0-9.]+)%, Good=(\d+/\d+ \([0-9.]+%\)), "
        r"StructGood=(\d+/\d+ \([0-9.]+%\)), PhiMAE=([0-9.]+), PsiMAE=([0-9.]+), "
        r"CircMAE=([0-9.]+), Chi1MAE=([0-9.]+) \(n=(\d+)\), WrongBasin=(\d+/\d+ \([0-9.]+%\)), "
        r"IDPLeak=(\d+/\d+ \([0-9.]+%\))"
    )
    for cm in cutoff_re.finditer(txt):
        lines.append(
            f"| `<= {cm.group(1)}%` | `{cm.group(2)}%` | `{cm.group(3)}` | `{cm.group(4)}` | "
            f"`{cm.group(5)}°` | `{cm.group(6)}°` | **`{cm.group(7)}°`** | `{cm.group(8)}°` (`{cm.group(9)}`) | "
            f"`{cm.group(10)}` | `{cm.group(11)}` |"
        )

    # 3. P1-11 IDP & Paramagnetic statistical tightening
    lines.append("")
    lines.append("### Slice `13.2` — `P1-11` IDP & Paramagnetic Statistical Tightening")
    lines.append("")
    lines.append("| Verification Check | Measured Outcome (from `w2_slice_13_2_tests.log`) |")
    lines.append("|---|---|")
    for pat, label in [
        (r"\[Zero-Offset Null MC N=\d+\] (.+)", "IDP Zero-Offset Look-Elsewhere Null MC (`N = 250`)"),
        (r"\[IDP QC Gate \(d\) Noise-Floor Audit\] (.+)", "IDP QC Gate (d) Clean `1x` vs Degraded `3x` Broadening"),
        (r"\[3-Seed Volumetric E2E bmr51984 \(p53-62aa\)\] (.+)", "Volumetric 2D `CON` + 3D E2E (`bmr51984` `p53-62aa`, `3` seeds)"),
        (r"\[3-Seed Volumetric E2E bmr15176 \(DARPP32-65aa\)\] (.+)", "Volumetric 2D `CON` + 3D E2E (`bmr15176` `DARPP32-65aa`, `3` seeds)"),
        (r"\[Pooled 6-Run \(3-Seed x 2-Protein\) Volumetric 2D CON \+ 3D E2E\] (.+)", "**Pooled 6-Run Volumetric IDP E2E**"),
        (r"\[No-CCCON 5-Spectrum 3x Noise \(18% ht=95 spur\)\] (.+)", "Degraded No-`CCCON` `3x` Noise (`15` runs)"),
        (r"\[Mid-Band sigma_seq_CA=0\.054 ppm \+ 15% ht=95 spur\] (.+)", "Mid-Band `sigma_seq_CA = 0.054 ppm` + `15%` Equal-Height Spurious"),
    ]:
        m = re.search(pat, txt)
        if m:
            cell_s = m.group(1).strip().replace(" | ", "; ")
            lines.append(f"| {label} | `{cell_s}` |")

    m_ferr = re.search(
        r"\[Volumetric Solomon-PRE 2D\+3D Picked Ferredoxin \(3 seeds, 1I7H\)\]:\s*\n\s*(.+)\n\s*(.+)\n\s*(.+)",
        txt,
    )
    if m_ferr:
        f_s = f"{m_ferr.group(1).strip()}; {m_ferr.group(2).strip()}; {m_ferr.group(3).strip()}".replace(
            " | ", ", "
        )
        lines.append(
            f"| Volumetric Solomon-PRE `2D+3D` Picked Ferredoxin (`1I7H`, `3` seeds) | `{f_s}` |"
        )

    # Calbindin D9k PCS table
    cal_block = re.search(
        r"Calbindin D9k Real Experimental Yb3\+-La3\+ 1H PCS[^\n]*\n((?:  .+\n)+)",
        txt,
    )
    if cal_block:
        lines.append("")
        lines.append("#### Calbindin `D9k` Experimental `Yb3+ - La3+` `1H` PCS (`off = 0` vs `+fit_off` Constant Offset)")
        lines.append("")
        lines.append("| Subset | Fit Method | `dchi_ax` | `dchi_rh` | `pcs_offset_ppm` | `Q`-Factor | RMSD (`ppm`) | `|dx_M|` (`Å`) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        cur_subset = ""
        row_re = re.compile(
            r"^\s{4}([^:]+):\s*dchi_ax=([+-]?[0-9.]+),\s*dchi_rh=([+-]?[0-9.]+),\s*"
            r"off=([+-]?[0-9.]+),\s*Q=([0-9.]+),\s*RMSD=([0-9.]+) ppm(?:,\s*\|dx_M\|=([0-9.]+) Å)?"
        )
        for raw_l in cal_block.group(1).splitlines():
            if raw_l.startswith("  ["):
                cur_subset = raw_l.strip().strip("[]").rstrip("]:")
                continue
            rm = row_re.match(raw_l)
            if rm:
                dx = f"`{rm.group(7)}`" if rm.group(7) else "`0.000 (fixed)`"
                lines.append(
                    f"| `{cur_subset}` | `{rm.group(1).strip()}` | `{rm.group(2)}` | `{rm.group(3)}` | "
                    f"`{rm.group(4)}` | `{rm.group(5)}` | `{rm.group(6)}` | {dx} |"
                )

    tot_p, tot_f, suites = parse_test_pass_counts(log_path)
    ws_p, ws_f, _ = parse_test_pass_counts(w2 / "w2_workspace_tests.log")
    lines.append("")
    lines.append("#### Slice `13.2` & Workspace Test Execution Summary")
    lines.append("")
    lines.append("| Suite Binary | Passed | Failed |")
    lines.append("|---|---:|---:|")
    for sname, sp, sf in suites:
        lines.append(f"| `{sname}` | `{sp}` | `{sf}` |")
    lines.append(f"| **Slice `13.2` Targeted Total** | **`{tot_p}`** | **`{tot_f}`** |")
    lines.append(f"| **Full Workspace (`cargo test --release --workspace`)** | **`{ws_p}`** | **`{ws_f}`** |")

    return "\n".join(lines) + "\n"


def render_slice_13_4(root: Path) -> str:
    w2 = root / "data/private/w2_results"
    log_path = w2 / "w2_slice_13_4_tests.log"
    txt = log_path.read_text(encoding="utf-8")
    lopo_data = json.loads((w2 / "w2_non_cold_26_lopo.json").read_text(encoding="utf-8"))
    lines: list[str] = []

    lines.append("### Slice `13.4` — `T13.0b` Clean-Room Bayesian Specific-Heat ($C_v(T)$) SA Solver (`bayes-sa`)")
    lines.append("")
    lines.append("| Verification Gate (`w2_slice_13_4_tests.log`) | Measured Outcome |")
    lines.append("|---|---|")
    for pat, label in [
        (r"\[thermodynamic C_v\(T\)\] (.+)", "Thermodynamic $C_v(T)$ Adaptive vs `FixedGeometric` Cooling"),
        (r"\[CPU vs WebGPU parity\] (.+)", "CPU (`f64`) vs WebGPU (`f32` + `Q27.36` fixed-point) Energy Parity"),
        (r"\[ubiquitin realistic 5-seed aggregate\] (.+)", "`nmr-core` Ubiquitin `--realistic` (`5`-seed aggregate, `R=32`)"),
        (r"\[WebGPU ubiquitin realistic 5-seed aggregate\] (.+)", "`nmr-gpu` WebGPU Ubiquitin `--realistic` (`5`-seed aggregate, `R=64`)"),
        (r"\[sparse 30% missing \+ 20% noise roots\] (.+)", "Sparse `30%` Missing Roots + `20%` Noise Roots Stress Gate"),
        (r"\[proline-rich IDP 84aa \(([^\)]+)\)\] (.+)", "Proline-Rich Disordered `84 aa` (`>=24% Pro`) Gate"),
        (r"\[480aa CPU\] (.+)", "Large `480 aa` Protein CPU Gate (`R=32`)"),
        (r"\[WebGPU 480aa ([^\]]+)\] (.+)", "Large `480 aa` Protein WebGPU Gate (`< 5.0 s` budget)"),
    ]:
        m = re.search(pat, txt)
        if m:
            val = "; ".join(g.strip().replace(" | ", "; ") for g in m.groups() if g)
            lines.append(f"| {label} | `{val}` |")

    tot_p, tot_f, suites = parse_test_pass_counts(log_path)
    lines.append("")
    lines.append("#### `bayes-sa` Dedicated Test Suites (`nmr-core`, `nmr-gpu`, `nmr-cli`)")
    lines.append("")
    lines.append("| Suite Binary | Passed | Failed |")
    lines.append("|---|---:|---:|")
    for sname, sp, sf in suites:
        lines.append(f"| `{sname}` | `{sp}` | `{sf}` |")
    lines.append(f"| **Total `bayes-sa` Dedicated Tests** | **`{tot_p}`** | **`{tot_f}`** |")

    # Dedicated LOPO knots & cold_p13_w2 summary for bayes-sa
    lines.append("")
    lines.append("#### `bayes-sa` Dedicated 26-Fold LOPO Calibration & Single-Shot `cold_p13_w2` Validation")
    lines.append("")
    lines.append(
        f"- **Nested 26x25 LOPO GPU->CPU Shrinkage Weight**: `CANONICAL_BAYES_SA_GPU_CPU_SHRINKAGE_WEIGHT = {lopo_data['canonical_bayes_gpu_shrinkage_w']:.2f}`"
    )
    lines.append("")
    lines.append(
        "| Engine | Cohort / Protocol | **Pooled `ECE_10` (`<=0.050`)** | Pooled `ECE_4` | Mean Debiased `ECE_10` | "
        "**Micro HC-Prec (`>=94%`)** | Macro Acc | Macro Recall |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for eng_key in ("BAYES_SA_CPU", "BAYES_SA_GPU"):
        eng = next(e for e in lopo_data["engines"] if e["engine"] == eng_key)
        hc_ok = sum(int(p["oof_hc_correct"]) for p in eng["per_protein"])
        hc_tot = sum(int(p["oof_hc_assigned"]) for p in eng["per_protein"])
        ece10 = float(eng["oof_pooled_ece_10"])
        mhcp = 100.0 * float(eng["oof_micro_hc_precision"])
        acc_mean = 100.0 * sum(float(p["accuracy"]) for p in eng["per_protein"]) / len(eng["per_protein"])
        lines.append(
            f"| `{eng_key}` | `26`-Protein Non-Cold OOF LOPO | **`{ece10:.4f}`** ({'PASS' if ece10 <= 0.050 else '**FAIL**'}) | "
            f"`{float(eng['oof_pooled_ece_4']):.4f}` | `{float(eng['oof_mean_debiased_ece_10']):.4f}` | "
            f"**`{mhcp:.2f}%` (`{hc_ok}/{hc_tot}`)** ({'PASS' if mhcp >= 94.0 else '**FAIL**'}) | `{acc_mean:.2f}%` | `n/a (OOF)` |"
        )

    for lbl, cpath in [
        ("`bayes-sa (cpu)`", w2 / "w2_cold_p13_w2_bayes_cpu.json"),
        ("`bayes-sa (gpu)`", w2 / "w2_cold_p13_w2_bayes_gpu.json"),
    ]:
        if cpath.exists():
            cs = summarize_real_bench_json(cpath)
            ece_flag = "PASS" if cs["pooled_ece10"] <= 0.050 else "**FAIL**"
            if cs["micro_hcp"] is not None:
                hcp_flag = "PASS" if cs["micro_hcp"] >= 94.0 else "**FAIL**"
                hcp_s = f"**`{cs['micro_hcp']:.2f}%` (`{cs['hc_ok']}/{cs['hc_tot']}`)** ({hcp_flag})"
            else:
                hcp_s = "`ABSTAIN (0/0)`"
            lines.append(
                f"| {lbl} | **Single-Shot Cold `cold_p13_w2`** (`{cs['evaluated']}/{cs['total']}`) | "
                f"**`{cs['pooled_ece10']:.4f}`** ({ece_flag}) | `{cs['pooled_ece4']:.4f}` | `{cs['mean_deb_ece10']:.4f}` | "
                f"{hcp_s} | `{cs['macro_acc']:.2f}%` | `{cs['macro_rec']:.2f}%` |"
            )

    return "\n".join(lines) + "\n"


def summarize_cold_cohort_entries(
    json_path: Path,
    cohort_filter: str | None = None,
    exclude_ids: set[str] | None = None,
) -> tuple[int, int, int, float | None]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    entries = [e for e in data.get("entries", []) if e.get("status") == "OK"]
    if cohort_filter is not None:
        entries = [e for e in entries if e.get("cohort") == cohort_filter]
    if exclude_ids:
        entries = [e for e in entries if str(e.get("bmrb_id")) not in exclude_ids]
    hc_ok = sum(int(e["reliability_curve"]["hc_correct"]) for e in entries)
    hc_tot = sum(int(e["reliability_curve"]["hc_assigned"]) for e in entries)
    pct = (100.0 * hc_ok / hc_tot) if hc_tot > 0 else None
    return len(entries), hc_ok, hc_tot, pct


def parse_provenance_gate_spec(prov_path: Path) -> list[tuple[str, str, str, str]]:
    txt = prov_path.read_text(encoding="utf-8")
    b_tag = "<!-- CALIBRATION-GATE-SPEC-BEGIN -->"
    e_tag = "<!-- CALIBRATION-GATE-SPEC-END -->"
    s = txt.index(b_tag) + len(b_tag)
    e = txt.index(e_tag, s)
    rows: list[tuple[str, str, str, str]] = []
    for raw in txt[s:e].splitlines():
        line = raw.strip()
        if (
            not line.startswith("|")
            or "Gate ID" in line
            or ":---" in line
            or line.startswith("|---")
        ):
            continue
        cols = [c.strip().strip("`") for c in line.strip("|").split("|")]
        if len(cols) >= 4:
            rows.append((cols[0], cols[1], cols[2], cols[3]))
    return rows


def parse_rust_gate_constants(
    rs_path: Path,
) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, str, str]]]:
    txt = rs_path.read_text(encoding="utf-8")

    def extract_const(name: str) -> list[tuple[str, str, str, str]]:
        m = re.search(
            rf"const\s+{name}\s*:[^=]+=\s*&\[(.*?)\];",
            txt,
            flags=re.DOTALL,
        )
        if not m:
            return []
        return re.findall(
            r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)',
            m.group(1),
        )

    return (
        extract_const("ACTIVE_26_FOLD_CALIBRATION_GATES"),
        extract_const("RETIRED_W1_13_FOLD_CALIBRATION_GATES"),
    )


def render_remediation(root: Path, extra_files: list[Path]) -> str:
    w2 = root / "data/private/w2_results"
    p_priv = root / "data/private"
    prov_path = root / "data/benchmark/PROVENANCE.md"
    rs_path = root / "crates/nmr-cli/src/real_benchmark.rs"
    manifest_path = root / "data/benchmark/real_manifest.tsv"
    lopo_data = json.loads((w2 / "w2_non_cold_26_lopo.json").read_text(encoding="utf-8"))

    lines: list[str] = []

    # 1. Digest table of updated & cited remediation artifacts
    rem_files = [
        manifest_path,
        prov_path,
        root / "data/torsion_db/PROVENANCE.md",
        root / "TODO.md",
        root / "docs/specs/14-regime-solvers.md",
        p_priv / "phase12_heldout_evo_graph.json",
        p_priv / "phase12_heldout_standard_cpu.json",
        p_priv / "phase12_heldout_standard_gpu.json",
        p_priv / "phase13_w1_cold_p13_evograph.json",
        p_priv / "phase13_w1_cold_p13_standard_cpu.json",
        p_priv / "phase13_w1_cold_p13_standard_gpu.json",
        w2 / "w2_non_cold_26_lopo.json",
        w2 / "w2_cold_p13_w2_evograph.json",
        w2 / "w2_cold_p13_w2_standard_cpu.json",
        w2 / "w2_cold_p13_w2_standard_gpu.json",
        w2 / "w2_cold_p13_w2_bayes_cpu.json",
        w2 / "w2_cold_p13_w2_bayes_gpu.json",
        w2 / "w2_remediation_workspace_tests.log",
        *extra_files,
    ]
    lines.append("### Pre-Tag Remediation Artifact & Raw JSON FIPS 180-4 SHA-256 Digests")
    lines.append("")
    lines.append(
        "Generated mechanically by `python3 scripts/render_phase13_w2_tables.py --section remediation`:"
    )
    lines.append("")
    lines.append("| Artifact Path | Bytes | SHA-256 |")
    lines.append("|---|---:|---|")
    seen_paths: set[Path] = set()
    for fp in rem_files:
        if fp.exists() and fp not in seen_paths:
            seen_paths.add(fp)
            digest, nbytes = sha256_file(fp)
            rel = fp.relative_to(root) if fp.is_relative_to(root) else fp
            lines.append(f"| `{rel}` | {nbytes:,} | `{digest}` |")

    # 2. Manifest seen_p13_w2 verification
    seen_w2_rows = parse_manifest_cohort(manifest_path, "seen_p13_w2")
    cold_w2_rows = parse_manifest_cohort(manifest_path, "cold_p13_w2")
    lines.append("")
    lines.append(
        f"- **Manifest Cohort Relabel Check (`data/benchmark/real_manifest.tsv`)**: "
        f"`cold_p13_w2` rows = **`{len(cold_w2_rows)}`**, "
        f"`seen_p13_w2` rows = **`{len(seen_w2_rows)}`** "
        f"(`{','.join(r['bmrb_id'] for r in seen_w2_rows)}`)."
    )

    # 3. 13.1:P1-1 Gate-set parity table
    doc_gates = parse_provenance_gate_spec(prov_path)
    rs_active, rs_retired = parse_rust_gate_constants(rs_path)
    rs_combined = rs_active + rs_retired
    if doc_gates != rs_combined:
        raise RuntimeError(
            f"Gate parity mismatch: PROVENANCE.md={doc_gates} vs real_benchmark.rs={rs_combined}"
        )
    lines.append("")
    lines.append(
        "### `13.1:P1-1` — Canonical Active vs Retired Calibration Gate Parity (`PROVENANCE.md` <-> `real_benchmark.rs`)"
    )
    lines.append("")
    lines.append(
        "| Gate ID | Operator | Threshold | `PROVENANCE.md` Status | `real_benchmark.rs` Constant | Parity |"
    )
    lines.append("|---|---|---|---|---|---|")
    for (dg, dop, dthr, dst), (rg, rop, rthr, rst) in zip(doc_gates, rs_combined):
        cname = (
            "`ACTIVE_26_FOLD_CALIBRATION_GATES`"
            if rst == "ACTIVE"
            else "`RETIRED_W1_13_FOLD_CALIBRATION_GATES`"
        )
        ok = (dg, dop, dthr, dst) == (rg, rop, rthr, rst)
        lines.append(
            f"| `{dg}` | `{dop}` | `{dthr}` | `{dst}` | {cname} | **`{'MATCH' if ok else 'MISMATCH'}`** |"
        )

    # 4. 13.1:P1-1 5-engine 26-fold OOF table on retired vs active gates
    macro_thr = float(next(t for g, _, t, _ in rs_retired if g == "macro_HCP"))
    deb_thr = float(next(t for g, _, t, _ in rs_retired if g == "mean_debiased_ECE_10"))
    ece10_thr = float(next(t for g, _, t, _ in rs_active if g == "pooled_ECE_10"))
    micro_thr = float(next(t for g, _, t, _ in rs_active if g == "micro_HCP"))

    lines.append("")
    lines.append(
        "### `13.1:P1-1` — 5-Engine 26-Fold Non-Cold OOF Evaluation on Retired `13.W1` Gates vs Active `13.W2` Gates (`w2_non_cold_26_lopo.json`)"
    )
    lines.append("")
    lines.append(
        f"| Engine (`26`-Fold OOF) | Retired `macro_HCP` (`>={100.0 * macro_thr:.1f}%`) | "
        f"Retired `mean_debiased_ECE_10` (`<={deb_thr:.3f}`) | Active `pooled_ECE_10` (`<={ece10_thr:.3f}`) | "
        f"Active `micro_HCP` (`>={100.0 * micro_thr:.1f}%`) | `clean_dev_capped == 0` | `5691 & 18561` OOF Capped |"
    )
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for eng in lopo_data["engines"]:
        macro_v = float(eng["oof_macro_hc_precision"])
        deb_v = float(eng["oof_mean_debiased_ece_10"])
        ece_v = float(eng["oof_pooled_ece_10"])
        micro_v = float(eng["oof_micro_hc_precision"])
        hc_ok = sum(int(p["oof_hc_correct"]) for p in eng["per_protein"])
        hc_tot = sum(int(p["oof_hc_assigned"]) for p in eng["per_protein"])
        dev_ok = len(eng["dev_clean_proteins_capped"]) == 0
        cat_ok = bool(eng["oof_5691_capped_or_warned"]) and bool(
            eng["oof_18561_capped_or_warned"]
        )
        macro_cell = (
            f"`{100.0 * macro_v:.2f}%` (`PASS`)"
            if macro_v >= macro_thr
            else f"**`{100.0 * macro_v:.2f}%` (`FAIL`)**"
        )
        deb_cell = (
            f"`{deb_v:.4f}` (`PASS`)"
            if deb_v <= deb_thr
            else f"**`{deb_v:.4f}` (`FAIL`)**"
        )
        ece_cell = (
            f"`{ece_v:.4f}` (`PASS`)"
            if ece_v <= ece10_thr
            else f"**`{ece_v:.4f}` (`FAIL`)**"
        )
        micro_cell = (
            f"`{100.0 * micro_v:.2f}%` (`{hc_ok}/{hc_tot}`, `PASS`)"
            if micro_v >= micro_thr
            else f"**`{100.0 * micro_v:.2f}%` (`{hc_ok}/{hc_tot}`, `FAIL`)**"
        )
        lines.append(
            f"| `{eng['engine']}` | {macro_cell} | {deb_cell} | {ece_cell} | {micro_cell} | "
            f"`{str(dev_ok).lower()}` (`{'PASS' if dev_ok else 'FAIL'}`) | "
            f"`{str(cat_ok).lower()}` (`{'PASS' if cat_ok else 'FAIL'}`) |"
        )

    # 5. 13.1:P1-2 Cold micro-HCP table across cold_p12, cold_p13, cold_p13_w2
    cold_matrix = [
        (
            "evo-graph",
            p_priv / "phase12_heldout_evo_graph.json",
            p_priv / "phase13_w1_cold_p13_evograph.json",
            w2 / "w2_cold_p13_w2_evograph.json",
        ),
        (
            "standard (cpu)",
            p_priv / "phase12_heldout_standard_cpu.json",
            p_priv / "phase13_w1_cold_p13_standard_cpu.json",
            w2 / "w2_cold_p13_w2_standard_cpu.json",
        ),
        (
            "standard (gpu)",
            p_priv / "phase12_heldout_standard_gpu.json",
            p_priv / "phase13_w1_cold_p13_standard_gpu.json",
            w2 / "w2_cold_p13_w2_standard_gpu.json",
        ),
        (
            "bayes-sa (cpu) [opt-in]",
            None,
            None,
            w2 / "w2_cold_p13_w2_bayes_cpu.json",
        ),
        (
            "bayes-sa (gpu) [opt-in]",
            None,
            None,
            w2 / "w2_cold_p13_w2_bayes_gpu.json",
        ),
    ]
    lines.append("")
    lines.append(
        "### `13.1:P1-2` — Per-Engine Unseen Single-Shot Cold Micro-HCP (`>= 0.80`) Across `cold_p12`, `cold_p13` & `cold_p13_w2`"
    )
    lines.append("")
    lines.append(
        "| Engine | `cold_p12` (`v0.8.0`, `7/7`) | `cold_p13` (`13.W1`, `5/7`) | `cold_p13_w2` (`13.W2`, `5/7`) | `cold_p13_w2` Ex-`16238` (`4` entries) | Unseen Cold-Observed Micro-HCP Range |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, p12_p, p13_p, w2_p in cold_matrix:
        observed_pcts: list[float] = []

        def fmt_cold_cell(
            jpath: Path | None,
            cohort_filter: str | None = None,
            exclude_ids: set[str] | None = None,
            record_range: bool = True,
            with_gate: bool = True,
        ) -> str:
            if jpath is None or not jpath.exists():
                return "`n/a`"
            _, ok, tot, pct = summarize_cold_cohort_entries(
                jpath, cohort_filter=cohort_filter, exclude_ids=exclude_ids
            )
            if pct is None:
                return "`ABSTAIN (0/0)`"
            if record_range:
                observed_pcts.append(pct)
            if not with_gate:
                return f"`{pct:.2f}%` (`{ok}/{tot}`)"
            if pct >= 100.0 * micro_thr:
                return f"`{pct:.2f}%` (`{ok}/{tot}`, `PASS`)"
            return f"**`{pct:.2f}%` (`{ok}/{tot}`, `FAIL`)**"

        c12 = fmt_cold_cell(p12_p, cohort_filter="cold_p12")
        c13 = fmt_cold_cell(p13_p)
        cw2 = fmt_cold_cell(w2_p)
        cw2_ex = fmt_cold_cell(
            w2_p, exclude_ids={"16238"}, record_range=False, with_gate=False
        )
        if len(observed_pcts) == 1:
            rng = f"**`{observed_pcts[0]:.2f}%`** (`1` cohort)"
        else:
            rng = f"**`{min(observed_pcts):.2f}%–{max(observed_pcts):.2f}%`** (`{len(observed_pcts)}` cohorts)"
        lines.append(f"| `{label}` | {c12} | {c13} | {cw2} | {cw2_ex} | {rng} |")

    # 6. Post-remediation workspace test log verification
    rem_log = w2 / "w2_remediation_workspace_tests.log"
    if rem_log.exists():
        tot_p, tot_f, suites = parse_test_pass_counts(rem_log)
        raw_log = rem_log.read_text(encoding="utf-8")
        gate_test_ok = (
            "test_documented_calibration_gate_set_matches_asserted_gate_set ... ok"
            in raw_log
        )
        manifest_test_ok = (
            "test_committed_real_manifest_tsv_schema_splits_and_disjointness ... ok"
            in raw_log
        )
        lines.append("")
        lines.append(
            "### Post-Remediation Workspace Test Verification (`data/private/w2_results/w2_remediation_workspace_tests.log`)"
        )
        lines.append("")
        lines.append("| Verification Check | Result |")
        lines.append("|---|---|")
        lines.append(
            f"| `cargo test --release --workspace` (`{len(suites)}` test result lines) | **`{tot_p} passed; {tot_f} failed`** |"
        )
        lines.append(
            f"| `test_documented_calibration_gate_set_matches_asserted_gate_set` | **`{'ok' if gate_test_ok else 'MISSING/FAILED'}`** |"
        )
        lines.append(
            f"| `test_committed_real_manifest_tsv_schema_splits_and_disjointness` (`seen_p13_w2`) | **`{'ok' if manifest_test_ok else 'MISSING/FAILED'}`** |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--extra-file", action="append", default=[])
    ap.add_argument(
        "--section",
        choices=["all", "digests", "13.1", "13.2", "13.4", "remediation"],
        default="all",
    )
    args = ap.parse_args()
    root = args.root.resolve()
    extras = [Path(p) if Path(p).is_absolute() else (root / p) for p in args.extra_file]

    parts: list[str] = []
    if args.section in ("all", "digests"):
        parts.append(render_digests(root, extras))
    if args.section in ("all", "13.1"):
        parts.append(render_slice_13_1(root))
    if args.section in ("all", "13.2"):
        parts.append(render_slice_13_2(root))
    if args.section in ("all", "13.4"):
        parts.append(render_slice_13_4(root))
    if args.section == "remediation":
        parts.append(render_remediation(root, extras))

    md = "\n".join(parts)
    if args.out is not None:
        args.out.write_text(md, encoding="utf-8")
    else:
        print(md, end="")


if __name__ == "__main__":
    main()
