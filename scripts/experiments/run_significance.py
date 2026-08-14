"""Phase 5.2: significance testing for the Fused-5 / joint-training comparison.

Two layers of evidence:

  1. **Pooled paired test across 145 (region x seed) pairs** (primary).
     Each pair yields one aggregate MAE per method. We test whether the MAE
     difference is systematically non-zero across the 145 independent pairs
     (Wilcoxon signed-rank + paired t + bootstrap 95% CI on the median delta),
     with Holm-Bonferroni correction across the family of comparisons.

  2. **Per-region temporal Diebold-Mariano on a subset** (robustness; written
     by run_residual_dump + analyze step). This script consumes the dumped
     per-origin residuals if present and appends a DM block.

Outputs:
  results/fused_five_significance.json
  results/fused_five_significance.md

Usage:
    .venv/bin/python scripts/experiments/run_significance.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from transcif.evaluation.stats import (
    diebold_mariano,
    holm_bonferroni,
    paired_bootstrap_ci,
    paired_t_test,
    wilcoxon_test,
)

HERE = Path(__file__).resolve().parent.parent.parent
RESULTS = HERE / "results"

# External reference number (single point, no per-pair data).
PATCHTST_SUPERVISED_MAE = 41.47

# Comparisons in the primary family. Each entry: (label, method_a, method_b).
# method_a - method_b > 0 means a is worse (higher MAE).
# We report "A vs B" as A's MAE minus B's MAE.
PRIMARY_COMPARISONS = [
    # headline: joint_trained vs each baseline (positive delta = joint better)
    ("joint_trained - basismix_plus", "joint_trained", "basismix_plus"),
    ("joint_trained - causal_plus", "joint_trained", "causal_plus"),
    ("joint_trained - equal_plus", "joint_trained", "equal_plus"),
    ("joint_trained - persistence", "joint_trained", "persistence"),
    ("joint_trained - best_single", "joint_trained", "best_single"),
    # DoD literal: basismix_plus vs each
    ("basismix_plus - causal_plus", "basismix_plus", "causal_plus"),
    ("basismix_plus - equal_plus", "basismix_plus", "equal_plus"),
    ("basismix_plus - persistence", "basismix_plus", "persistence"),
]

SINGLE_DIRECTIONS = ["rag", "phys", "causal", "icl", "hier"]


def _load_fused_rows():
    """Load fused_five_full.json, falling back to the committed git version.

    A pre-existing long-running ``run_fused_five_full.py`` re-run can leave the
    working copy with only the latest row (its non-resume incremental write
    clobbers the file mid-run). When the working copy is degenerate (<100
    rows), use ``git show HEAD:results/fused_five_full.json`` instead so the
    significance test runs on the canonical 145-row result.
    """
    path = RESULTS / "fused_five_full.json"
    rows = json.loads(path.read_text())
    if len(rows) >= 100:
        return rows, str(path)
    # fall back to committed version
    import subprocess
    committed = subprocess.run(
        ["git", "show", "HEAD:results/fused_five_full.json"],
        capture_output=True, text=True, cwd=HERE,
    )
    if committed.returncode != 0:
        raise RuntimeError(
            f"fused_five_full.json has only {len(rows)} rows and git fallback "
            f"failed: {committed.stderr}"
        )
    rows = json.loads(committed.stdout)
    print(f"[FALLBACK] working-copy fused_five_full.json degenerate "
          f"({len(rows)} rows from git HEAD used)", flush=True)
    return rows, "git:HEAD:results/fused_five_full.json"


def load_paired_maes():
    """Merge fused_five_full.json + joint_train_full.json on (target, seed).

    Returns:
        dict method -> np.array of MAE (length n_pairs, aligned across methods),
        plus the list of (target, seed) pairs in order.
    """
    fused, fused_source = _load_fused_rows()
    joint = json.loads((RESULTS / "joint_train_full.json").read_text())

    # index joint by (target, seed) -> held_out_mae
    joint_map = {}
    for r in joint:
        if "held_out_mae" in r and r["held_out_mae"] is not None:
            joint_map[(r["target"], r["seed"])] = r["held_out_mae"]

    pairs = []
    methods = {}
    # methods present in fused rows
    fused_methods = [
        "rag", "rag_plus", "phys", "phys_plus", "causal", "causal_plus",
        "icl", "icl_plus", "hier", "hier_plus",
        "equal", "equal_plus", "basismix", "basismix_plus", "persistence",
    ]
    for m in fused_methods + ["joint_trained"]:
        methods[m] = []
    best_single_vals = []

    for r in fused:
        key = (r["target"], r["seed"])
        if key not in joint_map:
            continue
        pairs.append(key)
        for m in fused_methods:
            methods[m].append(r[m]["mae"])
        methods["joint_trained"].append(joint_map[key])
        # best single direction (ZS+) per pair
        single_plus = [r[f"{d}_plus"]["mae"] for d in SINGLE_DIRECTIONS]
        best_single_vals.append(min(single_plus))

    methods_arr = {m: np.array(v, dtype=np.float64) for m, v in methods.items()}
    methods_arr["best_single"] = np.array(best_single_vals, dtype=np.float64)
    return methods_arr, pairs


def run_primary_family(methods):
    """Run Wilcoxon + paired-t + bootstrap on each primary comparison."""
    raw_results = []
    for label, a, b in PRIMARY_COMPARISONS:
        ma = methods[a]
        mb = methods[b]
        delta = mb - ma  # positive => a (first named) is better (lower MAE)
        w = wilcoxon_test(mb, ma)  # tests median(mb - ma)
        t = paired_t_test(mb, ma)
        boot = paired_bootstrap_ci(delta, n_boot=20000, seed=0)
        win_rate = float(np.mean(delta > 0))
        raw_results.append({
            "comparison": label,
            "method_a": a,
            "method_b": b,
            "n_pairs": int(len(delta)),
            "median_delta_mae": boot["median"],  # mb - ma
            "mean_delta_mae": t["mean_delta"],
            "bootstrap_ci95": [boot["ci_lo"], boot["ci_hi"]],
            "wilcoxon_p": w["p_value"],
            "wilcoxon_effect_r": w["effect_size_r"],
            "paired_t_p": t["p_value"],
            "paired_t_stat": t["statistic"],
            "cohen_d": t["cohen_d"],
            "win_rate_a_better": win_rate,  # fraction where a has lower MAE
            "median_mae_a": float(np.median(ma)),
            "median_mae_b": float(np.median(mb)),
        })
    # Holm-Bonferroni across the Wilcoxon p-values
    pvals = [r["wilcoxon_p"] for r in raw_results]
    holm = holm_bonferroni(pvals, alpha=0.05)
    for r, h in zip(raw_results, holm):
        r["holm_p"] = h["holm_p"]
        r["reject_holm_0.05"] = h["reject"]
    return raw_results


def patchtst_one_sample(methods):
    """One-sample test: is joint_trained MAE systematically < PatchTST 41.47?

    Uses a one-sample Wilcoxon on (joint_trained - 41.47) testing < 0.
    """
    from scipy import stats as sp_stats
    jt = methods["joint_trained"]
    diff = jt - PATCHTST_SUPERVISED_MAE
    res = sp_stats.wilcoxon(diff, alternative="less")
    win = float(np.mean(jt < PATCHTST_SUPERVISED_MAE))
    return {
        "patchtst_supervised_mae": PATCHTST_SUPERVISED_MAE,
        "joint_trained_median_mae": float(np.median(jt)),
        "n_pairs": int(len(jt)),
        "win_rate_joint_beats_patchtst": win,
        "wilcoxon_p_one_sided_less": float(res.pvalue),
        "note": ("PatchTST-supervised is a single external number (no per-pair "
                 "data), so this is a one-sample test on joint_trained MAE vs "
                 "the constant 41.47, not a paired comparison."),
    }


def run_subset_dm():
    """If residual npz files exist, run per-region DM (joint vs basismix_plus,
    joint vs persistence) and return a summary block. Else return None."""
    residuals_dir = RESULTS / "residuals"
    files = sorted(residuals_dir.glob("*_seed*.npz")) if residuals_dir.exists() else []
    if not files:
        return None
    from scipy import stats as sp_stats
    rows = []
    for f in files:
        d = np.load(f)
        name = f.stem
        y = d["y_true"]
        out = {"pair": name}
        if "err_joint_trained" in d.files and "err_basismix_plus" in d.files:
            # DM on per-hour errors (n_eval*HORIZON), h=HORIZON
            e_jt = (d["pred_joint_trained"] - y).reshape(-1)
            e_bm = (d["pred_basismix_plus"] - y).reshape(-1)
            dm = diebold_mariano(e_bm, e_jt, horizon=24)
            # positive mean_loss_diff => |e_bm|>|e_jt| => joint better
            out["dm_joint_vs_basismix_plus"] = dm
            out["joint_better_than_basismix_plus"] = bool(dm["mean_loss_diff"] > 0)
        if "err_joint_trained" in d.files and "err_persistence" in d.files:
            e_jt = (d["pred_joint_trained"] - y).reshape(-1)
            e_p = (d["pred_persistence"] - y).reshape(-1)
            dm = diebold_mariano(e_p, e_jt, horizon=24)
            out["dm_joint_vs_persistence"] = dm
            out["joint_better_than_persistence"] = bool(dm["mean_loss_diff"] > 0)
        rows.append(out)
    # Holm across the joint-vs-basismix_plus p-values (one family)
    pvals_bm = [r["dm_joint_vs_basismix_plus"]["p_value"]
                for r in rows if "dm_joint_vs_basismix_plus" in r]
    holm_bm = holm_bonferroni(pvals_bm, alpha=0.05) if pvals_bm else []
    sig_bm = sum(1 for h in holm_bm if h["reject"])
    return {
        "n_pairs": len(rows),
        "per_pair": rows,
        "holm_joint_vs_basismix_plus": {
            "n_significant": sig_bm,
            "details": holm_bm,
        },
        "note": ("Per-region Harvey-adjusted DM test on per-hour errors "
                 "(n_eval*24, h=24). Positive DM stat => joint model has "
                 "larger loss => baseline better; negative => joint better."),
    }


def write_markdown(primary, patchtst, subset_dm, out_md):
    lines = []
    lines.append("# Phase 5.2 — Significance Test Summary\n")
    lines.append("## 1. Primary: pooled paired test across 145 (region × seed) pairs\n")
    lines.append("Each pair contributes one aggregate MAE per method. "
                 "`delta = MAE(B) - MAE(A)`; **positive delta means A is better** "
                 "(lower MAE). Holm-Bonferroni correction applied across the family.\n")
    lines.append("| Comparison | median ΔMAE | 95% bootstrap CI | Wilcoxon p | Holm p | reject@0.05 | win-rate A better |")
    lines.append("|---|---:|---|---:|---:|:---:|---:|")
    for r in primary:
        ci = f"[{r['bootstrap_ci95'][0]:.2f}, {r['bootstrap_ci95'][1]:.2f}]"
        reject = "✅" if r["reject_holm_0.05"] else "—"
        lines.append(
            f"| {r['comparison']} | {r['median_delta_mae']:+.2f} | {ci} | "
            f"{r['wilcoxon_p']:.2e} | {r['holm_p']:.2e} | {reject} | "
            f"{r['win_rate_a_better']*100:.1f}% |"
        )
    lines.append("")
    lines.append("Effect sizes (Wilcoxon r, |Z|/√N):  "
                 + ", ".join(f"{r['comparison'].split(' - ')[0]}: {r['wilcoxon_effect_r']:.2f}"
                              for r in primary[:5]))
    lines.append("")

    lines.append("## 2. PatchTST-supervised (external reference, 41.47)\n")
    lines.append(f"- joint_trained median MAE: **{patchtst['joint_trained_median_mae']:.2f}**")
    lines.append(f"- joint_trained beats 41.47 on **{patchtst['win_rate_joint_beats_patchtst']*100:.1f}%** of pairs")
    lines.append(f"- one-sample Wilcoxon (joint − 41.47 < 0), p = {patchtst['wilcoxon_p_one_sided_less']:.2e}")
    lines.append(f"- _{patchtst['note']}_\n")

    if subset_dm is not None:
        lines.append("## 3. Robustness: per-region Diebold-Mariano (temporal)\n")
        lines.append(f"Harvey-adjusted DM on per-hour errors (n={subset_dm['n_pairs']} subset pairs). "
                     f"Joint vs BasisMix+: **{subset_dm['holm_joint_vs_basismix_plus']['n_significant']}**/"
                     f"{subset_dm['n_pairs']} significant after Holm.\n")
        lines.append("| pair | joint MAE < basismix+ MAE? | joint MAE < persistence? |")
        lines.append("|---|:---:|:---:|")
        for r in subset_dm["per_pair"]:
            jb = "✅" if r.get("joint_better_than_basismix_plus") else "❌"
            jp = "✅" if r.get("joint_better_than_persistence") else "❌"
            lines.append(f"| {r['pair']} | {jb} | {jp} |")
        lines.append("")
        lines.append(f"_{subset_dm['note']}_\n")

    out_md.write_text("\n".join(lines))


def main():
    methods, pairs = load_paired_maes()
    n = len(pairs)
    print(f"[LOAD] {n} paired (region, seed) observations", flush=True)

    primary = run_primary_family(methods)
    patchtst = patchtst_one_sample(methods)
    subset_dm = run_subset_dm()

    out = {
        "n_pairs": n,
        "primary_family": primary,
        "patchtst_reference": patchtst,
        "subset_dm_robustness": subset_dm,
        "methodology": {
            "primary": ("Pooled paired test across n_pairs (region x seed) "
                        "aggregate MAEs. Wilcoxon signed-rank + paired t + "
                        "bootstrap 95% CI (20000 resamples) on the median "
                        "delta. Holm-Bonferroni across the 8-comparison family."),
            "interpretation": ("delta = MAE(B) - MAE(A); positive => A better. "
                               "win_rate_a_better = fraction of pairs where A "
                               "has strictly lower MAE."),
        },
    }
    (RESULTS / "fused_five_significance.json").write_text(json.dumps(out, indent=2))
    write_markdown(primary, patchtst, subset_dm, RESULTS / "fused_five_significance.md")
    print(f"[WRITE] {RESULTS/'fused_five_significance.json'}")
    print(f"[WRITE] {RESULTS/'fused_five_significance.md'}")

    # console summary
    print("\n=== Primary family (Holm-corrected) ===")
    for r in primary:
        flag = "REJECT" if r["reject_holm_0.05"] else "keep"
        print(f"  {r['comparison']:38s} Δ={r['median_delta_mae']:+6.2f} "
              f"p_holm={r['holm_p']:.3e} [{flag}]  win={r['win_rate_a_better']*100:.0f}%")
    print(f"\nPatchTST one-sample p = {patchtst['wilcoxon_p_one_sided_less']:.2e}")
    if subset_dm:
        print(f"Subset DM: {subset_dm['holm_joint_vs_basismix_plus']['n_significant']}/"
              f"{subset_dm['n_pairs']} joint<basismix+ significant")


if __name__ == "__main__":
    main()
