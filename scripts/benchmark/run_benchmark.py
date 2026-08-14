#!/usr/bin/env python
"""TransCIF benchmark leaderboard builder (Phase FD-3).

Aggregates evaluation artifacts into ``results/leaderboard.json`` following
the schema in ``docs/BENCHMARK.md``:

    python scripts/benchmark/run_benchmark.py                 # default sources
    python scripts/benchmark/run_benchmark.py --fd results/fuel_decomp_eval_full.json

Sources:
    fuel_decomp_eval_*.json   FuelDecompNet I_cfg / I_0 tiers +
                              annual/monthly-constant + persistence baselines
    unified_eval_full.json    legacy TransCIF ZS / ZS+ / PatchTST ladder
                              (optional --unified)
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

RESULTS = Path(__file__).resolve().parent.parent.parent / "results"


def _median(d):
    return float(np.median(d)) if len(d) else None


def _entry(method, tier, rows, key, baseline_key=None, extra=None):
    vals = [r[key] for r in rows if r.get(key)]
    if not vals:
        return None
    e = {
        "method": method,
        "tier": tier,
        "n_pairs": len(vals),
        "median_mae": _median([v["mae"] for v in vals]),
        "mean_mae": float(np.mean([v["mae"] for v in vals])),
        "median_diurnal_mae": _median([v["diurnal_mae"] for v in vals]),
        "median_monthly_shape_mae": _median([v["monthly_shape_mae"] for v in vals]),
        "median_spearman": _median([v["spearman"] for v in vals]),
        "median_bias": _median([v["bias"] for v in vals]),
    }
    if baseline_key:
        pairs = [(r[key]["mae"], r[baseline_key]["mae"]) for r in rows
                 if r.get(key) and r.get(baseline_key)]
        if len(pairs) >= 5:
            method_mae = [p[0] for p in pairs]
            base_mae = [p[1] for p in pairs]
            e["win_rate_vs_baseline"] = float(np.mean(
                [m < b for m, b in pairs]))
            try:
                stat, p = wilcoxon(method_mae, base_mae)
                e["paired_wilcoxon_p_vs_baseline"] = float(p)
            except ValueError:
                pass
    if extra:
        e.update(extra)
    return e


def from_fd_rows(rows):
    """Leaderboard entries from a fuel-decomp eval artifact."""
    out = []
    for method, tier, key, base in (
        ("fuel_decomp", "I_0", "fuel_i0", "persistence"),
        ("fuel_decomp", "I_cfg", "fuel_i_cfg", "config_constant"),
        ("annual_constant", "I_cfg", "config_constant", None),
        ("monthly_constant", "I_cfg(oracle)", "monthly_constant", None),
        ("persistence_lag24", "reference", "persistence", None),
    ):
        e = _entry(method, tier, rows, key, base)
        if e:
            out.append(e)
    return out


def from_unified_rows(rows):
    """Leaderboard entries from the legacy unified-eval artifact."""
    out = []
    for method, tier, key, base in (
        ("transcif_zs", "I_0", "transcif_zs", "persistence"),
        ("transcif_zs_plus", "I_+", "transcif_zs_plus", "persistence"),
        ("patchtst_supervised", "I_S", "patchtst_sup", None),
        ("persistence_lag24", "reference", "persistence", None),
    ):
        vals = [r for r in rows if r.get(key)]
        if not vals:
            continue
        maes = [r[key]["mae"] for r in vals]
        e = {
            "method": method, "tier": tier, "n_pairs": len(vals),
            "median_mae": _median(maes), "mean_mae": float(np.mean(maes)),
        }
        if base:
            pairs = [(r[key]["mae"], r[base]["mae"]) for r in rows
                     if r.get(key) and r.get(base)]
            if len(pairs) >= 5:
                try:
                    _, p = wilcoxon([a for a, _ in pairs], [b for _, b in pairs])
                    e["paired_wilcoxon_p_vs_baseline"] = float(p)
                    e["win_rate_vs_baseline"] = float(np.mean(
                        [a < b for a, b in pairs]))
                except ValueError:
                    pass
        out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", nargs="+", default=[
        str(RESULTS / "fuel_decomp_eval_quick.json"),
        str(RESULTS / "fuel_decomp_eval_full.json")])
    ap.add_argument("--unified", default="scripts/results/unified_eval_full.json")
    ap.add_argument("--out", default=str(RESULTS / "leaderboard.json"))
    args = ap.parse_args()

    entries = []
    for path in args.fd:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                rows = json.load(f).get("rows", [])
            if rows:
                entries += from_fd_rows(rows)
                print(f"[bench] {p.name}: {len(rows)} pairs")

    uni = Path(args.unified)
    if uni.exists():
        with open(uni) as f:
            doc = json.load(f)
        rows = doc if isinstance(doc, list) else doc.get("results", doc.get("rows", []))
        if rows:
            entries += from_unified_rows(rows)
            print(f"[bench] {uni.name}: {len(rows)} pairs")

    # Deduplicate by (method, tier), keeping the largest n_pairs.
    best = {}
    for e in entries:
        k = (e["method"], e["tier"])
        if k not in best or e["n_pairs"] > best[k]["n_pairs"]:
            best[k] = e
    board = sorted(best.values(), key=lambda e: (
        {"I_cfg": 0, "I_0": 1, "I_+": 2, "I_J": 3, "I_S": 4}.get(e["tier"], 9),
        e["median_mae"] if e.get("median_mae") is not None else 9e9))

    with open(args.out, "w") as f:
        json.dump({"leaderboard": board, "schema": "docs/BENCHMARK.md v0.1"},
                  f, indent=1)
    print(f"[bench] wrote {args.out} ({len(board)} entries)")
    for e in board:
        print(f"  {e['tier']:12s} {e['method']:24s} "
              f"median MAE {e.get('median_mae', float('nan')):7.2f}  "
              f"n={e['n_pairs']}")


if __name__ == "__main__":
    main()
