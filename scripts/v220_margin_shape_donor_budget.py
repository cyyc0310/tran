"""V220: donor-budget sensitivity for V219 fine-margin shape transfer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import v219_monthly_margin_constrained_shape_transfer as v219

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v220-margin-shape-donor-budget-r2"
K_VALUES = (1, 3, 5, 10, 15, 23)
ALPHA = 0.10


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    series, hashes = v219.load_series()
    cache = v219.ENRICHED_CACHE
    with cache.open("rb") as stream:
        series, cache_hashes = __import__("pickle").load(stream)
    if hashes != cache_hashes:
        raise RuntimeError("V219 enriched cache source hashes mismatch")
    regions = v219.eligible_regions(series)
    target_records = {k: {} for k in K_VALUES}
    source_scores = {k: [] for k in K_VALUES}

    for excluded in regions:
        pool = [r for r in regions if r != excluded]
        for q in pool:
            allowed = [r for r in pool if r != q]
            for month in range(1, 13):
                key = (q, 2025, month)
                if key not in series:
                    continue
                d = series[key]
                for k in K_VALUES:
                    profile, _ = v219.profile_for(key, allowed, series, fine=True, k=k)
                    source_scores[k].extend(np.abs(d["cif"] - profile))

    radii = {k: float(np.quantile(source_scores[k], 1 - ALPHA, method="higher")) for k in K_VALUES}
    for region in regions:
        allowed = [r for r in regions if r != region]
        for month in range(1, 7):
            key = (region, 2026, month)
            if key not in series:
                continue
            d = series[key]
            for k in K_VALUES:
                profile, _ = v219.profile_for(key, allowed, series, fine=True, k=k)
                target_records[k][(region, month)] = {
                    "actual": np.asarray(d["cif"], float),
                    "prediction": profile,
                }

    rows = []
    region_rows = []
    region_vectors = {k: {} for k in K_VALUES}
    for k in K_VALUES:
        grouped = {}
        for (region, month), record in target_records[k].items():
            ae = np.abs(record["actual"] - record["prediction"])
            radius = radii[k]
            metrics = {
                "coverage": float(np.mean(ae <= radius)),
                "width": float(2 * radius),
                "winkler90": float(np.mean(2 * radius + 20 * np.maximum(ae - radius, 0))),
                "point_mae": float(ae.mean()),
            }
            rows.append({"k": k, "region": region, "month": month, "n_hours": len(ae), **metrics})
            grouped.setdefault(region, []).append(metrics)
        for region, values in grouped.items():
            avg = {name: float(np.mean([value[name] for value in values]))
                   for name in ("coverage", "width", "winkler90", "point_mae")}
            region_rows.append({"k": k, "region": region, "n_months": len(values), **avg})
            region_vectors[k][region] = avg

    region_df = pd.DataFrame(region_rows)
    macro = {}
    for k in K_VALUES:
        d = region_df.loc[region_df.k.eq(k)]
        macro[k] = {name: float(d[name].mean()) for name in ("coverage", "width", "winkler90", "point_mae")}
        macro[k]["worst_region_coverage"] = float(d.coverage.min())

    rng = np.random.default_rng(220)
    boot_rows = []
    base_k = 10
    names = ("coverage", "width", "winkler90", "point_mae")
    for k in K_VALUES:
        if k == base_k:
            continue
        delta = np.array([[region_vectors[k][r][name] - region_vectors[base_k][r][name]
                           for name in names] for r in sorted(region_vectors[k])])
        draws = delta[rng.integers(0, len(delta), size=(20000, len(delta)))].mean(axis=1)
        for j, name in enumerate(names):
            lo, hi = np.quantile(draws[:, j], [.025, .975])
            boot_rows.append({"k": k, "baseline_k": base_k, "metric": name,
                              "delta": float(delta[:, j].mean()), "ci95_low": float(lo),
                              "ci95_high": float(hi), "n_resamples": 20000, "cluster": "region"})

    OUT.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(OUT / "query_metrics.csv", index=False, float_format="%.10g")
    region_df.to_csv(OUT / "region_metrics.csv", index=False, float_format="%.10g")
    pd.DataFrame(boot_rows).to_csv(OUT / "paired_region_bootstrap.csv", index=False, float_format="%.10g")
    summary = {
        "status": "RETROSPECTIVE_EIA_DONOR_BUDGET_DEVELOPMENT",
        "regions": list(regions), "n_regions": len(regions),
        "n_target_region_months": sum(len(x) for x in target_records.values()) // len(K_VALUES),
        "target_window": "2026-01..2026-06", "source_window": "2025-01..2025-12",
        "k_donors": list(K_VALUES), "calibration_q90": radii,
        "macro_metrics": macro, "input_hashes": hashes,
        "limits": [
            "same EIA hourly generation proxy family and fixed EF labels as V219",
            "target fuel margins are post-month information; this is not a month-ahead forecast",
            "donor k grid is development tuning on already examined regions and 2026H1",
            "donor transfer is conventional and no novelty claim is made",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(macro, indent=2), flush=True)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
