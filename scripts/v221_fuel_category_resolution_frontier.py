"""V221: how fine must target monthly fuel categories be for shape transfer?"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import v219_monthly_margin_constrained_shape_transfer as v219

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v221-fuel-category-resolution-frontier-r1"
ALPHA = 0.10


def category_groups():
    # Fuel order: coal, gas, oil, other, nuclear, hydro, wind, solar.
    def rows(spec):
        matrix = np.zeros((len(spec), 8), dtype=float)
        for i, names in enumerate(spec):
            for name in names:
                matrix[i, v219.FUELS.index(name)] = 1.0
        return matrix
    return {
        "5_thermal_renewable": rows([
            ("coal", "gas", "oil", "other"), ("nuclear",), ("hydro",), ("wind",), ("solar",)
        ]),
        "6_coal_vs_other_thermal": rows([
            ("coal",), ("gas", "oil", "other"), ("nuclear",), ("hydro",), ("wind",), ("solar",)
        ]),
        "7_coal_gas_vs_other_thermal": rows([
            ("coal",), ("gas",), ("oil", "other"), ("nuclear",), ("hydro",), ("wind",), ("solar",)
        ]),
        "8_fuel_specific": np.eye(8),
    }


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    series, hashes = v219.load_series()
    with v219.ENRICHED_CACHE.open("rb") as stream:
        series, cache_hashes = pickle.load(stream)
    if hashes != cache_hashes:
        raise RuntimeError("source hash mismatch in enriched cache")
    regions = v219.eligible_regions(series)
    groups_by_level = category_groups()
    names = tuple(groups_by_level)
    scores = {name: [] for name in names}
    targets = {name: {} for name in names}

    # Nested source residuals: target and calibration source are both absent
    # from the permitted donor pool and point-level training pool.
    for excluded in regions:
        pool = [r for r in regions if r != excluded]
        for q in pool:
            allowed = [r for r in pool if r != q]
            for month in range(1, 13):
                key = (q, 2025, month)
                if key not in series:
                    continue
                d = series[key]
                for name, groups in groups_by_level.items():
                    pred, _ = v219.profile_for(key, allowed, series, fine=False, k=10, groups_override=groups)
                    scores[name].extend(np.abs(d["cif"] - pred))

    radii = {name: float(np.quantile(np.asarray(value), 1 - ALPHA, method="higher"))
             for name, value in scores.items()}
    for region in regions:
        allowed = [r for r in regions if r != region]
        for month in range(1, 7):
            key = (region, 2026, month)
            if key not in series:
                continue
            d = series[key]
            for name, groups in groups_by_level.items():
                pred, _ = v219.profile_for(key, allowed, series, fine=False, k=10, groups_override=groups)
                targets[name][(region, month)] = (np.asarray(d["cif"], float), pred)

    query_rows = []
    region_rows = []
    region_scores = {name: {} for name in names}
    for name in names:
        by_region = {}
        for (region, month), (actual, pred) in targets[name].items():
            ae = np.abs(actual - pred)
            radius = radii[name]
            metrics = {
                "coverage": float(np.mean(ae <= radius)), "width": float(2 * radius),
                "winkler90": float(np.mean(2 * radius + 20 * np.maximum(ae - radius, 0))),
                "point_mae": float(ae.mean()),
            }
            query_rows.append({"category_level": name, "region": region, "month": month,
                               "n_hours": len(actual), **metrics})
            by_region.setdefault(region, []).append(metrics)
        for region, values in by_region.items():
            avg = {metric: float(np.mean([v[metric] for v in values])
                   ) for metric in ("coverage", "width", "winkler90", "point_mae")}
            region_scores[name][region] = avg
            region_rows.append({"category_level": name, "region": region,
                                "n_months": len(values), **avg})

    region_df = pd.DataFrame(region_rows)
    macro = {}
    for name in names:
        x = region_df.loc[region_df.category_level.eq(name)]
        macro[name] = {metric: float(x[metric].mean()) for metric in
                       ("coverage", "width", "winkler90", "point_mae")}
        macro[name]["worst_region_coverage"] = float(x.coverage.min())

    rng = np.random.default_rng(221)
    boot_rows = []
    base = "5_thermal_renewable"
    metric_names = ("coverage", "width", "winkler90", "point_mae")
    for name in names:
        if name == base:
            continue
        delta = np.array([[region_scores[name][r][metric] - region_scores[base][r][metric]
                           for metric in metric_names] for r in sorted(region_scores[name])])
        draw = delta[rng.integers(0, len(delta), size=(20000, len(delta)))].mean(axis=1)
        for j, metric in enumerate(metric_names):
            lo, hi = np.quantile(draw[:, j], [.025, .975])
            boot_rows.append({"candidate": name, "baseline": base, "metric": metric,
                              "delta": float(delta[:, j].mean()), "ci95_low": float(lo),
                              "ci95_high": float(hi), "n_resamples": 20000, "cluster": "region"})

    OUT.mkdir(parents=True)
    pd.DataFrame(query_rows).to_csv(OUT / "query_metrics.csv", index=False, float_format="%.10g")
    region_df.to_csv(OUT / "region_metrics.csv", index=False, float_format="%.10g")
    pd.DataFrame(boot_rows).to_csv(OUT / "paired_region_bootstrap.csv", index=False, float_format="%.10g")
    summary = {
        "status": "RETROSPECTIVE_EIA_MONTHLY_FUEL_CATEGORY_RESOLUTION_FRONTIER",
        "regions": list(regions), "n_regions": len(regions),
        "n_target_region_months": len(targets[names[0]]),
        "category_levels": {name: [list(np.array(v219.FUELS)[np.flatnonzero(row)]) for row in groups]
                            for name, groups in groups_by_level.items()},
        "k_donors": 10, "calibration": "nested leave-two-region-out absolute residual q90",
        "target_window": "2026-01..2026-06", "source_window": "2025-01..2025-12",
        "fixed_ef_gco2_per_kwh": dict(zip(v219.FUELS, v219.EF.tolist())),
        "radii": radii, "macro_metrics": macro, "input_hashes": hashes,
        "limits": [
            "EIA hourly generation estimates converted to CIF with fixed EF, not measured CIF",
            "target monthly categories are post-month information; not a month-ahead forecast",
            "2026H1 and target regions were already examined in prior development work",
            "unreported internal thermal split is allocated by source donor mean fuel shares",
            "no method novelty or China field-schema validation claim",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(macro, indent=2), flush=True)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
