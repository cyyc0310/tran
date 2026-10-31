"""V112: conservatively calibrate V111's robust anchor by worst source region."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V111 = ROOT / "results/v111-factor-robust-monthly-anchor-r1"
V105_ROWS = ROOT / "results/v105-unseen-region-capacity-ablation-r1/paired_rows.csv"
OUT = ROOT / "results/v112-worst-region-robust-anchor-r1"
PROTOCOL = ROOT / "docs/V112_WORST_REGION_ROBUST_ANCHOR_PROTOCOL.md"
RATES = (0.01, 0.05, 0.10)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    calibration = pd.read_csv(V111 / "source_calibration_rows.csv")
    target = pd.read_csv(V111 / "target_rows.csv")
    v105 = pd.read_csv(V105_ROWS).drop_duplicates(["region", "month", "tau"])
    radii = {}
    region_quantiles = []
    for tau, subset in calibration.groupby("tau"):
        by_region = subset.groupby("region").worst_case_absolute_error.quantile(0.90)
        radii[int(tau)] = float(by_region.max())
        for region, value in by_region.items():
            region_quantiles.append({"tau": int(tau), "region": region, "q90": float(value),
                                     "selected_as_max": bool(value == by_region.max())})

    rows = []
    for item in target.itertuples(index=False):
        radius = radii[int(item.tau)]
        lower_fraction = max(0.0, float(item.anchor_base_lower_fraction) - radius)
        upper_fraction = min(1.0, float(item.anchor_base_upper_fraction) + radius)
        lower = max(0, min(int(item.hours), math.ceil(lower_fraction * int(item.hours) - 1e-12)))
        upper = max(0, min(int(item.hours), math.floor(upper_fraction * int(item.hours) + 1e-12)))
        if lower > upper:
            raise ValueError("empty interval after integer conversion")
        covered = lower <= int(item.factor_box_K_lower) and upper >= int(item.factor_box_K_upper)
        partial = v105.loc[v105.region.eq(item.region) & v105.month.astype(str).str[:10].eq(str(item.month)[:10])
                           & v105.tau.eq(item.tau)].iloc[0]
        row = {"region": item.region, "month": item.month, "tau": int(item.tau), "hours": int(item.hours),
               "radius_max_region_q90": radius, "anchor_base_lower_fraction": item.anchor_base_lower_fraction,
               "anchor_base_upper_fraction": item.anchor_base_upper_fraction,
               "anchor_robust_lower": lower, "anchor_robust_upper": upper,
               "anchor_robust_width": upper - lower,
               "factor_box_K_lower": int(item.factor_box_K_lower),
               "factor_box_K_upper": int(item.factor_box_K_upper), "factor_box_covered": bool(covered),
               "monthly_robust_lower": int(partial.monthly_lower),
               "monthly_robust_upper": int(partial.monthly_upper),
               "monthly_robust_width": int(partial.monthly_upper - partial.monthly_lower),
               "capacity_robust_lower": int(partial.capacity_robust_lower),
               "capacity_robust_upper": int(partial.capacity_robust_upper),
               "capacity_robust_width": int(partial.capacity_robust_width)}
        for rate in RATES:
            required = int(np.ceil(rate * int(item.hours)))
            for prefix, lo, hi in (("anchor", lower, upper),
                                   ("monthly", row["monthly_robust_lower"], row["monthly_robust_upper"]),
                                   ("capacity", row["capacity_robust_lower"], row["capacity_robust_upper"])):
                positive, negative = lo >= required, hi < required
                row[f"{prefix}_positive_{rate:g}"] = positive
                row[f"{prefix}_negative_{rate:g}"] = negative
                row[f"{prefix}_abstain_{rate:g}"] = not (positive or negative)
        rows.append(row)
    detail = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=False)
    detail.to_csv(OUT / "target_rows.csv", index=False)
    pd.DataFrame(region_quantiles).to_csv(OUT / "source_region_q90.csv", index=False)
    metrics = ["anchor_robust_width", "monthly_robust_width", "capacity_robust_width", "factor_box_covered"]
    per_region = detail.groupby(["region", "tau"])[metrics].mean().reset_index()
    for rate in RATES:
        for prefix in ("anchor", "monthly", "capacity"):
            for state in ("positive", "negative", "abstain"):
                col = f"{prefix}_{state}_{rate:g}"
                per_region[col] = detail.groupby(["region", "tau"])[col].mean().to_numpy()
    per_region.to_csv(OUT / "per_region_metrics.csv", index=False)
    macro = per_region.groupby("tau").mean(numeric_only=True).reset_index()
    macro.to_csv(OUT / "region_equal_metrics.csv", index=False)
    summary = {"status": "WORST_SOURCE_REGION_Q90_FACTOR_ROBUST_ANCHOR",
               "target_regions": sorted(detail.region.unique().tolist()),
               "calibration_regions": sorted(calibration.region.unique().tolist()),
               "target_region_months": int(detail[["region", "month"]].drop_duplicates().shape[0]),
               "max_region_q90_radius_by_tau": radii,
               "region_equal_metrics": macro.to_dict("records"),
               "target_factor_box_coverage": bool(detail.factor_box_covered.all()),
               "target_factor_box_covered_rows": int(detail.factor_box_covered.sum()),
               "target_rows": len(detail), "v111_calibration_sha256": sha(V111 / "source_calibration_rows.csv"),
               "v111_target_sha256": sha(V111 / "target_rows.csv"), "v105_rows_sha256": sha(V105_ROWS),
               "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
               "limitations": ["Worst-source-region empirical quantile is not a finite-sample guarantee for a new region.",
                               "Same retrospective EIA fixed-proxy regions and stress factor box as V111.",
                               "No actual China CIF labels or future-time confirmation."]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
