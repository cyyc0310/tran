"""Descriptive input-support audit for the frozen V140 source folds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v139_physical_reserve_calibration import ROOT, sha

RUN = ROOT / "results/v140-monthly-conditional-risk-r1"
OUT = ROOT / "results/v140-source-support-audit-r1"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    panel_path, roles_path, rows_path = (RUN / name for name in
                                        ("monthly_feature_panel.csv", "region_roles.csv", "outer_fold_rows.csv"))
    panel = pd.read_csv(panel_path).query("tau == 900").copy()
    roles, rows = pd.read_csv(roles_path), pd.read_csv(rows_path)
    fuels = [c for c in panel if c.startswith("fuelshare_")]
    support = []
    for (target_region, seed), roster in roles.groupby(["target_region", "seed"]):
        fit_regions = roster.loc[roster.role.eq("fit"), "region"]
        fit, target = panel.loc[panel.region.isin(fit_regions)], panel.loc[panel.region.eq(target_region)]
        all_sources = panel.loc[panel.region.ne(target_region)]
        for row in target.itertuples(index=False):
            x = np.array([getattr(row, c) for c in fuels], dtype=float)
            distance = np.abs(fit[fuels].to_numpy(float) - x).sum(axis=1)
            nearest = fit.iloc[int(distance.argmin())]
            support.append({"region": target_region, "seed": seed, "month": row.month,
                            "coal_share": row.fuelshare_coal,
                            "fit_coal_max": float(fit.fuelshare_coal.max()),
                            "all_source_coal_max": float(all_sources.fuelshare_coal.max()),
                            "above_fit_coal_support": row.fuelshare_coal > fit.fuelshare_coal.max() + 1e-12,
                            "above_all_source_coal_support": row.fuelshare_coal > all_sources.fuelshare_coal.max() + 1e-12,
                            "nearest_fit_share_L1": float(distance.min()),
                            "nearest_fit_region": nearest.region, "nearest_fit_month": nearest.month})
    support = pd.DataFrame(support)
    paired = rows.loc[rows.alpha.eq(.1)].merge(support, on=["region", "seed", "month"], validate="many_to_one")
    summaries = []
    for (budget, seed, tau, method), group in paired.groupby(["budget", "seed", "tau", "method"]):
        for stratum, subset in (("all", group),
                                ("above_fit_coal_support", group.loc[group.above_fit_coal_support]),
                                ("above_all_source_coal_support", group.loc[group.above_all_source_coal_support])):
            summaries.append({"budget": budget, "seed": seed, "tau": tau, "method": method,
                              "stratum": stratum, "rows": len(subset),
                              "pooled_coverage": float(subset.covered.mean()) if len(subset) else None,
                              "macro_coverage": (float(subset.groupby("region").covered.mean().mean())
                                                 if len(subset) else None)})
    region = panel.assign(tail_upper_fraction=panel.factor_box_K_upper / panel.hours).groupby("region").agg(
        coal_mean=("fuelshare_coal", "mean"), coal_min=("fuelshare_coal", "min"),
        coal_max=("fuelshare_coal", "max"), max_tail_upper_fraction=("tail_upper_fraction", "max"))
    OUT.mkdir(parents=True)
    support.to_csv(OUT / "fold_month_input_support.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "coverage_by_support.csv", index=False)
    region.to_csv(OUT / "region_coal_and_tail_inventory.csv")
    sc = support.loc[support.region.eq("SC")]
    payload = {"status": "POSTHOC_SOURCE_SUPPORT_DIAGNOSIS_NOT_CAUSAL", "support_rows": len(support),
               "SC_months_above_all_other_source_coal_support_per_seed": int(sc.loc[sc.seed.eq(41)].above_all_source_coal_support.sum()),
               "SC_months_per_seed": int(sc.loc[sc.seed.eq(41)].shape[0]),
               "input_hashes": {str(p.relative_to(ROOT)): sha(p) for p in (panel_path, roles_path, rows_path)},
               "script_sha256": sha(Path(__file__)),
               "limitation": "One observed coordinate can certify extrapolation but cannot prove its causal role or diagnose all hidden dispatch shifts."}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(region.sort_values("coal_mean", ascending=False).head(9).to_string())
    print(f"SC months above all other source coal support: {payload['SC_months_above_all_other_source_coal_support_per_seed']}/24")


if __name__ == "__main__":
    main()
