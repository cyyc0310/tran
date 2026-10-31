"""V354: diagnose mix-capacity interactions by EF-support position vs tau."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from v342_china_sharp_group_hull import ROOT
from v348_ember_share_error_frontier import FUEL_NAMES, robust_ef_bounds

V341 = ROOT / "results/v341-china-complete-category-hull-r2"
V352 = ROOT / "results/v352-threshold-factorial-sweep-r1/paired_threshold_factorial.csv"
EMBER = ROOT / "results/v294-china-ember-lineage-r1/ember_china_monthly_generation.csv"
PROTOCOL = ROOT / "docs/V354_THRESHOLD_SIDE_SUPPORT_DIAGNOSIS_PROTOCOL.md"
OUT = ROOT / "results/v354-threshold-side-support-diagnosis-r1"
SHARE_ERROR = 1.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    ember = pd.read_csv(EMBER)
    ember = ember.loc[
        ember.Province.notna() & ember.Province.ne("China (total)")
        & ember["Is aggregated source"].eq(False)
        & ember["Electricity source"].isin(
            ["Coal", "Gas", "Other fossil", "Bioenergy", "Hydro", "Nuclear", "Wind", "Solar"]
        )
    ].copy()
    ember["Province"] = ember.Province.replace({"Tibet (Xizang)": "Tibet"})
    ember["month"] = pd.to_datetime(ember.Date).dt.month
    ember["group"] = ember["Electricity source"].map(
        {"Coal": "coal", "Gas": "gas", "Other fossil": "other_fossil", "Bioenergy": "bioenergy",
         "Hydro": "hydro", "Nuclear": "nuclear", "Wind": "wind", "Solar": "solar"}
    )
    ember = ember.loc[pd.to_datetime(ember.Date).dt.year.eq(2023)]
    wide = ember.pivot(index=["Province", "month"], columns="group", values="Generation (TWh)").reset_index()
    columns = list(wide.columns.drop(["Province", "month"]))
    bounds = []
    for province in sorted(wide.Province.unique()):
        sub = wide.loc[wide.Province.eq(province)]
        for month in range(1, 13):
            period = "Jan-Feb" if month == 1 else f"{month:02d}"
            block = sub.loc[sub.month.isin([1, 2]), columns] if month == 1 else sub.loc[sub.month.eq(month), columns]
            row = block.sum() if month == 1 else block.iloc[0]
            fuels = [float(row[g]) for g in FUEL_NAMES]
            total = sum(fuels)
            if total <= 0:
                continue
            shares = [v / total for v in fuels]
            low, high = robust_ef_bounds(shares, SHARE_ERROR)
            bounds.append({"province": province, "period": period,
                           "thermal_ef_low": low, "thermal_ef_high": high,
                           "share_error_relative": SHARE_ERROR})
    support = pd.DataFrame(bounds)
    raw = pd.read_csv(V352)
    data = raw.merge(support, on=["province", "period"], validate="many_to_one")
    data["threshold_side"] = "crosses_or_touches"
    data.loc[data.thermal_ef_high.le(data.tau_g_kwh), "threshold_side"] = "all_thermal_ef_at_or_below_tau"
    data.loc[data.thermal_ef_low.gt(data.tau_g_kwh), "threshold_side"] = "all_thermal_ef_above_tau"

    below = data.threshold_side.eq("all_thermal_ef_at_or_below_tau")
    if not data.loc[below, ["mix_pm100_no_capacity_width_h", "mix_pm100_with_capacity_width_h"]].eq(0).all().all():
        raise ValueError("threshold-side implication failed: EF support below tau must imply zero mix K width")
    neg = data.complementarity_h.lt(0)
    if not data.loc[neg, "threshold_side"].eq("all_thermal_ef_at_or_below_tau").all():
        raise ValueError("negative complementarity found outside below-threshold EF support")

    summaries = []
    for (tau, side), frame in data.groupby(["tau_g_kwh", "threshold_side"]):
        summaries.append({"tau_g_kwh": int(tau), "threshold_side": side, "province_periods": len(frame),
                          "mean_thermal_ef_low": float(frame.thermal_ef_low.mean()),
                          "mean_thermal_ef_high": float(frame.thermal_ef_high.mean()),
                          "mean_mix_width_no_capacity_h": float(frame.mix_pm100_no_capacity_width_h.mean()),
                          "mean_mix_width_with_capacity_h": float(frame.mix_pm100_with_capacity_width_h.mean()),
                          "mean_complementarity_h": float(frame.complementarity_h.mean()),
                          "negative_complementarity_cells": int(frame.complementarity_h.lt(0).sum())})
    OUT.mkdir(parents=True)
    data.to_csv(OUT / "province_period_threshold_support.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "threshold_side_summary.csv", index=False)
    neg_rows = data.loc[neg, ["province", "period", "tau_g_kwh", "thermal_ef_low", "thermal_ef_high",
                              "mix_pm100_no_capacity_width_h", "mix_pm100_with_capacity_width_h",
                              "complementarity_h", "threshold_side"]]
    neg_rows.to_csv(OUT / "negative_complementarity_diagnosis.csv", index=False)
    summary = {"status": "COMPLETED_MECHANISM_DIAGNOSIS_WITHIN_DECLARED_EF_SUPPORT",
               "rows": len(data), "negative_interaction_rows": len(neg_rows),
               "negative_rows_all_below_threshold_support": bool(neg_rows.threshold_side.eq("all_thermal_ef_at_or_below_tau").all()),
               "all_below_support_mix_widths_zero": bool(data.loc[below, ["mix_pm100_no_capacity_width_h", "mix_pm100_with_capacity_width_h"]].eq(0).all().all()),
               "frontier": summaries,
               "limits": ["EF bounds derive from Ember source mix and analyst-defined ±100% share perturbation.",
                          "This is a conditional model-support diagnosis, not validation of true provincial EF or hourly CIF.",
                          "Nonthermal source direct operational factors are set to zero in the declared model."],
               "sha256": {str(p.relative_to(ROOT)): sha(p) for p in (V341 / "monthly_margin_capacity_audit.csv", V352, EMBER, PROTOCOL, Path(__file__))}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(pd.DataFrame(summaries).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"negative={len(neg_rows)}; all negative interaction cells are below-threshold support")


if __name__ == "__main__":
    main()
