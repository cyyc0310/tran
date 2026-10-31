#!/usr/bin/env python3
"""Cross-province audit of lagged composition threshold-state identification."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd


def crossing(share: pd.Series) -> pd.Series:
    return ((400 * share <= 300) & (300 < 1000 * share)).where(share.notna())


def metrics(frame: pd.DataFrame, prediction: str) -> dict[str, object]:
    valid = frame.dropna(subset=["target_crossing", prediction])
    y = valid["target_crossing"].astype(int)
    pred = valid[prediction].astype(int)
    positives, negatives = int(y.sum()), int((1 - y).sum())
    sensitivity = float(((y == 1) & (pred == 1)).sum() / positives) if positives else None
    specificity = float(((y == 0) & (pred == 0)).sum() / negatives) if negatives else None
    return {
        "prediction": prediction,
        "n": int(len(valid)),
        "target_crossing_prevalence": float(y.mean()) if len(valid) else None,
        "agreement": float((y == pred).mean()) if len(valid) else None,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None,
        "errors": int((y != pred).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/v394-nbs-all-province-long-panel-r1"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/v394-all-province-lopo-r1"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    data = pd.read_csv(args.input_dir / "official_current_values.csv")
    baseline_path = Path("results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv")
    baseline = pd.read_csv(baseline_path)
    crosscheck = data.merge(baseline, on=["province", "year", "month", "technology"], how="left")
    crosscheck["official_value_gwh"] = crosscheck["value_100m_kwh"] * 100
    crosscheck["difference_gwh"] = crosscheck["official_value_gwh"] - crosscheck["generation_gwh"]
    crosscheck["within_half_10gwh_display_step"] = crosscheck["difference_gwh"].abs() <= 5 + 1e-9
    crosscheck["oreo_rounded_to_nbs_10gwh_step"] = crosscheck["generation_gwh"].map(
        lambda value: float((Decimal(str(value)) / Decimal("10")).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * 10)
        if pd.notna(value) else float("nan")
    )
    crosscheck["rounded_match"] = (
        crosscheck["oreo_rounded_to_nbs_10gwh_step"] - crosscheck["official_value_gwh"]
    ).abs() < 1e-8
    crosscheck.to_csv(args.output_dir / "official_vs_oreo_crosscheck.csv", index=False)
    v392 = pd.read_csv("results/v392-nbs-official-provincial-long-panel-r1/official_current_values.csv")
    v394_selected = data[data["province"].isin(v392["province"].unique())]
    selected_check = v392.merge(
        v394_selected, on=["province", "year", "month", "technology"], suffixes=("_v392", "_v394"), how="outer"
    )
    same = (selected_check["value_100m_kwh_v392"] == selected_check["value_100m_kwh_v394"]) | (
        selected_check["value_100m_kwh_v392"].isna() & selected_check["value_100m_kwh_v394"].isna()
    )
    selected_check["same_as_v392"] = same
    selected_check.to_csv(args.output_dir / "five_province_snapshot_recheck.csv", index=False)
    wide = data.pivot_table(
        index=["province", "year", "month"], columns="technology", values="value_100m_kwh", aggfunc="first"
    ).reset_index()
    wide["period"] = wide["year"] * 12 + wide["month"]
    wide["local_share"] = wide["thermal"] / wide["total"]
    wide["target_crossing"] = crossing(wide["local_share"])

    national = wide.groupby("period", as_index=False).agg(
        pooled_thermal=("thermal", "sum"), pooled_total=("total", "sum")
    )
    national["pooled_share"] = national["pooled_thermal"] / national["pooled_total"]
    national["pooled_crossing"] = crossing(national["pooled_share"])
    wide = wide.merge(national[["period", "pooled_share", "pooled_crossing"]], on="period", how="left")

    for lag in (1, 2):
        lagged_local = wide[["province", "period", "local_share", "target_crossing"]].copy()
        lagged_local["period"] += lag
        lagged_local = lagged_local.rename(columns={
            "local_share": f"local_share_lag{lag}",
            "target_crossing": f"local_crossing_lag{lag}",
        })
        wide = wide.merge(lagged_local, on=["province", "period"], how="left")

        # Exclude the test province from the pooled donor mix at each source month.
        donors = wide[["province", "period", "thermal", "total"]].copy()
        donors["donor_thermal"] = donors.groupby("period")["thermal"].transform("sum") - donors["thermal"].fillna(0)
        donors["donor_total"] = donors.groupby("period")["total"].transform("sum") - donors["total"].fillna(0)
        donors[f"lopo_share_lag{lag}"] = donors["donor_thermal"] / donors["donor_total"]
        donors[f"lopo_crossing_lag{lag}"] = crossing(donors[f"lopo_share_lag{lag}"])
        donors["period"] += lag
        wide = wide.merge(
            donors[["province", "period", f"lopo_share_lag{lag}", f"lopo_crossing_lag{lag}"]],
            on=["province", "period"], how="left"
        )

        pooled_lag = national[["period", "pooled_share", "pooled_crossing"]].copy()
        pooled_lag["period"] += lag
        pooled_lag = pooled_lag.rename(columns={
            "pooled_share": f"pooled_share_lag{lag}",
            "pooled_crossing": f"pooled_crossing_lag{lag}",
        })
        wide = wide.merge(pooled_lag, on="period", how="left")

    # A static province classification is fitted only on target labels through 2022.
    train = wide[(wide["year"] <= 2022) & wide["target_crossing"].notna()]
    province_modes = train.groupby("province")["target_crossing"].agg(lambda values: int(values.mean() >= 0.5))
    wide["province_static_train_through_2022"] = wide["province"].map(province_modes)
    wide["pooled_static_train_through_2022"] = int(train["target_crossing"].mean() >= 0.5)

    # Remove pivot-only rows with no target mix and preserve all unavailable lag values as NA.
    wide["lopo_crossing_lag2"] = wide["lopo_crossing_lag2"].where(wide["target_crossing"].notna())
    wide.to_csv(args.output_dir / "all_province_lopo_panel.csv", index=False)

    prediction_columns = [
        "local_crossing_lag2",
        "lopo_crossing_lag2",
        "pooled_crossing_lag2",
        "province_static_train_through_2022",
        "pooled_static_train_through_2022",
    ]
    dynamic_predictions = prediction_columns[:3]
    common = wide.dropna(subset=["target_crossing", *dynamic_predictions]).copy()
    pd.DataFrame([metrics(common, pred) for pred in dynamic_predictions]).to_csv(
        args.output_dir / "common_sample_metrics.csv", index=False
    )
    by_province = []
    for province, rows in common.groupby("province"):
        for pred in dynamic_predictions:
            item = metrics(rows, pred)
            item["province"] = province
            by_province.append(item)
    pd.DataFrame(by_province).to_csv(args.output_dir / "metrics_by_province.csv", index=False)

    years = []
    for year, rows in common.groupby("year"):
        for pred in dynamic_predictions:
            item = metrics(rows, pred)
            item["year"] = int(year)
            years.append(item)
    pd.DataFrame(years).to_csv(args.output_dir / "metrics_by_year.csv", index=False)

    forward = wide[wide["year"] >= 2023].dropna(subset=["target_crossing", *prediction_columns]).copy()
    pd.DataFrame([metrics(forward, pred) for pred in prediction_columns]).to_csv(
        args.output_dir / "forward_holdout_metrics.csv", index=False
    )
    forward.to_csv(args.output_dir / "forward_holdout_panel.csv", index=False)

    summary = {
        "source": str(args.input_dir / "official_current_values.csv"),
        "provinces": int(data["province"].nunique()),
        "expected_provinces": 31,
        "source_months": [str(data["period_code"].min()), str(data["period_code"].max())],
        "target": "V392 binary mean-support cross-threshold state: 400*s <= 300 < 1000*s",
        "feature_timing": "t-2 monthly generation mix is used as the last potentially available month-start NBS composition",
        "lopo_definition": "For each target province and source month, compute thermal share from the other 30 provinces only, then classify the held-out target state.",
        "historical_vintage_limit": "Current historical API values are not archived as-published vintages; release-lag convention is timing-informed but revisions are unobserved.",
        "common_sample_n": int(len(common)),
        "official_vs_oreo": {
            "compared_nonmissing_pairs": int(crosscheck.dropna(subset=["official_value_gwh", "generation_gwh"]).shape[0]),
            "exact_value_pairs": int((crosscheck["difference_gwh"].abs() < 1e-8).sum()),
            "within_half_display_step_pairs": int(crosscheck["within_half_10gwh_display_step"].sum()),
            "rounded_to_nbs_display_step_pairs": int(crosscheck["rounded_match"].sum()),
            "max_abs_difference_gwh": float(crosscheck["difference_gwh"].abs().max()),
        },
        "v392_five_province_recheck": {
            "rows": int(len(selected_check)),
            "identical_including_missing": int(selected_check["same_as_v392"].sum()),
        },
        "common_sample_metrics": [metrics(common, pred) for pred in dynamic_predictions],
        "forward_holdout": {
            "training_cutoff_year": 2022,
            "test_start_year": 2023,
            "test_end_month": str(data["period_code"].max()),
            "common_sample_n": int(len(forward)),
            "metrics": [metrics(forward, pred) for pred in prediction_columns],
        },
        "caveat": "This evaluates a coarse state computed from the same generation series, not hourly CIF accuracy, probabilistic coverage, or interval efficiency.",
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
