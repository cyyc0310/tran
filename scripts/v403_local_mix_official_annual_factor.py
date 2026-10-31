"""Rebuild V403's post-hoc local-mix annual-factor diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import v327_nbs_mix_transfer as v327


ROOT = Path(__file__).resolve().parents[1]
FACTORS = ROOT / "data/china/mee_provincial_grid_co2_factors_2021_2023.csv"
MONTHLY = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
V327_PREDICTIONS = ROOT / "results/v327-nbs-mix-transfer-r1/province_predictions.csv"
DEFAULT_OUT = ROOT / "results/v403-local-mix-official-ef-threshold-audit-r1-repro"


def annual_thermal_share(monthly: pd.DataFrame, provinces: list[str]) -> pd.DataFrame:
    """Sum monthly report buckets Feb-Dec; Feb is the Jan-Feb cumulative bucket."""
    data = monthly.loc[
        monthly["province"].isin(provinces)
        & monthly["year"].isin([2020, 2021, 2022])
        & monthly["technology"].isin(["thermal", "total"])
        & monthly["month"].between(2, 12)
    ]
    annual = data.pivot_table(
        index=["province", "year"], columns="technology", values="generation_gwh", aggfunc="sum"
    ).reset_index()
    if annual[["thermal", "total"]].isna().any().any() or (annual["total"] <= 0).any():
        raise ValueError("Missing or invalid annual thermal/total generation")
    annual["thermal_share"] = annual["thermal"] / annual["total"]
    return annual


def binary_scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    truth = actual.astype(bool)
    guess = predicted.astype(bool)
    tp = int(np.sum(truth & guess))
    tn = int(np.sum(~truth & ~guess))
    fp = int(np.sum(~truth & guess))
    fn = int(np.sum(truth & ~guess))
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "n": int(len(truth)), "accuracy": float((tp + tn) / len(truth)),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "sensitivity": float(sensitivity), "specificity": float(specificity),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def continuous_scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    error = predicted - actual
    return {
        "n": int(len(error)), "mae_gco2_kwh": float(np.mean(np.abs(error))),
        "rmse_gco2_kwh": float(np.sqrt(np.mean(error**2))),
        "mean_signed_error_gco2_kwh": float(np.mean(error)),
        "p90_absolute_error_gco2_kwh": float(np.quantile(np.abs(error), 0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {args.out}")

    factors = pd.read_csv(FACTORS)
    provinces = sorted(factors["province"].astype(str))
    monthly = pd.read_csv(MONTHLY)
    mix = annual_thermal_share(monthly, provinces)
    targets = factors.melt(id_vars="province", var_name="target_year", value_name="actual_ef_kgco2_kwh")
    targets["target_year"] = targets["target_year"].str.removeprefix("ef_").str.removesuffix("_kgco2_kwh").astype(int)
    targets = targets.loc[targets["target_year"].between(2021, 2023)].copy()
    targets["feature_year"] = targets["target_year"] - 1
    pred = targets.merge(mix, left_on=["province", "feature_year"], right_on=["province", "year"], validate="one_to_one")
    pred = pred.drop(columns="year")
    pred["actual_ef_gco2_kwh"] = pred["actual_ef_kgco2_kwh"] * 1000
    pred["midpoint_ef_700share_gco2_kwh"] = 700 * pred["thermal_share"]
    pred["support_low_400share_gco2_kwh"] = 400 * pred["thermal_share"]
    pred["support_high_1000share_gco2_kwh"] = 1000 * pred["thermal_share"]
    threshold = 300.0
    pred["actual_above_300"] = pred["actual_ef_gco2_kwh"] > threshold
    pred["midpoint_predict_above_300"] = pred["midpoint_ef_700share_gco2_kwh"] > threshold
    pred["support_certificate_300"] = np.select(
        [pred["support_low_400share_gco2_kwh"] > threshold,
         pred["support_high_1000share_gco2_kwh"] <= threshold],
        ["above", "at_or_below"], default="ambiguous",
    )
    pred["support_contains_actual_factor"] = pred["actual_ef_gco2_kwh"].between(
        pred["support_low_400share_gco2_kwh"], pred["support_high_1000share_gco2_kwh"]
    )
    # The national comparator uses all NBS provincial rows, including Tibet,
    # even though the MEE target table has only 30 provincial labels.
    national_mix = annual_thermal_share(monthly, sorted(monthly["province"].dropna().unique()))
    national = national_mix.groupby("year")[["thermal", "total"]].sum()
    pred["national_share"] = pred["feature_year"].map((national["thermal"] / national["total"]).to_dict())
    pred["national_midpoint_700share_gco2_kwh"] = 700 * pred["national_share"]
    pred["national_predict_above_300"] = pred["national_midpoint_700share_gco2_kwh"] > threshold

    v327 = pd.read_csv(V327_PREDICTIONS)
    v327 = v327.rename(columns={"nbs_mix_pred": "nbs_mix_pred_kgco2_kwh"})
    pred = pred.merge(v327[["province", "target_year", "nbs_mix_pred_kgco2_kwh"]], on=["province", "target_year"], how="left", validate="one_to_one")

    panel_summary: dict[str, dict] = {}
    for year, subset in pred.groupby("target_year"):
        actual = subset["actual_ef_gco2_kwh"].to_numpy(float)
        local = subset["midpoint_ef_700share_gco2_kwh"].to_numpy(float)
        nat = subset["national_midpoint_700share_gco2_kwh"].to_numpy(float)
        decisive = subset["support_certificate_300"].ne("ambiguous")
        decisions = subset.loc[decisive]
        panel_summary[str(year)] = {
            "n_provinces": int(len(subset)),
            "local_midpoint_regression": continuous_scores(actual, local),
            "national_midpoint_regression": continuous_scores(actual, nat),
            "local_midpoint_threshold": binary_scores(subset["actual_above_300"].to_numpy(bool), subset["midpoint_predict_above_300"].to_numpy(bool)),
            "national_midpoint_threshold": binary_scores(subset["actual_above_300"].to_numpy(bool), subset["national_predict_above_300"].to_numpy(bool)),
            "support_factor_containment": float(subset["support_contains_actual_factor"].mean()),
            "support_mean_width_gco2_kwh": float((subset["support_high_1000share_gco2_kwh"] - subset["support_low_400share_gco2_kwh"]).mean()),
            "decisive_coverage": float(decisive.mean()), "decisive_n": int(decisive.sum()),
            "decisive_accuracy": float((decisions["actual_above_300"] == decisions["support_certificate_300"].eq("above")).mean()),
            "above_calls": int(decisions["support_certificate_300"].eq("above").sum()),
            "at_or_below_calls": int(decisions["support_certificate_300"].eq("at_or_below").sum()),
            "false_decisions": int((decisions["actual_above_300"] != decisions["support_certificate_300"].eq("above")).sum()),
        }
    all_actual = pred["actual_ef_gco2_kwh"].to_numpy(float)
    all_local = pred["midpoint_ef_700share_gco2_kwh"].to_numpy(float)
    all_nat = pred["national_midpoint_700share_gco2_kwh"].to_numpy(float)
    all_decisive = pred.loc[pred["support_certificate_300"].ne("ambiguous")]
    panel_summary["2021-2023"] = {
        "n_province_years": int(len(pred)),
        "local_midpoint_regression": continuous_scores(all_actual, all_local),
        "national_midpoint_regression": continuous_scores(all_actual, all_nat),
        "local_midpoint_threshold": binary_scores(pred["actual_above_300"].to_numpy(bool), pred["midpoint_predict_above_300"].to_numpy(bool)),
        "support_factor_containment": float(pred["support_contains_actual_factor"].mean()),
        "support_mean_width_gco2_kwh": float((pred["support_high_1000share_gco2_kwh"] - pred["support_low_400share_gco2_kwh"]).mean()),
        "decisive_coverage": float(len(all_decisive) / len(pred)), "decisive_n": int(len(all_decisive)),
        "decisive_errors": int((all_decisive["actual_above_300"] != all_decisive["support_certificate_300"].eq("above")).sum()),
    }
    matched = pred.loc[pred["target_year"].isin([2022, 2023])].dropna(subset="nbs_mix_pred_kgco2_kwh").copy()
    nnls_summary = continuous_scores(
        matched["actual_ef_gco2_kwh"].to_numpy(float),
        matched["nbs_mix_pred_kgco2_kwh"].to_numpy(float) * 1000,
    )
    local_error = np.abs(matched["midpoint_ef_700share_gco2_kwh"] - matched["actual_ef_gco2_kwh"])
    nnls_error = np.abs(matched["nbs_mix_pred_kgco2_kwh"] * 1000 - matched["actual_ef_gco2_kwh"])
    delta = (local_error - nnls_error).groupby(matched["province"]).mean().to_numpy(float)
    rng = np.random.default_rng(403)
    boot = np.array([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(50000)])
    summary = {
        "experiment": "posthoc; local lagged-mix proxy and threshold support vs official annual consumption-side CO2 factor",
        "target": "2021-2023 MEE/NBS provincial average electricity CO2 factors; 30 provinces/year; no hourly labels",
        "feature": "NBS-derived above-scale annual thermal generation share from previous calendar year; February Jan-Feb cumulative bucket counted once",
        "assumptions": "operational thermal EF support [400,1000] gCO2/kWh; nonthermal operational EF=0; midpoint=700",
        "threshold_gco2_kwh": threshold,
        "panels": panel_summary,
        "matched_V327_forward_years": {
            "n_province_years": int(len(matched)),
            "local_midpoint_regression": continuous_scores(
                matched["actual_ef_gco2_kwh"].to_numpy(float),
                matched["midpoint_ef_700share_gco2_kwh"].to_numpy(float),
            ),
            "nbs_nnls_regression": nnls_summary,
        },
        "paired_local_midpoint_minus_NBS_NNLS_abs_error_2022_2023": {
            "mean_difference_gco2_kwh": float(delta.mean()),
            "province_cluster_bootstrap_95pct_gco2_kwh": np.quantile(boot, [0.025, 0.975]).tolist(),
            "provinces_local_better": int((delta < 0).sum()), "provinces_tied": int((np.abs(delta) <= 1e-12).sum()),
        },
    }
    args.out.mkdir(parents=True)
    pred.to_csv(args.out / "province_year_predictions.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
