#!/usr/bin/env python3
"""Post-hoc upper-bound decomposition of CN5 level and hourly-shape error.

This is a diagnostic over already sealed 2024 predictions. Target residual means
are used only as oracle corrections to estimate how much error could be removed
by perfect level information. They are not deployable corrections.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_energy_charts_nine_country_cn5_loco as nine  # noqa: E402
import run_cn5_fuel_resolved_response as response  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402

PREDICTIONS = ROOT / "results/cn5-fuel-factor-head-blend-r1"
OUTER_COUNTRIES = nine.COUNTRIES
METHODS = ("factor_history", "fuelshare_history", "ratio_recomposition", "blend")


def truth_for(raw: pd.DataFrame, part: pd.DataFrame) -> np.ndarray:
    fuel = response.fuel_for(raw, part)
    return fuel @ model.FACTORS / fuel.sum(axis=1)


def scores(y: np.ndarray, p: np.ndarray, radius: float) -> dict[str, float]:
    error = y - p
    lower, upper = p - radius, p + radius
    miss = np.maximum(lower - y, 0) + np.maximum(y - upper, 0)
    winkler = (upper - lower) + 20.0 * miss
    return {
        "mae": float(np.mean(np.abs(error))),
        "bias_truth_minus_prediction": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "coverage90": float(np.mean((y >= lower) & (y <= upper))),
        "width90": float(2 * radius),
        "winkler90": float(np.mean(winkler)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    rows = []
    for country in OUTER_COUNTRIES:
        raw, features, _ = nine.prepare_country(country)
        query = features.loc[features.year.eq(2024)].copy()
        target = raw.iloc[query.raw_row.to_numpy()].copy()
        truth = truth_for(raw, query)
        pred_path = PREDICTIONS / f"{country.upper()}_outer_2024_predictions.csv.gz"
        pred = pd.read_csv(pred_path)
        if len(pred) != len(query):
            raise ValueError(f"prediction length mismatch for {country}: {len(pred)} vs {len(query)}")
        pred_time = pd.to_datetime(pred.time, utc=True)
        target_time = pd.to_datetime(target.time, utc=True)
        if not np.array_equal(pred_time.to_numpy(), target_time.to_numpy()):
            raise ValueError(f"prediction timestamps do not align for {country}")
        calibration = json.loads((PREDICTIONS / f"{country.upper()}_selection_calibration.json")
                                .read_text())
        radii = calibration["radii_h2"]
        month = pd.to_datetime(target.time, utc=True).dt.month.to_numpy()

        for method in METHODS:
            point = pred[method].to_numpy(dtype=float)
            radius = float(radii[method])
            correction = truth - point
            annual_oracle = float(correction.mean())
            corrected = point + annual_oracle
            raw_scores = scores(truth, point, radius)
            annual_scores = scores(truth, corrected, radius)
            mse = float(np.mean(correction**2))
            centered_mse = float(np.mean((correction - annual_oracle)**2))
            raw_scores.update({
                "scope": "country_year",
                "country": country.upper(),
                "method": method,
                "oracle_correction": annual_oracle,
                "oracle_centered_mae": annual_scores["mae"],
                "oracle_centered_coverage90": annual_scores["coverage90"],
                "oracle_centered_winkler90": annual_scores["winkler90"],
                "fraction_mse_attributable_to_global_bias": (
                    (mse - centered_mse) / mse if mse > 0 else 0.0),
                "radius_source_h2": radius,
            })
            rows.append(raw_scores)

            for m in np.unique(month):
                mask = month == m
                local = correction[mask]
                local_correction = float(local.mean())
                monthly_scores = scores(truth[mask], point[mask], radius)
                monthly_oracle = scores(truth[mask], point[mask] + local_correction, radius)
                month_mse = float(np.mean(local**2))
                month_centered_mse = float(np.mean((local - local_correction)**2))
                monthly_scores.update({
                    "scope": "country_month",
                    "country": country.upper(),
                    "method": method,
                    "month": int(m),
                    "oracle_correction": local_correction,
                    "oracle_centered_mae": monthly_oracle["mae"],
                    "oracle_centered_coverage90": monthly_oracle["coverage90"],
                    "oracle_centered_winkler90": monthly_oracle["winkler90"],
                    "fraction_mse_attributable_to_monthly_bias": (
                        (month_mse - month_centered_mse) / month_mse if month_mse > 0 else 0.0),
                    "radius_source_h2": radius,
                })
                rows.append(monthly_scores)

    result = pd.DataFrame(rows)
    result.to_csv(out / "level_shape_oracle_decomposition.csv", index=False)
    summary = {}
    for scope in ("country_year", "country_month"):
        subset = result[result.scope.eq(scope)]
        summary[scope] = {}
        for method, part in subset.groupby("method"):
            summary[scope][method] = {
                "n_units": int(len(part)),
                "macro_mae": float(part.mae.mean()),
                "macro_oracle_centered_mae": float(part.oracle_centered_mae.mean()),
                "macro_coverage90": float(part.coverage90.mean()),
                "macro_oracle_centered_coverage90": float(part.oracle_centered_coverage90.mean()),
                "macro_winkler90": float(part.winkler90.mean()),
                "macro_oracle_centered_winkler90": float(part.oracle_centered_winkler90.mean()),
                "mean_bias_mse_fraction": float(part[
                    "fraction_mse_attributable_to_global_bias" if scope == "country_year"
                    else "fraction_mse_attributable_to_monthly_bias"].mean()),
                "median_absolute_oracle_correction": float(part.oracle_correction.abs().median()),
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
