"""V407: past-only temporal check of region-selected physical-anchor shrinkage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

import v327_nbs_mix_transfer as v327
import v329_china_region_block_coldstart as v329
from v404_nested_region_conformal_mix_intervals import (
    FACTORS, MONTHLY, ROOT, conformal_quantile, interval_score,
)
from v406_nested_physical_anchor_shrinkage import region_bootstrap


DEFAULT_OUT = ROOT / "results/v407_temporal_forward_shrinkage-r1"
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
NOMINALS = (0.90, 0.95)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_rows(provinces: list[str], assignment: dict[str, str], mix: dict,
                  factors: pd.DataFrame, excluded: set[str]) -> pd.DataFrame:
    """Use only 2021 labels and their legally lagged 2020 NBS mix."""
    rows = []
    for province in provinces:
        if assignment[province] in excluded:
            continue
        values = mix[(province, 2020)]
        row = {"province": province,
               "actual": float(factors.loc[province, "ef_2021_kgco2_kwh"]),
               "thermal_share": values["thermal"]}
        row.update({f"nbs_{group}": values[group] for group in v327.GROUPS})
        rows.append(row)
    return pd.DataFrame(rows)


def predict_region(region: str, train: pd.DataFrame, year: int, mix: dict,
                   factors: pd.DataFrame) -> pd.DataFrame:
    coef = nnls(train[[f"nbs_{group}" for group in v327.GROUPS].__iter__()].to_numpy(float), train["actual"].to_numpy(float))[0]
    feature_year = year - 1
    rows = []
    for province in v329.REGIONS[region]:
        values = mix[(province, feature_year)]
        x = np.array([values[group] for group in v327.GROUPS], dtype=float)
        rows.append({
            "region": region, "province": province,
            "actual": float(factors.loc[province, f"ef_{year}_kgco2_kwh"]),
            "thermal_share": values["thermal"],
            "local_center": 0.7 * values["thermal"],
            "nnls_center": float(x @ coef),
        })
    return pd.DataFrame(rows)


def select_lambda(excluded_regions: set[str], provinces: list[str], assignment: dict[str, str],
                  mix: dict, factors: pd.DataFrame) -> tuple[float, dict[float, float]]:
    cv_parts = []
    for validation_region in v329.REGIONS:
        if validation_region in excluded_regions:
            continue
        train = training_rows(provinces, assignment, mix, factors,
                              excluded_regions | {validation_region})
        cv_parts.append(predict_region(validation_region, train, 2021, mix, factors))
    oof = pd.concat(cv_parts, ignore_index=True)
    scores = {}
    for lam in LAMBDAS:
        center = lam * oof["local_center"] + (1 - lam) * oof["nnls_center"]
        per_region = (center - oof["actual"]).abs().groupby(oof["region"]).mean()
        scores[lam] = float(per_region.mean())
    choice = min(LAMBDAS, key=lambda lam: (scores[lam], lam != 1.0))
    return choice, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    out = parser.parse_args().out
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {out}")

    factors = pd.read_csv(FACTORS).set_index("province")
    provinces = sorted(factors.index.astype(str))
    mix, _ = v327.nbs_annual_mix(MONTHLY, provinces)
    assignment = {p: r for r, members in v329.REGIONS.items() for p in members}
    if set(assignment) != set(provinces):
        raise ValueError("Frozen region assignment does not partition factor provinces")

    predictions, selection_rows = [], []
    for outer_region, test_members in v329.REGIONS.items():
        selected_lambda, outer_cv_scores = select_lambda(
            {outer_region}, provinces, assignment, mix, factors
        )
        selection_rows.append({
            "held_region": outer_region, "selected_lambda": selected_lambda,
            **{f"2021_inner_macro_mae_lambda{lam:g}_kgco2_kwh": score for lam, score in outer_cv_scores.items()},
        })

        # Each calibration region has its own nested lambda selected without its labels.
        cal_parts = []
        cal_lambda_rows = []
        for cal_region in v329.REGIONS:
            if cal_region == outer_region:
                continue
            cal_lambda, _ = select_lambda(
                {outer_region, cal_region}, provinces, assignment, mix, factors
            )
            train = training_rows(provinces, assignment, mix, factors,
                                  {outer_region, cal_region})
            cal = predict_region(cal_region, train, 2021, mix, factors)
            cal["lambda"] = cal_lambda
            cal["selected_center"] = cal_lambda * cal["local_center"] + (1 - cal_lambda) * cal["nnls_center"]
            cal_parts.append(cal)
            cal_lambda_rows.append({"cal_region": cal_region, "lambda": cal_lambda})
        calibration = pd.concat(cal_parts, ignore_index=True)

        final_train = training_rows(provinces, assignment, mix, factors, {outer_region})
        test = predict_region(outer_region, final_train, 2022, mix, factors)
        test["selected_center"] = selected_lambda * test["local_center"] + (1 - selected_lambda) * test["nnls_center"]

        for nominal in NOMINALS:
            calibration_centers = {
                "local_anchor": calibration["local_center"],
                "nnls": calibration["nnls_center"],
                "selected_shrinkage": calibration["selected_center"],
                "fixed_lambda_0.75": 0.75 * calibration["local_center"] + 0.25 * calibration["nnls_center"],
            }
            test_centers = {
                "local_anchor": test["local_center"],
                "nnls": test["nnls_center"],
                "selected_shrinkage": test["selected_center"],
                "fixed_lambda_0.75": 0.75 * test["local_center"] + 0.25 * test["nnls_center"],
            }
            for method, cal_center in calibration_centers.items():
                scores = np.abs(calibration["actual"].to_numpy(float) - cal_center.to_numpy(float))
                radius = conformal_quantile(scores, nominal)
                for _, row in test.iterrows():
                    center = float(test_centers[method].loc[row.name])
                    actual = float(row["actual"])
                    lower, upper = center - radius, center + radius
                    score = interval_score(actual, lower, upper, nominal)
                    predictions.append({
                        "held_region": outer_region, "province": row["province"],
                        "method": method, "nominal_coverage": nominal,
                        "selected_lambda": selected_lambda,
                        "actual_kgco2_kwh": actual, "center_kgco2_kwh": center,
                        "radius_kgco2_kwh": radius, "lower_kgco2_kwh": lower,
                        "upper_kgco2_kwh": upper, "covered": bool(lower <= actual <= upper),
                        "width_gco2_kwh": (upper - lower) * 1000,
                        "interval_score_gco2_kwh": score,
                    })

    pred = pd.DataFrame(predictions)
    region_metrics = pred.groupby(["held_region", "method", "nominal_coverage"]).agg(
        n=("province", "size"), coverage=("covered", "mean"),
        mean_width_gco2_kwh=("width_gco2_kwh", "mean"),
        mean_interval_score_gco2_kwh=("interval_score_gco2_kwh", "mean"),
    ).reset_index()
    pooled = pred.groupby(["method", "nominal_coverage"]).agg(
        n=("province", "size"), coverage=("covered", "mean"),
        mean_width_gco2_kwh=("width_gco2_kwh", "mean"),
        mean_interval_score_gco2_kwh=("interval_score_gco2_kwh", "mean"),
    ).reset_index()
    paired = {}
    for nominal in NOMINALS:
        sub = pred.loc[pred.nominal_coverage.eq(nominal)]
        wide = sub.pivot(index=["held_region", "province"], columns="method", values="interval_score_gco2_kwh").reset_index()
        for baseline in ("local_anchor", "nnls", "fixed_lambda_0.75"):
            delta = wide.assign(delta=wide["selected_shrinkage"] - wide[baseline]).groupby("held_region")["delta"].mean()
            paired[f"selected_minus_{baseline}_interval_score_nom{nominal}"] = region_bootstrap(
                delta, 407 + int(nominal * 100) + len(paired)
            )

    selections = pd.DataFrame(selection_rows)
    summary = {
        "experiment": "past-only region-held-out shrinkage transfer; 2021 labels tune/calibrate, 2022 labels test",
        "target": "2022 MEE/NBS provincial annual consumption-side CO2 factor; 30 provinces; 7 statistical regions",
        "feature_availability": "2020 NBS mix for 2021 source labels; 2021 NBS mix for 2022 target; February Jan-Feb cumulative bucket counted once",
        "pooled_test_metrics": pooled.to_dict("records"),
        "paired_interval_region_bootstrap": paired,
        "outer_region_lambda": selections.set_index("held_region")["selected_lambda"].to_dict(),
        "lambda_selection_counts": selections["selected_lambda"].value_counts().sort_index().to_dict(),
        "limitations": [
            "This is a temporal-forward diagnostic on a previously explored target table, not a blinded external confirmation.",
            "Only one test year (2022) and seven correlated region clusters are available.",
            "Target is an annual consumption-side MEE factor, not hourly CIF.",
            "NBS historical feature vintages are retrospective and publication-time revisions are not archived.",
            "Empirical residual quantiles and region bootstrap are descriptive; dependence precludes a general finite-sample guarantee.",
        ],
        "sha256": {
            str(FACTORS.relative_to(ROOT)): sha256(FACTORS),
            str(MONTHLY.relative_to(ROOT)): sha256(MONTHLY),
            "scripts/v327_nbs_mix_transfer.py": sha256(ROOT / "scripts/v327_nbs_mix_transfer.py"),
            "scripts/v329_china_region_block_coldstart.py": sha256(ROOT / "scripts/v329_china_region_block_coldstart.py"),
            "scripts/v407_temporal_forward_shrinkage_confirmation.py": sha256(Path(__file__).resolve()),
        },
    }
    out.mkdir(parents=True)
    pred.to_csv(out / "outer_predictions.csv", index=False)
    region_metrics.to_csv(out / "region_metrics.csv", index=False)
    selections.to_csv(out / "lambda_selection_by_outer_region.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"outer_region_lambda": summary["outer_region_lambda"],
                      "pooled_test_metrics": summary["pooled_test_metrics"],
                      "paired_interval_region_bootstrap": summary["paired_interval_region_bootstrap"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
