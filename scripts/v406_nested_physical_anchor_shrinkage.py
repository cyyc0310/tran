"""V406: select shrinkage from a learned mix model toward a local physical anchor."""

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


DEFAULT_OUT = ROOT / "results/v406_nested_physical_anchor_shrinkage-r1"
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
NOMINALS = (0.90, 0.95)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_mix(rows: pd.DataFrame) -> np.ndarray:
    return nnls(rows[[f"nbs_{g}" for g in v327.GROUPS]].to_numpy(float), rows["actual"].to_numpy(float))[0]


def train_rows(
    provinces: list[str], assignment: dict[str, str], mix: dict,
    factors: pd.DataFrame, excluded: set[str], years: tuple[int, ...] = (2021, 2022),
) -> pd.DataFrame:
    rows = []
    for target_year in years:
        feature_year = target_year - 1
        for province in provinces:
            if assignment[province] in excluded:
                continue
            row = {
                "province": province,
                "actual": float(factors.loc[province, f"ef_{target_year}_kgco2_kwh"]),
                "thermal_share": mix[(province, feature_year)]["thermal"],
            }
            row.update({f"nbs_{g}": mix[(province, feature_year)][g] for g in v327.GROUPS})
            rows.append(row)
    return pd.DataFrame(rows)


def predictions_for_region(
    region: str, train: pd.DataFrame, provinces: list[str], mix: dict,
    factors: pd.DataFrame, year: int,
) -> pd.DataFrame:
    beta = fit_mix(train)
    feature_year = year - 1
    rows = []
    for province in v329.REGIONS[region]:
        shares = mix[(province, feature_year)]
        x = np.array([shares[g] for g in v327.GROUPS], dtype=float)
        local = 0.7 * shares["thermal"]
        learned = float(x @ beta)
        rows.append({
            "held_region": region, "province": province,
            "actual": float(factors.loc[province, f"ef_{year}_kgco2_kwh"]),
            "thermal_share": shares["thermal"],
            "local_center": local, "nnls_center": learned,
        })
    return pd.DataFrame(rows)


def interval_metrics(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, nominal: float) -> tuple[np.ndarray, np.ndarray]:
    covered = (lower <= actual) & (actual <= upper)
    score = np.array([interval_score(y, lo, hi, nominal) for y, lo, hi in zip(actual, lower, upper)])
    return covered, score


def region_bootstrap(delta: pd.Series, seed: int) -> dict[str, object]:
    values = delta.sort_index().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(50000)])
    return {
        "macro_difference": float(values.mean()),
        "region_cluster_bootstrap_95pct": np.quantile(draws, [0.025, 0.975]).tolist(),
        "regions_selected_better": int((values < 0).sum()),
        "region_differences": {str(k): float(v) for k, v in delta.sort_index().items()},
    }


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
        raise ValueError("Frozen seven-region assignment does not partition factor provinces")

    output_rows, selection_rows = [], []
    for outer_region, outer_members in v329.REGIONS.items():
        # Select the mix weight using only 2022 region-held-out point errors.
        oof_parts = []
        for validation_region in v329.REGIONS:
            if validation_region == outer_region:
                continue
            train = train_rows(provinces, assignment, mix, factors, {outer_region, validation_region})
            oof_parts.append(predictions_for_region(validation_region, train, provinces, mix, factors, 2022))
        oof = pd.concat(oof_parts, ignore_index=True)
        candidate_scores = {}
        for lam in LAMBDAS:
            oof["blend"] = lam * oof["local_center"] + (1 - lam) * oof["nnls_center"]
            region_maes = oof.assign(abs_error=(oof["blend"] - oof["actual"]).abs()).groupby("held_region")["abs_error"].mean()
            candidate_scores[lam] = float(region_maes.mean())
        selected_lambda = min(LAMBDAS, key=lambda lam: (candidate_scores[lam], lam != 1.0))
        selection_rows.append({
            "held_region": outer_region, "selected_lambda_local_anchor": selected_lambda,
            **{f"inner_macro_mae_lambda{lam:g}_gco2_kwh": 1000 * score for lam, score in candidate_scores.items()},
        })

        # OOF calibration predictions for 2022; the outer test region is always excluded.
        calibration_parts = []
        for cal_region in v329.REGIONS:
            if cal_region == outer_region:
                continue
            train = train_rows(provinces, assignment, mix, factors, {outer_region, cal_region})
            calibration_parts.append(predictions_for_region(cal_region, train, provinces, mix, factors, 2022))
        calibration = pd.concat(calibration_parts, ignore_index=True)
        calibration["selected_center"] = (
            selected_lambda * calibration["local_center"]
            + (1 - selected_lambda) * calibration["nnls_center"]
        )

        # Outer test predictions use 2021+2022 labels from the six other regions.
        final_train = train_rows(provinces, assignment, mix, factors, {outer_region})
        test = predictions_for_region(outer_region, final_train, provinces, mix, factors, 2023)
        test["selected_center"] = selected_lambda * test["local_center"] + (1 - selected_lambda) * test["nnls_center"]

        for nominal in NOMINALS:
            method_cal = {
                "local_anchor": calibration["local_center"],
                "nnls": calibration["nnls_center"],
                "selected_shrinkage": calibration["selected_center"],
            }
            method_test = {
                "local_anchor": test["local_center"],
                "nnls": test["nnls_center"],
                "selected_shrinkage": test["selected_center"],
            }
            for method, cal_center in method_cal.items():
                residual = np.abs(calibration["actual"].to_numpy(float) - cal_center.to_numpy(float))
                radius = conformal_quantile(residual, nominal)
                centers = method_test[method].to_numpy(float)
                actual = test["actual"].to_numpy(float)
                lower, upper = centers - radius, centers + radius
                covered, scores = interval_metrics(actual, lower, upper, nominal)
                for idx, province in enumerate(test["province"]):
                    output_rows.append({
                        "held_region": outer_region, "province": province,
                        "method": method, "nominal_coverage": nominal,
                        "selected_lambda_local_anchor": selected_lambda,
                        "actual_kgco2_kwh": actual[idx], "center_kgco2_kwh": centers[idx],
                        "radius_kgco2_kwh": radius, "lower_kgco2_kwh": lower[idx],
                        "upper_kgco2_kwh": upper[idx], "covered": bool(covered[idx]),
                        "width_gco2_kwh": (upper[idx] - lower[idx]) * 1000,
                        "interval_score_gco2_kwh": scores[idx],
                    })
    pred = pd.DataFrame(output_rows)
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
    macro = region_metrics.groupby(["method", "nominal_coverage"])[
        ["coverage", "mean_width_gco2_kwh", "mean_interval_score_gco2_kwh"]
    ].mean().reset_index()
    paired_interval = {}
    for nominal in NOMINALS:
        subset = pred.loc[pred.nominal_coverage.eq(nominal)]
        for baseline in ("local_anchor", "nnls"):
            for metric in ("interval_score_gco2_kwh", "width_gco2_kwh"):
                region_values = subset.pivot(
                    index=["held_region", "province"], columns="method", values=metric
                ).reset_index()
                region_diff = region_values.assign(
                    delta=region_values["selected_shrinkage"] - region_values[baseline]
                ).groupby("held_region")["delta"].mean()
                paired_interval[f"selected_minus_{baseline}_{metric}_nom{nominal}"] = region_bootstrap(
                    region_diff, 406 + int(nominal * 100) + len(paired_interval)
                )
    point_rows = pred.loc[pred.nominal_coverage.eq(0.90)].drop_duplicates(
        ["held_region", "province", "method"]
    ).copy()
    point_rows["absolute_error_gco2_kwh"] = (
        point_rows["center_kgco2_kwh"] - point_rows["actual_kgco2_kwh"]
    ).abs() * 1000
    point_metrics = point_rows.groupby("method")["absolute_error_gco2_kwh"].mean().to_dict()
    paired_point = {}
    point_wide = point_rows.pivot(
        index=["held_region", "province"], columns="method", values="absolute_error_gco2_kwh"
    ).reset_index()
    for baseline in ("local_anchor", "nnls"):
        region_diff = point_wide.assign(
            delta=point_wide["selected_shrinkage"] - point_wide[baseline]
        ).groupby("held_region")["delta"].mean()
        paired_point[f"selected_minus_{baseline}_mae"] = region_bootstrap(region_diff, 406 + len(paired_point))
    selections = pd.DataFrame(selection_rows)
    summary = {
        "experiment": "nested region-held-out selection of shrinkage between local physical anchor and five-group NNLS",
        "target": "2023 MEE/NBS provincial annual consumption-side CO2 factor; 30 provinces; 7 statistical regions",
        "selection": "For each outer test region, choose lambda from {0,.25,.5,.75,1} by equal-region 2022 held-region MAE; lambda=1 is the local anchor.",
        "pooled_test_metrics": pooled.to_dict("records"),
        "equal_region_macro_test_metrics": macro.to_dict("records"),
        "paired_interval_region_bootstrap": paired_interval,
        "point_mae_gco2_kwh": point_metrics,
        "paired_point_mae_region_bootstrap": paired_point,
        "lambda_selection_counts": selections["selected_lambda_local_anchor"].value_counts().sort_index().to_dict(),
        "outer_region_lambda": selections.set_index("held_region")["selected_lambda_local_anchor"].to_dict(),
        "limitations": [
            "One principal outer target year and seven correlated regions; the result is exploratory.",
            "Annual consumption-side factor target is not hourly generation CIF.",
            "NBS predictor is a retrospective revision snapshot; as-published vintages are unavailable.",
            "Residual-quantile intervals are empirical and do not give a formal finite-sample guarantee under regional dependence.",
        ],
        "sha256": {
            str(FACTORS.relative_to(ROOT)): sha256(FACTORS),
            str(MONTHLY.relative_to(ROOT)): sha256(MONTHLY),
            "scripts/v327_nbs_mix_transfer.py": sha256(ROOT / "scripts/v327_nbs_mix_transfer.py"),
            "scripts/v329_china_region_block_coldstart.py": sha256(ROOT / "scripts/v329_china_region_block_coldstart.py"),
            "scripts/v406_nested_physical_anchor_shrinkage.py": sha256(Path(__file__).resolve()),
        },
    }
    out.mkdir(parents=True)
    pred.to_csv(out / "outer_predictions.csv", index=False)
    region_metrics.to_csv(out / "region_metrics.csv", index=False)
    selections.to_csv(out / "lambda_selection_by_outer_region.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"outer_region_lambda": summary["outer_region_lambda"],
                      "pooled_test_metrics": summary["pooled_test_metrics"],
                      "equal_region_macro_test_metrics": summary["equal_region_macro_test_metrics"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
