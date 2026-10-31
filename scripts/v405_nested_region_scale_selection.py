"""V405: nested region-CV selection of local-composition interval scaling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import v327_nbs_mix_transfer as v327
import v329_china_region_block_coldstart as v329
from v404_nested_region_conformal_mix_intervals import (
    FACTORS, MONTHLY, ROOT, conformal_quantile, interval_score, predict_center,
)


DEFAULT_OUT = ROOT / "results/v405_nested_region_scale_selection-r1"
BETAS = (0.0, 0.5, 1.0)
NOMINALS = (0.90, 0.95)
BASELINES = ("nbsnnls", "globalmean")


def training_rows(
    provinces: list[str], assignment: dict[str, str], mix: dict,
    factors: pd.DataFrame, excluded_regions: set[str],
) -> pd.DataFrame:
    rows = []
    for target_year in (2021, 2022):
        feature_year = target_year - 1
        for province in provinces:
            if assignment[province] in excluded_regions:
                continue
            row = {
                "province": province,
                "actual": float(factors.loc[province, f"ef_{target_year}_kgco2_kwh"]),
                "thermal_share": mix[(province, feature_year)]["thermal"],
            }
            row.update({f"nbs_{group}": mix[(province, feature_year)][group] for group in v327.GROUPS})
            rows.append(row)
    return pd.DataFrame(rows)


def crossfit_2022(
    outer_region: str, inner_region: str | None,
    provinces: list[str], assignment: dict[str, str], mix: dict,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-fit 2022 residuals, excluding outer, inner-validation, and residual regions."""
    rows = []
    excluded = {outer_region}
    if inner_region is not None:
        excluded.add(inner_region)
    residual_regions = [r for r in v329.REGIONS if r not in excluded]
    for residual_region in residual_regions:
        train = training_rows(
            provinces, assignment, mix, factors, excluded | {residual_region}
        )
        for province in v329.REGIONS[residual_region]:
            feature_year = 2021
            row = {"thermal_share": mix[(province, feature_year)]["thermal"]}
            row.update({f"nbs_{group}": mix[(province, feature_year)][group] for group in v327.GROUPS})
            actual = float(factors.loc[province, "ef_2022_kgco2_kwh"])
            for method in ("local700", *BASELINES):
                center, _ = predict_center(method, train, pd.Series(row))
                rows.append({
                    "residual_region": residual_region, "province": province,
                    "method": method, "actual": actual, "center": center,
                    "thermal_share": row["thermal_share"],
                })
    return pd.DataFrame(rows)


def make_interval(actual: float, center: float, radius: float, nominal: float) -> tuple[float, float, float]:
    lower, upper = center - radius, center + radius
    return lower, upper, interval_score(actual, lower, upper, nominal)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    out = parser.parse_args().out
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {out}")
    factors = pd.read_csv(FACTORS).set_index("province")
    provinces = sorted(factors.index.astype(str))
    mix, _ = v327.nbs_annual_mix(MONTHLY, provinces)
    assignment = {province: region for region, members in v329.REGIONS.items() for province in members}
    if set(assignment) != set(provinces):
        raise ValueError("Frozen statistical regions do not partition the target provinces")

    prediction_rows = []
    selection_rows = []
    for outer_region, outer_members in v329.REGIONS.items():
        # Tune beta only from nested, region-disjoint 2022 interval scores.
        beta_region_scores: dict[float, list[float]] = {beta: [] for beta in BETAS}
        selection_nominal = 0.90
        for validation_region in v329.REGIONS:
            if validation_region == outer_region:
                continue
            cal = crossfit_2022(
                outer_region, validation_region, provinces, assignment, mix, factors
            )
            excluded = {outer_region, validation_region}
            train = training_rows(provinces, assignment, mix, factors, excluded)
            val_scores_by_beta = {beta: [] for beta in BETAS}
            for province in v329.REGIONS[validation_region]:
                feature_year = 2021
                row = {"thermal_share": mix[(province, feature_year)]["thermal"]}
                row.update({f"nbs_{group}": mix[(province, feature_year)][group] for group in v327.GROUPS})
                actual = float(factors.loc[province, "ef_2022_kgco2_kwh"])
                center, _ = predict_center("local700", train, pd.Series(row))
                local_cal = cal.loc[cal.method.eq("local700")]
                residual = np.abs(local_cal["actual"].to_numpy(float) - local_cal["center"].to_numpy(float))
                cal_scale = np.maximum(local_cal["thermal_share"].to_numpy(float), 1e-12)
                for beta in BETAS:
                    q = conformal_quantile(residual / cal_scale**beta, selection_nominal)
                    radius = q * max(row["thermal_share"], 1e-12) ** beta
                    _, _, score = make_interval(actual, center, radius, selection_nominal)
                    val_scores_by_beta[beta].append(score)
            for beta in BETAS:
                beta_region_scores[beta].append(float(np.mean(val_scores_by_beta[beta])))
        means = {beta: float(np.mean(scores)) for beta, scores in beta_region_scores.items()}
        # Stable tie-break: prefer the simpler unscaled residual (beta=0).
        selected_beta = min(BETAS, key=lambda beta: (means[beta], beta != 0.0))
        selection_rows.append({
            "held_region": outer_region, "selection_nominal_coverage": selection_nominal,
            "selected_beta": selected_beta,
            **{f"cv_interval_score_beta{beta:g}": means[beta] for beta in BETAS},
        })

        # Calibrate outer-test intervals from cross-fitted 2022 residuals.
        outer_cal = crossfit_2022(outer_region, None, provinces, assignment, mix, factors)
        final_train = training_rows(provinces, assignment, mix, factors, {outer_region})
        for province in outer_members:
            feature_year = 2022
            row = {"thermal_share": mix[(province, feature_year)]["thermal"]}
            row.update({f"nbs_{group}": mix[(province, feature_year)][group] for group in v327.GROUPS})
            actual = float(factors.loc[province, "ef_2023_kgco2_kwh"])
            centers = {
                method: predict_center(method, final_train, pd.Series(row))[0]
                for method in ("local700", *BASELINES)
            }
            for nominal in NOMINALS:
                local_cal = outer_cal.loc[outer_cal.method.eq("local700")]
                residual = np.abs(local_cal["actual"].to_numpy(float) - local_cal["center"].to_numpy(float))
                cal_scale = np.maximum(local_cal["thermal_share"].to_numpy(float), 1e-12)
                for beta in BETAS:
                    q = conformal_quantile(residual / cal_scale**beta, nominal)
                    radius = q * max(row["thermal_share"], 1e-12) ** beta
                    lower, upper, score = make_interval(actual, centers["local700"], radius, nominal)
                    prediction_rows.append({
                        "held_region": outer_region, "province": province,
                        "method": "local700_fixed",
                        "beta": beta, "nominal_coverage": nominal,
                        "selected_beta": selected_beta, "center": centers["local700"],
                        "actual": actual, "lower": lower, "upper": upper,
                        "covered": bool(lower <= actual <= upper),
                        "width_gco2_kwh": (upper - lower) * 1000,
                        "interval_score_gco2_kwh": score,
                    })
                beta = selected_beta
                q = conformal_quantile(residual / cal_scale**beta, nominal)
                radius = q * max(row["thermal_share"], 1e-12) ** beta
                lower, upper, score = make_interval(actual, centers["local700"], radius, nominal)
                prediction_rows.append({
                    "held_region": outer_region, "province": province,
                    "method": "local700_nested_selected", "beta": beta,
                    "nominal_coverage": nominal, "selected_beta": selected_beta,
                    "center": centers["local700"], "actual": actual,
                    "lower": lower, "upper": upper, "covered": bool(lower <= actual <= upper),
                    "width_gco2_kwh": (upper - lower) * 1000,
                    "interval_score_gco2_kwh": score,
                })
                for method in BASELINES:
                    cal_model = outer_cal.loc[outer_cal.method.eq(method)]
                    model_residual = np.abs(cal_model["actual"].to_numpy(float) - cal_model["center"].to_numpy(float))
                    q = conformal_quantile(model_residual, nominal)
                    radius = q
                    lower, upper, score = make_interval(actual, centers[method], radius, nominal)
                    prediction_rows.append({
                        "held_region": outer_region, "province": province,
                        "method": method, "beta": 0.0,
                        "nominal_coverage": nominal, "selected_beta": selected_beta,
                        "center": centers[method], "actual": actual,
                        "lower": lower, "upper": upper, "covered": bool(lower <= actual <= upper),
                        "width_gco2_kwh": (upper - lower) * 1000,
                        "interval_score_gco2_kwh": score,
                    })
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    metrics = predictions.groupby(["held_region", "method", "beta", "nominal_coverage"], dropna=False).agg(
        n=("province", "size"), coverage=("covered", "mean"),
        mean_width_gco2_kwh=("width_gco2_kwh", "mean"),
        mean_interval_score_gco2_kwh=("interval_score_gco2_kwh", "mean"),
    ).reset_index()
    overall = predictions.groupby(["method", "beta", "nominal_coverage"], dropna=False).agg(
        n=("province", "size"), coverage=("covered", "mean"),
        mean_width_gco2_kwh=("width_gco2_kwh", "mean"),
        mean_interval_score_gco2_kwh=("interval_score_gco2_kwh", "mean"),
    ).reset_index()
    macro = metrics.groupby(["method", "beta", "nominal_coverage"], dropna=False)[
        ["coverage", "mean_width_gco2_kwh", "mean_interval_score_gco2_kwh"]
    ].mean().reset_index()
    selected_region_metrics = predictions.loc[predictions.method.eq("local700_nested_selected")].groupby(
        ["held_region", "nominal_coverage"]
    ).agg(
        n=("province", "size"), coverage=("covered", "mean"),
        mean_width_gco2_kwh=("width_gco2_kwh", "mean"),
        mean_interval_score_gco2_kwh=("interval_score_gco2_kwh", "mean"),
    ).reset_index()
    selected_vs_beta0 = {}
    rng = np.random.default_rng(405)
    for nominal in NOMINALS:
        selected_rows = selected_region_metrics.loc[selected_region_metrics.nominal_coverage.eq(nominal)].set_index("held_region")
        fixed_rows = metrics.loc[(metrics.method.eq("local700_fixed"))
                                 & np.isclose(metrics.beta, 0.0)
                                 & np.isclose(metrics.nominal_coverage, nominal)].set_index("held_region")
        diffs = {}
        for metric in ("coverage", "mean_width_gco2_kwh", "mean_interval_score_gco2_kwh"):
            delta = (selected_rows[metric] - fixed_rows[metric]).sort_index().to_numpy(float)
            boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(50000)])
            diffs[metric] = {
                "macro_difference": float(delta.mean()),
                "region_cluster_bootstrap_95pct": np.quantile(boot, [0.025, 0.975]).tolist(),
                "regions_selected_better": int((delta < 0).sum()) if metric != "coverage" else int((delta > 0).sum()),
            }
        selected_vs_beta0[str(nominal)] = diffs
    summary = {
        "experiment": "nested region CV beta selection on 2022; outer region-held-out evaluation on 2023",
        "target": "MEE/NBS 2023 provincial annual consumption-side CO2 factor; 30 provinces; 7 statistical regions",
        "selection": "For each outer test region, choose beta in {0,.5,1} by equal-region 2022 interval score. Inner validation region excluded from its calibration residual models; outer test region excluded throughout.",
        "outer_predictions": overall.to_dict("records"),
        "equal_region_macro": macro.to_dict("records"),
        "selected_method_macro_by_region": selected_region_metrics.to_dict("records"),
        "selected_vs_fixed_beta0_region_bootstrap": selected_vs_beta0,
        "beta_selection_counts_by_nominal": {
            "selection_nominal_0.9_applied_to_both_0.9_and_0.95": selections["selected_beta"].value_counts().sort_index().to_dict()
        },
        "limitations": [
            "Posthoc method proposal, but beta selection does not use outer 2023 target labels.",
            "Only seven correlated regions and 30 provinces; region-level coverage is low-powered.",
            "2022 calibration labels and 2023 evaluation labels are annual consumption-side factors, not hourly CIF.",
            "Historical NBS features are retrospective revisions, not archived as-published vintages.",
            "Regional residual quantiles do not establish a distribution-free guarantee under dependence.",
        ],
        "sha256": {
            str(FACTORS.relative_to(ROOT)): sha256(FACTORS),
            str(MONTHLY.relative_to(ROOT)): sha256(MONTHLY),
            "scripts/v327_nbs_mix_transfer.py": sha256(ROOT / "scripts/v327_nbs_mix_transfer.py"),
            "scripts/v329_china_region_block_coldstart.py": sha256(ROOT / "scripts/v329_china_region_block_coldstart.py"),
            "scripts/v404_nested_region_conformal_mix_intervals.py": sha256(ROOT / "scripts/v404_nested_region_conformal_mix_intervals.py"),
            "scripts/v405_nested_region_scale_selection.py": sha256(Path(__file__).resolve()),
        },
    }
    out.mkdir(parents=True)
    predictions.to_csv(out / "outer_predictions.csv", index=False)
    metrics.to_csv(out / "region_metrics.csv", index=False)
    selections.to_csv(out / "beta_selection_by_outer_region.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"beta_selection_counts_by_nominal": summary["beta_selection_counts_by_nominal"],
                      "equal_region_macro": summary["equal_region_macro"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
