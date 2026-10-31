"""Rebuild V404's post-hoc nested region-held-out annual-factor intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

import v327_nbs_mix_transfer as v327
import v329_china_region_block_coldstart as v329


ROOT = Path(__file__).resolve().parents[1]
FACTORS = ROOT / "data/china/mee_provincial_grid_co2_factors_2021_2023.csv"
MONTHLY = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
DEFAULT_OUT = ROOT / "results/v404_nested_region_conformal_mix_intervals-r1-repro"
BETAS = (0.0, 0.5, 1.0)
NOMINALS = (0.90, 0.95)
METHODS = ("local700", "nbsnnls", "globalmean")


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    """Finite-sample split-conformal order statistic, with 1-based rank ceil((n+1)c)."""
    scores = np.asarray(scores, dtype=float)
    rank = int(np.ceil((len(scores) + 1) * coverage))
    if rank > len(scores):
        return float("inf")
    return float(np.sort(scores)[rank - 1])


def fit_nbs(train: pd.DataFrame) -> np.ndarray:
    x = train[[f"nbs_{group}" for group in v327.GROUPS]].to_numpy(float)
    y = train["actual"].to_numpy(float)
    return nnls(x, y)[0]


def predict_center(method: str, train: pd.DataFrame, row: pd.Series) -> tuple[float, float]:
    if method == "local700":
        return float(0.7 * row["thermal_share"]), float("nan")
    if method == "globalmean":
        return float(train["actual"].mean()), float(len(train))
    beta = fit_nbs(train)
    x = np.array([row[f"nbs_{group}"] for group in v327.GROUPS], dtype=float)
    return float(x @ beta), float(len(train))


def interval_score(actual: float, lower: float, upper: float, nominal: float) -> float:
    alpha = 1.0 - nominal
    score = upper - lower
    if actual < lower:
        score += 2.0 / alpha * (lower - actual)
    elif actual > upper:
        score += 2.0 / alpha * (actual - upper)
    return float(score * 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {args.out}")

    factors = pd.read_csv(FACTORS).set_index("province")
    provinces = sorted(factors.index.astype(str))
    monthly = pd.read_csv(MONTHLY)
    mix, _ = v327.nbs_annual_mix(MONTHLY, provinces)
    assignment = {province: region for region, members in v329.REGIONS.items() for province in members}
    if set(assignment) != set(provinces):
        raise ValueError("Frozen region assignment does not partition the factor dataset")

    rows = []
    # Calibration target year 2022: each calibration region is cross-fitted,
    # and neither it nor the outer test region supplies a training label.
    for held_region, test_members in v329.REGIONS.items():
        calibration_members = [p for p in provinces if assignment[p] != held_region]
        calibration_predictions: dict[str, list[dict]] = {m: [] for m in METHODS}
        for cal_region in v329.REGIONS:
            if cal_region == held_region:
                continue
            train_rows = []
            for target_year in (2021, 2022):
                for p in provinces:
                    if assignment[p] in (held_region, cal_region):
                        continue
                    feature_year = target_year - 1
                    row = {"province": p, "actual": float(factors.loc[p, f"ef_{target_year}_kgco2_kwh"])}
                    row["thermal_share"] = mix[(p, feature_year)]["thermal"]
                    row.update({f"nbs_{g}": mix[(p, feature_year)][g] for g in v327.GROUPS})
                    train_rows.append(row)
            train = pd.DataFrame(train_rows)
            for p in v329.REGIONS[cal_region]:
                actual = float(factors.loc[p, "ef_2022_kgco2_kwh"])
                row = {"thermal_share": mix[(p, 2021)]["thermal"]}
                row.update({f"nbs_{g}": mix[(p, 2021)][g] for g in v327.GROUPS})
                for method in METHODS:
                    center, _ = predict_center(method, train, pd.Series(row))
                    calibration_predictions[method].append({
                        "province": p, "actual": actual, "center": center,
                        "thermal_share": row["thermal_share"],
                    })

        cal = {m: pd.DataFrame(v) for m, v in calibration_predictions.items()}
        scores: dict[tuple[str, float], np.ndarray] = {}
        for beta in BETAS:
            residual = np.abs(cal["local700"]["actual"] - cal["local700"]["center"]).to_numpy(float)
            scale = np.maximum(cal["local700"]["thermal_share"].to_numpy(float), 1e-12) ** beta
            scores[("local700", beta)] = residual / scale
        for method in ("nbsnnls", "globalmean"):
            scores[(method, 0.0)] = np.abs(cal[method]["actual"] - cal[method]["center"]).to_numpy(float)

        # Fit final centers on 2021+2022 from regions outside the outer test region.
        final_train_rows = []
        for year in (2021, 2022):
            for p in provinces:
                if assignment[p] == held_region:
                    continue
                row = {"province": p, "actual": float(factors.loc[p, f"ef_{year}_kgco2_kwh"]),
                       "thermal_share": mix[(p, year - 1)]["thermal"]}
                row.update({f"nbs_{g}": mix[(p, year - 1)][g] for g in v327.GROUPS})
                final_train_rows.append(row)
        final_train = pd.DataFrame(final_train_rows)

        for p in test_members:
            actual = float(factors.loc[p, "ef_2023_kgco2_kwh"])
            row = {"thermal_share": mix[(p, 2022)]["thermal"]}
            row.update({f"nbs_{g}": mix[(p, 2022)][g] for g in v327.GROUPS})
            centers: dict[str, tuple[float, float]] = {
                method: predict_center(method, final_train, pd.Series(row)) for method in METHODS
            }
            for method in METHODS:
                beta_values = BETAS if method == "local700" else (0.0,)
                for beta in beta_values:
                    for nominal in NOMINALS:
                        q = conformal_quantile(scores[(method, beta)], nominal)
                        scale = max(row["thermal_share"], 1e-12) ** beta if method == "local700" else 1.0
                        radius = q * scale
                        center, n_train = centers[method]
                        lower, upper = center - radius, center + radius
                        rows.append({
                            "held_region": held_region, "province": p, "method": method,
                            "beta_scale": beta, "nominal_coverage": nominal,
                            "n_cal_provinces": int(len(cal[method])),
                            "n_test_model_train_rows": n_train,
                            "actual_kgco2_kwh": actual, "center_kgco2_kwh": center,
                            "radius_kgco2_kwh": radius, "lower_kgco2_kwh": lower,
                            "upper_kgco2_kwh": upper, "covered": bool(lower <= actual <= upper),
                            "width_gco2_kwh": (upper - lower) * 1000,
                            "interval_score_gco2_kwh": interval_score(actual, lower, upper, nominal),
                        })
    predictions = pd.DataFrame(rows)
    group = predictions.groupby(["held_region", "method", "beta_scale", "nominal_coverage"], dropna=False)
    region_metrics = group.agg(
        n=("province", "size"), coverage=("covered", "mean"),
        mean_width_g=("width_gco2_kwh", "mean"),
        mean_interval_score_g=("interval_score_gco2_kwh", "mean"),
    ).reset_index()
    models = {}
    for (method, beta, nominal), subset in predictions.groupby(["method", "beta_scale", "nominal_coverage"], dropna=False):
        key = f"{method}_beta{beta}_nom{nominal}"
        models[key] = {
            "n": int(len(subset)), "coverage": float(subset["covered"].mean()),
            "mean_width_gco2_kwh": float(subset["width_gco2_kwh"].mean()),
            "mean_interval_score_gco2_kwh": float(subset["interval_score_gco2_kwh"].mean()),
            "misses": int((~subset["covered"]).sum()),
        }
    summaries = {}
    for nom in NOMINALS:
        for method, beta in (("local700", 0.0), ("local700", 0.5), ("local700", 1.0), ("nbsnnls", 0.0), ("globalmean", 0.0)):
            key = f"{method}_beta{beta}_nom{nom}"
            vals = region_metrics.loc[(region_metrics.method == method)
                                      & np.isclose(region_metrics.beta_scale, beta)
                                      & np.isclose(region_metrics.nominal_coverage, nom)]
            summaries[key] = {
                "macro_coverage": float(vals.coverage.mean()),
                "macro_width_g": float(vals.mean_width_g.mean()),
                "macro_interval_score_g": float(vals.mean_interval_score_g.mean()),
            }
    paired = {}
    rng = np.random.default_rng(404)
    for nominal in NOMINALS:
        local05 = region_metrics.loc[(region_metrics.method == "local700")
                                     & np.isclose(region_metrics.beta_scale, 0.5)
                                     & np.isclose(region_metrics.nominal_coverage, nominal)].set_index("held_region")["mean_interval_score_g"]
        for other_method, other_beta, label in (("nbsnnls", 0.0, "nbsnnls"), ("globalmean", 0.0, "globalmean"), ("local700", 0.0, "local700_beta0")):
            other = region_metrics.loc[(region_metrics.method == other_method)
                                       & np.isclose(region_metrics.beta_scale, other_beta)
                                       & np.isclose(region_metrics.nominal_coverage, nominal)].set_index("held_region")["mean_interval_score_g"]
            delta = (local05 - other).sort_index().to_numpy(float)
            boots = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(50000)])
            paired[f"local700_beta0.5_minus_{label}_interval_score_nom{nominal}"] = {
                "macro_mean_difference_g": float(delta.mean()),
                "cluster_bootstrap_95pct_g": np.quantile(boots, [0.025, 0.975]).tolist(),
                "regions_local_better": int((delta < 0).sum()),
            }
    summary = {
        "experiment": "posthoc, fully nested region-held-out rolling residual interval diagnostic",
        "target": "2023 MEE/NBS provincial annual consumption-side CO2 factor; 30 provinces, 7 pre-existing statistical regions",
        "feature": "NBS-derived above-designated-size five-group annual generation mix lagged one year; Feb is Jan-Feb cumulative bucket",
        "target_time": "2023 official factor; 2022 factor used for calibration; NBS historical as-published vintages unavailable",
        "models": models, "paired_region_bootstrap": paired,
        "all_region_equal_macro": summaries,
        "limitations": [
            "Annual official consumption-side factor only; not hourly generation CIF or hourly coverage.",
            "Posthoc exploratory comparison; 2023 factor labels were visible during method development.",
            "NBS historical as-published feature vintages are unavailable.",
            "Seven regions and one principal test target year give low-powered region-level evidence.",
            "Empirical rolling residual quantiles are not a formal distribution-free guarantee under spatial/temporal dependence.",
            "Thermal midpoint 700 gCO2/kWh is the midpoint of the pre-existing [400,1000] support assumption.",
        ],
    }
    args.out.mkdir(parents=True)
    predictions.to_csv(args.out / "nested_region_predictions.csv", index=False)
    region_metrics.to_csv(args.out / "region_metrics.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"models": models, "paired_region_bootstrap": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
