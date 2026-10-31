"""V330: nested spatial tuning of ridge-regularized NBS mix transfer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

import v300_official_provincial_ef_mix_transfer as v300
import v327_nbs_mix_transfer as v327


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V330_SPATIAL_NESTED_RIDGE_PROTOCOL.md"
FACTORS = ROOT / "data/china/mee_provincial_grid_co2_factors_2021_2023.csv"
MONTHLY = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
EMBER = ROOT / "results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
OUT = ROOT / "results/v330-spatial-nested-ridge-transfer-r1"
REGIONS = {
    "North": ("Beijing", "Tianjin", "Hebei", "Shanxi", "Inner Mongolia"),
    "Northeast": ("Liaoning", "Jilin", "Heilongjiang"),
    "East": ("Shanghai", "Jiangsu", "Zhejiang", "Anhui", "Fujian", "Jiangxi", "Shandong"),
    "Central": ("Henan", "Hubei", "Hunan"),
    "South": ("Guangdong", "Guangxi", "Hainan"),
    "Southwest": ("Chongqing", "Sichuan", "Guizhou", "Yunnan"),
    "Northwest": ("Shaanxi", "Gansu", "Qinghai", "Ningxia", "Xinjiang"),
}
LAMBDAS = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)
GROUPS = tuple(v327.GROUPS)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_ridge_nnls(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    if lam == 0:
        return nnls(x, y)[0]
    n = len(y)
    augmented_x = np.vstack([x, np.sqrt(n * lam) * np.eye(x.shape[1])])
    augmented_y = np.concatenate([y, np.zeros(x.shape[1])])
    return nnls(augmented_x, augmented_y)[0]


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def choose_lambda(train: pd.DataFrame, target_year: int) -> tuple[float, dict[str, float]]:
    # A spatial CV that respects target-year availability: 2022 is validated
    # from 2021 labels; for the 2022 outer target, use region CV within 2021.
    validation_year = 2021 if target_year == 2022 else 2022
    validation = train.loc[train.target_year.eq(validation_year)]
    fit_years = train.loc[train.target_year.lt(validation_year)]
    if validation_year == 2021:
        validation = train.loc[train.target_year.eq(2021)]
        fit_years = validation
    fold_losses: dict[float, list[float]] = {lam: [] for lam in LAMBDAS}
    for held_region in sorted(train.region.unique()):
        valid = validation.loc[validation.region.eq(held_region)]
        if valid.empty:
            continue
        if validation_year == 2021:
            fit = fit_years.loc[fit_years.region.ne(held_region)]
        else:
            fit = fit_years.loc[fit_years.region.ne(held_region)]
        if fit.empty:
            continue
        x_fit = fit[[f"nbs_{g}" for g in GROUPS]].to_numpy(float)
        y_fit = fit.actual.to_numpy(float)
        x_valid = valid[[f"nbs_{g}" for g in GROUPS]].to_numpy(float)
        y_valid = valid.actual.to_numpy(float)
        for lam in LAMBDAS:
            beta = fit_ridge_nnls(x_fit, y_fit, lam)
            fold_losses[lam].extend(np.abs(y_valid - x_valid @ beta).tolist())
    losses = {f"{lam:g}": mae(np.zeros(len(v)), np.asarray(v)) for lam, v in fold_losses.items() if v}
    if not losses:
        raise RuntimeError("No inner spatial validation predictions available")
    minimum = min(losses.values())
    selected = min((lam for lam in LAMBDAS if abs(losses[f"{lam:g}"] - minimum) <= 1e-12))
    return selected, losses


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen results: {OUT}")
    factors = pd.read_csv(FACTORS).set_index("province")
    provinces = sorted(factors.index.astype(str))
    assignment = {province: region for region, members in REGIONS.items() for province in members}
    if set(assignment) != set(provinces):
        raise ValueError("Frozen region blocks do not partition target provinces")
    nbs_mix, _ = v327.nbs_annual_mix(MONTHLY, provinces)
    ember_mix = v300.monthly_mix(EMBER)

    rows = []
    for target_year in (2021, 2022):
        feature_year = target_year - 1
        for province in provinces:
            nbs = nbs_mix[(province, feature_year)]
            ember = v300.aggregate_coarse(ember_mix[(province, feature_year)])
            rows.append({"province": province, "region": assignment[province],
                         "target_year": target_year, "feature_year": feature_year,
                         "actual": float(factors.loc[province, f"ef_{target_year}_kgco2_kwh"]),
                         **{f"nbs_{g}": float(nbs[g]) for g in GROUPS},
                         **{f"ember_{g}": float(value) for g, value in zip(GROUPS, ember)}})
    train_all = pd.DataFrame(rows)
    predictions = []
    tuning_records = []
    for held_region, members in REGIONS.items():
        for target_year in (2022, 2023):
            feature_year = target_year - 1
            outer_train = train_all.loc[train_all.target_year.lt(target_year) & train_all.region.ne(held_region)]
            selected, inner_losses = choose_lambda(outer_train, target_year)
            nbs_x = np.vstack([[nbs_mix[(p, feature_year)][g] for g in GROUPS] for p in members])
            ember_x = np.vstack([v300.aggregate_coarse(ember_mix[(p, feature_year)]) for p in members])
            y_fit = outer_train.actual.to_numpy(float)
            x_nbs_fit = outer_train[[f"nbs_{g}" for g in GROUPS]].to_numpy(float)
            x_ember_fit = outer_train[[f"ember_{g}" for g in GROUPS]].to_numpy(float)
            beta_ols = fit_ridge_nnls(x_nbs_fit, y_fit, 0.0)
            beta_ridge = fit_ridge_nnls(x_nbs_fit, y_fit, selected)
            beta_ember = nnls(x_ember_fit, y_fit)[0]
            tuning_records.append({"held_region": held_region, "target_year": target_year,
                                   "selected_lambda": selected, "inner_mae_by_lambda": json.dumps(inner_losses),
                                   "ridge_beta": json.dumps(dict(zip(GROUPS, beta_ridge.tolist()))),
                                   "nnls_beta": json.dumps(dict(zip(GROUPS, beta_ols.tolist())))})
            for i, province in enumerate(members):
                predictions.append({"province": province, "held_region": held_region,
                                    "target_year": target_year,
                                    "actual": float(factors.loc[province, f"ef_{target_year}_kgco2_kwh"]),
                                    "outside_mean_pred": float(y_fit.mean()),
                                    "nbs_nnls_pred": float(nbs_x[i] @ beta_ols),
                                    "nbs_ridge_pred": float(nbs_x[i] @ beta_ridge),
                                    "ember_nnls_pred": float(ember_x[i] @ beta_ember),
                                    "selected_lambda": selected})
    pred = pd.DataFrame(predictions)
    methods = {"outside_mean": "outside_mean_pred", "NBS_NNLS": "nbs_nnls_pred",
               "NBS_nested_ridge_NNLS": "nbs_ridge_pred", "Ember_NNLS": "ember_nnls_pred"}

    def score(frame: pd.DataFrame, col: str) -> dict[str, float]:
        e = frame[col].to_numpy(float) - frame.actual.to_numpy(float)
        return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
                "mean_signed_error": float(np.mean(e)), "p90_absolute_error": float(np.quantile(np.abs(e), .9))}

    scores = []
    for name, subset in [("pooled", pred), ("2022", pred.loc[pred.target_year.eq(2022)]),
                         ("2023", pred.loc[pred.target_year.eq(2023)])]:
        for method, col in methods.items():
            scores.append({"evaluation": name, "weighting": "province_weighted", "method": method, **score(subset, col)})
            region_scores = [score(group, col) for _, group in subset.groupby("held_region")]
            scores.append({"evaluation": name, "weighting": "region_equal", "method": method,
                           **{k: float(np.mean([item[k] for item in region_scores])) for k in region_scores[0]}})
    region_errors = pred.groupby("held_region").apply(
        lambda f: pd.Series({m: float(np.mean(np.abs(f[c] - f.actual))) for m, c in methods.items()}),
        include_groups=False)
    rng = np.random.default_rng(330)
    deltas = {}
    for comparator in ("NBS_NNLS", "Ember_NNLS", "outside_mean"):
        d = (region_errors["NBS_nested_ridge_NNLS"] - region_errors[comparator]).to_numpy(float)
        boot = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(50000)])
        deltas[comparator] = {"region_equal_mae_difference": float(d.mean()),
                              "region_bootstrap_95pct": np.quantile(boot, [.025, .975]).tolist(),
                              "regions_ridge_better": int((d < 0).sum())}
    region_year_rows = []
    for (region, year), group in pred.groupby(["held_region", "target_year"]):
        for method, col in methods.items():
            region_year_rows.append({"held_region": region, "target_year": int(year), "method": method,
                                     "n_test_provinces": int(group.province.nunique()), **score(group, col)})
    OUT.mkdir(parents=True)
    pred.to_csv(OUT / "province_predictions.csv", index=False)
    pd.DataFrame(scores).to_csv(OUT / "model_scores.csv", index=False)
    pd.DataFrame(region_year_rows).to_csv(OUT / "region_year_scores.csv", index=False)
    pd.DataFrame(tuning_records).to_csv(OUT / "tuning_and_coefficients.csv", index=False)
    summary = {"status": "NESTED_SPATIAL_REGULARIZATION_RETROSPECTIVE_ANNUAL_MEE_FACTOR_TRANSFER",
               "n_regions": len(REGIONS), "n_provinces": len(provinces), "lambda_candidates": list(LAMBDAS),
               "models": scores, "paired_region_bootstrap": deltas,
               "selected_lambda_counts": pred.groupby(["held_region", "target_year"]).selected_lambda.first().value_counts().sort_index().to_dict(),
               "limitations": ["Targets are annual MEE consumption-side factors, not hourly CIF.",
                               "Only seven correlated region blocks and two outer target years.",
                               "The 2022 target tunes lambda by within-2021 geographic CV because no earlier MEE year is available.",
                               "Monthly NBS generation is above-designated-size only; data vintage legality is unverified."]}
    summary["sha256"] = {str(path.relative_to(ROOT)): sha(path) for path in (MONTHLY, FACTORS, EMBER, PROTOCOL, Path(__file__))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
    print(pd.DataFrame(scores).to_string(index=False))
    print(json.dumps({"paired_region_bootstrap": deltas, "selected_lambda_counts": summary["selected_lambda_counts"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
