"""V329: leave entire Chinese statistical regions out of annual-factor transfer."""

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
PROTOCOL = ROOT / "docs/V329_CHINA_REGION_BLOCK_COLDSTART_PROTOCOL.md"
FACTORS = ROOT / "data/china/mee_provincial_grid_co2_factors_2021_2023.csv"
MONTHLY = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
EMBER = ROOT / "results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
OUT = ROOT / "results/v329-china-region-block-coldstart-r1"
TARGET_YEARS = (2022, 2023)
REGIONS = {
    "North": ("Beijing", "Tianjin", "Hebei", "Shanxi", "Inner Mongolia"),
    "Northeast": ("Liaoning", "Jilin", "Heilongjiang"),
    "East": ("Shanghai", "Jiangsu", "Zhejiang", "Anhui", "Fujian", "Jiangxi", "Shandong"),
    "Central": ("Henan", "Hubei", "Hunan"),
    "South": ("Guangdong", "Guangxi", "Hainan"),
    "Southwest": ("Chongqing", "Sichuan", "Guizhou", "Yunnan"),
    "Northwest": ("Shaanxi", "Gansu", "Qinghai", "Ningxia", "Xinjiang"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_scores(frame: pd.DataFrame, col: str) -> dict[str, float]:
    err = frame[col].to_numpy(float) - frame["actual"].to_numpy(float)
    return {"mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mean_signed_error": float(np.mean(err)),
            "p90_absolute_error": float(np.quantile(np.abs(err), 0.9))}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen result directory: {OUT}")
    factors = pd.read_csv(FACTORS).set_index("province")
    provinces = sorted(factors.index.astype(str))
    assignment = {province: region for region, members in REGIONS.items() for province in members}
    if set(assignment) != set(provinces) or len(assignment) != 30:
        raise ValueError(f"Frozen geographic regions do not partition target provinces: missing={set(provinces)-set(assignment)}, extra={set(assignment)-set(provinces)}")

    nbs_mix, nbs_audit = v327.nbs_annual_mix(MONTHLY, provinces)
    ember_mix = v300.monthly_mix(EMBER)
    train_rows = []
    for target_year in (2021, 2022):
        feature_year = target_year - 1
        for province in provinces:
            nbs = nbs_mix[(province, feature_year)]
            ember = v300.aggregate_coarse(ember_mix[(province, feature_year)])
            train_rows.append({"province": province, "region": assignment[province],
                               "target_year": target_year, "feature_year": feature_year,
                               "actual": float(factors.loc[province, f"ef_{target_year}_kgco2_kwh"]),
                               **{f"nbs_{group}": float(nbs[group]) for group in v327.GROUPS},
                               **{f"ember_{group}": float(value) for group, value in zip(v327.GROUPS, ember)}})
    train = pd.DataFrame(train_rows)
    predictions: list[dict[str, object]] = []
    for held_region, members in REGIONS.items():
        for target_year in TARGET_YEARS:
            feature_year = target_year - 1
            fit = train.loc[train["target_year"].lt(target_year) & train["region"].ne(held_region)]
            if fit["province"].isin(members).any():
                raise AssertionError("Held-out region leaked into training rows")
            y = fit["actual"].to_numpy(float)
            beta_nbs, _ = nnls(fit[[f"nbs_{group}" for group in v327.GROUPS]].to_numpy(float), y)
            beta_ember, _ = nnls(fit[[f"ember_{group}" for group in v327.GROUPS]].to_numpy(float), y)
            for province in members:
                nbs_x = np.array([nbs_mix[(province, feature_year)][group] for group in v327.GROUPS])
                ember_x = v300.aggregate_coarse(ember_mix[(province, feature_year)])
                predictions.append({
                    "province": province, "held_region": held_region,
                    "target_year": target_year, "feature_year": feature_year,
                    "n_train_rows": int(len(fit)), "n_train_provinces": int(fit.province.nunique()),
                    "actual": float(factors.loc[province, f"ef_{target_year}_kgco2_kwh"]),
                    "outside_region_mean_pred": float(y.mean()),
                    "nbs_mix_pred": float(nbs_x @ beta_nbs),
                    "ember_mix_pred": float(ember_x @ beta_ember),
                    "nbs_beta": json.dumps(dict(zip(v327.GROUPS, beta_nbs.tolist())), ensure_ascii=False),
                    "ember_beta": json.dumps(dict(zip(v327.GROUPS, beta_ember.tolist())), ensure_ascii=False),
                })
    pred = pd.DataFrame(predictions)
    methods = {"outside_region_mean": "outside_region_mean_pred",
               "NBS_lagged_mix_NNLS": "nbs_mix_pred",
               "Ember_lagged_mix_NNLS": "ember_mix_pred"}
    score_rows = []
    for evaluation, subset in [("pooled_2022_2023", pred), *((str(y), pred.loc[pred.target_year.eq(y)]) for y in TARGET_YEARS)]:
        for method, col in methods.items():
            score_rows.append({"evaluation": evaluation, "weighting": "province_weighted",
                               "method": method, **model_scores(subset, col)})
        for method, col in methods.items():
            region_metrics = [model_scores(group, col) for _, group in subset.groupby("held_region")]
            score_rows.append({"evaluation": evaluation, "weighting": "region_equal",
                               "method": method,
                               **{key: float(np.mean([item[key] for item in region_metrics]))
                                  for key in ("mae", "rmse", "mean_signed_error", "p90_absolute_error")}})
    scores = pd.DataFrame(score_rows)

    pred["abs_nbs"] = (pred.nbs_mix_pred - pred.actual).abs()
    pred["abs_ember"] = (pred.ember_mix_pred - pred.actual).abs()
    pred["abs_mean"] = (pred.outside_region_mean_pred - pred.actual).abs()
    region_errors = pred.groupby("held_region")[["abs_nbs", "abs_ember", "abs_mean"]].mean()
    delta = (region_errors.abs_nbs - region_errors.abs_ember).to_numpy(float)
    rng = np.random.default_rng(329)
    bootstrap = np.array([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(50000)])
    paired_nbs_ember = {
        "region_equal_mean_absolute_error_difference_kgco2_kwh": float(delta.mean()),
        "region_cluster_bootstrap_95pct": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "regions_nbs_better": int((delta < 0).sum()),
        "per_region_mean_abs_error_difference": {region: float(value) for region, value in zip(region_errors.index, delta)},
    }
    region_year = []
    for (region, year), group in pred.groupby(["held_region", "target_year"]):
        for method, col in methods.items():
            region_year.append({"held_region": region, "target_year": int(year),
                                "n_test_provinces": int(group.province.nunique()),
                                "n_train_provinces": int(group.n_train_provinces.iloc[0]),
                                "method": method, **model_scores(group, col),
                                "fraction_better_than_outside_mean": float(np.mean(np.abs(group[col] - group.actual) < np.abs(group.outside_region_mean_pred - group.actual)))})

    OUT.mkdir(parents=True)
    pred.to_csv(OUT / "province_predictions.csv", index=False)
    scores.to_csv(OUT / "model_scores.csv", index=False)
    pd.DataFrame(region_year).to_csv(OUT / "region_year_scores.csv", index=False)
    pd.DataFrame(nbs_audit).to_csv(OUT / "nbs_feature_audit.csv", index=False)
    summary = {
        "status": "WHOLE_STATISTICAL_REGION_COLDSTART_RETROSPECTIVE_ANNUAL_MEE_FACTOR_TRANSFER",
        "n_regions": len(REGIONS), "n_provinces": len(provinces),
        "region_membership": {region: list(members) for region, members in REGIONS.items()},
        "target_years": list(TARGET_YEARS),
        "training_targets_for_each_test": {str(year): [target for target in (2021, 2022) if target < year] for year in TARGET_YEARS},
        "nbs_vs_ember_region_cluster_comparison": paired_nbs_ember,
        "models": scores.to_dict("records"),
        "severe_region_failures": [
            row for row in region_year
            if row["method"] == "NBS_lagged_mix_NNLS" and row["target_year"] == 2023
            and row["mae"] > 2 * next(r["mae"] for r in region_year if r["held_region"] == row["held_region"] and r["target_year"] == row["target_year"] and r["method"] == "outside_region_mean")
        ],
        "limitations": [
            "Targets are annual MEE consumption-accounting factors, not hourly generation CIF.",
            "Only two forward annual targets and seven correlated statistical regions are available.",
            "This is retrospective geographic-block transfer; target-vintage and predictor as-of availability are not verified.",
            "NBS monthly generation covers above-designated-size enterprises and omits below-size generation.",
            "The NBS-versus-Ember uncertainty interval uses only seven regional clusters and is low-power.",
        ],
        "sha256": {
            str(MONTHLY.relative_to(ROOT)): sha(MONTHLY),
            str(FACTORS.relative_to(ROOT)): sha(FACTORS),
            str(EMBER.relative_to(ROOT)): sha(EMBER),
            "docs/V329_CHINA_REGION_BLOCK_COLDSTART_PROTOCOL.md": sha(PROTOCOL),
            "scripts/v329_china_region_block_coldstart.py": sha(Path(__file__)),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(scores.to_string(index=False))
    print(json.dumps({"paired_nbs_vs_ember": paired_nbs_ember,
                      "severe_region_failures": summary["severe_region_failures"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
