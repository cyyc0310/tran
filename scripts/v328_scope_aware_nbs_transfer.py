"""V328: test an explicit unobserved-generation share in NBS mix transfer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

import v300_official_provincial_ef_mix_transfer as v300


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V328_SCOPE_AWARE_NBS_TRANSFER_PROTOCOL.md"
FACTORS = ROOT / "data/china/mee_provincial_grid_co2_factors_2021_2023.csv"
MONTHLY = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
ANNUAL = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_annual_generation_by_tech.csv"
EMBER = ROOT / "results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
OUT = ROOT / "results/v328-scope-aware-nbs-transfer-r1"
GROUPS = ("thermal", "hydro_or_remainder", "nuclear", "wind", "solar")
TARGET_YEARS = (2022, 2023)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nbs_annual_mix(path: Path, annual_path: Path, provinces: list[str]) -> tuple[dict[tuple[str, int], dict[str, float]], list[dict[str, object]]]:
    raw = pd.read_csv(path)
    annual_raw = pd.read_csv(annual_path)
    raw = raw.loc[raw["province"].isin(provinces) & raw["year"].isin([2020, 2021, 2022])].copy()
    result: dict[tuple[str, int], dict[str, float]] = {}
    audits: list[dict[str, object]] = []
    technologies = ("total", "thermal", "hydro", "nuclear", "wind", "solar")
    for province in provinces:
        for year in (2020, 2021, 2022):
            subset = raw.loc[raw["province"].eq(province) & raw["year"].eq(year)
                             & raw["month"].isin([2, *range(3, 13)])]
            annual: dict[str, float] = {}
            for tech in technologies:
                vals = subset.loc[subset["technology"].eq(tech), "generation_gwh"]
                annual[tech] = float(vals.sum()) if not vals.empty else float("nan")
            if not np.isfinite(annual["total"]) or annual["total"] <= 0:
                raise ValueError(f"Missing/nonpositive NBS total for {province} {year}")
            full_rows = annual_raw.loc[
                annual_raw["province"].eq(province) & annual_raw["year"].eq(year)
                & annual_raw["technology"].eq("total"), "generation_gwh"
            ]
            if len(full_rows) != 1 or float(full_rows.iloc[0]) <= 0:
                raise ValueError(f"Missing/nonpositive annual all-plant total for {province} {year}")
            full_total = float(full_rows.iloc[0])
            coverage_raw = annual["total"] / full_total
            if coverage_raw > 1.0 + 1e-6 or coverage_raw <= 0:
                raise ValueError(f"Monthly/all-plant generation scope mismatch for {province} {year}: {coverage_raw:.6f}")
            coverage = min(coverage_raw, 1.0)
            hydro_missing = not np.isfinite(annual["hydro"])
            if hydro_missing:
                hydro = max(annual["total"] - sum(annual[t] for t in ("thermal", "nuclear", "wind", "solar")), 0.0)
            else:
                hydro = annual["hydro"]
            amounts = np.array([annual["thermal"], hydro, annual["nuclear"], annual["wind"], annual["solar"]], dtype=float)
            if not np.isfinite(amounts).all() or (amounts < -1e-9).any() or amounts.sum() <= 0:
                raise ValueError(f"Invalid NBS category amounts for {province} {year}: {amounts}")
            shares = np.maximum(amounts, 0.0) / np.maximum(amounts, 0.0).sum()
            result[(province, year)] = {**dict(zip(GROUPS, shares.tolist())),
                                        "coverage_ratio": coverage,
                                        "unobserved_share": 1.0 - coverage}
            audits.append({"province": province, "year": year,
                           "reported_total_gwh": annual["total"],
                           "hydro_missing_imputed_as_unresolved_remainder": bool(hydro_missing),
                           "reported_category_sum_gwh": float(amounts.sum()),
                           "annual_all_plant_total_gwh": full_total,
                           "coverage_ratio": coverage,
                           "unobserved_share": 1.0 - coverage,
                           **dict(zip(GROUPS, shares.tolist()))})
    return result, audits


def score_predictions(pred: pd.DataFrame, value_col: str) -> dict[str, float]:
    error = pred[value_col].to_numpy(float) - pred["actual"].to_numpy(float)
    return {"mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mean_signed_error": float(np.mean(error)),
            "p90_absolute_error": float(np.quantile(np.abs(error), 0.9))}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen result directory: {OUT}")
    factors = pd.read_csv(FACTORS).set_index("province")
    provinces = sorted(factors.index.astype(str))
    nbs_mix, nbs_audit = nbs_annual_mix(MONTHLY, ANNUAL, provinces)
    ember_mix = v300.monthly_mix(EMBER)

    train_rows = []
    for target_year in (2021, 2022):
        for province in provinces:
            feature_year = target_year - 1
            nbs_values = nbs_mix[(province, feature_year)]
            ember_values = v300.aggregate_coarse(ember_mix[(province, feature_year)])
            train_rows.append({"province": province, "target_year": target_year,
                               "feature_year": feature_year,
                               "actual": float(factors.loc[province, f"ef_{target_year}_kgco2_kwh"]),
                               **{f"nbs_{k}": v for k, v in nbs_values.items()},
                               **{f"scope_{g}": nbs_values[g] * nbs_values["coverage_ratio"] for g in GROUPS},
                               "scope_unobserved": nbs_values["unobserved_share"],
                               **{f"ember_{k}": float(v) for k, v in zip(GROUPS, ember_values)}})
    train = pd.DataFrame(train_rows)

    predictions: list[dict[str, object]] = []
    for target_year in TARGET_YEARS:
        feature_year = target_year - 1
        for province in provinces:
            fit = train.loc[train["target_year"].lt(target_year) & train["province"].ne(province)]
            y = fit["actual"].to_numpy(float)
            if not len(y):
                raise ValueError(f"No training observations for {province} {target_year}")
            beta_nbs, _ = nnls(fit[[f"nbs_{g}" for g in GROUPS]].to_numpy(float), y)
            beta_scope, _ = nnls(fit[[*(f"scope_{g}" for g in GROUPS), "scope_unobserved"]].to_numpy(float), y)
            beta_ember, _ = nnls(fit[[f"ember_{g}" for g in GROUPS]].to_numpy(float), y)
            nbs_x = np.array([nbs_mix[(province, feature_year)][g] for g in GROUPS], dtype=float)
            ember_x = v300.aggregate_coarse(ember_mix[(province, feature_year)])
            predictions.append({
                "province": province,
                "target_year": target_year,
                "feature_year": feature_year,
                "n_train_rows": int(len(fit)),
                "actual": float(factors.loc[province, f"ef_{target_year}_kgco2_kwh"]),
                "training_mean_pred": float(y.mean()),
                "nbs_mix_pred": float(nbs_x @ beta_nbs),
                "scope_aware_nbs_pred": float(np.r_[nbs_x * nbs_mix[(province, feature_year)]["coverage_ratio"], nbs_mix[(province, feature_year)]["unobserved_share"]] @ beta_scope),
                "ember_mix_pred": float(ember_x @ beta_ember),
                "lagged_official_factor_reference": float(factors.loc[province, f"ef_{target_year - 1}_kgco2_kwh"]),
                "nbs_beta": json.dumps(dict(zip(GROUPS, beta_nbs.tolist())), ensure_ascii=False),
                "scope_aware_beta": json.dumps(dict(zip([*GROUPS, "unobserved_remainder"], beta_scope.tolist())), ensure_ascii=False),
                "ember_beta": json.dumps(dict(zip(GROUPS, beta_ember.tolist())), ensure_ascii=False),
            })
    pred = pd.DataFrame(predictions)
    methods = {
        "training_mean": "training_mean_pred",
        "NBS_lagged_monthly_mix_NNLS": "nbs_mix_pred",
        "NBS_scope_aware_six_group_NNLS": "scope_aware_nbs_pred",
        "Ember_lagged_monthly_mix_NNLS": "ember_mix_pred",
        "same_province_lagged_official_factor_reference": "lagged_official_factor_reference",
    }
    score_rows = []
    for year, subset in [("pooled_2022_2023", pred), *((str(y), pred.loc[pred.target_year.eq(y)]) for y in TARGET_YEARS)]:
        for name, col in methods.items():
            score_rows.append({"evaluation": year, "method": name, **score_predictions(subset, col)})
    scores = pd.DataFrame(score_rows)
    for year, subset in scores.groupby("evaluation"):
        base = float(subset.loc[subset.method.eq("training_mean"), "mae"].iloc[0])
        scores.loc[scores.evaluation.eq(year), "mae_reduction_vs_training_mean_pct"] = 100 * (
            base - scores.loc[scores.evaluation.eq(year), "mae"]
        ) / base

    paired = pred.pivot(index="province", columns="target_year", values="nbs_mix_pred")
    abs_nbs = pred.assign(abs_nbs=(pred.nbs_mix_pred - pred.actual).abs(),
                          abs_scope=(pred.scope_aware_nbs_pred - pred.actual).abs(),
                          abs_ember=(pred.ember_mix_pred - pred.actual).abs())
    paired_abs = abs_nbs.groupby("province")[["abs_nbs", "abs_scope", "abs_ember"]].mean()
    delta = (paired_abs.abs_nbs - paired_abs.abs_ember).to_numpy(float)
    scope_delta = (paired_abs.abs_scope - paired_abs.abs_nbs).to_numpy(float)
    rng = np.random.default_rng(328)
    boot = np.array([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(20000)])
    nbs_minus_ember = {
        "mean_paired_absolute_error_difference_kgco2_kwh": float(delta.mean()),
        "province_cluster_bootstrap_95pct": np.quantile(boot, [0.025, 0.975]).tolist(),
        "provinces_nbs_better": int((delta < 0).sum()),
        "provinces_tied": int((np.abs(delta) <= 1e-12).sum()),
    }
    scope_boot = np.array([rng.choice(scope_delta, size=len(scope_delta), replace=True).mean() for _ in range(20000)])
    scope_minus_nbs = {
        "mean_paired_absolute_error_difference_kgco2_kwh": float(scope_delta.mean()),
        "province_cluster_bootstrap_95pct": np.quantile(scope_boot, [0.025, 0.975]).tolist(),
        "provinces_scope_aware_better": int((scope_delta < 0).sum()),
    }

    OUT.mkdir(parents=True)
    pred.to_csv(OUT / "province_predictions.csv", index=False)
    scores.to_csv(OUT / "model_scores.csv", index=False)
    pd.DataFrame(nbs_audit).to_csv(OUT / "nbs_feature_audit.csv", index=False)
    pd.DataFrame(nbs_audit).to_csv(OUT / "scope_feature_audit.csv", index=False)
    summary = {
        "status": "OFFICIAL_NBS_SCOPE_AWARE_LAGGED_MIX_PROVINCIAL_ANNUAL_EF_LOPO_TRANSFER_RETROSPECTIVE",
        "n_provinces": len(provinces),
        "target_years": list(TARGET_YEARS),
        "feature_years": {str(y): y - 1 for y in TARGET_YEARS},
        "training_target_years_for_each_test": {str(y): [t for t in (2021, 2022) if t < y] for y in TARGET_YEARS},
        "feature_groups": list(GROUPS),
        "scope_aware_feature_groups": [*GROUPS, "unobserved_remainder"],
        "paired_scope_aware_vs_nbs_five": scope_minus_nbs,
        "category_scope": "monthly above-designated-size NBS-derived generation, Feb cumulative Jan-Feb through Dec",
        "tianjin_hydro_handling": "missing hydro reconstructed as nonnegative remainder after subtracting reported thermal, nuclear, wind, solar from reported total; not set to zero",
        "paired_NBS_vs_Ember": nbs_minus_ember,
        "models": scores.to_dict("records"),
        "limitations": [
            "Official MEE target is an annual consumption-accounting CO2 factor, not generation-side hourly CIF.",
            "NBS-derived monthly features cover above-designated-size enterprises, not all plants; monthly scope is incomplete relative to annual all-plant totals.",
            "The OREO/NBS and MEE files are retrospective snapshots with no verified as-of publication/revision vintage for each feature year.",
            "Tianjin hydro is not observed in the monthly table and is represented by an unresolved residual feature.",
            "Only two temporally forward target years are evaluated; LOPO provinces are not independent systems.",
            "No hourly shape, hourly coverage, threshold-risk prediction, or consumption-flow mechanism is evaluated.",
        ],
        "sha256": {
            str(MONTHLY.relative_to(ROOT)): sha(MONTHLY),
            str(ANNUAL.relative_to(ROOT)): sha(ANNUAL),
            str(FACTORS.relative_to(ROOT)): sha(FACTORS),
            str(EMBER.relative_to(ROOT)): sha(EMBER),
            "docs/V328_SCOPE_AWARE_NBS_TRANSFER_PROTOCOL.md": sha(PROTOCOL),
            "scripts/v328_scope_aware_nbs_transfer.py": sha(Path(__file__)),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(scores.to_string(index=False))
    print(json.dumps({"paired_NBS_vs_Ember": nbs_minus_ember,
                      "paired_scope_aware_vs_nbs_five": scope_minus_nbs,
                      "nbs_feature_rows": len(nbs_audit)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
