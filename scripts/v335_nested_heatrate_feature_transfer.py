"""V335: nested ridge-NNLS test of annual local heat-rate information."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import v300_official_provincial_ef_mix_transfer as v300
import v327_nbs_mix_transfer as v327
import v329_china_region_block_coldstart as v329
import v330_spatial_nested_ridge_transfer as v330
import v334_local_thermal_heatrate_transfer as v334


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V335_NESTED_HEATRATE_FEATURE_TRANSFER_PROTOCOL.md"
FACTORS = ROOT / "data/china/mee_provincial_grid_co2_factors_2021_2023.csv"
MONTHLY = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
EMBER = ROOT / "results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
RATES = ROOT / "data/china/cec_province_thermal_coal_consumption_rates_2020_2022.csv"
OUT = ROOT / "results/v335-nested-heatrate-feature-transfer-r1"
REGIONS = v329.REGIONS
NBS_GROUPS = tuple(v327.GROUPS)
EMBER_GROUPS = tuple(v300.COARSE)
LAMBDAS = tuple(v330.LAMBDAS)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(frame: pd.DataFrame, col: str) -> dict[str, float]:
    e = frame[col].to_numpy(float) - frame.actual.to_numpy(float)
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mean_signed_error": float(np.mean(e)), "p90_absolute_error": float(np.quantile(np.abs(e), .9))}


def choose_lambda(train: pd.DataFrame, target_year: int, cols: tuple[str, ...]) -> tuple[float, dict[str, float]]:
    validation_year = 2021 if target_year == 2022 else 2022
    validation = train.loc[train.target_year.eq(validation_year)]
    fit_pool = train.loc[train.target_year.lt(validation_year)]
    if validation_year == 2021:
        fit_pool = validation
    errors = {lam: [] for lam in LAMBDAS}
    for region in sorted(train.region.unique()):
        valid = validation.loc[validation.region.eq(region)]
        fit = fit_pool.loc[fit_pool.region.ne(region)]
        if fit.empty or valid.empty:
            continue
        x = fit[list(cols)].to_numpy(float); y = fit.actual.to_numpy(float)
        xv = valid[list(cols)].to_numpy(float); yv = valid.actual.to_numpy(float)
        for lam in LAMBDAS:
            beta = v330.fit_ridge_nnls(x, y, lam)
            errors[lam].extend(np.abs(yv - xv @ beta).tolist())
    losses = {f"{lam:g}": float(np.mean(e)) for lam, e in errors.items() if e}
    if not losses:
        raise RuntimeError("No inner spatial validation predictions")
    minimum = min(losses.values())
    selected = min(lam for lam in LAMBDAS if abs(losses[f"{lam:g}"] - minimum) <= 1e-12)
    return selected, losses


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen results: {OUT}")
    factors = pd.read_csv(FACTORS).set_index("province")
    provinces = sorted(factors.index.astype(str))
    rates = pd.read_csv(RATES)
    nbs_mix, _ = v327.nbs_annual_mix(MONTHLY, provinces)
    ember_mix = v300.monthly_mix(EMBER)
    rows = v334.add_rows(factors, provinces, nbs_mix, ember_mix, rates)

    schemas: dict[str, tuple[str, ...]] = {
        "NBS5_mix": tuple(f"nbs5_{g}" for g in NBS_GROUPS),
        "Ember5_mix": tuple(f"ember5_{g}" for g in EMBER_GROUPS),
    }
    for source, groups in (("nbs5", NBS_GROUPS), ("ember5", EMBER_GROUPS)):
        for rate_col in v334.RATE_COLS:
            schemas[f"{source}_{rate_col}"] = (
                f"{source}_{rate_col}_thermal_scaled",
                *(f"{source}_{g}" for g in groups[1:]))

    predictions, tuning = [], []
    for held_region, members in REGIONS.items():
        for target_year in (2022, 2023):
            train = rows.loc[rows.target_year.lt(target_year) & rows.region.ne(held_region)]
            target = rows.loc[rows.region.eq(held_region) & rows.target_year.eq(target_year)]
            if target.empty or train.region.eq(held_region).any():
                raise AssertionError("empty target or held-region leakage")
            for method, cols in schemas.items():
                lam, inner = choose_lambda(train, target_year, cols)
                beta = v330.fit_ridge_nnls(train[list(cols)].to_numpy(float), train.actual.to_numpy(float), lam)
                pred = target[list(cols)].to_numpy(float) @ beta
                tuning.append({"held_region": held_region, "target_year": target_year, "method": method,
                               "selected_lambda": lam, "inner_mae_by_lambda": json.dumps(inner),
                               "beta": json.dumps(dict(zip(cols, beta.tolist())))})
                for i, (_, row) in enumerate(target.iterrows()):
                    predictions.append({"province": row.province, "held_region": held_region,
                                        "target_year": target_year, "method": method,
                                        "actual": float(row.actual), "prediction": float(pred[i]),
                                        "selected_lambda": lam})
    pred = pd.DataFrame(predictions)
    pred["abs_error"] = (pred.prediction - pred.actual).abs()
    score_rows, region_year_rows = [], []
    for evaluation, sub in [("pooled_2022_2023", pred),
                            *((str(year), pred.loc[pred.target_year.eq(year)]) for year in (2022, 2023))]:
        for method, group in sub.groupby("method", sort=False):
            score_rows.append({"evaluation": evaluation, "weighting": "province_weighted", "method": method,
                               **score(group, "prediction")})
            reg = [score(g, "prediction") for _, g in group.groupby("held_region")]
            score_rows.append({"evaluation": evaluation, "weighting": "region_equal", "method": method,
                               **{k: float(np.mean([x[k] for x in reg])) for k in reg[0]}})
    for (region, year, method), group in pred.groupby(["held_region", "target_year", "method"]):
        region_year_rows.append({"held_region": region, "target_year": int(year), "method": method,
                                 "n_test_provinces": int(group.province.nunique()), **score(group, "prediction")})

    errors = pred.groupby("held_region").apply(
        lambda g: pd.Series({m: float(np.mean(g.loc[g.method.eq(m), "abs_error"])) for m in schemas}),
        include_groups=False)
    rng = np.random.default_rng(335)
    comparisons = {}
    for source in ("nbs5", "ember5"):
        base = "NBS5_mix" if source == "nbs5" else "Ember5_mix"
        for rate_col in v334.RATE_COLS:
            adjusted = f"{source}_{rate_col}"
            d = (errors[adjusted] - errors[base]).to_numpy(float)
            b = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(50000)])
            comparisons[f"{adjusted}_minus_{base}_region_mae"] = {
                "difference": float(d.mean()), "region_cluster_bootstrap_95pct": np.quantile(b, [.025, .975]).tolist(),
                "regions_adjusted_better": int((d < 0).sum()),
                "per_region_delta": dict(zip(errors.index, d.tolist()))}
    OUT.mkdir(parents=True)
    pred.to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(score_rows).to_csv(OUT / "model_scores.csv", index=False)
    pd.DataFrame(region_year_rows).to_csv(OUT / "region_year_scores.csv", index=False)
    pd.DataFrame(tuning).to_csv(OUT / "tuning.csv", index=False)
    summary = {"status": "NESTED_RIDGE_LOCAL_HEATRATE_WHOLE_REGION_ANNUAL_MEE_TRANSFER",
               "lambda_candidates": list(LAMBDAS), "paired_region_bootstrap": comparisons,
               "models": score_rows,
               "limitations": ["Targets are annual MEE consumption-side factors, not hourly CIF.",
                               "Only two forward years and seven correlated region clusters.",
                               "CEC yearbook values are retrospective; as-of forecast legality is unverified.",
                               "Standard-coal consumption is an efficiency proxy, not CO2 emission factor.",
                               "CEC and NBS plant-size scopes differ."]}
    summary["sha256"] = {str(path.relative_to(ROOT)): sha(path) for path in (FACTORS, MONTHLY, EMBER, RATES, PROTOCOL, Path(__file__))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(score_rows).to_string(index=False))
    print(json.dumps(comparisons, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
