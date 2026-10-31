"""V334: test lagged provincial thermal coal-consumption rates in annual EF transfer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

import v300_official_provincial_ef_mix_transfer as v300
import v327_nbs_mix_transfer as v327
import v329_china_region_block_coldstart as v329


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V334_LOCAL_THERMAL_HEATRATE_TRANSFER_PROTOCOL.md"
FACTORS = ROOT / "data/china/mee_provincial_grid_co2_factors_2021_2023.csv"
MONTHLY = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
EMBER = ROOT / "results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
RATES = ROOT / "data/china/cec_province_thermal_coal_consumption_rates_2020_2022.csv"
OUT = ROOT / "results/v334-local-thermal-heatrate-transfer-r2"
REGIONS = v329.REGIONS
NBS_GROUPS = tuple(v327.GROUPS)
EMBER_GROUPS = tuple(v300.COARSE)
RATE_COLS = ("generation_coal_consumption_gce_kwh", "supply_coal_consumption_gce_kwh")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(frame: pd.DataFrame, col: str) -> dict[str, float]:
    e = frame[col].to_numpy(float) - frame.actual.to_numpy(float)
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mean_signed_error": float(np.mean(e)), "p90_absolute_error": float(np.quantile(np.abs(e), .9))}


def add_rows(factors: pd.DataFrame, provinces: list[str], nbs_mix, ember_mix,
             rates: pd.DataFrame) -> pd.DataFrame:
    assignment = {p: r for r, members in REGIONS.items() for p in members}
    rate_dict = rates.set_index(["province", "year"]).to_dict("index")
    rows = []
    for target_year in (2021, 2022, 2023):
        feature_year = target_year - 1
        for p in provinces:
            nbs = nbs_mix[(p, feature_year)]
            ember_full = ember_mix[(p, feature_year)]
            ember5 = v300.aggregate_coarse(ember_full)
            r = rate_dict[(p, feature_year)]
            row = {"province": p, "region": assignment[p], "target_year": target_year,
                   "feature_year": feature_year,
                   "actual": float(factors.loc[p, f"ef_{target_year}_kgco2_kwh"])}
            row.update({f"nbs5_{k}": float(nbs[k]) for k in NBS_GROUPS})
            row.update({f"ember5_{k}": float(v) for k, v in zip(EMBER_GROUPS, ember5)})
            for rate_col in RATE_COLS:
                rate = float(r[rate_col])
                row[f"{rate_col}_gce_kwh"] = rate
                row[f"nbs5_{rate_col}_thermal_scaled"] = float(nbs["thermal"] * rate / 300.0)
                row[f"ember5_{rate_col}_thermal_scaled"] = float(ember5[0] * rate / 300.0)
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate(fit_rows: pd.DataFrame, target_rows: pd.DataFrame, methods: dict[str, tuple[str, ...]],
             grouping: str, held_unit: str) -> list[dict[str, object]]:
    out = []
    for method, columns in methods.items():
        x = fit_rows[list(columns)].to_numpy(float)
        y = fit_rows.actual.to_numpy(float)
        beta, _ = nnls(x, y)
        pred = target_rows[list(columns)].to_numpy(float) @ beta
        for i, (_, row) in enumerate(target_rows.iterrows()):
            out.append({"split_type": grouping, "held_unit": held_unit,
                        "province": row.province, "region": row.region,
                        "target_year": int(row.target_year), "method": method,
                        "actual": float(row.actual), "prediction": float(pred[i]),
                        "beta": json.dumps(dict(zip(columns, beta.tolist())))})
    return out


def cluster_bootstrap_delta(frame: pd.DataFrame, method_a: str, method_b: str,
                            cluster_col: str, seed: int) -> dict[str, object]:
    index_cols = (["province", "target_year"] if cluster_col == "province"
                  else [cluster_col, "province", "target_year"])
    pivot = frame.pivot_table(index=index_cols, columns="method",
                              values="abs_error", aggfunc="first").reset_index()
    cluster_losses = pivot.groupby(cluster_col).apply(
        lambda g: float(np.mean(g[method_a] - g[method_b])), include_groups=False)
    d = cluster_losses.to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(50000)])
    return {"mean_cluster_delta_method_a_minus_b": float(d.mean()),
            "bootstrap_95pct": np.quantile(boot, [.025, .975]).tolist(),
            "clusters_method_a_better": int((d < 0).sum()),
            "per_cluster_delta": dict(zip(cluster_losses.index.astype(str), d.tolist()))}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen results: {OUT}")
    factors = pd.read_csv(FACTORS).set_index("province")
    provinces = sorted(factors.index.astype(str))
    rates = pd.read_csv(RATES)
    expected = {(p, y) for p in provinces for y in (2020, 2021, 2022)}
    observed = set(zip(rates.province, rates.year))
    if expected != observed:
        raise ValueError(f"rate-table support mismatch: missing={expected-observed}, extra={observed-expected}")
    nbs_mix, _ = v327.nbs_annual_mix(MONTHLY, provinces)
    ember_mix = v300.monthly_mix(EMBER)
    rows = add_rows(factors, provinces, nbs_mix, ember_mix, rates)

    methods: dict[str, tuple[str, ...]] = {"NBS5_mix": tuple(f"nbs5_{g}" for g in NBS_GROUPS),
                                          "Ember5_mix": tuple(f"ember5_{g}" for g in EMBER_GROUPS)}
    for source, groups in (("nbs5", NBS_GROUPS), ("ember5", EMBER_GROUPS)):
        for rate_col in RATE_COLS:
            methods[f"{source}_with_{rate_col}"] = (
                f"{source}_{rate_col}_thermal_scaled",
                *(f"{source}_{g}" for g in groups[1:]))

    predictions = []
    # Province LOPO: target province excluded, all earlier target years only.
    for province in provinces:
        for target_year in (2022, 2023):
            fit = rows.loc[rows.target_year.lt(target_year) & rows.province.ne(province)]
            target = rows.loc[rows.province.eq(province) & rows.target_year.eq(target_year)]
            predictions.extend(evaluate(fit, target, methods, "province_LOPO", province))
    # Whole-region cold start: all provinces in held block excluded from every fit.
    for region, members in REGIONS.items():
        for target_year in (2022, 2023):
            fit = rows.loc[rows.target_year.lt(target_year) & rows.region.ne(region)]
            target = rows.loc[rows.region.eq(region) & rows.target_year.eq(target_year)]
            predictions.extend(evaluate(fit, target, methods, "whole_region_LOCO", region))
    pred = pd.DataFrame(predictions)
    pred["abs_error"] = (pred.prediction - pred.actual).abs()

    score_rows = []
    for split_type in pred.split_type.unique():
        sub = pred.loc[pred.split_type.eq(split_type)]
        for evaluation, test in [("pooled_2022_2023", sub),
                                 *((str(year), sub.loc[sub.target_year.eq(year)]) for year in (2022, 2023))]:
            for method, group in test.groupby("method", sort=False):
                score_rows.append({"split_type": split_type, "evaluation": evaluation,
                                   "weighting": "province_weighted", "method": method,
                                   **score(group, "prediction")})
                if split_type == "whole_region_LOCO":
                    reg_scores = [score(g, "prediction") for _, g in group.groupby("held_unit")]
                    score_rows.append({"split_type": split_type, "evaluation": evaluation,
                                       "weighting": "region_equal", "method": method,
                                       **{k: float(np.mean([s[k] for s in reg_scores])) for k in reg_scores[0]}})
    score_df = pd.DataFrame(score_rows)
    comparisons = {}
    for split_type, cluster in (("province_LOPO", "province"), ("whole_region_LOCO", "region")):
        sub = pred.loc[pred.split_type.eq(split_type)].copy()
        sub[cluster] = sub[cluster]
        for source in ("nbs5", "ember5"):
            base = "NBS5_mix" if source == "nbs5" else "Ember5_mix"
            for rate_col in RATE_COLS:
                adjusted = f"{source}_with_{rate_col}"
                comparisons[f"{split_type}_{adjusted}_minus_{base}"] = cluster_bootstrap_delta(
                    sub, adjusted, base, cluster, 334 + len(comparisons))

    OUT.mkdir(parents=True)
    pred.to_csv(OUT / "predictions.csv", index=False)
    score_df.to_csv(OUT / "model_scores.csv", index=False)
    summary = {"status": "LAGGED_LOCAL_THERMAL_RATE_ANNUAL_MEE_TRANSFER", "n_provinces": len(provinces),
               "n_regions": len(REGIONS), "target_years": [2022, 2023],
               "rate_columns": list(RATE_COLS), "paired_cluster_bootstrap": comparisons,
               "models": score_rows,
               "limitations": ["Targets are annual MEE consumption-side factors, not hourly CIF.",
                               "CEC rate tables cover 6000-kW-and-above plants; NBS monthly features use above-designated-size enterprises.",
                               "CEC Yearbook 2024 is a retrospective vintage published after target years; as-of forecast legality is not established.",
                               "Standard-coal consumption rate is a conversion-efficiency proxy, not a measured CO2 emission factor.",
                               "Only two forward target years and seven correlated region clusters."]}
    summary["sha256"] = {str(path.relative_to(ROOT)): sha(path) for path in (FACTORS, MONTHLY, EMBER, RATES, PROTOCOL, Path(__file__))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(score_df.to_string(index=False))
    print(json.dumps(comparisons, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
