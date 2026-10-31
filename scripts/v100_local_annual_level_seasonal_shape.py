"""V100: separate target-local annual fuel level from source-transferred month shape."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
SH_NBS = ROOT / "results/v378-shanghai-nbs-export-audit-r1/reported_monthly_values.csv"
USER_EXPORTS = (
    ROOT / "data/nbs_user_exports/shanghai/thermal.csv",
    ROOT / "data/nbs_user_exports/shanghai/wind.csv",
    ROOT / "data/nbs_user_exports/shanghai/solar.csv",
)
PROTOCOL = ROOT / "docs/V100_LOCAL_ANNUAL_LEVEL_SEASONAL_SHAPE_PROTOCOL.md"
OUT = ROOT / "results/v100-local-annual-level-seasonal-shape-r3"
FUELS = ("Coal", "Gas", "Other fossil", "Bioenergy", "Hydro", "Nuclear", "Wind", "Solar")
FEATURES = ("share_Hydro", "share_Nuclear", "share_Wind", "share_Solar", "thermal_share",
            "month_sin", "month_cos", "year", "lagged_annual_side_share")
METHODS = ("pooled_month", "local_annual_anchor", "annual_plus_source_season",
           "annual_coarse_ridge", "annual_coarse_ET")
SEED = 20261031


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def contract() -> dict:
    files = (INPUT, SH_NBS, PROTOCOL, *USER_EXPORTS)
    return {"version": 100, "protocol_sha256": sha(PROTOCOL),
            "script_sha256": sha(Path(__file__)),
            "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in files},
            "fit_years": [2017, 2021], "calibration_years": [2022, 2023],
            "test_years": [2024, 2025], "outer_unit": "province",
            "threshold_gco2_kwh": 300, "target": "(Coal+Gas)/coarse_thermal",
            "seed": SEED, "target_hourly_labels_used": False,
            "independent_confirmation": False, "official_annual_fuel_field_verified": False}


def make_panel() -> tuple[pd.DataFrame, dict[tuple[str, int], float]]:
    raw = pd.read_csv(INPUT)
    raw = raw.loc[raw["Province code"].notna() & raw["Province"].ne("China (total)")].copy()
    panel = raw.pivot(index=["Province code", "Province", "Date"], columns="Electricity source",
                       values="Generation (TWh)").reset_index()
    if set(FUELS + ("Total generation",)) - set(panel.columns):
        raise ValueError("V60 eight-fuel and generation total schema required")
    panel["Date"] = pd.to_datetime(panel["Date"], errors="raise")
    panel["year"] = panel["Date"].dt.year.astype(int)
    panel["month"] = panel["Date"].dt.month.astype(int)
    panel["thermal_TWh"] = panel[["Coal", "Gas", "Other fossil", "Bioenergy"]].sum(axis=1)
    panel["side_share"] = (panel["Coal"] + panel["Gas"]) / panel["thermal_TWh"].replace(0, np.nan)
    for fuel in ("Hydro", "Nuclear", "Wind", "Solar"):
        panel[f"share_{fuel}"] = panel[fuel] / panel["Total generation"].replace(0, np.nan)
    panel["thermal_share"] = panel["thermal_TWh"] / panel["Total generation"].replace(0, np.nan)
    angle = 2 * np.pi * panel["month"] / 12
    panel["month_sin"], panel["month_cos"] = np.sin(angle), np.cos(angle)
    annual = panel.groupby(["Province code", "year"], sort=True).apply(
        lambda frame: float(frame["Coal"].sum() + frame["Gas"].sum()) / float(frame["thermal_TWh"].sum()),
        include_groups=False)
    annual_lookup = {(str(province), int(year)): float(value)
                     for (province, year), value in annual.items() if np.isfinite(value)}
    panel["lagged_annual_side_share"] = [annual_lookup.get((str(province), int(year) - 1), np.nan)
                                         for province, year in zip(panel["Province code"], panel["year"])]
    return panel.dropna(subset=["side_share", "lagged_annual_side_share", *FEATURES]).copy(), annual_lookup


def fit_predict(train: pd.DataFrame, calibration: pd.DataFrame, target: pd.DataFrame) -> tuple[dict, dict]:
    month_mean = train.groupby("month")["side_share"].mean()
    residual_month = (train["side_share"] - train["lagged_annual_side_share"]).groupby(train["month"]).mean()
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(train[list(FEATURES)], train["side_share"])
    forest = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=25, max_features=.8,
                                 n_jobs=-1, random_state=SEED).fit(train[list(FEATURES)], train["side_share"])

    def predict(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        annual = frame["lagged_annual_side_share"].to_numpy(float)
        return {
            "pooled_month": frame["month"].map(month_mean).to_numpy(float),
            "local_annual_anchor": annual,
            "annual_plus_source_season": annual + frame["month"].map(residual_month).to_numpy(float),
            "annual_coarse_ridge": ridge.predict(frame[list(FEATURES)]),
            "annual_coarse_ET": forest.predict(frame[list(FEATURES)]),
        }

    cp = predict(calibration)
    tp = predict(target)
    calibration_meta = {}
    for method in METHODS:
        residual = np.abs(calibration["side_share"].to_numpy(float) - cp[method])
        # Equal numbers of months per calibration province; macro marginal conformal quantile.
        q_marginal = float(np.quantile(residual, .90, method="higher"))
        grouped = calibration.assign(abs_error=residual).groupby("Province code")["abs_error"].max().to_numpy(float)
        q_region_max = float(np.quantile(grouped, .90, method="higher"))
        calibration_meta[method] = {"q90_marginal": q_marginal, "q90_region_max": q_region_max,
                                    "calibration_rows": int(len(residual)),
                                    "calibration_regions": int(len(grouped))}
    return {"calibration": calibration_meta, "prediction": tp}, cp


def cluster_bootstrap(region_table: pd.DataFrame, metric: str, a: str, b: str) -> dict:
    paired = region_table.pivot(index="province_code", columns="method", values=metric).dropna(subset=[a, b])
    delta = (paired[a] - paired[b]).to_numpy(float)
    rng = np.random.default_rng(SEED)
    samples = rng.choice(delta, size=(20000, len(delta)), replace=True).mean(axis=1)
    return {"contrast": f"{a} minus {b}", "clusters": int(len(delta)), "mean_difference": float(delta.mean()),
            "bootstrap_95pct_percentile": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))],
            "negative_favors_first_method": True}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite V100 output: {OUT}")
    frozen = contract()
    OUT.mkdir(parents=True)
    write_json(OUT / "freeze.json", {"created_utc": datetime.now(timezone.utc).isoformat(), **frozen})
    panel, _ = make_panel()
    prediction_rows = []
    for region in sorted(panel["Province code"].astype(str).unique()):
        source = panel.loc[panel["Province code"].astype(str).ne(region)]
        train = source.loc[source["year"].between(2017, 2021)].copy()
        calibration = source.loc[source["year"].between(2022, 2023)].copy()
        target = panel.loc[panel["Province code"].astype(str).eq(region) & panel["year"].isin([2024, 2025])].copy()
        if not len(target) or not len(train) or not len(calibration):
            raise ValueError(f"Missing fixed source or test rows for {region}")
        fitted, _ = fit_predict(train, calibration, target)
        predictions, intervals = fitted["prediction"], fitted["calibration"]
        for pos, (_, row) in enumerate(target.iterrows()):
            truth = float(row["side_share"])
            thermal = float(row["thermal_TWh"])
            for method in METHODS:
                point = float(np.clip(predictions[method][pos], 0., 1.))
                cal = intervals[method]
                for scope, qkey in (("marginal", "q90_marginal"), ("region_max", "q90_region_max")):
                    q = cal[qkey]
                    lo, hi = max(0., point - q), min(1., point + q)
                    prediction_rows.append({"province_code": region, "province": row["Province"],
                        "year": int(row["year"]), "month": int(row["month"]), "method": method,
                        "calibration_scope": scope, "side_share_truth": truth, "side_share_prediction": point,
                        "thermal_TWh": thermal, "side_TWh_truth": truth * thermal,
                        "side_TWh_prediction": point * thermal, "absolute_side_TWh_error": abs(point-truth)*thermal,
                        "absolute_share_error": abs(point-truth), "lower_share": lo, "upper_share": hi,
                        "covered": bool(lo <= truth <= hi), "interval_width_share": hi-lo,
                        "interval_width_TWh": (hi-lo)*thermal,
                        "lagged_annual_side_share": float(row["lagged_annual_side_share"])})
    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty or predictions.duplicated(["province_code", "year", "month", "method", "calibration_scope"]).any():
        raise ValueError("V100 prediction grid is incomplete or duplicated")
    macro_rows, region_rows = [], []
    metric_source = predictions.loc[predictions.calibration_scope.eq("marginal")]
    for (province, method), group in metric_source.groupby(["province_code", "method"], sort=True):
        region_rows.append({"province_code": province, "province": group.province.iloc[0], "method": method,
            "months": int(len(group)), "share_mae": float(group.absolute_share_error.mean()),
            "share_rmse": float(np.sqrt(np.mean((group.side_share_prediction-group.side_share_truth)**2))),
            "side_energy_mae_TWh": float(group.absolute_side_TWh_error.mean()),
            "side_energy_wape": float(group.absolute_side_TWh_error.sum()/group.side_TWh_truth.sum()),
            "marginal_coverage90": float(group.covered.mean()), "marginal_width90_share": float(group.interval_width_share.mean()),
            "marginal_width90_TWh": float(group.interval_width_TWh.mean())})
    regions = pd.DataFrame(region_rows)
    for method, group in metric_source.groupby("method", sort=True):
        reg = regions.loc[regions.method.eq(method)]
        macro_rows.append({"method": method, "provinces": int(len(reg)),
            "equal_province_share_mae": float(reg.share_mae.mean()),
            "equal_province_share_rmse": float(np.sqrt(np.mean(reg.share_rmse**2))),
            "equal_province_side_energy_mae_TWh": float(reg.side_energy_mae_TWh.mean()),
            "generation_weighted_side_energy_wape": float(group.absolute_side_TWh_error.sum()/group.side_TWh_truth.sum()),
            "equal_province_coverage90": float(reg.marginal_coverage90.mean()),
            "worst_province_coverage90": float(reg.marginal_coverage90.min()),
            "equal_province_width90_share": float(reg.marginal_width90_share.mean())})
    macro = pd.DataFrame(macro_rows)
    group_rows = []
    for method, group in predictions.loc[predictions.calibration_scope.eq("region_max")].groupby("method"):
        reg = group.groupby(["province_code", "year"], as_index=False).agg(
            coverage90=("covered", "mean"), width90_share=("interval_width_share", "mean"),
            width90_TWh=("interval_width_TWh", "mean"))
        group_rows.append({"method": method, "equal_region_year_coverage90": float(reg.coverage90.mean()),
            "worst_region_year_coverage90": float(reg.coverage90.min()),
            "equal_region_year_width90_share": float(reg.width90_share.mean()),
            "equal_region_year_width90_TWh": float(reg.width90_TWh.mean())})
    group_macro = pd.DataFrame(group_rows)

    # Pair bootstrap uses province as the independent cluster and averages the two target years within each province.
    region_paired = regions.groupby(["province_code", "method"], as_index=False).agg(
        share_mae=("share_mae", "mean"), side_energy_mae_TWh=("side_energy_mae_TWh", "mean"))
    contrasts = [cluster_bootstrap(region_paired, "share_mae", "annual_plus_source_season", "local_annual_anchor"),
                 cluster_bootstrap(region_paired, "share_mae", "annual_plus_source_season", "pooled_month"),
                 cluster_bootstrap(region_paired, "side_energy_mae_TWh", "annual_plus_source_season", "local_annual_anchor")]
    # Verify the user's direct Shanghai coarse thermal export on the four overlapping 2025 months.
    nbs = pd.read_csv(SH_NBS)
    nbs = nbs.loc[nbs.province.eq("Shanghai") & nbs.technology.eq("thermal") & nbs.year.eq(2025)
                  & nbs.month.isin([9, 10, 11, 12]) & nbs.observation_kind.eq("current")].copy()
    sh_ember = panel.loc[panel["Province code"].astype(str).eq("SH") & panel.year.eq(2025)
                         & panel.month.isin([9, 10, 11, 12]), ["month", "thermal_TWh"]].copy()
    bridge = nbs.merge(sh_ember, on="month", validate="one_to_one")
    bridge["nbs_thermal_TWh"] = bridge.value_100m_kwh.astype(float) * .1
    bridge["difference_TWh"] = bridge.nbs_thermal_TWh - bridge.thermal_TWh
    bridge["relative_difference"] = bridge.difference_TWh / bridge.thermal_TWh
    bridge.to_csv(OUT / "shanghai_nbs_coarse_bridge.csv", index=False)

    predictions.to_csv(OUT / "monthly_predictions.csv", index=False)
    regions.to_csv(OUT / "province_metrics.csv", index=False)
    macro.to_csv(OUT / "macro_metrics.csv", index=False)
    group_macro.to_csv(OUT / "region_max_interval_metrics.csv", index=False)
    write_json(OUT / "paired_cluster_bootstrap.json", {"contrasts": contrasts, "bootstrap_replicates": 20000,
        "seed": SEED, "cluster_unit": "province"})
    summary = {"status": "RETROSPECTIVE_MONTHLY_HIDDEN_SIDE_PROXY_NO_HOURLY_VALIDATION",
        "provinces": int(panel["Province code"].nunique()), "test_month_rows": int(panel["year"].isin([2024, 2025]).sum()),
        "target_years": [2024, 2025], "methods": list(METHODS), "macro": macro.to_dict(orient="records"),
        "region_max_intervals": group_macro.to_dict(orient="records"), "shanghai_nbs_bridge_months": int(len(bridge)),
        "shanghai_nbs_thermal_relative_abs_difference_max": float(bridge.relative_difference.abs().max()),
        "interpretation": "Lagged local annual fuel structure is a useful level anchor in the secondary panel; source seasonality is a candidate shape transfer. The annual subtype input is not verified as an official target-region field; hourly CIF and event risk remain untested.",
        "limitations": ["Ember fuel shares are secondary estimates with no historical release-vintage chain.",
            "The lagged local annual coal-plus-gas share is an assumed information-budget arm, not verified official Chinese monthly NBS data.",
            "Metrics evaluate a monthly proxy share and side energy, not hourly CIF or hourly threshold-event count.",
            "Province-level conformal residual calibration does not guarantee coverage under arbitrary structural shift.",
            "Shanghai NBS coarse-category bridge covers only four 2025 months and validates coarse thermal alignment, not hidden coal/gas mix."]}
    write_json(OUT / "summary.json", summary)
    outputs = ["shanghai_nbs_coarse_bridge.csv", "monthly_predictions.csv", "province_metrics.csv",
        "macro_metrics.csv", "region_max_interval_metrics.csv", "paired_cluster_bootstrap.json", "summary.json"]
    write_json(OUT / "completed.json", {"freeze_sha256": sha(OUT / "freeze.json"),
        "output_sha256": {name: sha(OUT / name) for name in outputs},
        "target_hourly_labels_used": False, "independent_confirmation": False,
        "algorithm_breakthrough": False, "official_annual_fuel_field_verified": False})
    print(macro.to_string(index=False))
    print("\nRegion-max interval summary:\n", group_macro.to_string(index=False))
    print("\nShanghai NBS coarse bridge max absolute relative difference:", summary["shanghai_nbs_thermal_relative_abs_difference_max"])


if __name__ == "__main__":
    main()
