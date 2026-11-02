"""Apply the frozen source-pseudotarget risk model to sealed China monthly inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_loco_public_input_support_risk import X_COLUMNS, scores, unique_months
from analyze_expansion_public_support_risk import load_common_and_new, OLD_DATA, NEW_DATA
from run_source_pseudotarget_risk_meta import (C, METHODS, SUPPORT_COLUMNS, METRICS,
                                                make_domain_features, unobserved_group_count)
import run_energy_charts_loco as base


ROOT = Path(__file__).resolve().parents[1]
RISK_PROTOCOL = ROOT / "docs/SOURCE_PSEUDOTARGET_RISK_META_PROTOCOL_20261101.md"
RISK_RESULTS = ROOT / "docs/evidence/energy-charts-czech-expansion-r4/monthly_metrics.csv"
CHINA_MONTHS = ROOT / "docs/evidence/china-province-public-inference-r2/accepted_months.csv"
NBS_PANEL = ROOT / "results/v394-nbs-all-province-long-panel-r1/provincial_mix_and_closure.csv"
NBS_MANIFEST = ROOT / "results/v394-nbs-all-province-long-panel-r1/manifest.json"
NBS_SEAL = ROOT / "docs/evidence/shanghai-public-inference-r2/nbs-public-input-seal-r1.json"
MONTH_GROUPS = ("thermal", "hydro", "nuclear", "wind", "solar")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def china_features(months: pd.DataFrame, origin: str, source: np.ndarray) -> pd.DataFrame:
    origin_period = pd.Period(origin, freq="M")
    records = []
    for province, block in months.groupby("province", sort=True):
        block = block.copy()
        block["period"] = pd.PeriodIndex(block.period.astype(str), freq="M")
        block = block.set_index("period").sort_index()
        periods = [origin_period - lag for lag in range(7, 1, -1)]  # oldest to newest: p-7 ... p-2
        if not set(periods).issubset(block.index):
            raise ValueError(f"{province}: incomplete six-month t-2..t-7 input window")
        selected = block.loc[periods, list(MONTH_GROUPS)].astype(float)
        totals = selected.sum(axis=1)
        if (selected < 0).any().any() or (totals <= 0).any():
            raise ValueError(f"{province}: invalid generation values in the legal input window")
        shares = selected.div(totals, axis=0)
        row: dict[str, float | str | int] = {"province": province}
        for lag in range(2, 8):
            position = 7 - lag
            for group in MONTH_GROUPS:
                row[f"{group}_lag{lag}"] = float(shares.iloc[position][group])
        thermal = shares.thermal.to_numpy(float)
        total = totals.to_numpy(float)
        row.update({
            "thermal_mean": float(thermal.mean()),
            "thermal_std": float(thermal.std(ddof=0)),
            "thermal_change": float(thermal[-1] - thermal[0]),
            "total_cv": float(total.std(ddof=0) / total.mean()),
            "observations": 1.0,
            "unobserved_group_count": 0,
            "month_sin": float(np.sin(2 * np.pi * origin_period.month / 12.)),
            "month_cos": float(np.cos(2 * np.pi * origin_period.month / 12.)),
        })
        records.append(row)
    target = pd.DataFrame(records)
    support = scores(source, target[list(X_COLUMNS)].to_numpy(float))
    for column, values in support.items():
        target[column] = values
    return target


def fit_full_risk_models(panels: dict[str, pd.DataFrame], monthly_metrics: pd.DataFrame):
    countries = ["NL", "ES", "BE", "AT", "SE", "CZ"]
    feature_columns = [*X_COLUMNS, *SUPPORT_COLUMNS, "unobserved_group_count", "month_sin", "month_cos"]
    source_vectors = np.vstack([unique_months(panels[country], 2022)[list(X_COLUMNS)].to_numpy(float)
                                for country in countries])
    fitted = {}
    for method in METHODS:
        outcomes = monthly_metrics.loc[monthly_metrics.method.eq(method)]
        training_blocks = []
        for country in countries:
            target_features = make_domain_features(country, countries, panels)
            target_outcomes = outcomes.loc[outcomes.target_country.eq(country),
                                           ["target_country", "month", "coverage90"]]
            block = target_features.merge(target_outcomes, on=["target_country", "month"],
                                          how="inner", validate="one_to_one")
            block["event"] = block.coverage90.lt(.85).astype(int)
            training_blocks.append(block)
        training = pd.concat(training_blocks, ignore_index=True)
        p0 = float(training.event.mean())
        range_model = make_pipeline(StandardScaler(), LogisticRegression(
            C=C, l1_ratio=0, solver="lbfgs", max_iter=2000, random_state=17023))
        range_model.fit(training[["mean_range_excursion_z"]], training.event)
        full_model = make_pipeline(StandardScaler(), LogisticRegression(
            C=C, l1_ratio=0, solver="lbfgs", max_iter=2000, random_state=17023))
        full_model.fit(training[feature_columns], training.event)
        fitted[method] = {"prevalence": p0, "range": range_model, "full": full_model,
                          "feature_columns": feature_columns}
    return fitted, source_vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin-month", default="2026-11")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    months = pd.read_csv(CHINA_MONTHS)
    if months.province.nunique() != 31:
        raise ValueError("expected 31 Chinese provinces in the sealed public-input artifact")
    panels = load_common_and_new()
    outcomes = pd.read_csv(RISK_RESULTS)
    fitted, source_vectors = fit_full_risk_models(panels, outcomes)
    target = china_features(months, args.origin_month, source_vectors)
    rows = []
    for method, models in fitted.items():
        full = np.clip(models["full"].predict_proba(target[models["feature_columns"]])[:, 1], .01, .99)
        ranges = np.clip(models["range"].predict_proba(target[["mean_range_excursion_z"]])[:, 1], .01, .99)
        for i, row in target.iterrows():
            rows.append({"province": row.province, "origin_month": args.origin_month,
                         "forecast_method": method,
                         "source_meta_constant_risk": models["prevalence"],
                         "source_range_only_risk": float(ranges[i]),
                         "source_full_meta_risk": float(full[i]),
                         "mean_range_excursion_z": float(row.mean_range_excursion_z),
                         "outside_coordinate_count": int(row.outside_coordinate_count),
                         "nearest_source_month_zrms": float(row.nearest_source_month_zrms),
                         "source_shrunk_mahalanobis": float(row.source_shrunk_mahalanobis),
                         "unobserved_group_count": int(row.unobserved_group_count)})
    forecast = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    forecast.to_csv(args.output / "province_risk_scores.csv", index=False)
    input_paths = [CHINA_MONTHS, NBS_PANEL, NBS_MANIFEST, NBS_SEAL, RISK_RESULTS, RISK_PROTOCOL,
                   OLD_DATA / "acquisition_manifest.json", OLD_DATA / "audit.json",
                   *(OLD_DATA / f"{code}_2022_2024.json" for code in base.COUNTRIES),
                   NEW_DATA / "cz_acquisition_manifest.json", NEW_DATA / "cz_2022_2024.json",
                   NEW_DATA / "acquisition_manifest.json", NEW_DATA / "rs_2022_2024.json"]
    code_paths = [Path(__file__).resolve(), ROOT / "scripts/run_source_pseudotarget_risk_meta.py",
                  ROOT / "scripts/analyze_loco_public_input_support_risk.py",
                  ROOT / "scripts/analyze_expansion_public_support_risk.py",
                  ROOT / "scripts/run_energy_charts_loco.py",
                  ROOT / "scripts/run_energy_charts_czech_expansion.py",
                  ROOT / "scripts/run_energy_charts_serbia_expansion.py",
                  ROOT / "scripts/run_episodic_coldstart_transfer.py",
                  ROOT / "src/transcif/research/multicountry.py",
                  ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    summary = {
        "analysis": "no-label application of source-pseudotarget undercoverage risk model to China public monthly inputs",
        "origin_month": args.origin_month,
        "input_window": [str(pd.Period(args.origin_month, freq="M") - lag) for lag in range(7, 1, -1)],
        "province_count": int(months.province.nunique()),
        "target_hourly_labels_read": False,
        "china_target_coverage_or_accuracy_measured": False,
        "risk_models": "full logistic models refit on all six source-country pseudo-target tasks; C=0.05 fixed",
        "risk_event_definition": "source-calibrated 90% interval month coverage <85% on Energy-Charts proxy tasks",
        "interpretation": "model-generated extrapolation diagnostics only; not calibrated China probabilities or accuracy claims",
        "caveats": [
            "trained on six European domains and one proxy target year",
            "China NBS above-designated-size generation differs from Energy-Charts source scope",
            "all 31 China province features may be outside the source joint support",
            "the fitted probability model may extrapolate beyond its six-domain training support",
        ],
        "input_sha256": {str(path): sha256(path) for path in input_paths},
        "code_sha256": {str(path): sha256(path) for path in code_paths},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(forecast.groupby("forecast_method").agg(
        mean_full_risk=("source_full_meta_risk", "mean"),
        min_full_risk=("source_full_meta_risk", "min"),
        max_full_risk=("source_full_meta_risk", "max"),
        mean_range_excursion=("mean_range_excursion_z", "mean"),
        provinces_any_coordinate_oos=("outside_coordinate_count", lambda x: int(x.gt(0).sum())),
    ).to_string())


if __name__ == "__main__":
    main()
