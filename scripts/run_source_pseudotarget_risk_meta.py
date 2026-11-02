"""Nested country-held-out evaluation of source-pseudotarget undercoverage risk."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_loco_public_input_support_risk import X_COLUMNS, scores, unique_months
from analyze_expansion_public_support_risk import (load_common_and_new, CZ_RESULTS, RS_RESULTS,
                                                   OLD_DATA, NEW_DATA)
import run_energy_charts_loco as base
import run_energy_charts_serbia_expansion as serbia


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/SOURCE_PSEUDOTARGET_RISK_META_PROTOCOL_20261101.md"
METRICS = ROOT / "docs/evidence/energy-charts-czech-expansion-r4/monthly_metrics.csv"
METHODS = ("ratio_recomposition", "factor_history")
SUPPORT_COLUMNS = ("nearest_source_month_zrms", "source_shrunk_mahalanobis",
                   "mean_range_excursion_z", "outside_coordinate_count")
MODEL_NAMES = ("constant", "range_logistic", "full_meta_logistic")
C = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unobserved_group_count(code: str) -> int:
    mapping = serbia.RS_MAP if code == "RS" else base.TYPE_MAP[code.lower()]
    groups = {
        "thermal": any(mapping[name] for name in ("coal", "gas", "oil", "other")),
        "hydro": bool(mapping["hydro"]),
        "nuclear": bool(mapping["nuclear"]),
        "wind": bool(mapping["wind"]),
        "solar": bool(mapping["solar"]),
    }
    return int(sum(not observed for observed in groups.values()))


def make_domain_features(target: str, roster: list[str], panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sources = [country for country in roster if country != target]
    source = np.vstack([unique_months(panels[country], 2022)[list(X_COLUMNS)].to_numpy(float)
                        for country in sources])
    months = unique_months(panels[target], 2024, expected=12)
    public = months[list(X_COLUMNS)].reset_index(drop=True)
    support = scores(source, public.to_numpy(float))
    out = public.copy()
    for column, values in support.items():
        out[column] = values
    out["unobserved_group_count"] = unobserved_group_count(target)
    month = months.month.to_numpy(float)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.)
    out["target_country"] = target
    out["month"] = month.astype(int)
    out["source_country_month_count"] = len(source)
    return out


def fit_probability_model(train_x: pd.DataFrame, train_y: np.ndarray,
                          test_x: pd.DataFrame, model_name: str) -> np.ndarray:
    prevalence = float(np.mean(train_y))
    if model_name == "constant" or np.unique(train_y).size < 2:
        return np.full(len(test_x), prevalence, dtype=float)
    columns = ["mean_range_excursion_z"] if model_name == "range_logistic" else list(FEATURE_COLUMNS)
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=C, l1_ratio=0, solver="lbfgs", max_iter=2000, random_state=17023))
    model.fit(train_x[columns], train_y)
    return np.clip(model.predict_proba(test_x[columns])[:, 1], .01, .99)


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    result = {"n": int(len(y)), "events": int(np.sum(y)),
              "event_rate": float(np.mean(y)), "mean_predicted_risk": float(np.mean(p)),
              "brier": float(brier_score_loss(y, p)),
              "log_loss": float(log_loss(y, p, labels=[0, 1])),
              "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
              "average_precision": float(average_precision_score(y, p)) if np.sum(y) else None}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panels = load_common_and_new()
    countries = ["NL", "ES", "BE", "AT", "SE", "CZ"]
    metrics_frame = pd.read_csv(METRICS)
    metrics_frame = metrics_frame.loc[metrics_frame.method.isin(METHODS)].copy()
    features = {country: make_domain_features(country, countries, panels) for country in countries}
    feature_columns = [*X_COLUMNS, *SUPPORT_COLUMNS, "unobserved_group_count", "month_sin", "month_cos"]
    global FEATURE_COLUMNS
    FEATURE_COLUMNS = feature_columns

    predictions: list[dict] = []
    for method in METHODS:
        outcomes = metrics_frame.loc[metrics_frame.method.eq(method)].copy()
        targets = []
        for country in countries:
            block = features[country].merge(
                outcomes.loc[outcomes.target_country.eq(country), ["target_country", "month", "coverage90", "mae"]],
                on=["target_country", "month"], how="inner", validate="one_to_one")
            if len(block) != 12:
                raise ValueError(f"{country}/{method}: expected 12 paired target months")
            block["undercoverage_85"] = block.coverage90.lt(.85).astype(int)
            targets.append(block)
        data = pd.concat(targets, ignore_index=True)
        for heldout in countries:
            train = data.loc[data.target_country.ne(heldout)].copy()
            test = data.loc[data.target_country.eq(heldout)].copy()
            y_train = train.undercoverage_85.to_numpy(int)
            y_test = test.undercoverage_85.to_numpy(int)
            for model_name in MODEL_NAMES:
                p = fit_probability_model(train, y_train, test, model_name)
                for index, (_, row) in enumerate(test.iterrows()):
                    predictions.append({
                        "method": method, "risk_model": model_name,
                        "heldout_country": heldout, "month": int(row.month),
                        "event_undercoverage_85": int(y_test[index]),
                        "predicted_risk": float(p[index]),
                        "mae": float(row.mae), "coverage90": float(row.coverage90),
                        "mean_range_excursion_z": float(row.mean_range_excursion_z),
                        "unobserved_group_count": int(row.unobserved_group_count),
                        "meta_training_countries": 5,
                        "meta_training_rows": int(len(train)),
                    })
    prediction_frame = pd.DataFrame(predictions)
    domain_rows = []
    for (method, model_name, country), group in prediction_frame.groupby(
        ["method", "risk_model", "heldout_country"]
    ):
        result = metrics(group.event_undercoverage_85.to_numpy(int), group.predicted_risk.to_numpy(float))
        domain_rows.append({"method": method, "risk_model": model_name,
                            "heldout_country": country, **result})
    summary_rows = []
    for (method, model_name), group in prediction_frame.groupby(["method", "risk_model"]):
        result = metrics(group.event_undercoverage_85.to_numpy(int), group.predicted_risk.to_numpy(float))
        matching_domains = [row for row in domain_rows
                            if row["method"] == method and row["risk_model"] == model_name]
        defined_auc = [row["roc_auc"] for row in matching_domains if row["roc_auc"] is not None]
        summary_rows.append({"method": method, "risk_model": model_name, **result,
                             "domain_macro_brier": float(np.mean([row["brier"] for row in matching_domains])),
                             "domain_macro_log_loss": float(np.mean([row["log_loss"] for row in matching_domains])),
                             "mean_defined_within_domain_auc": float(np.mean(defined_auc)) if defined_auc else None,
                             "domains_with_defined_auc": len(defined_auc),
                             "pooled_auc_is_primary": False})

    # RS is not part of the primary cohort. Score it after fitting on all six
    # source pseudo-target folds; its response is joined only for stress reporting.
    rs_feature = make_domain_features("RS", ["NL", "ES", "BE", "AT", "SE", "RS"], panels)
    rs_outcomes = pd.read_csv(RS_RESULTS)
    rs_rows = []
    for method in METHODS:
        train_outcomes = metrics_frame.loc[metrics_frame.method.eq(method)].copy()
        blocks = []
        for country in countries:
            blocks.append(features[country].merge(
                train_outcomes.loc[train_outcomes.target_country.eq(country),
                                   ["target_country", "month", "coverage90"]],
                on=["target_country", "month"], how="inner", validate="one_to_one"))
        risk_train = pd.concat(blocks, ignore_index=True)
        y_train = risk_train.coverage90.lt(.85).astype(int).to_numpy()
        observed = rs_outcomes.loc[(rs_outcomes.target_country.eq("RS")) &
                                   rs_outcomes.method.eq(method)].copy()
        rs_test = rs_feature.merge(observed[["target_country", "month", "coverage90", "mae"]],
                                   on=["target_country", "month"], how="inner", validate="one_to_one")
        y_rs = rs_test.coverage90.lt(.85).astype(int).to_numpy()
        for model_name in MODEL_NAMES:
            p = fit_probability_model(risk_train, y_train, rs_test, model_name)
            for i, (_, row) in enumerate(rs_test.iterrows()):
                rs_rows.append({"method": method, "risk_model": model_name, "target_country": "RS",
                                "month": int(row.month), "predicted_risk": float(p[i]),
                                "event_undercoverage_85": int(y_rs[i]), "coverage90": float(row.coverage90),
                                "mae": float(row.mae), "unobserved_group_count": int(row.unobserved_group_count)})

    args.output.mkdir(parents=True, exist_ok=False)
    prediction_frame.to_csv(args.output / "nested_country_predictions.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output / "overall_scores.csv", index=False)
    pd.DataFrame(domain_rows).to_csv(args.output / "heldout_country_scores.csv", index=False)
    rs_frame = pd.DataFrame(rs_rows)
    rs_frame.to_csv(args.output / "rs_missing_schema_stress.csv", index=False)
    # Summarize RS alert probability and response only; never pool it with the
    # primary six-domain score table.
    rs_summary = rs_frame.groupby(["method", "risk_model"], as_index=False).agg(
        mean_predicted_risk=("predicted_risk", "mean"), actual_event_rate=("event_undercoverage_85", "mean"),
        mean_coverage90=("coverage90", "mean"), mean_mae=("mae", "mean"),
        unobserved_group_count=("unobserved_group_count", "first"))
    rs_summary.to_csv(args.output / "rs_stress_summary.csv", index=False)

    inputs = [PROTOCOL, METRICS, CZ_RESULTS, RS_RESULTS,
              OLD_DATA / "acquisition_manifest.json", OLD_DATA / "audit.json",
              *(OLD_DATA / f"{code}_2022_2024.json" for code in base.COUNTRIES),
              NEW_DATA / "cz_acquisition_manifest.json", NEW_DATA / "cz_2022_2024.json",
              NEW_DATA / "acquisition_manifest.json", NEW_DATA / "rs_2022_2024.json",
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
        "analysis": "nested country-held-out source-pseudotarget risk learning",
        "primary_countries": countries, "target_year": 2024,
        "event": "country-month source-calibrated 90% interval coverage <85%",
        "risk_features": feature_columns, "fixed_logistic_C": C,
        "outer_validation": "leave one entire country out; train meta-risk model on other five pseudo-target countries",
        "risk_model_fitted_without_outer_target_labels": True,
        "target_monthly_outcomes_used_only_as_heldout_evaluation": True,
        "rs": "separate missing-schema stress evaluation; excluded from primary risk scores",
        "all_domains_previously_viewed": True,
        "limits": [
            "six domains and one target year; nested cross-fitting is development evidence, not blind confirmation",
            "target labels are fixed-factor Energy-Charts proxies and monthly inputs are reconstructed, not issued vintages",
            "risk outcomes are correlated by country; 72 month rows do not equal 72 independent domains",
            "logistic probability calibration may be unstable with five meta-training domains",
        ],
        "sha256": {str(path): sha256(path) for path in inputs},
        "runner_sha256": sha256(Path(__file__).resolve()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("RS stress:")
    print(rs_summary.to_string(index=False))


if __name__ == "__main__":
    FEATURE_COLUMNS: list[str]
    main()
