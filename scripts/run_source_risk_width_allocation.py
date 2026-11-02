"""Test cross-fitted source risk scores as fixed-budget monthly interval allocators."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_energy_charts_czech_expansion as czech  # noqa: E402

EVIDENCE = ROOT / "docs/evidence/energy-charts-czech-expansion-r4"
RISK = ROOT / "results/source-pseudotarget-risk-meta-r2/nested_country_predictions.csv"
MONTHLY = EVIDENCE / "monthly_metrics.csv"
COUNTRIES = ("NL", "ES", "BE", "AT", "SE", "CZ")
METHODS = ("ratio_recomposition", "factor_history")
RISK_MODELS = ("constant", "range_logistic", "full_meta_logistic")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    old_manifest = json.loads(czech.OLD_MANIFEST.read_text())
    cz_manifest = json.loads(czech.CZ_MANIFEST.read_text())
    manifests = {code: old_manifest for code in czech.base.COUNTRIES}
    manifests["cz"] = {"countries": [{"country": "cz", "sha256": cz_manifest["sha256"]}]}
    prepared = {}
    for code in COUNTRIES:
        key = code.lower()
        raw, features, _ = czech.prepare_country(key, manifests[key])
        prepared[code] = (raw, features)

    risk_frame = pd.read_csv(RISK)
    risk_frame = risk_frame.loc[risk_frame.method.isin(METHODS)
                                & risk_frame.risk_model.isin(RISK_MODELS)].copy()
    monthly = pd.read_csv(MONTHLY)
    rows: list[dict] = []
    for country in COUNTRIES:
        raw, features = prepared[country]
        query = features.loc[features.year.eq(2024)].copy()
        truth_fuel = czech.fuel_for(raw, query)
        truth = truth_fuel @ czech.FACTORS / truth_fuel.sum(axis=1)
        times = raw.iloc[query.raw_row.to_numpy()].time.astype(str).reset_index(drop=True)
        if len(query) != len(times) or len(query) != len(truth):
            raise ValueError(f"{country}: target rows are misaligned")
        pred_path = EVIDENCE / f"{country}_heldout_predictions.csv.gz"
        seal = json.loads((EVIDENCE / "prediction_seal.json").read_text())
        expected = next(item["sha256"] for item in seal if item["target"] == country)
        if sha(pred_path) != expected:
            raise ValueError(f"sealed hourly predictions changed: {pred_path}")
        forecast = pd.read_csv(pred_path)
        if not forecast.time.astype(str).reset_index(drop=True).equals(times):
            raise ValueError(f"{country}: sealed timestamps do not match target truth")
        months = query.clock.dt.month.to_numpy(int)

        for method in METHODS:
            point = forecast[method].to_numpy(float)
            base_rows = monthly.loc[monthly.target_country.eq(country)
                                    & monthly.method.eq(method)].sort_values("month")
            widths = base_rows.width90.to_numpy(float)
            if len(widths) != 12 or not np.allclose(widths, widths[0], rtol=0, atol=1e-9):
                raise ValueError(f"{country}/{method}: expected one frozen radius for all months")
            base_radius = float(widths[0] / 2.)
            errors = np.abs(truth - point)

            for risk_model in RISK_MODELS:
                probs = risk_frame.loc[risk_frame.heldout_country.eq(country)
                                       & risk_frame.method.eq(method)
                                       & risk_frame.risk_model.eq(risk_model)].sort_values("month")
                if len(probs) != 12 or probs.month.astype(int).tolist() != list(range(1, 13)):
                    raise ValueError(f"{country}/{method}/{risk_model}: incomplete monthly risks")
                p = probs.predicted_risk.to_numpy(float)
                multiplier = np.exp(p - p.mean())
                multiplier /= multiplier.mean()
                radius_by_month = {month: base_radius for month in range(1, 13)}
                if risk_model != "constant":
                    radius_by_month = {month: base_radius * float(multiplier[month - 1])
                                       for month in range(1, 13)}
                radii = np.array([radius_by_month[int(month)] for month in months])
                for month in range(1, 13):
                    mask = months == month
                    radius = radius_by_month[month]
                    err = errors[mask]
                    rows.append({
                        "country": country, "method": method, "risk_model": risk_model,
                        "month": month, "hours": int(mask.sum()),
                        "predicted_risk": float(p[month - 1]),
                        "radius": radius, "width": 2. * radius,
                        "coverage90": float(np.mean(err <= radius)),
                        "interval_score90": float(np.mean(2. * radius + 20. * np.maximum(err - radius, 0.))),
                    })
                if not np.isclose(float(np.mean(list(radius_by_month.values()))), base_radius,
                                  rtol=0, atol=1e-10):
                    raise ValueError(f"{country}/{method}/{risk_model}: width budget changed")

    detail = pd.DataFrame(rows)
    detail.to_csv(output / "monthly_allocation_scores.csv", index=False)
    grouped = detail.groupby(["country", "method", "risk_model"], as_index=False).agg(
        coverage90=("coverage90", "mean"), interval_score90=("interval_score90", "mean"),
        mean_width=("width", "mean"), mean_risk=("predicted_risk", "mean"))
    grouped.to_csv(output / "country_method_scores.csv", index=False)
    summary = grouped.groupby(["method", "risk_model"], as_index=False).agg(
        macro_country_coverage90=("coverage90", "mean"),
        macro_country_interval_score90=("interval_score90", "mean"),
        macro_country_mean_width=("mean_width", "mean"),
        country_count=("country", "nunique"))
    uniform = summary.loc[summary.risk_model.eq("constant")].set_index("method")
    summary["coverage_delta_vs_uniform"] = summary.apply(
        lambda row: row.macro_country_coverage90 - uniform.loc[row.method, "macro_country_coverage90"], axis=1)
    summary["score_relative_delta_vs_uniform"] = summary.apply(
        lambda row: row.macro_country_interval_score90 / uniform.loc[row.method, "macro_country_interval_score90"] - 1., axis=1)
    summary.to_csv(output / "summary.csv", index=False)

    paths = [RISK, MONTHLY, ROOT / "scripts/run_source_pseudotarget_risk_meta.py",
             Path(__file__), czech.PROTOCOL]
    hashes = {str(path.resolve()): sha(path) for path in paths}
    hashes.update({str((EVIDENCE / f"{country}_heldout_predictions.csv.gz").resolve()):
                   sha(EVIDENCE / f"{country}_heldout_predictions.csv.gz") for country in COUNTRIES})
    (output / "input_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
