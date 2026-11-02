#!/usr/bin/env python3
"""Post-hoc test whether public-input model disagreement signals hourly error."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_energy_charts_nine_country_cn5_loco as nine  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402

PREDICTIONS = ROOT / "results/cn5-fuel-resolved-response-r1"
OUT = ROOT / "results/cn5-prediction-disagreement-risk-r1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    daily_parts = []
    seals = []
    for country in nine.COUNTRIES:
        raw, features, _ = nine.prepare_country(country)
        query = features.loc[features.year.eq(2024)].copy()
        truth_fuel = nine.fuel_for(raw, query)
        truth = truth_fuel @ model.FACTORS / truth_fuel.sum(axis=1)
        pred_path = PREDICTIONS / f"{country.upper()}_heldout_predictions.csv.gz"
        pred = pd.read_csv(pred_path)
        expected_time = raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy()
        if len(pred) != len(query) or not np.array_equal(pred.time.astype(str).to_numpy(), expected_time):
            raise ValueError(f"prediction/label alignment failure: {country}")
        ratio = pred.ratio_recomposition.to_numpy(float)
        factor = pred.factor_history.to_numpy(float)
        fuelshare = pred.fuelshare_history.to_numpy(float)
        disagreement = np.abs(factor - ratio)
        three_way_spread = np.maximum.reduce([ratio, factor, fuelshare]) - np.minimum.reduce(
            [ratio, factor, fuelshare])
        frame = pd.DataFrame({
            "country": country.upper(),
            "date": query.clock.dt.strftime("%Y-%m-%d").to_numpy(),
            "ratio_abs_error": np.abs(truth - ratio),
            "factor_abs_error": np.abs(truth - factor),
            "fuelshare_abs_error": np.abs(truth - fuelshare),
            "factor_ratio_disagreement": disagreement,
            "three_model_spread": three_way_spread,
        })
        daily = frame.groupby(["country", "date"], as_index=False).mean(numeric_only=True)
        daily_parts.append(daily)
        seals.append({"country": country.upper(), "hours": len(pred),
                      "prediction_sha256": sha(pred_path)})

    daily = pd.concat(daily_parts, ignore_index=True)
    daily.to_csv(OUT / "daily_risk.csv", index=False)
    rows = []
    for signal in ("factor_ratio_disagreement", "three_model_spread"):
        per_country = []
        for country, part in daily.groupby("country"):
            rho = spearmanr(part[signal], part.ratio_abs_error).statistic
            low = part[signal] <= part[signal].quantile(.1)
            high = part[signal] >= part[signal].quantile(.9)
            low_error = float(part.loc[low, "ratio_abs_error"].mean())
            high_error = float(part.loc[high, "ratio_abs_error"].mean())
            item = {"country": country, "days": len(part), "spearman_rho": float(rho),
                    "low_decile_mae": low_error, "high_decile_mae": high_error,
                    "high_low_error_ratio": high_error / max(low_error, 1e-12)}
            per_country.append(item)
            rows.append({"signal": signal, **item})
        pooled_rho = spearmanr(daily[signal], daily.ratio_abs_error).statistic
        macro_ratio = float(np.mean([x["high_low_error_ratio"] for x in per_country]))
        rows.append({"signal": signal, "country": "MACRO", "days": len(daily),
                     "spearman_rho": float(pooled_rho),
                     "low_decile_mae": np.nan, "high_decile_mae": np.nan,
                     "high_low_error_ratio": macro_ratio})
    pd.DataFrame(rows).to_csv(OUT / "risk_signal_metrics.csv", index=False)

    by_country = daily.groupby("country", as_index=False).agg(
        ratio_mae=("ratio_abs_error", "mean"),
        factor_mae=("factor_abs_error", "mean"),
        fuelshare_mae=("fuelshare_abs_error", "mean"),
        factor_ratio_disagreement=("factor_ratio_disagreement", "mean"),
        three_model_spread=("three_model_spread", "mean"),
    )
    by_country.to_csv(OUT / "country_metrics.csv", index=False)
    hashes = {str(p.resolve()): sha(p) for p in [Path(__file__).resolve(),
              nine.PROTOCOL, *[nine.six_group.data_locations(c)[1] for c in nine.COUNTRIES],
              *[nine.six_group.data_locations(c)[0] / f"{c}_2022_2024.json"
                for c in nine.COUNTRIES],
              *[PREDICTIONS / f"{c.upper()}_heldout_predictions.csv.gz"
                for c in nine.COUNTRIES]]}
    (OUT / "audit.json").write_text(json.dumps(
        {"status": "POSTHOC_DIAGNOSTIC_TARGET_LABELS_ALREADY_VIEWED",
         "target_year": 2024, "countries": [c.upper() for c in nine.COUNTRIES],
         "risk_target": "daily mean absolute error of ratio_recomposition",
         "signals": ["factor_history minus ratio_recomposition absolute disagreement",
                     "range across ratio, factor, and fuelshare predictions"],
         "interpretation": "diagnostic only; no inference or calibrated coverage claim",
         "prediction_seals": seals, "sha256": hashes}, indent=2) + "\n", encoding="utf-8")
    print(pd.read_csv(OUT / "risk_signal_metrics.csv").to_string(index=False))
    print("country metrics")
    print(by_country.to_string(index=False))


if __name__ == "__main__":
    main()
