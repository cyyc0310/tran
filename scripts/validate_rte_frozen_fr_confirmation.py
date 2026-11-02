"""Audit frozen France 2024 forecasts against RTE's definitive CO2 intensity.

This is a label-source robustness audit, not a new country holdout: forecasts,
models, features, and source-calibrated interval radii are read from the prior
Energy-Charts France confirmation and are never refit here.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "rte-eco2mix-2024-frozen-fr-audit-r1"
RTE_EXPORT = (
    "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "eco2mix-national-cons-def/exports/csv"
)
WHERE = (
    "perimetre='France' AND date_heure >= '2024-01-01T00:00:00+00:00' "
    "AND date_heure < '2025-01-01T00:00:00+00:00'"
)
PREDICTIONS = ROOT / "results/cn5-fuel-resolved-fr-confirmation-r1/FR_heldout_predictions.csv.gz"
RADII = ROOT / "results/cn5-fuel-resolved-fr-confirmation-r1/source_crossfit_radii.json"
# The frozen file stores real UTC instants for France local-hour starts. The
# March–December local window therefore begins at 2024-02-29 23:00 UTC.
START = pd.Timestamp("2024-02-29 23:00", tz="UTC")
END = pd.Timestamp("2024-12-31 23:00", tz="UTC")
ALPHA = 0.10


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def fetch_rte() -> tuple[pd.DataFrame, str]:
    url = RTE_EXPORT + "?" + urllib.parse.urlencode({"where": WHERE})
    request = urllib.request.Request(url, headers={"User-Agent": "TransCIF research"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    frame = pd.read_csv(io.BytesIO(payload), sep=";", encoding="utf-8-sig")
    return frame, sha256(payload)


def score(prediction: pd.DataFrame, target: pd.DataFrame, radii: dict[str, float]):
    joined = prediction.merge(target, on="time", how="inner", validate="one_to_one")
    joined = joined[(joined.time >= START) & (joined.time < END)].copy()
    if joined.empty:
        raise ValueError("No forecast/label overlap in the frozen March–December window")
    joined["local_time"] = joined.time.dt.tz_convert("Europe/Paris")

    methods = [column for column in prediction.columns if column != "time"]
    monthly_rows = []
    overall_rows = []
    for method in methods:
        radius = float(radii[method])
        errors = joined[method] - joined.rte_final_g_kwh
        lower, upper = joined[method] - radius, joined[method] + radius
        interval_score = (upper - lower) + (2 / ALPHA) * np.maximum(lower - joined.rte_final_g_kwh, 0) + (2 / ALPHA) * np.maximum(joined.rte_final_g_kwh - upper, 0)
        daily_shape_errors = []
        for _, day in joined.groupby(joined.local_time.dt.date):
            if len(day) < 20:
                continue
            p_shape = day[method] - day[method].mean()
            y_shape = day.rte_final_g_kwh - day.rte_final_g_kwh.mean()
            daily_shape_errors.extend(np.abs(p_shape - y_shape))

        month_values = []
        for month, part in joined.assign(month=joined.local_time.dt.month).groupby("month"):
            p = part[method]
            y = part.rte_final_g_kwh
            lo, hi = p - radius, p + radius
            iscore = (hi - lo) + (2 / ALPHA) * np.maximum(lo - y, 0) + (2 / ALPHA) * np.maximum(y - hi, 0)
            row = {
                "method": method,
                "month": int(month),
                "n_hours": int(len(part)),
                "mae_g_kwh": float(np.mean(np.abs(p - y))),
                "bias_g_kwh": float(np.mean(p - y)),
                "coverage_90": float(((y >= lo) & (y <= hi)).mean()),
                "width_g_kwh": float(2 * radius),
                "interval_score_90": float(iscore.mean()),
            }
            monthly_rows.append(row)
            month_values.append(row)

        overall_rows.append(
            {
                "method": method,
                "n_hours": int(len(joined)),
                "pooled_mae_g_kwh": float(np.mean(np.abs(errors))),
                "month_equal_mae_g_kwh": float(np.mean([r["mae_g_kwh"] for r in month_values])),
                "pooled_rmse_g_kwh": float(np.sqrt(np.mean(errors**2))),
                "pooled_bias_g_kwh": float(np.mean(errors)),
                "daily_demeaned_shape_mae_g_kwh": float(np.mean(daily_shape_errors)),
                "coverage_90": float(((joined.rte_final_g_kwh >= lower) & (joined.rte_final_g_kwh <= upper)).mean()),
                "width_g_kwh": float(2 * radius),
                "pooled_interval_score_90": float(interval_score.mean()),
                "month_equal_interval_score_90": float(np.mean([r["interval_score_90"] for r in month_values])),
            }
        )
    return joined, overall_rows, monthly_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rte, source_hash = fetch_rte()
    if len(rte) != 35_136:
        raise ValueError(f"Expected 35,136 quarter-hour 2024 records; found {len(rte)}")
    nature_counts = Counter(rte["nature"].fillna("<null>"))
    if nature_counts != {"Données définitives": 35_136}:
        raise ValueError(f"2024 series is not wholly definitive: {nature_counts}")

    rte["date_heure"] = pd.to_datetime(rte["date_heure"], utc=True)
    rte["taux_co2"] = pd.to_numeric(rte["taux_co2"], errors="coerce")
    intensity = rte.dropna(subset=["taux_co2"]).copy()
    interval_counts = intensity.groupby(intensity.date_heure.dt.floor("h")).size()
    hourly_target = (
        intensity.assign(time=intensity.date_heure.dt.floor("h"))
        .groupby("time", as_index=False)["taux_co2"]
        .mean()
        .rename(columns={"taux_co2": "rte_final_g_kwh"})
    )

    prediction = pd.read_csv(PREDICTIONS)
    prediction["time"] = pd.to_datetime(prediction.time, utc=True)
    radii = json.loads(RADII.read_text())
    joined, overall, monthly = score(prediction, hourly_target, radii)
    pd.DataFrame(overall).to_csv(OUT / "overall_metrics.csv", index=False)
    pd.DataFrame(monthly).to_csv(OUT / "monthly_metrics.csv", index=False)

    audit = {
        "purpose": "re-score frozen France forecasts against an independent official definitive RTE generation-intensity label",
        "not_a_new_country_holdout": True,
        "dataset_id": "eco2mix-national-cons-def",
        "source_url": "https://odre.opendatasoft.com/explore/dataset/eco2mix-national-cons-def/api/",
        "source_query": WHERE,
        "source_csv_sha256": source_hash,
        "source_rows": int(len(rte)),
        "nature_counts": dict(nature_counts),
        "taux_co2_non_null_half_hour_rows": int(len(intensity)),
        "hourly_label_hours_full_year": int(len(hourly_target)),
        "hourly_label_hours_with_two_half_hour_values": int((interval_counts == 2).sum()),
        "hourly_label_hours_with_other_interval_count": int((interval_counts != 2).sum()),
        "prediction_file_sha256": sha256(PREDICTIONS.read_bytes()),
        "source_radius_file_sha256": sha256(RADII.read_bytes()),
        "forecast_hours_in_march_december": int(((prediction.time >= START) & (prediction.time < END)).sum()),
        "matched_hours_in_march_december": int(len(joined)),
        "unmatched_forecast_hours": int(((prediction.time >= START) & (prediction.time < END)).sum() - len(joined)),
        "matched_range_utc": [joined.time.min().isoformat(), joined.time.max().isoformat()],
        "hourly_aggregation": "arithmetic mean of official definitive taux_co2 records within each UTC clock hour; paired by actual instant, then summarized in Europe/Paris local dates; one hour lacks an official value",
        "target_unit": "gCO2e/kWh",
        "target_scope": "France generation-side operational emissions intensity; excludes interconnection exchanges",
        "prediction_training_and_calibration_changed": False,
        "interpretation": "robustness to official label source; not independent geographic evidence and not evidence for consumption-side CIF",
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"matched_hours": audit["matched_hours_in_march_december"], "missing_hours": audit["unmatched_forecast_hours"], "source_sha256": source_hash, "metrics": overall}, indent=2))


if __name__ == "__main__":
    main()
