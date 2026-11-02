"""Score the sealed US->GB November 2026 forecast after response maturity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_official_intensity_transfer as official

FORECAST = ROOT / "results/prospective-us-to-gb-neso-nov2026-r1"
GB_PANEL = ROOT / "data/neso-gb/df_fuel_ckan.csv"
OUT = FORECAST / "posthoc-score"
TARGET_START = pd.Timestamp("2026-11-01T00:00:00Z")
TARGET_STOP = pd.Timestamp("2026-12-01T00:00:00Z")
MIN_SCORE_DATE = TARGET_STOP + pd.Timedelta(days=2)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    now = pd.Timestamp.now(tz="UTC")
    if now < MIN_SCORE_DATE:
        raise RuntimeError(f"target response is not mature before {MIN_SCORE_DATE}; current time is {now}")
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    seal = json.loads((FORECAST / "prediction_seal.json").read_text())
    prediction_path = FORECAST / seal["path"]
    if sha(prediction_path) != seal["sha256"]:
        raise ValueError("sealed prediction hash mismatch")
    predictions = pd.read_csv(prediction_path)
    times = pd.to_datetime(predictions.time_utc, utc=True, errors="raise")
    if len(predictions) != 720 or times.min() != TARGET_START or times.max() != TARGET_STOP - pd.Timedelta(hours=1):
        raise ValueError("prediction timestamps do not cover the frozen 720-hour target")
    if times.duplicated().any() or not times.sort_values().diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("prediction timestamps are not a unique hourly sequence")

    # This response field is intentionally loaded only after the maturity guard
    # and sealed-forecast verification above.
    raw = pd.read_csv(GB_PANEL, usecols=["DATETIME", "CARBON_INTENSITY"])
    raw["time"] = pd.to_datetime(raw.DATETIME, utc=True, errors="raise")
    raw["y"] = pd.to_numeric(raw.CARBON_INTENSITY, errors="coerce")
    raw = raw.loc[raw.time.ge(TARGET_START) & raw.time.lt(TARGET_STOP)].dropna(subset=["y"])
    raw["hour"] = raw.time.dt.floor("h")
    counts = raw.groupby("hour").size()
    response = raw.groupby("hour").y.mean()
    response = response.loc[counts.eq(2)]
    expected = pd.date_range(TARGET_START, TARGET_STOP, inclusive="left", freq="h")
    response = response.reindex(expected)
    if len(response) != 720 or response.isna().any():
        raise RuntimeError(f"NESO response is not complete: {int(response.notna().sum())}/720 hourly labels")

    forecast_local = pd.to_datetime(predictions.local_time, errors="raise")
    y = response.to_numpy(float)
    methods = json.loads((FORECAST / "prediction_seal.json").read_text())["methods"]
    calibration = json.loads((FORECAST / "source_calibration.json").read_text())
    rows, daily = [], []
    for method in methods:
        metrics = official.score(y, predictions[method].to_numpy(float),
                                 calibration[method]["bias"], calibration[method]["sigma"],
                                 pd.Series(forecast_local))
        rows.append({"method": method, **metrics})
        for day, indices in pd.Series(forecast_local).groupby(pd.Series(forecast_local).dt.strftime("%Y-%m-%d")):
            idx = indices.index.to_numpy()
            detail = official.score(y[idx], predictions.loc[idx, method].to_numpy(float),
                                    calibration[method]["bias"], calibration[method]["sigma"],
                                    pd.Series(forecast_local.iloc[idx].reset_index(drop=True)))
            daily.append({"method": method, "local_day": day, **detail})
    OUT.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(daily).to_csv(OUT / "daily_metrics.csv", index=False)
    provenance = {
        "status": "SCORED_PROSPECTIVE_US_TO_GB_NOV2026",
        "scored_at_utc": str(now), "forecast_sha256": seal["sha256"],
        "response_path": str(GB_PANEL.relative_to(ROOT)), "response_sha256": sha(GB_PANEL),
        "response_field": "CARBON_INTENSITY", "response_hours": int(response.notna().sum()),
        "response_semantics": "NESO current revised official consumption-intensity estimate",
        "forecast_methods": methods,
        "warning": "single prospective month in a previously seen geography; not an unseen-country confirmation",
    }
    (OUT / "scoring_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (OUT / "status.json").write_text(json.dumps({"status": provenance["status"],
        "target_response_used_after_forecast_seal": True,
        "complete_response_hours": int(response.notna().sum())}, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
