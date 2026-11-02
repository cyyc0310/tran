"""Score the sealed GB->US EIA-923 November 2026 forecast after maturity."""
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
from transcif.data.eia930 import normalize_balance
from transcif.research.cross_country import FUELS, ZONES, labels, metrics
from transcif.research.eia_proxy import _proxy_rows

FORECAST = ROOT / "results/prospective-gb-to-us-eia923-nov2026-r1"
OUT = FORECAST / "posthoc-score"
BAS = ("ERCO", "FPL")
LOCAL_START = pd.Timestamp("2026-11-01 00:00:00")
LOCAL_STOP = pd.Timestamp("2026-12-01 00:00:00")
MIN_SCORE_DATE = pd.Timestamp("2026-12-03T00:00:00Z")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_response() -> tuple[pd.DataFrame, dict[str, str]]:
    paths = [ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jan_Jun.csv",
             ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jul_Dec.csv"]
    parts, schemas = [], {}
    for snapshot, path in enumerate(paths):
        native = pd.read_csv(path)
        native = native.loc[native["Balancing Authority"].isin(BAS)].copy()
        normalized = normalize_balance(native)
        normalized["source_snapshot"] = snapshot
        schemas[str(snapshot)] = {"fuel_columns": [c for c in normalized
                                                   if c.startswith("generation_")]}
        parts.append(normalized)
    combined = pd.concat(parts, ignore_index=True)
    rows_by_ba = []
    for ba, rows in combined.groupby("region", sort=True):
        rows = rows.copy()
        rows.index = pd.DatetimeIndex(rows.valid_at)
        rows = rows.sort_index()
        hourly, _ = _proxy_rows(rows, ba, schemas)
        hourly = hourly.loc[hourly.valid.astype(bool)].copy()
        hourly["time"] = pd.to_datetime(hourly.time, utc=True, errors="raise")
        zone = ZONES[ba]
        start = LOCAL_START.tz_localize(zone).tz_convert("UTC")
        stop = LOCAL_STOP.tz_localize(zone).tz_convert("UTC")
        hourly = hourly.loc[hourly.time.ge(start) & hourly.time.lt(stop)].copy()
        hourly["clock"] = hourly.time.dt.tz_convert(zone).dt.tz_localize(None)
        hourly["ba"] = ba
        expected = pd.date_range(start, stop, inclusive="left", freq="h")
        times = pd.DatetimeIndex(hourly.time)
        if len(hourly) != 721 or times.has_duplicates or not times.equals(expected):
            raise RuntimeError(f"{ba} target response incomplete or noncontinuous: {len(hourly)}/721")
        rows_by_ba.append(hourly)
    response = pd.concat(rows_by_ba, ignore_index=True)
    return response, {str(path.relative_to(ROOT)): sha(path) for path in paths}


def pooled_score(pred: pd.DataFrame, response: pd.DataFrame,
                 methods: list[str], calibration: dict) -> pd.DataFrame:
    frame = response[["ba", "time", "clock"]].merge(
        pred, left_on=["ba", "time"], right_on=["ba", "time_utc"],
        how="inner", validate="one_to_one")
    rows = []
    for method in methods:
        bias, sigma = calibration[method]["bias"], calibration[method]["sigma"]
        y = frame.y.to_numpy(float)
        point = frame[method].to_numpy(float) + bias
        base = metrics(pd.DataFrame({"clock": frame.clock}), y,
                       frame[method].to_numpy(float), bias, sigma)
        daily = pd.DataFrame({"ba": frame.ba,
                              "day": frame.clock.dt.strftime("%Y-%m-%d"),
                              "y": y, "p": point})
        centered = daily[["y", "p"]] - daily.groupby(["ba", "day"])[["y", "p"]].transform("mean")
        base["shape_mae"] = float(np.abs(centered.y - centered.p).mean())
        rows.append({"method": method, **base})
    return pd.DataFrame(rows)


def main() -> None:
    now = pd.Timestamp.now(tz="UTC")
    if now < MIN_SCORE_DATE:
        raise RuntimeError(f"target response is not mature before {MIN_SCORE_DATE}; current time is {now}")
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite posthoc score {OUT}")
    seal = json.loads((FORECAST / "prediction_seal.json").read_text())
    prediction_path = FORECAST / seal["path"]
    if sha(prediction_path) != seal["sha256"]:
        raise ValueError("sealed prediction hash mismatch")
    predictions = pd.read_csv(prediction_path)
    predictions["time_utc"] = pd.to_datetime(predictions.time_utc, utc=True, errors="raise")
    if predictions.groupby("ba").size().to_dict() != {ba: 721 for ba in BAS}:
        raise ValueError("sealed prediction row counts differ from the frozen BA queue")
    if predictions.duplicated(["ba", "time_utc"]).any():
        raise ValueError("duplicate sealed BA-hour predictions")

    # Target EIA-930 response is loaded only after the maturity and seal checks.
    response, response_hashes = target_response()
    response_fuels = response.rename(columns={f"generation_{fuel}": fuel for fuel in FUELS})
    response["y"] = labels(response_fuels)
    calibration = json.loads((FORECAST / "source_calibration.json").read_text())
    rows = []
    for ba in BAS:
        observed = response.loc[response.ba.eq(ba)].sort_values("time").reset_index(drop=True)
        pred = predictions.loc[predictions.ba.eq(ba)].sort_values("time_utc").reset_index(drop=True)
        if not observed.time.reset_index(drop=True).equals(pred.time_utc.reset_index(drop=True)):
            raise ValueError(f"prediction/response timestamp mismatch for {ba}")
        for method in seal["methods"]:
            rows.append({"ba": ba, "method": method,
                         **official.score(observed.y.to_numpy(float), pred[method].to_numpy(float),
                                          calibration[method]["bias"], calibration[method]["sigma"],
                                          observed.clock.reset_index(drop=True))})
    per_ba = pd.DataFrame(rows)
    pooled = pooled_score(predictions, response, seal["methods"], calibration)
    OUT.mkdir(parents=True)
    per_ba.to_csv(OUT / "per_ba_metrics.csv", index=False)
    pooled.to_csv(OUT / "pooled_metrics.csv", index=False)
    scoring = {"status": "SCORED_PROSPECTIVE_GB_TO_US_EIA923_NOV2026",
               "scored_at_utc": str(now), "forecast_sha256": seal["sha256"],
               "scoring_code_sha256": sha(Path(__file__).resolve()),
               "response_hours": {ba: int(response.ba.eq(ba).sum()) for ba in BAS},
               "response_source": [str(Path(p).relative_to(ROOT)) for p in response_hashes],
               "response_hashes": response_hashes,
               "response_label": "EIA-930 current revised fixed-EF generation intensity proxy",
               "methods": seal["methods"],
               "warning": "two previously studied BAs and one target month; not unseen-geography validation"}
    (OUT / "scoring_provenance.json").write_text(json.dumps(scoring, indent=2) + "\n")
    print(per_ba.to_string(index=False))
    print("\nPooled:\n", pooled.to_string(index=False))


if __name__ == "__main__":
    main()
