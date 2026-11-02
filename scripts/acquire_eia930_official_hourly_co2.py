#!/usr/bin/env python3
"""Acquire current EIA-930 hourly CO2 generation and consumption intensity series.

The resulting files are official estimates as returned at retrieval time, not
plant-metered emissions and not a historical as-issued snapshot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

API = "https://www.eia.gov/electricity/930-api/region_data/series_data"
TARGET_TYPES = {"CO2.GER": "generated", "CO2.CER": "consumed"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acquire(ba: str, year: int, out: Path) -> tuple[pd.DataFrame, dict]:
    start, end = f"0101{year} 00:00:00", f"0101{year + 1} 00:00:00"
    params = {
        "respondent[]": [ba],
        "type[]": list(TARGET_TYPES),
        "frequency": "hourly",
        "start": start,
        "end": end,
    }
    response = requests.get(API, params=params, timeout=90)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not payload or not payload[0].get("data"):
        raise ValueError(f"no EIA-930 intensity series for {ba}/{year}: {payload}")
    rows = payload[0]["data"]
    series = {row["TYPE_ID"]: row for row in rows}
    if set(series) != set(TARGET_TYPES) or any(row["RESPONDENT_ID"] != ba for row in rows):
        raise ValueError(f"unexpected EIA-930 series returned for {ba}: {list(series)}")
    zones = {row["TIME_ZONE"] for row in rows}
    if len(zones) != 1:
        raise ValueError(f"inconsistent EIA reporting time zones for {ba}: {zones}")
    frame = None
    for type_id, label in TARGET_TYPES.items():
        row = series[type_id]
        values = row["VALUES"]
        part = pd.DataFrame({
            "local_hour_ending": values["DATES"],
            f"co2_{label}_intensity_lb_per_kwh": values["DATA"],
        })
        if frame is None:
            frame = part
        else:
            if frame.local_hour_ending.tolist() != part.local_hour_ending.tolist():
                raise ValueError(f"generation/consumption timestamps differ for {ba}")
            frame[f"co2_{label}_intensity_lb_per_kwh"] = part[f"co2_{label}_intensity_lb_per_kwh"]
    frame.insert(0, "reporting_timezone", next(iter(zones)))
    frame.insert(0, "ba", ba)
    frame.insert(0, "year", year)
    raw_path = out / "raw" / f"{ba}_{year}_co2.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps({"request_url": response.url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "payload": payload}, indent=2) + "\n")
    meta = {
        "ba": ba,
        "year": year,
        "request_url": response.url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "reporting_timezone": next(iter(zones)),
        "series": {type_id: {"type_name": series[type_id]["TYPE_NAME"], "rows": len(series[type_id]["VALUES"]["DATES"]), "missing_values": sum(v is None for v in series[type_id]["VALUES"]["DATA"])} for type_id in TARGET_TYPES},
        "raw_json_sha256": digest(raw_path),
    }
    return frame, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--bas", nargs="+", default=["BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames, metadata = [], []
    for ba in args.bas:
        frame, meta = acquire(ba, args.year, args.output_dir)
        frames.append(frame)
        metadata.append(meta)
    result = pd.concat(frames, ignore_index=True)
    csv_path = args.output_dir / f"eia930_official_co2_{args.year}.csv"
    result.to_csv(csv_path, index=False)
    summary = {
        "provider": "U.S. Energy Information Administration, EIA-930 Hourly Electric Grid Monitor",
        "endpoint": API,
        "target_year": args.year,
        "target_tracks": ["generation intensity CO2.GER", "consumption/load intensity CO2.CER"],
        "units": "lb CO2/kWh",
        "timestamp_contract": "EIA hourly-ending timestamps in UTC; valid-hour-start = timestamp - 1 hour; request spans Jan 1 year 00:00 through Jan 1 next year 00:00 inclusive, then filter valid hour starts to target year",
        "status": "current published history snapshot; not a historical as-issued vintage; EIA estimates, not direct emissions measurements",
        "balancing_authorities": metadata,
        "csv": str(csv_path),
        "csv_sha256": digest(csv_path),
        "total_rows": len(result),
        "missing_by_track": {c: int(result[c].isna().sum()) for c in result.columns if c.startswith("co2_")},
        "rows_by_ba": {str(ba): int(len(group)) for ba, group in result.groupby("ba")},
    }
    (args.output_dir / "acquisition_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
