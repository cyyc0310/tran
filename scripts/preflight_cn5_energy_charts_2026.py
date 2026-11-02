#!/usr/bin/env python3
"""Metadata-only availability audit for a frozen 2026 CN5 time holdout."""
from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COUNTRIES = ("nl", "es", "be", "at", "se", "cz", "sk", "hu", "si")
START, END = "2026-01-01", "2026-11-01"
SCORE_START, SCORE_END = "2026-01-01", "2026-10-01"
OUT = ROOT / "data/energy-charts-cn5-2026-preflight-r1/metadata_preflight.json"


def get_document(url: str) -> tuple[bytes, dict]:
    request = Request(url, headers={"User-Agent": "TransCIF research metadata preflight/1.0"})
    for attempt in range(5):
        try:
            with urlopen(request, timeout=120) as response:
                return response.read(), dict(response.headers.items())
        except HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
            delay = int(exc.headers.get("Retry-After", "5"))
            time.sleep(max(1, delay))
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def expected_timestamps(start: str, end: str, tz_name: str, interval_minutes: int):
    zone = ZoneInfo(tz_name)
    start_at = pd.Timestamp(start, tz=zone)
    end_at = pd.Timestamp(end, tz=zone)
    return pd.date_range(start_at, end_at, inclusive="left", freq=f"{interval_minutes}min")


def stable_payload_hash(payload: dict) -> str:
    stable = dict(payload)
    stable.pop("generated_at", None)
    canonical = json.dumps(stable, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def main() -> None:
    rows = []
    for country in COUNTRIES:
        url = ("https://api.energy-charts.info/v2/public_power"
               f"?country={country}&start={START}&end={END}")
        payload_bytes, headers = get_document(url)
        payload = json.loads(payload_bytes)
        timestamps = pd.DatetimeIndex(pd.to_datetime(
            [row["timestamp"] for row in payload["data"]], utc=True
        )).tz_convert(payload["timezone"])
        if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
            raise ValueError(f"duplicate or unordered timestamps for {country}")
        interval = int(payload["interval_minutes"])
        expected = expected_timestamps(SCORE_START, SCORE_END, payload["timezone"], interval)
        observed_score = timestamps[(timestamps >= expected[0]) & (timestamps < expected[-1] +
                                 pd.Timedelta(minutes=interval))]
        observed_index = set(observed_score.asi8.tolist())
        expected_index = set(expected.asi8.tolist())
        exact_hits = len(observed_index & expected_index)
        rows.append({
            "country": country,
            "url": url,
            "http_date": headers.get("Date"),
            "http_status": 200,
            "response_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "payload_sha256_excluding_generated_at": stable_payload_hash(payload),
            "response_bytes": len(payload_bytes),
            "schema_version": payload.get("schema_version"),
            "country_echo": payload.get("country"),
            "timezone": payload.get("timezone"),
            "resolution": payload.get("resolution"),
            "interval_minutes": interval,
            "generated_at": payload.get("generated_at"),
            "available_from": payload.get("available_from"),
            "available_until": payload.get("available_until"),
            "rows_returned": len(timestamps),
            "first_timestamp": timestamps[0].isoformat() if len(timestamps) else None,
            "last_timestamp": timestamps[-1].isoformat() if len(timestamps) else None,
            "series_ids": [item.get("id") for item in payload.get("series", [])],
            "score_start_local": str(expected[0]),
            "score_end_exclusive_local": str(expected[-1] + pd.Timedelta(minutes=interval)),
            "expected_2026_jan_sep_intervals": len(expected),
            "observed_2026_jan_sep_intervals": len(observed_index),
            "exact_expected_2026_jan_sep_hits": exact_hits,
            "coverage_fraction": exact_hits / len(expected),
        })
        print(f"metadata preflight {country.upper()}: {len(timestamps)} timestamps, "
              f"score-window coverage {exact_hits}/{len(expected)}", flush=True)
        time.sleep(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "protocol": "docs/CN5_TOTAL_VOLUME_TRAJECTORY_2026_TEMPORAL_PROTOCOL_20261102.md",
        "endpoint": "Energy-Charts API v2 public_power",
        "retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "request_start": START,
        "request_end_exclusive": END,
        "score_months": "2026-01 through 2026-09, local timestamps",
        "values_inspected": False,
        "countries": rows,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote metadata-only preflight to {OUT}", flush=True)


if __name__ == "__main__":
    main()
