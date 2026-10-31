"""Collect fixed-lead historical GFS point forecasts for the V10 information budget.

Open-Meteo Previous Runs gives nominal forecasts 72 hours before each valid
hour, but does not return an original per-hour issue/publication timestamp.
The output is therefore an audited forecast proxy, not certified issued_public.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


BASE = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "gfs_global"
LEAD_DAYS = 3
PUBLICATION_MARGIN_HOURS = 18
VARIABLES = {
    "temp": "temperature_2m_previous_day3",
    "wind": "wind_speed_10m_previous_day3",
    "solar": "shortwave_radiation_previous_day3",
}
UNITS = {"temp": "°C", "wind": "m/s", "solar": "W/m²"}
FIELD_TEMPORAL_SUPPORT = {
    "weather_temp_hh": "temperature at API valid time D hh:00 UTC",
    "weather_wind_hh": "10 m wind speed at API valid time D hh:00 UTC",
    "weather_solar_hh": "shortwave mean over preceding hour (D hh-1:00, D hh:00]; weather_solar_00 includes the hour before the CIF day",
    "hourly_cif": "CIF row h covers [D hh:00, D hh+1:00); solar covariate is shifted one hour earlier relative to this interval",
}
SOURCES = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
TARGETS = ("AZPS", "DUK", "LGEE", "PACE", "PSCO", "SRP")
REGIONS = SOURCES + TARGETS
PERIODS = (
    ("source_2024h1", SOURCES, "2024-01-25", "2024-06-30"),
    ("source_2024h2", SOURCES, "2024-07-01", "2024-12-31"),
    ("source_2025h1", SOURCES, "2025-01-01", "2025-06-30"),
    ("seen_target_2026h1", TARGETS, "2026-01-01", "2026-06-30"),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_points(path: Path) -> dict[str, dict]:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema_version") != 1 or not isinstance(raw.get("points"), list):
        raise ValueError("frozen schema_version=1 points required")
    points = {}
    for point in raw["points"]:
        region = point["region"]
        lat, lon = point["latitude"], point["longitude"]
        if (region in points or region not in REGIONS or
                not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)) or
                isinstance(lat, bool) or isinstance(lon, bool) or
                not math.isfinite(lat) or not math.isfinite(lon) or
                not -90 <= lat <= 90 or not -180 <= lon <= 180 or
                not point.get("proxy")):
            raise ValueError("unique registered finite geographic proxy points required")
        points[region] = point
    if set(points) != set(REGIONS):
        raise ValueError("exactly seven source and six seen-target points required")
    return points


def request_url(point: dict, start: str, end: str) -> str:
    params = {
        "latitude": point["latitude"], "longitude": point["longitude"],
        "start_date": start, "end_date": end,
        "hourly": ",".join(VARIABLES.values()), "models": MODEL,
        "wind_speed_unit": "ms", "timezone": "UTC",
    }
    return BASE + "?" + urlencode(params)


def fetch(url: str, timeout: float = 45., retries: int = 3) -> tuple[bytes, dict]:
    """Bounded GET retry for transient HTTP/network errors; no silent fallback."""
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers={"User-Agent": "TransCIF-V10-weather/1"}),
                         timeout=timeout) as response:
                if response.status != 200:
                    raise ValueError(f"weather HTTP {response.status}")
                content = response.read(8_000_001)
                if len(content) > 8_000_000:
                    raise ValueError("weather response exceeds 8 MB budget")
                # A gateway can transiently return HTTP 200 with an empty or
                # non-JSON body. Do not cache it as an immutable forecast.
                json.loads(content)
                return content, {
                    "http_date": response.headers.get("Date"),
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            if isinstance(error, HTTPError) and error.code != 429 and error.code < 500:
                raise
            if attempt == retries - 1:
                raise
            time.sleep(1.0 * 2 ** attempt)
    raise AssertionError("unreachable retry state")


def parse_response(raw: bytes, start: str, end: str) -> pd.DataFrame:
    """Require exact UTC hourly support; preserve null fields as NaN."""
    doc = json.loads(raw)
    if doc.get("error"):
        raise ValueError(f"weather API error: {doc.get('reason')}")
    if doc.get("utc_offset_seconds") != 0:
        raise ValueError("UTC weather response required")
    hourly, units = doc.get("hourly", {}), doc.get("hourly_units", {})
    expected = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(days=1),
                             freq="h", inclusive="left", tz="UTC")
    times = pd.DatetimeIndex(pd.to_datetime(hourly.get("time", []), utc=True, errors="raise"))
    if not times.equals(expected):
        raise ValueError("forecast hours must exactly cover requested UTC days")
    columns = {"valid_at": times}
    for name, field in VARIABLES.items():
        if units.get(field) != UNITS[name]:
            raise ValueError(f"unexpected weather unit for {field}")
        values = hourly.get(field)
        if not isinstance(values, list) or len(values) != len(times):
            raise ValueError(f"missing or short weather field {field}")
        numeric = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(float)
        if np.isinf(numeric).any() or (name != "temp" and np.nanmin(numeric, initial=0) < 0):
            raise ValueError(f"infinite or physically negative weather field {field}")
        columns[name] = numeric
    return pd.DataFrame(columns)


def daily_rows(hourly: pd.DataFrame, region: str, period: str) -> list[dict]:
    rows = []
    for day, frame in hourly.groupby(hourly.valid_at.dt.floor("D"), sort=True):
        if len(frame) != 24 or not frame.valid_at.reset_index(drop=True).equals(
                pd.Series(pd.date_range(day, periods=24, freq="h", tz="UTC"))):
            raise ValueError("one complete UTC valid-time grid per forecast day required")
        latest_nominal_init = frame.valid_at.iloc[-1] - pd.Timedelta(days=LEAD_DAYS)
        minimum_gap = (day - latest_nominal_init) / pd.Timedelta(hours=1)
        if minimum_gap <= PUBLICATION_MARGIN_HOURS:
            raise ValueError("nominal issue time and margin must precede common origin")
        row = {"region": region, "origin": day.isoformat(), "period": period}
        missing = 0
        for name in VARIABLES:
            values = frame[name].to_numpy(float)
            missing += int(np.isnan(values).sum())
            row.update({f"weather_{name}_{hour:02d}": value for hour, value in enumerate(values)})
        row["weather_complete"] = missing == 0
        row["weather_missing_values"] = missing
        rows.append(row)
    return rows


def collect(points_path: Path, output: Path, requester=fetch) -> dict:
    """Write raw immutable responses, then one daily table and audit on success."""
    points_path, output = Path(points_path), Path(output)
    points = load_points(points_path)
    point_hash = sha256(points_path.read_bytes())
    output.mkdir(parents=True, exist_ok=True)
    if (output / "daily_weather.csv").exists() or (output / "metadata.json").exists():
        raise FileExistsError("completed weather output is immutable")
    raw_dir = output / "raw"
    raw_dir.mkdir(exist_ok=True)
    all_rows, records = [], []
    for period, regions, start, end in PERIODS:
        for region in regions:
            url = request_url(points[region], start, end)
            raw_path = raw_dir / f"{period}_{region}.json"
            audit_path = raw_dir / f"{period}_{region}.audit.json"
            if raw_path.exists() != audit_path.exists():
                raise ValueError("incomplete immutable raw weather cache entry")
            if raw_path.exists():
                content = raw_path.read_bytes()
                audit = json.loads(audit_path.read_text())
                if audit.get("url") != url or audit.get("sha256") != sha256(content) or audit.get("points_sha256") != point_hash:
                    raise ValueError("cached weather URL/hash/points mismatch")
            else:
                content, headers = requester(url)
                if not isinstance(content, bytes) or not content:
                    raise ValueError("nonempty raw JSON bytes required")
                audit = {"url": url, "sha256": sha256(content), "points_sha256": point_hash,
                         "http_date": headers.get("http_date"), "fetched_at": headers.get("fetched_at")}
                # Parse before committing an immutable cache entry.
                parse_response(content, start, end)
                raw_path.write_bytes(content)
                audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
            hourly = parse_response(content, start, end)
            all_rows.extend(daily_rows(hourly, region, period))
            records.append({"region": region, "period": period, "start": start, "end": end,
                            "raw_file": str(raw_path.relative_to(output)),
                            "raw_sha256": audit["sha256"], "url": url,
                            "http_date": audit.get("http_date"),
                            "fetched_at": audit.get("fetched_at"),
                            "hours": len(hourly),
                            "missing_values": int(hourly[list(VARIABLES)].isna().sum().sum())})
    table = pd.DataFrame(all_rows).sort_values(["region", "origin"]).reset_index(drop=True)
    if table.duplicated(["region", "origin"]).any() or len(records) != 27:
        raise ValueError("registered cohort must have 27 unique point-period requests")
    if sha256(points_path.read_bytes()) != point_hash:
        raise ValueError("point registry changed during weather collection")
    output_csv = output / "daily_weather.csv"
    table.to_csv(output_csv, index=False)
    audit = {
        "schema_version": 1, "points_file": str(points_path.resolve()),
        "points_sha256": point_hash, "model": MODEL, "api": BASE,
        "variable_ids": VARIABLES, "units": UNITS,
        "field_temporal_support": FIELD_TEMPORAL_SUPPORT,
        "field_temporal_support_source": "https://open-meteo.com/en/docs/previous-runs-api",
        "fixed_lead_hours": 72, "publication_margin_hours_assumed": PUBLICATION_MARGIN_HOURS,
        "minimum_nominal_initialization_gap_before_origin_hours": 49,
        "interpretation": "fixed-lead historical operational-forecast proxy; API lacks per-hour run/publication timestamp; not certified issued_public",
        "availability_assumption": "Open-Meteo documents _previous_day3 as predicted 72h before each valid hour; all nominal contributing initializations precede common 00 UTC origin by >=49h, leaving >=31h after an assumed 18h publication margin. Historical actual publication is not certified.",
        "raw_requests": records, "rows": len(table),
        "complete_rows": int(table.weather_complete.sum()),
        "incomplete_rows": int((~table.weather_complete).sum()),
        "daily_weather_sha256": sha256(output_csv.read_bytes()),
        "target_hourly_labels_read": False,
    }
    (output / "metadata.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, default=Path("results/information-value-v10-weather/points.json"))
    parser.add_argument("--output", type=Path, default=Path("results/information-value-v10-weather"))
    args = parser.parse_args()
    print(json.dumps(collect(args.points, args.output), indent=2))


if __name__ == "__main__":
    main()
