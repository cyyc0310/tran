import json
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest
import scripts.collect_information_weather_v10 as weather

from scripts.collect_information_weather_v10 import (
    PERIODS, PUBLICATION_MARGIN_HOURS, REGIONS, UNITS, VARIABLES,
    collect, daily_rows, load_points, parse_response, request_url,
)


def response(start, end, *, missing_at=None):
    hours = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(days=1),
                          freq="h", inclusive="left", tz="UTC")
    data = {"time": hours.strftime("%Y-%m-%dT%H:%M").tolist()}
    for name, field in VARIABLES.items():
        data[field] = [None if i == missing_at else float(i % 10 + 1)
                       for i in range(len(hours))]
    return json.dumps({"utc_offset_seconds": 0,
                       "hourly_units": {field: UNITS[name] for name, field in VARIABLES.items()},
                       "hourly": data}).encode()


def test_request_is_fixed_model_fixed_lead_and_utc():
    url = request_url({"latitude": 31.8, "longitude": -99.3}, "2024-01-25", "2024-01-25")
    query = parse_qs(urlparse(url).query)
    assert query["models"] == ["gfs_global"]
    assert query["timezone"] == ["UTC"]
    assert query["wind_speed_unit"] == ["ms"]
    assert query["hourly"] == [",".join(VARIABLES.values())]
    assert PUBLICATION_MARGIN_HOURS < 49


def test_response_hours_units_and_nulls_are_audited_without_filling():
    parsed = parse_response(response("2024-01-25", "2024-01-25", missing_at=3),
                            "2024-01-25", "2024-01-25")
    rows = daily_rows(parsed, "ERCO", "source_2024h1")
    assert len(rows) == 1
    assert rows[0]["origin"] == "2024-01-25T00:00:00+00:00"
    assert rows[0]["weather_complete"] is False
    assert rows[0]["weather_missing_values"] == 3
    assert np.isnan(rows[0]["weather_temp_03"])
    assert rows[0]["weather_temp_04"] == 5
    assert len([key for key in rows[0] if key.startswith("weather_") and key[-2:].isdigit()]) == 72


def test_response_rejects_wrong_time_and_unit():
    body = json.loads(response("2024-01-25", "2024-01-25"))
    body["hourly"]["time"][0] = "2024-01-24T23:00"
    with pytest.raises(ValueError, match="exactly cover"):
        parse_response(json.dumps(body).encode(), "2024-01-25", "2024-01-25")
    body = json.loads(response("2024-01-25", "2024-01-25"))
    body["hourly_units"][VARIABLES["wind"]] = "km/h"
    with pytest.raises(ValueError, match="unit"):
        parse_response(json.dumps(body).encode(), "2024-01-25", "2024-01-25")


def test_only_declared_regions_and_finite_points(tmp_path):
    path = tmp_path / "points.json"
    points = [{"region": region, "latitude": 31., "longitude": -99., "proxy": "declared"}
              for region in REGIONS]
    path.write_text(json.dumps({"schema_version": 1, "points": points}))
    assert len(load_points(path)) == 13
    points[-1]["region"] = "UNSEEN"
    path.write_text(json.dumps({"schema_version": 1, "points": points}))
    with pytest.raises(ValueError, match="registered"):
        load_points(path)


def test_transient_http_200_non_json_body_is_retried(monkeypatch):
    bodies = iter([b"", b'{}'])

    class FakeResponse:
        status = 200
        headers = {"Date": "test"}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, *_):
            return next(bodies)

    monkeypatch.setattr(weather, "urlopen", lambda *args, **kwargs: FakeResponse())
    monkeypatch.setattr(weather.time, "sleep", lambda *_: None)
    assert weather.fetch("https://example.invalid", retries=2)[0] == b'{}'


def test_collector_raw_cache_and_full_cohort_contract(tmp_path):
    points = tmp_path / "points.json"
    points.write_text(json.dumps({"schema_version": 1, "points": [
        {"region": region, "latitude": 31., "longitude": -99., "proxy": "declared"}
        for region in REGIONS]}))
    calls = []

    def requester(url):
        query = parse_qs(urlparse(url).query)
        calls.append(url)
        return response(query["start_date"][0], query["end_date"][0]), {
            "http_date": "test", "fetched_at": "test"}

    output = tmp_path / "weather"
    audit = collect(points, output, requester=requester)
    assert len(calls) == 27 == len(audit["raw_requests"])
    expected_days = sum(len(regions) * len(pd.date_range(start, end, freq="D"))
                        for _, regions, start, end in PERIODS)
    assert audit["rows"] == audit["complete_rows"] == expected_days
    assert "preceding hour" in audit["field_temporal_support"]["weather_solar_hh"]
    frame = pd.read_csv(output / "daily_weather.csv")
    assert len(frame) == expected_days
    assert not frame.duplicated(["region", "origin"]).any()
    assert frame.filter(regex=r"^weather_(temp|wind|solar)_\d\d$").shape[1] == 72
    assert len(list((output / "raw").glob("*.json"))) == 54
    with pytest.raises(FileExistsError):
        collect(points, output, requester=requester)
