import csv
import json

import numpy as np
import pandas as pd
import pytest

from scripts.information_features_v10 import (
    BASE_NAMES, RANGE_NAMES, SHAPE_NAMES, build_features, extract_total_telemetry,
)
from scripts.coarse_dynamics_v9 import forecast_monthly
from transcif.research.coarse_forecast_v7 import COLUMNS


def monthly():
    starts = pd.date_range("2022-01-01", "2024-04-01", freq="MS", tz="UTC")
    frame = pd.DataFrame({"region": "BPAT", "period_start": starts,
                          "period_end": starts + pd.offsets.MonthBegin(1),
                          "available_at": starts + pd.offsets.MonthBegin(1) + pd.Timedelta(days=30),
                          "observed_hours": 700., "total_mwh": 700_000.})
    for column in COLUMNS:
        frame[column] = 0.
    frame[COLUMNS[0]] = .4
    frame[COLUMNS[1]] = .6
    return frame


def telemetry(start="2024-01-01", stop="2024-04-01"):
    hours = pd.date_range(start, stop, inclusive="left", freq="h", tz="UTC")
    return pd.DataFrame({"region": "BPAT", "hour_end": hours + pd.Timedelta(hours=1),
                         "g_mw": 1000 + 100 * np.sin(2 * np.pi * hours.hour / 24)})


def test_shapes_and_hour_end_cutoff_exclude_the_incomplete_lagged_day():
    origin = pd.Timestamp("2024-03-15", tz="UTC")
    result = build_features(monthly(), telemetry(), "BPAT", [origin])
    assert result["base"].shape == (1, len(BASE_NAMES))
    assert result["range"].shape == (1, len(RANGE_NAMES))
    assert result["shape"].shape == (1, len(SHAPE_NAMES))
    assert result["audit"]["rows"][0]["max_hour_end_allowed"] == "2024-03-08T00:00:00+00:00"
    assert result["audit"]["rows"][0]["max_hour_end_used"] == "2024-03-08T00:00:00+00:00"
    # The 7-day-prior same-weekday day is incomplete at the hour-end cutoff;
    # the 28 days preceding that cutoff contain four complete same weekdays.
    assert result["shape"][0, 24] == 4
    assert result["shape"][0, 25] == 0
    np.testing.assert_allclose(result["shape"][0, :24].mean(), 1)
    np.testing.assert_allclose(result["base"][0, 8:16], [.4, .6, 0, 0, 0, 0, 0, 0])


def test_cached_harmonic_point_matches_frozen_v9_forecaster():
    dates = pd.to_datetime(["2024-01-15", "2024-02-15", "2024-03-15"], utc=True)
    frame = monthly()
    frame.loc[frame.period_start.dt.month.eq(7), COLUMNS[0]] = .5
    frame.loc[frame.period_start.dt.month.eq(7), COLUMNS[1]] = .5
    got = build_features(frame, telemetry(), "BPAT", dates)["base"][:, :8]
    expected = forecast_monthly(frame, "BPAT", dates, n_samples=1)["harmonic_ridge"]["mix"]
    np.testing.assert_allclose(got, expected, atol=1e-14)


def test_future_monthly_and_total_values_are_invisible_even_when_poisoned():
    origin = "2024-03-15"
    baseline = build_features(monthly(), telemetry(), "BPAT", [origin])
    future_monthly = monthly().copy().astype({"total_mwh": object})
    future_monthly.loc[future_monthly.available_at.gt(pd.Timestamp(origin, tz="UTC")),
                       list(COLUMNS)] = -999.
    future_monthly.loc[future_monthly.available_at.gt(pd.Timestamp(origin, tz="UTC")),
                       "total_mwh"] = "unreleased poison"
    future_g = pd.concat([telemetry(), pd.DataFrame({"region": ["BPAT", "DUK"],
        "hour_end": ["2024-03-08T01:00:00Z", "not a relevant timestamp"],
        "g_mw": ["future poison", "fuel poison"], "generation_coal": ["poison", "poison"]})],
        ignore_index=True)
    got = build_features(future_monthly, future_g, "BPAT", [origin])
    for key in ("base", "range", "shape"):
        np.testing.assert_array_equal(got[key], baseline[key])
    assert got["audit"] == baseline["audit"]


def test_exact_cutoff_hour_is_usable_but_next_hour_is_not():
    origin = "2024-03-15"
    g = telemetry()
    boundary = g.hour_end.eq(pd.Timestamp("2024-03-08", tz="UTC"))
    changed = g.copy()
    changed.loc[boundary, "g_mw"] = 5000.
    before = build_features(monthly(), g, "BPAT", [origin])
    after = build_features(monthly(), changed, "BPAT", [origin])
    assert before["range"][0, 0] != after["range"][0, 0]
    later = g.copy()
    later.loc[later.hour_end.eq(pd.Timestamp("2024-03-08T01:00:00Z")), "g_mw"] = 9000.
    excluded = build_features(monthly(), later, "BPAT", [origin])
    for key in ("base", "range", "shape"):
        np.testing.assert_array_equal(excluded[key], before[key])
    delayed = build_features(monthly(), g, "BPAT", [origin], lag_days=30)
    assert delayed["audit"]["rows"][0]["max_hour_end_allowed"] == "2024-02-14T00:00:00+00:00"


def test_missing_total_is_explicit_and_not_replaced_from_fuel_proxy():
    empty = pd.DataFrame({"region": [], "hour_end": [], "g_mw": []})
    result = build_features(monthly(), empty, "BPAT", ["2024-03-15"])
    assert np.isnan(result["range"][0, :7]).all()
    np.testing.assert_array_equal(result["range"][0, 7:], [0, 0])
    assert np.isnan(result["shape"][0, :24]).all()
    np.testing.assert_array_equal(result["shape"][0, 24:], [0, 1, 0])
    assert result["audit"]["rows"][0]["max_hour_end_used"] is None


def test_extractor_selects_ba_before_parsing_and_writes_only_three_fields(tmp_path):
    source = tmp_path / "raw.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Balancing Authority", "UTC Time at End of Hour",
                                                    "Net Generation (MW)", "Net Generation (MW) from Coal"])
        writer.writeheader()
        writer.writerow({"Balancing Authority": "BPAT", "UTC Time at End of Hour": "01/01/2024 01:00:00 AM",
                         "Net Generation (MW)": "-3", "Net Generation (MW) from Coal": "secret"})
        writer.writerow({"Balancing Authority": "UNSEEN", "UTC Time at End of Hour": "bad",
                         "Net Generation (MW)": "poison", "Net Generation (MW) from Coal": "secret"})
    output = tmp_path / "telemetry"
    audit = extract_total_telemetry([source], output, regions=["BPAT"], chunksize=1)
    frame = pd.read_csv(output / "total_generation.csv")
    assert list(frame) == ["region", "hour_end", "g_mw"]
    assert len(frame) == 1 and frame.g_mw.iloc[0] == -3
    assert audit["raw_sources"][0]["registered_rows"] == 1
    assert json.loads((output / "audit.json").read_text())["telemetry_sha256"] == audit["telemetry_sha256"]
    with pytest.raises(FileExistsError):
        extract_total_telemetry([source], output, regions=["BPAT"])
