import numpy as np
import pytest

from transcif.data.neso import national_intervals, hourly_national


def test_neso_never_substitutes_forecasts_for_missing_actuals_or_partial_hours():
    payload = {"data": [
        {"from": "2023-01-01T00:00Z", "to": "2023-01-01T00:30Z", "intensity": {"actual": 100, "forecast": 300}},
        {"from": "2023-01-01T00:30Z", "to": "2023-01-01T01:00Z", "intensity": {"actual": None, "forecast": 400}},
        {"from": "2023-01-01T01:00Z", "to": "2023-01-01T01:30Z", "intensity": {"actual": 50, "forecast": 300}}]}
    start, end = "2023-01-01T00:00Z", "2023-01-01T02:00Z"
    intervals = national_intervals(payload, start, end)
    hourly = hourly_national(intervals, start, end)
    assert np.isnan(hourly.provider_actual_g_per_kwh).all()
    assert hourly.provider_forecast_g_per_kwh.iloc[0] == 350
    assert hourly.actual_intervals.tolist() == [1, 1]
    assert hourly.intervals_present.tolist() == [2, 1]
    assert "available_at" not in hourly
    with pytest.raises(ValueError, match="national intensity endpoint"):
        national_intervals({"data": {"data": payload["data"]}}, start, end)


def test_neso_complete_hour_and_half_open_chunk_bounds():
    payload = {"data": [
        {"from": "2023-01-01T00:00Z", "to": "2023-01-01T00:30Z", "intensity": {"actual": 100}},
        {"from": "2023-01-01T00:30Z", "to": "2023-01-01T01:00Z", "intensity": {"actual": 200}},
        {"from": "2023-01-01T01:00Z", "to": "2023-01-01T01:30Z", "intensity": {"actual": 500}}]}
    start, end = "2023-01-01T00:00Z", "2023-01-01T01:00Z"
    native = national_intervals(payload, start, end)
    assert len(native) == 2
    assert hourly_national(native, start, end).provider_actual_g_per_kwh.iloc[0] == 150
