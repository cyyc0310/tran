"""NESO national estimated actuals, kept separate from its forecast product."""

import numpy as np
import pandas as pd

from transcif.data.asof import utc


def national_intervals(payload, start, end):
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ValueError("national intensity endpoint required; regional forecasts are not actual labels")
    start, end = utc([start, end])
    if start >= end:
        raise ValueError("ordered UTC interval required")
    frame = pd.DataFrame({
        "valid_at": utc([row["from"] for row in rows]),
        "interval_end": utc([row["to"] for row in rows]),
        "provider_actual_g_per_kwh": [row["intensity"].get("actual") for row in rows],
        "provider_forecast_g_per_kwh": [row["intensity"].get("forecast") for row in rows]})
    frame = frame[(frame.valid_at >= start) & (frame.valid_at < end)].copy()
    if (frame.valid_at.duplicated().any() or
            (frame.interval_end - frame.valid_at != pd.Timedelta(minutes=30)).any() or
            not (frame.valid_at == frame.valid_at.dt.floor("30min")).all()):
        raise ValueError("unique half-hour national intervals required")
    for key in ("provider_actual_g_per_kwh", "provider_forecast_g_per_kwh"):
        frame[key] = pd.to_numeric(frame[key], errors="raise").astype(float)
        if np.isinf(frame[key]).any() or (frame[key].dropna() < 0).any():
            raise ValueError("nonnegative finite or missing native intensity required")
    return frame.sort_values("valid_at").reset_index(drop=True)


def hourly_national(intervals, start, end):
    start, end = utc([start, end])
    if start >= end or start != start.floor("h") or end != end.floor("h"):
        raise ValueError("ordered whole-hour bounds required")
    if intervals.valid_at.duplicated().any():
        raise ValueError("duplicate native intervals require explicit vintage resolution")
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    selected = intervals[(intervals.valid_at >= start) & (intervals.valid_at < end)]
    grouped = selected.groupby(selected.valid_at.dt.floor("h"))
    result = pd.DataFrame(index=hours)
    result.index.name = "valid_at"
    result["intervals_present"] = grouped.size().reindex(hours, fill_value=0)
    for field, count in (("provider_actual_g_per_kwh", "actual_intervals"),
                         ("provider_forecast_g_per_kwh", "forecast_intervals")):
        result[count] = grouped[field].count().reindex(hours, fill_value=0)
        result[field] = grouped[field].mean().reindex(hours).where(result[count] == 2)
    return result.reset_index()
