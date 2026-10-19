"""UTC publication-time joins. Missing information raises; nothing is backfilled."""

import numpy as np
import pandas as pd


def utc(values, name="timestamps"):
    index = pd.DatetimeIndex(values)
    if index.tz is None or index.hasnans:
        raise ValueError(f"{name} must be timezone-aware and nonmissing")
    return index.tz_convert("UTC")


def public_config_asof(releases, origin, fields):
    """Latest completed period released by origin, including eligible revisions."""
    origin = utc([origin], "origin")[0]
    frame = releases.copy()
    for name in ("period_end", "available_at"):
        frame[name] = utc(frame[name], name)
    if (frame.available_at < frame.period_end).any():
        raise ValueError("statistics released before their period ends")
    if frame.duplicated(["period_end", "available_at"]).any():
        raise ValueError("ambiguous statistical revisions")
    frame = frame[(frame.available_at <= origin) & (frame.period_end <= origin)]
    if frame.empty:
        raise ValueError(f"no config published by {origin}")
    row = frame.sort_values(["period_end", "available_at"]).iloc[-1]
    if not np.isfinite(row[list(fields)].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite public configuration")
    return row


def forecast_asof(forecasts, origin, valid_hours, members, fields):
    """Latest complete single run for all members and hours available at origin.

    Columns: issued_at, available_at, valid_at, member, and fields. Revisions
    use available_at. Different issue cycles are never stitched together.
    """
    origin = utc([origin], "origin")[0]
    valid = utc(valid_hours, "valid_at")
    if len(valid) == 0 or valid.has_duplicates or (valid < origin).any():
        raise ValueError("unique nonempty future valid hours required")
    members = list(members)
    if not members or len(set(members)) != len(members):
        raise ValueError("unique nonempty ensemble members required")
    frame = forecasts.copy()
    for name in ("issued_at", "available_at", "valid_at"):
        frame[name] = utc(frame[name], name)
    if (frame.available_at < frame.issued_at).any() or (frame.valid_at < frame.issued_at).any():
        raise ValueError("forecast time ordering is invalid")
    if frame.duplicated(["issued_at", "available_at", "valid_at", "member"]).any():
        raise ValueError("ambiguous forecast revisions")
    frame = frame[(frame.issued_at <= origin) & (frame.available_at <= origin)]
    index = pd.MultiIndex.from_product([members, valid], names=["member", "valid_at"])
    for issued in sorted(frame.issued_at.unique(), reverse=True):
        run = frame[frame.issued_at == issued]
        rows = (run.sort_values("available_at")
                .drop_duplicates(["member", "valid_at"], keep="last")
                .set_index(["member", "valid_at"]).reindex(index))
        if rows.issued_at.notna().all() and np.isfinite(
                rows[list(fields)].to_numpy(dtype=float)).all():
            return rows
    raise ValueError(f"no complete issued forecast available at {origin}")
