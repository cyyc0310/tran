"""Local-time calendar features for TransCIF-FD.

Datetime cyclicals in LOCAL time (EnsembleCI, arXiv 2505.01959, finds
sin/cos hour/day-of-week/day-of-year among the most important cross-grid
features).  Dataset timestamps are UTC; local time is reconstructed with
the per-region standard-time offset from ``region_meta`` (DST ignored,
< 1 h effect at hourly resolution).
"""

import numpy as np
import pandas as pd


def calendar_features(hours, tz_offset=0.0):
    """Cyclical local-time calendar features.

    Args:
        hours     : pd.DatetimeIndex / array of pd.Timestamp in UTC
        tz_offset : standard-time UTC offset in hours (e.g. -8 for PST)

    Returns (T, 6) float32: [sin h, cos h, sin dow, cos dow, sin doy, cos doy]
    with h = local hour-of-day / 24, dow = local day-of-week / 7 and
    doy = local day-of-year / 365.25, each mapped to the unit circle.
    """
    if not isinstance(hours, pd.DatetimeIndex):
        hours = pd.DatetimeIndex(hours)
    local = hours + pd.to_timedelta(tz_offset, unit="h")
    h = (local.hour.values + 0.5) / 24.0
    dow = local.dayofweek.values.astype(np.float64) / 7.0
    doy = (local.dayofyear.values - 0.5) / 365.25
    feats = np.stack([
        np.sin(2 * np.pi * h), np.cos(2 * np.pi * h),
        np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow),
        np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy),
    ], axis=1)
    return feats.astype(np.float32)
