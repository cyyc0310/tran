"""Calendar covariates with an explicit civil timezone or fixed market offset."""

import numpy as np
import pandas as pd

from transcif.data.asof import utc


def calendar_features(hours, timezone="UTC", tz_offset=None):
    times = utc(hours)
    local = (times + pd.to_timedelta(tz_offset, unit="h") if tz_offset is not None
             else times.tz_convert(timezone))
    phases = ((local.hour.to_numpy() + 0.5) / 24,
              local.dayofweek.to_numpy() / 7,
              (local.dayofyear.to_numpy() - 0.5) / 365.25)
    return np.stack([f(2 * np.pi * phase) for phase in phases
                     for f in (np.sin, np.cos)], axis=-1).astype(np.float32)
