"""Calendar covariates in the manifest's explicit civil timezone."""

import numpy as np

from transcif.data.asof import utc


def calendar_features(hours, timezone="UTC"):
    local = utc(hours).tz_convert(timezone)
    phases = ((local.hour.to_numpy() + 0.5) / 24,
              local.dayofweek.to_numpy() / 7,
              (local.dayofyear.to_numpy() - 0.5) / 365.25)
    return np.stack([f(2 * np.pi * phase) for phase in phases
                     for f in (np.sin, np.cos)], axis=-1).astype(np.float32)
