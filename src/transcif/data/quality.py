"""Data quality scoring for raw region timeseries."""

import numpy as np


def compute_data_quality(rs, cif):
    """Return a quality score in [0, 1] for a raw timeseries.

    Checks:  - missing fraction
             - physical range violations (share in [0,1], cif >= 0)
             - consecutive-identical blocks (stuck sensor)
             - variance / stdev sufficiency
    """
    rs = np.asarray(rs, dtype=np.float32)
    cif = np.asarray(cif, dtype=np.float32)
    n = len(rs)
    if n < 24:
        return 0.0

    missing_f = np.mean(~np.isfinite(rs)) + np.mean(~np.isfinite(cif))
    completeness = max(0.0, 1.0 - missing_f / 0.3)

    valid = (rs >= 0) & (rs <= 1) & (cif >= 0)
    validity = np.mean(valid.astype(float))

    n_unique_rs = len(set(np.round(rs[::24], 4)))
    stuck_penalty = min(1.0, n_unique_rs / max(1, n // 24 + 1))

    rs_std = np.nanstd(rs)
    suf = min(1.0, rs_std / 0.02)

    quality = (0.25 * completeness + 0.3 * validity +
               0.25 * stuck_penalty + 0.2 * suf)
    return float(np.clip(quality, 0.0, 1.0))
