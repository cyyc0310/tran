"""Origin-frozen monthly state for a future horizon, with known future calendars."""
from __future__ import annotations

import numpy as np
import pandas as pd

from transcif.research.multicountry import GROUPS, public_features

STATE = (*GROUPS, 'thermal_mean', 'thermal_std', 'thermal_change', 'total_cv', 'observations')


def public_horizon_features(months: pd.DataFrame, local_clock: pd.Series,
                            origin_local: pd.Timestamp) -> pd.DataFrame:
    """Pin t-2 to origin's month, never to the future valid hour's month.

    Both clocks must use the caller's same naive local-calendar convention;
    caller owns conversion from absolute UTC origin and weather valid times.
    This is the simulated lag contract, not an as-published release certificate.
    """
    origin = pd.Timestamp(origin_local)
    if (pd.isna(origin) or origin.tzinfo is not None or local_clock.dt.tz is not None
            or local_clock.isna().any() or local_clock.lt(origin).any()):
        raise ValueError('finite future local clocks and a matching naive local origin required')
    cutoff = origin.to_period('M')-2
    if cutoff not in months.index:
        raise ValueError('missing exact origin-lag2 public month')
    snapshot = public_features(months, pd.Series([origin])).iloc[0]
    values = snapshot[list(STATE)].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('invalid origin public state')
    out = pd.DataFrame({'clock': local_clock})
    out['period'] = local_clock.dt.to_period('M')
    for i, (value, denominator) in enumerate([(local_clock.dt.hour,24),
                                             (local_clock.dt.dayofweek,7),
                                             (local_clock.dt.month,12)]):
        out[f'cal{2*i}'] = np.sin(value*2*np.pi/denominator)
        out[f'cal{2*i+1}'] = np.cos(value*2*np.pi/denominator)
    for name, value in zip(STATE,values):
        out[name] = value
    for group in GROUPS:
        out[f'{group}_hsin'] = out[group]*out.cal0
        out[f'{group}_hcos'] = out[group]*out.cal1
    out['forecast_origin_local'] = origin
    out['latest_allowed_period'] = str(cutoff)
    return out
