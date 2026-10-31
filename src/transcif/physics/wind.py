"""Illustrative wind-power feature used by synthetic data and public baselines.

This curve does not represent outages, curtailment or turbine diversity.
"""

import numpy as np


def wind_capacity_factor(wind_speed_ms, v_in=3.0, v_rated=12.0, v_out=25.0):
    """Illustrative turbine capacity factor from 100 m wind speed.

    Piecewise: 0 below cut-in, cubic ramp ((v^3 - v_in^3) / (v_rated^3 -
    v_in^3)) between cut-in and rated, 1.0 between rated and cut-out, 0
    above cut-out.  Returns values in [0, 1].
    """
    v = np.asarray(wind_speed_ms, dtype=np.float64)
    cf = np.zeros_like(v)
    ramp = (v >= v_in) & (v < v_rated)
    cf[ramp] = (v[ramp] ** 3 - v_in ** 3) / (v_rated ** 3 - v_in ** 3)
    cf[(v >= v_rated) & (v <= v_out)] = 1.0
    return np.clip(cf, 0.0, 1.0)
