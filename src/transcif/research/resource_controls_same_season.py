"""Same-season issued-weather training control for observed-weather expansion."""
from __future__ import annotations

import numpy as np
import pandas as pd

from transcif.research import resource_controls as controls

METHODS=controls.METHODS
day_groups=controls.day_groups
day_inputs=controls.day_inputs
predict_methods=controls.predict_methods


def fit(frame: pd.DataFrame, coarse: np.ndarray, past_month_power: np.ndarray,
        climate: pd.DataFrame, *, allowed_countries: tuple) -> dict:
    """Exclude October–December before fitting; retain an explicit used-day audit."""
    mask=frame.clock.dt.month.le(9).to_numpy()
    groups=day_groups(frame)
    days=np.array([bool(mask[ix[0]]) for ix in groups])
    fitted=controls.fit(frame.loc[mask],np.asarray(coarse)[mask],np.asarray(past_month_power)[days],
                        climate,allowed_countries=allowed_countries)
    fitted['audit'].update(provided_source_hours=len(frame),used_source_hours=int(mask.sum()),
        source_selection='issued-weather pilot days through September 2022 only',
        used_source_days=[dict(region=frame.region.iloc[ix[0]],day=str(frame.clock.iloc[ix[0]].floor('D')))
                          for ix in groups if mask[ix[0]]])
    return fitted
