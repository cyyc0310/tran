"""Drop analytically zero daily harmonic inputs; preserve original hourly models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from transcif.research import resource_controls as controls
from transcif.research.multicountry import GROUPS

METHODS=controls.METHODS
day_groups=controls.day_groups
ZERO_DAILY=('cal0','cal1',*(f'{fuel}_h{phase}' for fuel in GROUPS for phase in ('sin','cos')))


def daily_public(frame: pd.DataFrame) -> pd.DataFrame:
    """A complete 24-clock-hour day's mean of these harmonics is exactly zero."""
    day_groups(frame)
    public=frame.copy()
    public.loc[:,list(ZERO_DAILY)]=0.
    return public


def day_inputs(frame: pd.DataFrame, climate: pd.DataFrame) -> tuple[pd.DataFrame,np.ndarray,np.ndarray]:
    return controls.day_inputs(daily_public(frame),climate)


def fit(frame: pd.DataFrame,coarse: np.ndarray,past_month_power: np.ndarray,
        climate: pd.DataFrame,*,allowed_countries: tuple) -> dict:
    result=controls.fit(daily_public(frame),coarse,past_month_power,climate,allowed_countries=allowed_countries)
    result['audit']['removed_analytically_zero_daily_features']=list(ZERO_DAILY)
    result['audit']['reason']='Daily harmonic means are zero by design; no floating-point residue as a feature.'
    return result


def predict_methods(bank: dict,fitted: dict,public: pd.DataFrame,
                    climate: pd.DataFrame) -> tuple[dict,pd.DataFrame]:
    return controls.predict_methods(bank,fitted,daily_public(public),climate)
