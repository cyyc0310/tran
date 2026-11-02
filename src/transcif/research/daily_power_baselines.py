"""Ordinary source-trained total/thermal power references, without target labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from transcif.research import resource_controls as controls
from transcif.research import resource_controls_daily as daily_controls
from transcif.research.multicountry import GROUPS, weights

OPTIONS = dict(max_iter=200, max_leaf_nodes=7, min_samples_leaf=5,
               learning_rate=.05, l2_regularization=1., early_stopping=False,
               random_state=416)


def source_inputs(frame: pd.DataFrame, coarse: np.ndarray, past: np.ndarray,
                  climate: pd.DataFrame, *, allowed_countries: tuple) -> tuple:
    """Daily amplitudes relative to released t-2 power, using source-2022 labels."""
    if (frame.empty or not set(frame.country) <= set(allowed_countries)
            or not frame.split.eq('train').all() or not frame.clock.dt.year.eq(2022).all()):
        raise ValueError('proper source-2022 training only')
    groups = daily_controls.day_groups(frame)
    coarse, past = np.asarray(coarse, float), np.asarray(past, float)
    if (coarse.shape != (len(frame), 6) or past.shape != (len(groups), 6)
            or not np.isfinite(coarse).all() or not np.isfinite(past).all()
            or (coarse < 0).any() or (past < 0).any()
            or (coarse.sum(axis=1) <= 0).any() or (past.sum(axis=1) <= 0).any()):
        raise ValueError('aligned finite nonnegative generation and past monthly power required')
    for d, ix in enumerate(groups):
        if not np.allclose(past[d]/past[d].sum(),
                           frame.iloc[ix[0]][list(GROUPS)].to_numpy(float), atol=1e-10, rtol=0.):
            raise ValueError('past power differs from the visible monthly composition')
    daily, _, _ = daily_controls.day_inputs(frame, climate)
    actual = np.stack([coarse[ix].mean(axis=0) for ix in groups])
    response = {'total': actual.sum(axis=1)/past.sum(axis=1)}
    response['thermal'] = np.divide(actual[:, 0], past[:, 0],
                                    out=np.full(len(daily), np.nan), where=past[:, 0] > 0)
    return daily, response


def fit_head(daily: pd.DataFrame, response: np.ndarray, *, weather: bool,
             learner: str, allowed_countries: tuple) -> dict:
    """Fit an ordinary log1p location; its inverse is not a physical mean."""
    response = np.asarray(response, float)
    if (daily.empty or not set(daily.country) <= set(allowed_countries)
            or not pd.to_datetime(daily.day).dt.year.eq(2022).all()
            or response.shape != (len(daily),)
            or not np.isfinite(response).all() or (response < 0).any()):
        raise ValueError('finite aligned proper-source daily responses required')
    x = controls.design(daily, weather)
    if not np.isfinite(x).all():
        raise ValueError('finite public features required')
    w = weights(daily); scaler = None
    if learner == 'ridge':
        scaler = StandardScaler().fit(x, sample_weight=w)
        x = scaler.transform(x); model = Ridge(alpha=10.)
    elif learner == 'hgb':
        model = HistGradientBoostingRegressor(**OPTIONS)
    else:
        raise ValueError('ordinary ridge/hgb reference required')
    model.fit(x, np.log1p(response), sample_weight=w*len(w))
    return dict(model=model, scaler=scaler, upper=float(response.max()), weather=weather,
                learner=learner, audit=dict(source_countries=sorted(set(daily.country)),
                source_days=len(daily), target_labels_used=False,
                clipping='proper source maximum amplitude, not a physical target bound',
                location='inverse log1p squared-loss location, not physical conditional mean'))


def predict_head(fitted: dict, daily: pd.DataFrame) -> np.ndarray:
    x = controls.design(daily, fitted['weather'])
    if not np.isfinite(x).all():
        raise ValueError('finite public features required')
    if fitted['scaler'] is not None:
        x = fitted['scaler'].transform(x)
    with np.errstate(over='raise', invalid='raise'):
        result = np.expm1(fitted['model'].predict(x))
    return np.clip(result, 0., fitted['upper'])
