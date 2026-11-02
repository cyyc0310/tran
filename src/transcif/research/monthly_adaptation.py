"""Public coarse-month forecasts and low-dimensional, label-free local adaptation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from transcif.research.cross_country import CALENDAR, COARSE, FUELS

METHODS = ('persistence', 'seasonal', 'damped_trend', 'local_ridge',
           'source_ridge', 'local_bias', 'shrunk_bias')
RISKS = ('global', 'hetero', 'cqr')


def monthly_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert the development hourly archive to a simulated public-month table."""
    period = raw.clock.dt.to_period('M')
    totals = raw.assign(period=period).groupby('period')[list(FUELS)].sum()
    coverage = raw.groupby(period).size()/raw.groupby(period).expected_hours.first()
    totals = totals.loc[coverage >= .99].copy()
    totals['thermal'] = totals[['coal', 'gas', 'oil', 'other']].sum(axis=1)
    return totals[list(COARSE)].div(totals[list(FUELS)].sum(axis=1), axis=0)


def clr(values: np.ndarray) -> np.ndarray:
    result = np.log(np.maximum(np.asarray(values, float), .001))
    return result-result.mean(axis=-1, keepdims=True)


def composition(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    exp = np.exp(values-values.max(axis=-1, keepdims=True))
    return exp/exp.sum(axis=-1, keepdims=True)


def inputs(history: pd.DataFrame, period: pd.Period) -> np.ndarray | None:
    if period-2 not in history.index:
        return None
    last = history.loc[period-2].to_numpy(float)
    previous = history.loc[period-3].to_numpy(float) if period-3 in history.index else last
    return np.r_[clr(last), clr(last)-clr(previous),
                 np.sin(period.month*2*np.pi/12), np.cos(period.month*2*np.pi/12),
                 float(period-3 in history.index)]


def supervised_months(history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pairs = [(inputs(history, p), clr(history.loc[p].to_numpy())-clr(history.loc[p-2].to_numpy()))
             for p in history.index if inputs(history, p) is not None]
    if not pairs:
        return np.empty((0, 13)), np.empty((0, 5))
    x, y = zip(*pairs)
    return np.asarray(x), np.asarray(y)


def fit_source(tables: dict[str, pd.DataFrame]) -> Ridge:
    xs, ys, ws = [], [], []
    for table in tables.values():
        x, y = supervised_months(table.loc[table.index.year <= 2022])
        if len(x):
            xs.append(x); ys.append(y); ws.append(np.ones(len(x))/len(x))
    if not xs:
        raise ValueError('no source monthly supervision')
    weights = np.concatenate(ws)
    weights *= len(weights)/weights.sum()
    return Ridge(alpha=10.).fit(np.concatenate(xs), np.concatenate(ys), sample_weight=weights)


def forecast(table: pd.DataFrame, period: pd.Period, source: Ridge, method: str) -> np.ndarray:
    """Only coarse observations through period-2 can affect any method."""
    if method not in METHODS:
        raise ValueError(method)
    history = table.loc[table.index <= period-2]
    x = inputs(history, period)
    if x is None:
        raise ValueError('missing exact lag-2 observation')
    last = history.loc[period-2].to_numpy(float)
    if method == 'persistence':
        return last
    if method == 'seasonal':
        return history.loc[period-12].to_numpy(float) if period-12 in history.index else last
    if method == 'damped_trend':
        return composition(clr(last) + np.clip(x[5:10], -.5, .5))
    if method == 'local_ridge':
        local_x, local_y = supervised_months(history)
        if len(local_x) < 6:
            return last
        delta = Ridge(alpha=10.).fit(local_x, local_y).predict(x[None])[0]
        return composition(clr(last)+delta)
    base = clr(last)+source.predict(x[None])[0]
    if method == 'source_ridge':
        return composition(base)
    errors = []
    for p in history.index[-6:]:
        xp = inputs(history, p)
        if xp is not None:
            errors.append(clr(history.loc[p].to_numpy()) - clr(history.loc[p-2].to_numpy())
                          - source.predict(xp[None])[0])
    if not errors:
        return composition(base)
    bias = np.mean(errors, axis=0)
    shrink = 1. if method == 'local_bias' else len(errors)/(len(errors)+6.)
    return composition(base+shrink*bias)


def predict_hours(frame: pd.DataFrame, table: pd.DataFrame, source: Ridge,
                  method: str, native: dict) -> tuple[np.ndarray, np.ndarray]:
    mixes = {p: forecast(table, p, source, method) for p in frame.period.unique()}
    mix = np.stack([mixes[p] for p in frame.period])
    q = mix*native['profile'][frame.clock.dt.hour.to_numpy()]
    point = native['thermal_ef']*q[:, 0]/q.sum(axis=1)
    risk_features = np.column_stack([frame[list(CALENDAR)].to_numpy(float), mix])
    return point, risk_features


def region_weights(regions: np.ndarray) -> np.ndarray:
    _, ix, counts = np.unique(regions, return_inverse=True, return_counts=True)
    weights = 1/counts[ix]
    return weights/weights.sum()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])/weights.sum()
    return float(values[order[min(np.searchsorted(cumulative, level), len(order)-1)]])


def fit_risks(x1: np.ndarray, error1: np.ndarray, region1: np.ndarray,
              x2: np.ndarray, error2: np.ndarray, region2: np.ndarray) -> dict:
    """H1 fits location/conditional risk; H2 calibrates radii, source labels only."""
    w1, w2 = region_weights(region1), region_weights(region2)
    bias = float(np.sum(w1*error1))
    r1, r2 = error1-bias, error2-bias
    options = dict(max_iter=80, max_leaf_nodes=15, min_samples_leaf=100,
                   l2_regularization=10., random_state=415)
    scale = HistGradientBoostingRegressor(loss='absolute_error', **options)
    scale.fit(x1, np.abs(r1), sample_weight=w1*len(w1))
    lo = HistGradientBoostingRegressor(loss='quantile', quantile=.05, **options)
    hi = HistGradientBoostingRegressor(loss='quantile', quantile=.95, **options)
    lo.fit(x1, r1, sample_weight=w1*len(w1))
    hi.fit(x1, r1, sample_weight=w1*len(w1))
    lower, upper = np.minimum(lo.predict(x2), hi.predict(x2)), np.maximum(lo.predict(x2), hi.predict(x2))
    scores = np.maximum.reduce([lower-r2, r2-upper, np.zeros(len(r2))])
    return {'bias': bias, 'global': weighted_quantile(np.abs(r2), w2, .9),
            'scale': scale, 'hetero': weighted_quantile(np.abs(r2)/np.maximum(scale.predict(x2), 1.), w2, .9),
            'lo': lo, 'hi': hi, 'cqr': weighted_quantile(scores, w2, .9)}


def risk_bounds(fitted: dict, x: np.ndarray, point: np.ndarray) -> dict:
    center = point+fitted['bias']
    radius = fitted['global']
    hetero = np.maximum(fitted['scale'].predict(x), 1.)*fitted['hetero']
    a, b = fitted['lo'].predict(x), fitted['hi'].predict(x)
    return {'global': (center-radius, center+radius), 'hetero': (center-hetero, center+hetero),
            'cqr': (center+np.minimum(a, b)-fitted['cqr'], center+np.maximum(a, b)+fitted['cqr'])}
