"""Source-only conditional interval references with separate temporal calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from transcif.research.capacity_transfer import COLUMNS
from transcif.research.monthly_adaptation import weighted_quantile
from transcif.research.multicountry import weights

KINDS = ('global_q2', 'bias_q2', 'hetero_q2', 'cqr_q2')


def risk_inputs(public: pd.DataFrame, point: np.ndarray) -> np.ndarray:
    """Only public history, public capacity proportion and the frozen forecast."""
    return np.column_stack([public[list(COLUMNS)].to_numpy(float), np.asarray(point) / 1000.])


def fit(train: pd.DataFrame, error_train: np.ndarray, point_train: np.ndarray,
        calibration: pd.DataFrame, error_calibration: np.ndarray,
        point_calibration: np.ndarray) -> dict:
    """Q1 learns error structure; Q2 calibrates; no target labels are accepted."""
    if train.empty or calibration.empty:
        raise ValueError('both source risk periods must be nonempty')
    w1, w2 = weights(train), weights(calibration)
    x1, x2 = risk_inputs(train, point_train), risk_inputs(calibration, point_calibration)
    e1, e2 = np.asarray(error_train), np.asarray(error_calibration)
    if not all(np.isfinite(v).all() for v in (x1, x2, e1, e2)):
        raise ValueError('nonfinite source risk input')
    bias = weighted_quantile(e1, w1, .5)
    r1, r2 = e1 - bias, e2 - bias
    options = dict(max_iter=80, max_leaf_nodes=15, min_samples_leaf=100,
                   l2_regularization=10., random_state=416, early_stopping=False)
    scale = HistGradientBoostingRegressor(loss='absolute_error', **options)
    scale.fit(x1, np.abs(r1), sample_weight=w1 * len(w1))
    lower = HistGradientBoostingRegressor(loss='quantile', quantile=.05, **options)
    upper = HistGradientBoostingRegressor(loss='quantile', quantile=.95, **options)
    lower.fit(x1, r1, sample_weight=w1 * len(w1))
    upper.fit(x1, r1, sample_weight=w1 * len(w1))
    a, b = lower.predict(x2), upper.predict(x2)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    scores = np.maximum.reduce([lo - r2, r2 - hi, np.zeros(len(r2))])
    return {
        'bias': bias,
        'global_q2': weighted_quantile(np.abs(e2), w2, .9),
        'bias_q2': weighted_quantile(np.abs(r2), w2, .9),
        'hetero_q2': weighted_quantile(np.abs(r2) / np.maximum(scale.predict(x2), 1.), w2, .9),
        'cqr_q2': weighted_quantile(scores, w2, .9),
        'scale': scale, 'lower': lower, 'upper': upper,
    }


def bounds(fitted: dict, public: pd.DataFrame, point: np.ndarray) -> dict:
    x = risk_inputs(public, point)
    center = point + fitted['bias']
    scale = np.maximum(fitted['scale'].predict(x), 1.) * fitted['hetero_q2']
    a, b = fitted['lower'].predict(x), fitted['upper'].predict(x)
    return {
        'global_q2': (point - fitted['global_q2'], point + fitted['global_q2']),
        'bias_q2': (center - fitted['bias_q2'], center + fitted['bias_q2']),
        'hetero_q2': (center - scale, center + scale),
        'cqr_q2': (center + np.minimum(a, b) - fitted['cqr_q2'],
                   center + np.maximum(a, b) + fitted['cqr_q2']),
    }


def interval_metrics(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    if (lower > upper).any() or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError('invalid interval endpoints')
    return {
        'coverage90': float(((y >= lower) & (y <= upper)).mean()),
        'width90': float((upper - lower).mean()),
        'interval_score90': float(np.mean(upper - lower + 20 * np.maximum(lower - y, 0)
                                          + 20 * np.maximum(y - upper, 0))),
    }
