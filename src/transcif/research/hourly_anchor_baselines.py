"""Ordinary monthly-anchor residual references, with physical-unit OOF scales."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from transcif.research.hourly_transfer_baselines import OPTIONS
from transcif.research.multicountry import weights
from transcif.research.resource_controls import day_groups

EPSILON = .001
LOGIT_LIMIT = float(np.log((1. - EPSILON) / EPSILON))
KINDS = ('additive', 'logit')


def anchor_values(anchor: np.ndarray) -> np.ndarray:
    """Only observed public thermal fractions are admissible anchors."""
    value = np.asarray(anchor, float)
    if value.ndim != 1 or not np.isfinite(value).all() or ((value < 0.) | (value > 1.)).any():
        raise ValueError('finite public anchor fractions in 0–1 required')
    return value


def logit(value: np.ndarray) -> np.ndarray:
    fraction = np.clip(value, EPSILON, 1. - EPSILON)
    return np.log(fraction / (1. - fraction))


def encode(response_permille: np.ndarray, anchor: np.ndarray, kind: str) -> np.ndarray:
    """Map bounded additive/logit increments to a common 0–1000 response scale."""
    prior = anchor_values(anchor)
    response = np.asarray(response_permille, float)
    if (response.shape != prior.shape or not np.isfinite(response).all()
            or ((response < 0.) | (response > 1000.)).any()):
        raise ValueError('aligned thermal response in permille required')
    if kind == 'additive':
        result = 500. + (response - 1000. * prior) / 2.
    elif kind == 'logit':
        result = 500. + (logit(response / 1000.) - logit(prior)) * 250. / LOGIT_LIMIT
    else:
        raise ValueError('unknown anchor response kind')
    return np.clip(result, 0., 1000.)


def decode(encoded: np.ndarray, anchor: np.ndarray, kind: str) -> np.ndarray:
    """Convert locations or ordered quantiles back to physical thermal permille."""
    prior = anchor_values(anchor)
    value = np.asarray(encoded, float)
    if value.ndim not in (1, 2) or len(value) != len(prior) or not np.isfinite(value).all():
        raise ValueError('aligned finite encoded predictions required')
    if value.ndim == 2:
        prior = prior[:, None]
    value = np.clip(value, 0., 1000.)
    if kind == 'additive':
        result = 1000. * prior + 2. * (value - 500.)
    elif kind == 'logit':
        shifted = logit(prior) + (value - 500.) * LOGIT_LIMIT / 250.
        result = 1000. / (1. + np.exp(-shifted))
    else:
        raise ValueError('unknown anchor response kind')
    return np.clip(result, 0., 1000.)


def fit(frame: pd.DataFrame, response_permille: np.ndarray, x: np.ndarray, *, kind: str,
        allowed_countries: tuple, progress: Callable[[str], None] | None = None) -> tuple[dict, dict]:
    """Match R5 capacity, folds and weights; scale uses decoded OOF errors."""
    if (frame.empty or set(frame.country) != set(allowed_countries)
            or not frame.split.eq('train').all() or not frame.clock.dt.year.eq(2022).all()
            or not frame.clock.dt.month.between(3, 9).all()):
        raise ValueError('proper source-2022 training rows and countries required')
    anchor = anchor_values(frame.thermal.to_numpy())
    physical = np.asarray(response_permille, float)
    y = encode(physical, anchor, kind)
    x = np.asarray(x, float)
    if x.ndim != 2 or len(x) != len(frame) or not np.isfinite(x).all():
        raise ValueError('aligned finite public design required')
    day_groups(frame)
    w = weights(frame)
    block = np.where(frame.clock.dt.month <= 4, 0, np.where(frame.clock.dt.month <= 6, 1, 2))
    if set(block) != {0, 1, 2}:
        raise ValueError('three nonempty source calendar blocks required')
    oof = np.empty(len(frame)); folds = []

    def notify(step: str) -> None:
        if progress is not None:
            progress(step)

    for held_block in (0, 1, 2):
        held = block == held_block; proper = ~held
        if set(frame.loc[proper, 'country']) != set(allowed_countries):
            raise ValueError('every source country needs proper rows in each fold')
        wf = weights(frame.loc[proper])
        model = HistGradientBoostingRegressor(**OPTIONS).fit(x[proper], y[proper], sample_weight=wf * len(wf))
        oof[held] = np.clip(model.predict(x[held]), 0., 1000.)
        folds.append(dict(held_block=held_block, proper_hours=int(proper.sum()), held_hours=int(held.sum()),
            proper_months=sorted(frame.loc[proper, 'clock'].dt.month.unique().astype(int).tolist()),
            held_months=sorted(frame.loc[held, 'clock'].dt.month.unique().astype(int).tolist())))
        notify(f'OOF block {held_block}')
    physical_oof = decode(oof, anchor, kind)
    location = HistGradientBoostingRegressor(**OPTIONS).fit(x, y, sample_weight=w * len(w))
    notify('final encoded location')
    scale = HistGradientBoostingRegressor(loss='absolute_error', **OPTIONS).fit(
        x, np.abs(physical - physical_oof), sample_weight=w * len(w))
    notify('physical-permille OOF absolute-error scale')
    quantiles = []
    for alpha in (.05, .5, .95):
        quantiles.append(HistGradientBoostingRegressor(loss='quantile', quantile=alpha, **OPTIONS).fit(
            x, y, sample_weight=w * len(w)))
        notify(f'quantile {alpha}')
    fitted = dict(location=location, scale=scale, quantiles=quantiles, kind=kind, epsilon=EPSILON,
        audit=dict(source_countries=sorted(allowed_countries), source_hours=len(frame), source_year=2022,
            source_region_hours=frame.groupby('region').size().to_dict(), feature_count=x.shape[1],
            source_country_weight={c: float(w[frame.country.eq(c)].sum()) for c in sorted(allowed_countries)},
            folds=folds, options=dict(OPTIONS), response=kind, encoded_response_range=[0., 1000.],
            physical_response_units='1000*thermal generation/total generation',
            scale_units='thermal permille absolute decoded OOF error, not encoded residual units',
            point_definition='inverse encoded squared-loss location; not asserted physical conditional mean',
            epsilon=EPSILON, target_fit=False))
    return fitted, dict(encoded=oof, permille=physical_oof)


def forecast(fitted: dict, x: np.ndarray, anchor: np.ndarray, factor: np.ndarray) -> dict:
    """CIF conversion is applied after decoding; scale already has physical units."""
    x = np.asarray(x, float)
    prior = anchor_values(anchor)
    conversion = np.asarray(factor, float)
    if (x.ndim != 2 or len(x) != len(prior) or not np.isfinite(x).all()
            or conversion.shape != prior.shape or not np.isfinite(conversion).all()
            or ((conversion <= 0.) | (conversion > 1.)).any() or fitted['epsilon'] != EPSILON):
        raise ValueError('aligned design, public anchor, conversion and fixed epsilon required')
    point = decode(fitted['location'].predict(x), prior, fitted['kind']) * conversion
    encoded = np.sort(np.clip(np.column_stack([model.predict(x) for model in fitted['quantiles']]),
                              0., 1000.), axis=1)
    quantiles = decode(encoded, prior, fitted['kind']) * conversion[:, None]
    scale = np.maximum(fitted['scale'].predict(x) * conversion, 1.)
    return dict(point=point, scale=scale, lower=quantiles[:, 0], median=quantiles[:, 1], upper=quantiles[:, 2])
