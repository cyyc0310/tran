"""Pair-excluded monthly center teachers and historical daily risk decomposition."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from scripts import coarse_context_v30 as coarse
from scripts import forward_context_v31 as forward
from scripts import fleet_information_v34 as fleet

SOURCES, EF, POS = coarse.SOURCES, coarse.EF, coarse.POS
DAYS = pd.date_range('2024-01-25', '2024-12-31', tz='UTC', freq='D').as_unit('ns')
HEADS = ('Size-ET', 'SizeCap-Ridge')
ARMS = ('Scale-anchor', 'Scale-recompose', 'B0-SizeET-KeepA', 'B0-SizeET-ReplaceR',
        'B0-SizeET-AlignedKeep', 'B1-SizeCapRidge-KeepA', 'B1-SizeCapRidge-ReplaceR',
        'B1-SizeCapRidge-AlignedKeep')
BUDGETS = ('B0',)*5 + ('B1',)*3
DATE_FIELDS = ('training_period_start_ns', 'training_period_end_ns', 'training_available_ns')


def pair_guard(pair: tuple[str, str]) -> None:
    if (type(pair) is not tuple or len(pair) != 2 or len(set(pair)) != 2
            or tuple(sorted(pair)) != pair or not set(pair) <= set(SOURCES)):
        raise ValueError('Canonical two-source exclusion required')


def real(value: Any, shape: tuple, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError('Real '+name+' required')
    a = np.asarray(raw, float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError('Finite '+name+' with shape '+str(shape)+' required')
    return a


def _metadata(record: Mapping[str, Any], arm: str, pair: tuple[str, str]) -> dict:
    family, method = arm.split('-')
    if (record['arm'] != arm or record['family'] != family or record['model'] != method
            or record['excluded_outer'] != pair[0] or record['parameters'] != forward.PARAMETERS[method]
            or tuple(record['feature_names']) != fleet.FEATURE_NAMES[family]
            or (method == 'ET' and type(record['parameters']['max_features']) is not float)):
        raise ValueError('Original center head identity and parameters required')
    keys, regions = list(record['training_keys']), np.asarray(record['training_regions'], str)
    n = len(keys)
    if not n or len(set(keys)) != n or regions.shape != (n,) or set(regions) != set(SOURCES)-{pair[0]}:
        raise ValueError('Original six-source training metadata required')
    dates = {}
    for name in DATE_FIELDS:
        value = np.asarray(record[name])
        if value.shape != (n,) or not np.issubdtype(value.dtype, np.integer):
            raise ValueError('Integer training timestamp metadata required')
        dates[name] = value
    for i, start in enumerate(dates[DATE_FIELDS[0]]):
        at = pd.Timestamp(int(start), tz='UTC'); end = pd.Timestamp(int(dates[DATE_FIELDS[1]][i]), tz='UTC')
        if (not coarse.START <= at < coarse.END or at.day != 1 or at != at.normalize()
                or end != at+pd.offsets.MonthBegin(1) or keys[i] != coarse._key(regions[i], at)
                or not end.value <= dates[DATE_FIELDS[2]][i] <= coarse.CUTOFF.value):
            raise ValueError('Canonical released pre-cutoff source labels required')
    ids = np.flatnonzero(~np.isin(regions, pair))
    if set(regions[ids]) != set(SOURCES)-set(pair):
        raise ValueError('Exactly five training sources after pair exclusion required')
    return dict(ids=ids, keys=[keys[i] for i in ids], regions=regions[ids],
                **{k: v[ids] for k, v in dates.items()})


def fit_centers(records: Sequence[Mapping[str, Any]], pair: tuple[str, str]) -> dict:
    pair_guard(pair)
    if len(records) != 2:
        raise ValueError('Two fixed center records required')
    meta = [_metadata(record, arm, pair) for record, arm in zip(records, HEADS)]
    if any(not np.array_equal(meta[0][k], meta[1][k]) for k in ('keys', 'regions', *DATE_FIELDS)):
        raise ValueError('Identical pair-excluded center cohorts required')
    numerics = []
    for record, arm, m in zip(records, HEADS, meta):
        family = arm.split('-')[0]; width = len(fleet.FEATURE_NAMES[family]); n = len(m['ids'])
        # Slice lists by validated metadata first; held-source fine values are never converted.
        x = real([record['training_X'][i] for i in m['ids']], (n, width), 'retained training X')
        y = real([record['training_y'][i] for i in m['ids']], (n, 4), 'retained training y')
        forward._simplex_rows(y, 4)
        w = np.array([n/(5*np.count_nonzero(m['regions'] == s)) for s in m['regions']])
        numerics.append((x, y, w))
    if not (np.array_equal(numerics[0][0], numerics[1][0][:, :31])
            and np.array_equal(numerics[0][1], numerics[1][1])):
        raise ValueError('Common pair center source features and labels required')
    models, audits = [], []
    for arm, m, (x, y, w) in zip(HEADS, meta, numerics):
        method = arm.split('-')[1]
        mean, scale = np.zeros(x.shape[1]), np.ones(x.shape[1])
        if method == 'Ridge':
            mean = np.average(x, axis=0, weights=w)
            scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w)); scale[scale <= 1e-12] = 1.
        fitted = Ridge(**forward.PARAMETERS[method]) if method == 'Ridge' else ExtraTreesRegressor(**forward.PARAMETERS[method])
        fitted.fit((x-mean)/scale, y, sample_weight=w)
        models.append((fitted, mean, scale))
        extra = dict(coefficients=fitted.coef_, intercept=fitted.intercept_) if method == 'Ridge' else dict(feature_importances=fitted.feature_importances_)
        audits.append(dict(arm=arm, excluded_regions=pair, training_keys=m['keys'], training_regions=m['regions'],
            training_X=x, training_y=y, sample_weights=w, scaler_mean=mean, scaler_scale=scale,
            parameters=forward.PARAMETERS[method], **{k: m[k] for k in DATE_FIELDS}, **extra))
    return dict(models=models, audit=dict(pair=pair, fit_count=2, ET_fits=1, Ridge_fits=1, fits=audits,
        training_snapshot='2025-01-01', authentic_2024_online_hindcast=False))


def features(rows: Sequence[Mapping[str, str]], region: str, MG: np.ndarray,
             q4: np.ndarray, Cpos: float) -> dict:
    if region not in SOURCES:
        raise ValueError('Original source region required')
    mg = real(MG, (342, 128, 5), 'historical MG'); forward._simplex_rows(mg, 5)
    q = real(q4, (4,), 'capacity shares'); forward._simplex_rows(q, 4)
    c = float(real(Cpos, (), 'POS capacity'))
    if c <= 0:
        raise ValueError('Positive year-minus-two capacity required')
    context = forward._context(rows, region, DAYS)
    size = forward._features(mg, context, DAYS)['Size']
    cap = fleet._features(size, np.tile(q, (342, 1)), np.full(342, c))['SizeCap']
    return dict(region=region, features=dict(Size=size, SizeCap=cap), MG_draws=mg, **context)


def predict_centers(fitted: dict, query: dict) -> dict:
    if query['region'] not in fitted['audit']['pair'] or not np.array_equal(query['origin_ns'], DAYS.asi8):
        raise ValueError('Held pair member and fixed historical daily query required')
    mg = real(query['MG_draws'], (342, 128, 5), 'query MG'); forward._simplex_rows(mg, 5)
    if not np.array_equal(np.asarray(query['features']['SizeCap'])[..., :31], query['features']['Size']):
        raise ValueError('Aligned Size and SizeCap feature families required')
    raw, ratio = [], []
    for arm, (estimator, mean, scale) in zip(HEADS, fitted['models']):
        family = arm.split('-')[0]; width = len(fleet.FEATURE_NAMES[family])
        x = real(query['features'][family], (342, 128, width), 'query '+family)
        if not np.array_equal(x[..., :5], mg):
            raise ValueError('Literal query particle feature coupling required')
        value = estimator.predict((x.reshape(-1, width)-mean)/scale).reshape(342, 128, 4)
        raw.append(value); ratio.append(coarse.project(value))
    ratio = np.stack(ratio); ef = ratio @ EF[POS]
    return dict(raw_ratios=np.stack(raw), ratios=ratio, EF_draws=ef, CIF_draws=ef*mg[None, ..., 0])


def bases(anchor: np.ndarray, recompose: np.ndarray, fine: np.ndarray, query_mean: np.ndarray,
          MG: np.ndarray, levels: np.ndarray) -> dict:
    a = real(anchor, (342, 128, 24), 'anchor'); r = real(recompose, a.shape, 'recompose')
    f = real(fine, (342, 128, 8), 'fine'); q = real(query_mean, (342, 128, 4), 'query mean')
    mg = real(MG, (342, 128, 5), 'MG'); s = real(levels, (2, 342, 128), 'levels')
    forward._simplex_rows(f, 8); forward._simplex_rows(q, 4); forward._simplex_rows(mg, 5)
    if (not np.allclose(f[..., POS].sum(-1), mg[..., 0], atol=1e-12, rtol=0)
            or not np.allclose(f[..., coarse.ZERO], mg[..., 1:], atol=1e-12, rtol=0)):
        raise ValueError('Fine/MG particle alignment required')
    if np.any(s < 400*mg[None, ..., 0]-1e-9) or np.any(s > 1000*mg[None, ..., 0]+1e-9):
        raise ValueError('Bounded same-particle monthly levels required')
    u, p = f @ EF, mg[..., 0]*(q @ EF[POS])
    paths = [a, r]
    for level in s:
        paths.extend((a-p[..., None]+level[..., None], r-u[..., None]+level[..., None],
                      r-r.mean(-1, keepdims=True)+a.mean(-1, keepdims=True)-p[..., None]+level[..., None]))
    raw = np.stack(paths)
    if not np.isfinite(raw).all():
        raise ValueError('Finite source risk base paths required')
    return dict(raw=raw, U=u, P=p)


def errors(point: np.ndarray, truth: np.ndarray) -> dict:
    p = np.asarray(point, float); y = np.asarray(truth, float)
    if p.ndim != 3 or p.shape[0] != 8 or p.shape[2] != 24 or y.shape != p.shape[1:] or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Finite eight-base points and matching day truth required')
    hourly = y[None] - p
    level = hourly.mean(-1)
    shape = (y-y.mean(-1, keepdims=True))[None] - (p-p.mean(-1, keepdims=True))
    error = max(float(abs(hourly-level[..., None]-shape).max()), float(abs(shape.mean(-1)).max()))
    if error > 1e-9:
        raise ValueError('Orthogonal hourly level/shape decomposition required')
    return dict(hourly_error=hourly, level_error=level, shape_error=shape, reconstruction_max_abs=error)
