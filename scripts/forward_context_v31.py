"""Frozen V30 head replay on supplied MG forecasts and legal past context only.

Pure rows/arrays; no I/O, current truth, hourly values, or new hidden labels.
The six refits reproduce the sealed L24 training contract, never 2025 labels.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from scripts import coarse_context_v30 as parent

SOURCES, POS, ZERO, EF = parent.SOURCES, parent.POS, parent.ZERO, parent.EF
CUTOFF = parent.CUTOFF
START, END = CUTOFF, pd.Timestamp('2025-07-01', tz='UTC')
FAMILIES, MODELS = ('G5', 'Trend', 'Size'), ('Ridge', 'ET')
ARMS = tuple(f'{family}-{model}' for family in FAMILIES for model in MODELS)
FEATURE_NAMES = {family: parent.FEATURE_NAMES[family] for family in FAMILIES}
PARAMETERS = {
    'Ridge': dict(alpha=1., fit_intercept=True),
    'ET': dict(n_estimators=128, min_samples_leaf=2, max_features=1., random_state=17023, n_jobs=1),
}


def _dates(origins: Sequence[Any]) -> pd.DatetimeIndex:
    values = []
    for value in origins:
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            at = pd.Timestamp(int(value), unit='ns', tz='UTC')
        else:
            at = parent._utc(value)
        if (pd.isna(at) or not START <= at < END or at.day != 1 or at != at.normalize()):
            raise ValueError('UTC month-start origins in 2025 January through June required')
        values.append(at)
    # pandas 3 preserves microsecond resolution; asi8 follows the index unit.
    # The public integer/provenance/age contract is always nanoseconds.
    dates = pd.DatetimeIndex(values).as_unit('ns')
    if dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError('Nonempty unique increasing month-start origins required')
    return dates


def _simplex_rows(values: np.ndarray, size: int) -> None:
    if (values.ndim < 1 or values.shape[-1] != size or not np.isfinite(values).all() or
            np.any(values < 0) or not np.allclose(values.sum(-1), 1., atol=1e-12, rtol=0)):
        raise ValueError('Finite nonnegative simplex rows required')


def _context(rows: Sequence[Mapping[str, str]], region: str,
             dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    records, seen = [], set()
    for row in rows:
        source = row['region']
        if source not in SOURCES:
            continue
        at = parent._utc(row['period_start'])
        if at < parent.START:
            continue
        end, release = parent._utc(row['period_end']), parent._utc(row['available_at'])
        if at != at.normalize() or at.day != 1 or end != at+pd.offsets.MonthBegin(1) or release < end:
            raise ValueError('Whole calendar months with legal release metadata required')
        key = parent._key(source, at)
        if key in seen:
            raise ValueError('Duplicate source region-month/vintage')
        seen.add(key)
        if source == region:
            records.append((at, end, release, key, row))
    records.sort(key=lambda item: item[0])
    chosen = []
    for at in dates:
        legal = [row for row in records if row[0] < at and row[1] <= at and row[2] <= at][-3:][::-1]
        if len(legal) != 3:
            raise ValueError('Three legal past monthly contexts required at every origin')
        chosen.append(legal)
    g = np.array([[parent._visible(item[4]) for item in row] for row in chosen])
    mw = np.array([[float(item[4]['total_mwh'])/((item[1]-item[0])/pd.Timedelta(hours=1))
                    for item in row] for row in chosen])
    if not np.isfinite(mw).all() or np.any(mw <= 0):
        raise ValueError('Finite strictly positive eligible past volumes required')
    starts = np.array([[item[0].value for item in row] for row in chosen], np.int64)
    ends = np.array([[item[1].value for item in row] for row in chosen], np.int64)
    releases = np.array([[item[2].value for item in row] for row in chosen], np.int64)
    month_age = np.array([[at.year*12+at.month-item[0].year*12-item[0].month
                           for item in row] for at, row in zip(dates, chosen)], float)
    return dict(origin_ns=dates.asi8.copy(), context_keys=np.array([[item[3] for item in row] for row in chosen]),
                context_start_ns=starts, context_end_ns=ends, context_release_ns=releases,
                context_g=g, context_MW=mw, context_month_age=month_age,
                context_release_age_days=(dates.asi8[:, None]-releases)/(86400.*1e9))


def _features(draws: np.ndarray, context: Mapping[str, np.ndarray],
              dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    n, particles = draws.shape[:2]
    phase = 2*np.pi*(dates.month.to_numpy()-1)/12
    history = np.column_stack([np.sin(phase), np.cos(phase), context['context_g'].reshape(n, 15),
                               context['context_month_age'], context['context_release_age_days']])
    logs = np.log(context['context_MW'])
    suffix = np.column_stack([history, logs[:, 0]-logs[:, 1], logs[:, 1]-logs[:, 2]])
    trend = np.concatenate([draws, np.broadcast_to(suffix[:, None, :], (n, particles, 25))], axis=-1)
    size = np.concatenate([trend, np.broadcast_to(logs[:, None, :1], (n, particles, 1))], axis=-1)
    return dict(G5=draws.copy(), Trend=trend, Size=size)


def _training(fit: Mapping[str, Any], family: str, name: str, region: str) -> dict[str, Any]:
    """Validate frozen training identity; checked statistics never replace scaler."""
    if (fit['arm'] != f'{family}-{name}' or fit['family'] != family or fit['model'] != name or
            fit['excluded_outer'] != region or fit['parameters'] != PARAMETERS[name] or
            tuple(fit['feature_names']) != FEATURE_NAMES[family]):
        raise ValueError('Frozen parent head identity/parameters/features required')
    keys = list(fit['training_keys'])
    regions = np.asarray(fit['training_regions'], dtype=str)
    x, y = np.asarray(fit['training_X'], float), np.asarray(fit['training_y'], float)
    w = np.asarray(fit['sample_weights'], float)
    n, d = len(keys), len(FEATURE_NAMES[family])
    if (not n or len(set(keys)) != n or regions.shape != (n,) or
            set(regions) != set(SOURCES)-{region} or x.shape != (n, d) or y.shape != (n, 4) or
            w.shape != (n,) or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w <= 0)):
        raise ValueError('Finite common six-source training cohort required')
    _simplex_rows(y, 4)
    metadata = {}
    for field in ('training_period_start_ns', 'training_period_end_ns', 'training_available_ns'):
        value = np.asarray(fit[field])
        if value.shape != (n,) or not np.issubdtype(value.dtype, np.integer):
            raise ValueError('Integer nanosecond training metadata required')
        metadata[field] = value.copy()
    for i, value in enumerate(metadata['training_period_start_ns']):
        at = pd.Timestamp(int(value), unit='ns', tz='UTC')
        end = pd.Timestamp(int(metadata['training_period_end_ns'][i]), unit='ns', tz='UTC')
        available = int(metadata['training_available_ns'][i])
        if (pd.isna(at) or not parent.START <= at < parent.END or at != at.normalize() or at.day != 1 or
                end != at+pd.offsets.MonthBegin(1) or not end.value <= available <= CUTOFF.value or
                keys[i] != parent._key(regions[i], at)):
            raise ValueError('Excluded-outer six-source L24 released labels required')
    expected_w = np.array([n/(6*np.count_nonzero(regions == source)) for source in regions])
    if not np.allclose(w, expected_w, atol=1e-12, rtol=0):
        raise ValueError('Frozen region-balanced training weights required')
    mean, scale = np.asarray(fit['scaler_mean'], float), np.asarray(fit['scaler_scale'], float)
    if mean.shape != (d,) or scale.shape != (d,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Finite frozen scaler required')
    if name == 'Ridge':
        checked_mean = np.average(x, axis=0, weights=w)
        checked_scale = np.sqrt(np.average((x-checked_mean)**2, axis=0, weights=w))
        checked_scale[checked_scale <= 1e-12] = 1.
    else:
        checked_mean, checked_scale = np.zeros(d), np.ones(d)
    if not (np.allclose(mean, checked_mean, atol=1e-12, rtol=0) and
            np.allclose(scale, checked_scale, atol=1e-12, rtol=0)):
        raise ValueError('Frozen parent scaler changed')
    return dict(training_keys=keys, training_regions=regions.copy(), training_X=x.copy(), training_y=y.copy(),
                sample_weights=w.copy(), scaler_mean=mean.copy(), scaler_scale=scale.copy(), **metadata)


def fit_predict(parent_audit: Mapping[str, Any], raw_month_rows: Sequence[Mapping[str, str]], *,
                region: str, origins: Sequence[Any], mg_draws: np.ndarray) -> dict[str, Any]:
    """Replay six frozen heads, project each MG particle once, then average CIF."""
    dates = _dates(origins)  # Date guard precedes any raw-row access or fitting.
    draws = np.asarray(mg_draws, float)
    if draws.shape != (len(dates), 128, 5):
        raise ValueError('MG forecast draws must have shape (origins,128,5)')
    _simplex_rows(draws, 5)
    if region not in SOURCES:
        raise ValueError('Registered source region required')
    try:
        if parent_audit['region'] != region or parent_audit['outer'] != region or parent_audit['fit_count'] != 10:
            raise ValueError('Matching sealed V30 outer audit required')
        fits = parent_audit['fits']
        by_arm = {fit['arm']: fit for fit in fits}
        if len(fits) != 10 or len(by_arm) != 10 or set(by_arm) != set(parent.ARMS):
            raise ValueError('Complete ten-head sealed V30 audit required')
        tables = {arm: _training(by_arm[arm], family, name, region)
                  for family in FAMILIES for name in MODELS for arm in (f'{family}-{name}',)}
        reference = tables[ARMS[0]]
        for arm, table in tables.items():
            for field in ('training_keys', 'training_regions', 'training_y', 'sample_weights',
                          'training_period_start_ns', 'training_period_end_ns', 'training_available_ns'):
                if not np.array_equal(table[field], reference[field]):
                    raise ValueError('All six frozen heads require identical training cohort/labels/weights')
            if not np.array_equal(table['training_X'][:, :5], reference['training_X']):
                raise ValueError('Frozen heads require identical current-group training covariates')
        for name in MODELS:
            if not np.array_equal(tables['Size-'+name]['training_X'][:, :30], tables['Trend-'+name]['training_X']):
                raise ValueError('Frozen Trend/Size training covariates disagree')
        context = _context(raw_month_rows, region, dates)
        features = _features(draws, context, dates)
        predictions, records = [], []
        for family in FAMILIES:
            for name in MODELS:
                arm = f'{family}-{name}'
                table, sealed = tables[arm], by_arm[arm]
                model = Ridge(**PARAMETERS[name]) if name == 'Ridge' else ExtraTreesRegressor(**PARAMETERS[name])
                x = table['training_X']
                query = features[family].reshape(-1, len(FEATURE_NAMES[family]))
                if name == 'Ridge':
                    fit_x = (x-table['scaler_mean'])/table['scaler_scale']
                    predict_x = (query-table['scaler_mean'])/table['scaler_scale']
                else:
                    fit_x, predict_x = x, query
                model.fit(fit_x, table['training_y'], sample_weight=table['sample_weights'])
                if name == 'Ridge':
                    coef, intercept = np.asarray(sealed['coefficients'], float), np.asarray(sealed['intercept'], float)
                    if coef.shape != model.coef_.shape or intercept.shape != model.intercept_.shape:
                        raise ValueError('Frozen Ridge coefficient shape changed')
                    error = max(float(np.max(np.abs(model.coef_-coef))), float(np.max(np.abs(model.intercept_-intercept))))
                    reproduced = dict(coefficients=model.coef_.copy(), intercept=model.intercept_.copy())
                    tolerance = 1e-10
                else:
                    importance = np.asarray(sealed['feature_importances'], float)
                    if importance.shape != model.feature_importances_.shape:
                        raise ValueError('Frozen ET importance shape changed')
                    error = float(np.max(np.abs(model.feature_importances_-importance)))
                    reproduced = dict(feature_importances=model.feature_importances_.copy())
                    tolerance = 1e-12
                if not np.isfinite(error) or error > tolerance:
                    raise ValueError('Frozen parent model reproduction failed: '+arm)
                predictions.append(model.predict(predict_x).reshape(len(dates), 128, 4))
                records.append(dict(arm=arm, family=family, model=name, excluded_outer=region,
                    parameters=PARAMETERS[name].copy(), feature_names=FEATURE_NAMES[family],
                    parent_reproduction_max_abs=error, **table, **reproduced))
    except KeyError as exc:
        raise ValueError('Missing frozen parent or eligible monthly field: '+str(exc)) from exc
    raw = np.stack(predictions)
    ratios = parent.project(raw)
    ef = ratios@EF[POS]
    cif = draws[None, ..., 0]*ef
    return dict(ratios_raw=raw, ratios=ratios, EF_draws=ef, CIF_draws=cif,
                EF_point=ef.mean(-1), CIF_point=cif.mean(-1), projection_l1=np.abs(ratios-raw).sum(-1),
                MG_draws=draws.copy(), **{'query_feature_'+k: v for k, v in features.items()}, **context,
                audit=dict(region=region, arms=list(ARMS), fit_count=6, fits=records,
                    hidden_fuel_fields_read=False, current_group_truth_read=False, current_total_read=False,
                    oracle_current_group=False, forward_forecast=True, hourly_values_read=False,
                    target_values_opened=False, unseen_values_opened=False, new_training_labels_used=False,
                    training_contract='Frozen V30 six-source L24 cohort; identical-head reproduction only',
                    point_definition='Particle mean EF and particle mean of MG_POS times EF'))
