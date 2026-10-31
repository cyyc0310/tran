"""Pure coarse conditional diagnostics; current G5 is an explicit oracle covariate.

No file I/O or hourly data. Hidden four-fuel fields are read only for training
responses, after excluding the outer region. This is not a month-start forecast.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

SOURCES = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
FUELS = ('coal', 'gas', 'oil', 'nuclear', 'hydro', 'wind', 'solar', 'other')
SHARES = tuple('fuelshare_'+fuel for fuel in FUELS)
POS, ZERO = np.array([0, 1, 2, 7]), np.array([3, 4, 5, 6])
EF = np.array([1000., 400., 800., 0., 0., 0., 0., 400.])
CUTOFF = pd.Timestamp('2025-01-01', tz='UTC')
START, END = pd.Timestamp('2022-01-01', tz='UTC'), pd.Timestamp('2024-07-01', tz='UTC')
FAMILIES, MODELS = ('G5', 'Season', 'History', 'Trend', 'Size'), ('Ridge', 'ET')
ARMS = tuple(f'{family}-{model}' for family in FAMILIES for model in MODELS)
FEATURE_NAMES: dict[str, tuple[str, ...]] = {'G5': ('POS', 'nuclear', 'hydro', 'wind', 'solar')}
FEATURE_NAMES['Season'] = FEATURE_NAMES['G5']+('month_sin', 'month_cos')
FEATURE_NAMES['History'] = (FEATURE_NAMES['Season']+
    tuple(f'lag{lag}_{name}' for lag in range(3) for name in FEATURE_NAMES['G5'])+
    tuple(f'lag{lag}_month_age' for lag in range(3))+
    tuple(f'lag{lag}_release_age_days' for lag in range(3)))
FEATURE_NAMES['Trend'] = FEATURE_NAMES['History']+('log_MW0_MW1', 'log_MW1_MW2')
FEATURE_NAMES['Size'] = FEATURE_NAMES['Trend']+('log_MW0',)


def _utc(value: Any) -> pd.Timestamp:
    at = pd.Timestamp(value)
    if pd.isna(at) or at.tzinfo is None:
        raise ValueError('Explicit finite timezone-aware monthly metadata required')
    return at.tz_convert('UTC')


def _simplex(value: np.ndarray, size: int) -> None:
    if (value.shape != (size,) or not np.isfinite(value).all() or np.any(value < 0)
            or not np.isclose(value.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError('Finite nonnegative simplex required')


def _visible(row: Mapping[str, str]) -> np.ndarray:
    zero = np.array([float(row[SHARES[i]]) for i in ZERO])
    complement = 1.-zero.sum()
    if -1e-12 <= complement < 0:
        complement = 0.
    result = np.r_[complement, zero]
    _simplex(result, 5)
    return result


def _key(region: str, at: pd.Timestamp) -> str:
    return f'{region}/{at.year:04d}-{at.month:02d}'


def response(row: Mapping[str, str]) -> np.ndarray:
    """Explicit hidden-label accessor, for training or scoring after sealing only."""
    mix = np.array([float(row[column]) for column in SHARES])
    _simplex(mix, 8)
    mass = mix[POS].sum()
    if mass <= 1e-12:
        raise ValueError('Positive POS response required')
    return mix[POS]/mass


def build_cases(rawrows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Build the common complete-history cohort without reading hidden shares.

    Only latest three metadata-eligible contexts are converted to numbers.
    Current total_mwh is never read for that case. A row can legally supply a
    past volume to a later case. Missing history skips; invalid eligible values
    fail, including missing or nonpositive past totals.
    """
    monthly: dict[str, list[tuple]] = {region: [] for region in SOURCES}
    seen: set[str] = set()
    for row in rawrows:
        region = row['region']
        if region not in SOURCES:
            continue
        start = _utc(row['period_start'])
        if start < START:
            continue
        end, release = _utc(row['period_end']), _utc(row['available_at'])
        if (start != start.normalize() or start.day != 1 or
                end != start+pd.offsets.MonthBegin(1) or release < end):
            raise ValueError('Whole calendar month and release at or after end required')
        key = _key(region, start)
        if key in seen:
            raise ValueError('Duplicate source region-month/vintage')
        seen.add(key)
        monthly[region].append((start, end, release, key, row))
    for records in monthly.values():
        records.sort(key=lambda item: item[0])
    kept, skips = [], []
    for region in SOURCES:
        records = monthly[region]
        for start, end, release, key, row in records:
            if not (START <= start < END and end <= CUTOFF and release <= CUTOFF):
                continue
            current = _visible(row)
            if current[0] <= 1e-12:
                skips.append(dict(key=key, reason='zero_POS'))
                continue
            contexts = [item for item in records if item[0] < start and
                        item[1] <= start and item[2] <= start][-3:][::-1]
            if len(contexts) < 3:
                skips.append(dict(key=key, reason='missing_three_legal_contexts',
                                  legal_context_count=len(contexts)))
                continue
            context_g = np.stack([_visible(item[4]) for item in contexts])
            mw = np.array([float(item[4]['total_mwh'])/
                           ((item[1]-item[0])/pd.Timedelta(hours=1)) for item in contexts])
            if not np.isfinite(mw).all() or np.any(mw <= 0):
                raise ValueError('Finite strictly positive eligible past monthly volumes required')
            kept.append((region, start, end, release, key, current, contexts, context_g, mw))
    n = len(kept)
    current = np.array([item[5] for item in kept], float).reshape(n, 5)
    context_g = np.array([item[7] for item in kept], float).reshape(n, 3, 5)
    mw = np.array([item[8] for item in kept], float).reshape(n, 3)
    origin = np.array([item[1].value for item in kept], np.int64)
    context_start = np.array([[ctx[0].value for ctx in item[6]] for item in kept], np.int64).reshape(n, 3)
    context_end = np.array([[ctx[1].value for ctx in item[6]] for item in kept], np.int64).reshape(n, 3)
    context_release = np.array([[ctx[2].value for ctx in item[6]] for item in kept], np.int64).reshape(n, 3)
    month_age = np.array([[item[1].year*12+item[1].month-ctx[0].year*12-ctx[0].month
                           for ctx in item[6]] for item in kept], float).reshape(n, 3)
    release_age = (origin[:, None]-context_release)/(86400.*1e9)
    phase = np.array([2*np.pi*(item[1].month-1)/12 for item in kept])
    x = {'G5': current.copy()}
    x['Season'] = np.column_stack([x['G5'], np.sin(phase), np.cos(phase)])
    x['History'] = np.column_stack([x['Season'], context_g.reshape(n, 15), month_age, release_age])
    logs = np.log(mw)
    x['Trend'] = np.column_stack([x['History'], logs[:, 0]-logs[:, 1], logs[:, 1]-logs[:, 2]])
    x['Size'] = np.column_stack([x['Trend'], logs[:, 0]])
    return dict(X_by_family=x, feature_names_by_family=FEATURE_NAMES.copy(),
        keys=[item[4] for item in kept], regions=np.array([item[0] for item in kept], dtype=str),
        origin_ns=origin, period_start_ns=origin.copy(),
        period_end_ns=np.array([item[2].value for item in kept], np.int64),
        available_ns=np.array([item[3].value for item in kept], np.int64),
        current_g=current, context_keys=np.array([[ctx[3] for ctx in item[6]] for item in kept], dtype=str).reshape(n, 3),
        context_start_ns=context_start, context_end_ns=context_end, context_release_ns=context_release,
        context_g=context_g, context_MW=mw, context_month_age=month_age,
        context_release_age_days=release_age,
        audit=dict(skips=skips, case_count=n, hidden_fields_read=False, current_total_read=False,
                   diagnostic='oracle-current-group conditional diagnostic; not month-start forecast',
                   cutoff=CUTOFF.isoformat(), label_start=START.isoformat(), label_end_exclusive=END.isoformat()))


def project(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the four-fuel simplex, once after regression."""
    x = np.asarray(values, float)
    if x.ndim < 1 or x.shape[-1] != 4 or not np.isfinite(x).all():
        raise ValueError('Finite four-coordinate regression output required')
    flat = x.reshape(-1, 4)
    ordered = np.sort(flat, axis=1)[:, ::-1]
    candidates = (np.cumsum(ordered, axis=1)-1)/np.arange(1, 5)
    rho = (ordered > candidates).sum(1)-1
    result = np.maximum(flat-candidates[np.arange(len(flat)), rho, None], 0)
    result /= result.sum(1)[:, None]
    return result.reshape(x.shape)


def fit_predict(rows: Sequence[Mapping[str, str]], cases: Mapping[str, Any], *, outer: str) -> dict[str, Any]:
    """Fit ten ordinary heads; never convert any outer hidden response field."""
    if outer not in SOURCES:
        raise ValueError('Registered outer source required')
    regions = np.asarray(cases['regions'])
    if not set(regions) <= set(SOURCES) or len(set(cases['keys'])) != len(regions):
        raise ValueError('Unique source cases required')
    train_ids, query_ids = np.flatnonzero(regions != outer), np.flatnonzero(regions == outer)
    if not len(train_ids) or not len(query_ids):
        raise ValueError('Nonempty training and outer query cohort required')
    wanted = {cases['keys'][i] for i in train_ids}
    labels: dict[str, np.ndarray] = {}
    for row in rows:
        region = row['region']
        if region == outer or region not in SOURCES:
            continue
        key = _key(region, _utc(row['period_start']))
        if key not in wanted:
            continue
        if key in labels:
            raise ValueError('Duplicate training response')
        labels[key] = response(row)
    if set(labels) != wanted:
        raise ValueError('Missing training responses')
    y = np.stack([labels[cases['keys'][i]] for i in train_ids])
    train_regions = regions[train_ids]
    if set(train_regions) != set(SOURCES)-{outer}:
        raise ValueError('All six non-outer training source regions required')
    represented = len(set(train_regions))
    weight = np.array([len(train_ids)/(represented*np.count_nonzero(train_regions == r)) for r in train_regions])
    predictions, fits = [], []
    for family in FAMILIES:
        all_x = np.asarray(cases['X_by_family'][family], float)
        if all_x.shape != (len(regions), len(FEATURE_NAMES[family])) or not np.isfinite(all_x).all():
            raise ValueError('Finite common-cohort feature matrices required')
        train_x, query_x = all_x[train_ids], all_x[query_ids]
        for name in MODELS:
            mean, scale = np.zeros(train_x.shape[1]), np.ones(train_x.shape[1])
            if name == 'Ridge':
                mean = np.average(train_x, axis=0, weights=weight)
                scale = np.sqrt(np.average((train_x-mean)**2, axis=0, weights=weight))
                scale[scale <= 1e-12] = 1.
                parameters = dict(alpha=1., fit_intercept=True)
                model = Ridge(**parameters)
                fit_x, predict_x = (train_x-mean)/scale, (query_x-mean)/scale
            else:
                parameters = dict(n_estimators=128, min_samples_leaf=2, max_features=1., random_state=17023, n_jobs=1)
                model = ExtraTreesRegressor(**parameters)
                fit_x, predict_x = train_x, query_x
            model.fit(fit_x, y, sample_weight=weight)
            predictions.append(model.predict(predict_x))
            meta = dict(arm=f'{family}-{name}', family=family, model=name, excluded_outer=outer,
                training_keys=[cases['keys'][i] for i in train_ids], query_keys=[cases['keys'][i] for i in query_ids],
                training_indices=train_ids.copy(), query_indices=query_ids.copy(), training_regions=train_regions.copy(),
                training_X=train_x.copy(), query_X=query_x.copy(), training_y=y.copy(), sample_weights=weight.copy(),
                scaler_mean=mean.copy(), scaler_scale=scale.copy(), parameters=parameters,
                feature_names=FEATURE_NAMES[family],
                weight_definition='N / (represented training regions * training region month count)',
                training_period_start_ns=np.asarray(cases['origin_ns'])[train_ids].copy(),
                training_period_end_ns=np.asarray(cases['period_end_ns'])[train_ids].copy(),
                training_available_ns=np.asarray(cases['available_ns'])[train_ids].copy(),
                audit=dict(outer_hidden_labels_read=False, train_only_scaler=True, train_only_weights=True))
            if name == 'Ridge':
                meta.update(coefficients=model.coef_.copy(), intercept=model.intercept_.copy())
            else:
                meta['feature_importances'] = model.feature_importances_.copy()
            fits.append(meta)
    raw = np.stack(predictions)
    ratios = project(raw)
    provenance = {key: np.asarray(cases[key])[query_ids].copy() for key in
                  ('origin_ns', 'period_end_ns', 'available_ns', 'current_g', 'context_keys',
                   'context_start_ns', 'context_end_ns', 'context_release_ns', 'context_g', 'context_MW')}
    provenance['keys'] = [cases['keys'][i] for i in query_ids]
    provenance['regions'] = regions[query_ids].copy()
    return dict(ratios_raw=raw, ratios=ratios, EF_point=ratios@EF[POS],
                projection_l1=np.abs(ratios-raw).sum(-1), train_ids=train_ids, query_ids=query_ids,
                query_keys=provenance['keys'], query_provenance=provenance, fits=fits,
                audit=dict(outer=outer, fit_count=len(fits), outer_hidden_labels_read=False,
                           diagnostic='oracle-current-group conditional diagnostic; not month-start forecast'))
