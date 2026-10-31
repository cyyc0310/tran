"""Pure fixed-donor coarse cohorts and six ordinary budget-controlled heads."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from scripts import coarse_context_v30 as v30
from scripts import fleet_information_v34 as v34
from scripts import forward_context_v31 as v31
from scripts.donor_inventory_v41 import ALL_SOURCES, DONOR_SOURCES, LEGACY_SOURCES


FAMILIES, MODELS = ('Size', 'SizeCap', 'DirectEF'), ('Ridge', 'ET')
ARMS = tuple(f'{family}-{model}' for family in FAMILIES for model in MODELS)
FEATURE_NAMES = {family: v34.FEATURE_NAMES[family] for family in FAMILIES}
PARAMETERS = {name: v34.PARAMETERS[name].copy() for name in MODELS}
EF_POS = v30.EF[v30.POS]


def build_cases(rawrows: Sequence[Mapping[str, str]], *, sources: tuple[str, ...] = ALL_SOURCES) -> dict[str, Any]:
    """V30 cohort/feature order with only the fixed seven/seventeen source tuple changed."""
    if type(sources) is not tuple or sources not in (LEGACY_SOURCES, ALL_SOURCES):
        raise ValueError('sources must equal ordered legacy-seven or fixed-seventeen tuple')
    monthly: dict[str, list[tuple]] = {region: [] for region in sources}
    seen: set[str] = set()
    for row in rawrows:
        region = row['region']
        if region not in sources:
            continue
        start = v30._utc(row['period_start'])
        if start < v30.START:
            continue
        end, release = v30._utc(row['period_end']), v30._utc(row['available_at'])
        if (start != start.normalize() or start.day != 1 or
                end != start+pd.offsets.MonthBegin(1) or release < end):
            raise ValueError('Whole calendar month and release at or after end required')
        key = v30._key(region, start)
        if key in seen:
            raise ValueError('Duplicate source region-month/vintage')
        seen.add(key)
        monthly[region].append((start, end, release, key, row))
    for records in monthly.values():
        records.sort(key=lambda item: item[0])
    kept, skips = [], []
    for region in sources:
        records = monthly[region]
        for start, end, release, key, row in records:
            if not (v30.START <= start < v30.END and end <= v30.CUTOFF and release <= v30.CUTOFF):
                continue
            current = v30._visible(row)
            if current[0] <= 1e-12:
                skips.append(dict(key=key, reason='zero_POS'))
                continue
            contexts = [item for item in records if item[0] < start and
                        item[1] <= start and item[2] <= start][-3:][::-1]
            if len(contexts) < 3:
                skips.append(dict(key=key, reason='missing_three_legal_contexts', legal_context_count=len(contexts)))
                continue
            context_g = np.stack([v30._visible(item[4]) for item in contexts])
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
    return dict(X_by_family=x, feature_names_by_family=v30.FEATURE_NAMES.copy(),
        keys=[item[4] for item in kept], regions=np.array([item[0] for item in kept], dtype=str),
        origin_ns=origin, period_start_ns=origin.copy(),
        period_end_ns=np.array([item[2].value for item in kept], np.int64),
        available_ns=np.array([item[3].value for item in kept], np.int64), current_g=current,
        context_keys=np.array([[ctx[3] for ctx in item[6]] for item in kept], dtype=str).reshape(n, 3),
        context_start_ns=context_start, context_end_ns=context_end, context_release_ns=context_release,
        context_g=context_g, context_MW=mw, context_month_age=month_age, context_release_age_days=release_age,
        audit=dict(skips=skips, case_count=n, hidden_fields_read=False, current_total_read=False,
                   diagnostic='oracle-current-group conditional diagnostic; not month-start forecast',
                   cutoff=v30.CUTOFF.isoformat(), label_start=v30.START.isoformat(), label_end_exclusive=v30.END.isoformat()))


def _real(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f'Real {name} required')
    try:
        array = np.asarray(raw, float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Real numeric {name} required') from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'Finite {name} with shape {shape} required')
    return array


def fit_heads(XSize: np.ndarray, Q: np.ndarray, Cpos: np.ndarray, y: np.ndarray,
              regions: np.ndarray, query_features: Mapping[str, np.ndarray], *, outer: str) -> dict[str, Any]:
    """Fixed total training weight, actual eligible donor count; caller binds keys/dates."""
    regions = np.asarray(regions, dtype=str)
    if (outer not in LEGACY_SOURCES or regions.ndim != 1 or not len(regions)
            or any(source not in ALL_SOURCES or source == outer for source in regions)):
        raise ValueError('Registered allowed training sources excluding the legacy outer required')
    n = len(regions)
    XSize = _real(XSize, (n, 31), 'Size training features')
    Q = _real(Q, (n, 4), 'training capacity shares')
    Cpos = _real(Cpos, (n,), 'training POS capacities')
    y = _real(y, (n, 4), 'training responses')
    v31._simplex_rows(Q, 4); v31._simplex_rows(y, 4)
    if np.any(Cpos <= 0):
        raise ValueError('Strictly positive training POS capacities required; no row drops')
    train = {'Size': XSize.copy(), **{k: v for k, v in v34._features(XSize, Q, Cpos).items() if k in FAMILIES}}
    try:
        query = {family: _real(query_features[family], (6, 128, len(FEATURE_NAMES[family])),
                              family+' query features') for family in FAMILIES}
    except KeyError as exc:
        raise ValueError('Missing frozen query feature family') from exc
    if (not np.array_equal(query['DirectEF'], query['SizeCap'])
            or not np.array_equal(query['SizeCap'][..., :31], query['Size'])):
        raise ValueError('Frozen query Size/SizeCap/DirectEF features must align')
    v31._simplex_rows(query['SizeCap'][..., 31:35], 4)
    if np.any(query['SizeCap'][..., -1] < 0):
        raise ValueError('Nonnegative frozen query inventory pressure required')
    mass = 156 if outer == 'FPL' else 154
    source_count = len(set(regions))
    weights = np.array([mass/(source_count*np.count_nonzero(regions == source)) for source in regions])
    records, ef_raw, ef, projections, kinds, raw_ratios, ratios, ratio_arms = [], [], [], [], [], [], [], []
    for family in FAMILIES:
        x = train[family]
        target = y@EF_POS/1000 if family == 'DirectEF' else y
        for name in MODELS:
            arm = family+'-'+name
            mean, scale = np.zeros(x.shape[1]), np.ones(x.shape[1])
            if name == 'Ridge':
                mean = np.average(x, axis=0, weights=weights)
                scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
                scale[scale <= 1e-12] = 1.
                model = Ridge(**PARAMETERS[name])
            else:
                model = ExtraTreesRegressor(**PARAMETERS[name])
            model.fit((x-mean)/scale, target, sample_weight=weights)
            flat = query[family].reshape(-1, x.shape[1])
            shape = (6*128,) if family == 'DirectEF' else (6*128, 4)
            raw = _real(model.predict((flat-mean)/scale), shape, 'raw query prediction')
            if family == 'DirectEF':
                raw = raw.reshape(6, 128); projected = np.clip(raw, .4, 1.)
                ef_raw.append(raw*1000); ef.append(projected*1000)
                projections.append(np.abs(raw-projected)); kinds.append('scalar_EF')
            else:
                raw = raw.reshape(6, 128, 4); projected = v30.project(raw)
                ratio_arms.append(arm); raw_ratios.append(raw); ratios.append(projected)
                ef_raw.append(raw@EF_POS); ef.append(projected@EF_POS)
                projections.append(np.abs(raw-projected).sum(-1)); kinds.append('simplex')
            record = dict(arm=arm, family=family, model=name, excluded_outer=outer,
                parameters=PARAMETERS[name].copy(), feature_names=FEATURE_NAMES[family],
                training_X=x.copy(), training_y=target.copy(), training_regions=regions.copy(),
                sample_weights=weights.copy(), scaler_mean=mean, scaler_scale=scale,
                projection_kind=kinds[-1], raw_query_prediction=raw.copy(), projected_query_prediction=projected.copy())
            if name == 'Ridge':
                record.update(coefficients=np.asarray(model.coef_).copy(), intercept=np.asarray(model.intercept_).copy())
            else:
                record['feature_importances'] = np.asarray(model.feature_importances_).copy()
            records.append(record)
    return dict(EF_raw=np.stack(ef_raw), EF_draws=np.stack(ef), projection_l1=np.stack(projections),
        projection_kind=np.asarray(kinds), ratio_arms=np.asarray(ratio_arms),
        ratios_raw=np.stack(raw_ratios), ratios=np.stack(ratios), training_features=train,
        sample_weights=weights, audit=dict(outer=outer, arms=ARMS, fits=records, fit_count=6,
            training_source_count=source_count, training_weight_mass=mass, current_totals_used_as_features=False,
            capacity_is_physical_output_bound=False, zero_capacity_response_rows_dropped=False))
