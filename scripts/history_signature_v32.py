"""Twelve-month visible history signatures, with pure row/array interfaces.

The matched L24 heads use six non-outer source labels. Forward current G5 is
supplied by MG forecast particles; raw query hidden fuels are never features.
No file I/O, hourly values, unknown regions, or learned gate.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from scripts import coarse_context_v30 as prior
from scripts import forward_context_v31 as forward

SOURCES, POS, ZERO, EF, SHARES = prior.SOURCES, prior.POS, prior.ZERO, prior.EF, prior.SHARES
START, END, CUTOFF = prior.START, prior.END, prior.CUTOFF
MODELS = ('Ridge', 'ET')
FAMILIES = ('G5', 'Trend', 'Size', 'LongTotal', 'JointLevel', 'JointTrend',
            'JointAmplitude', 'JointShape', 'PhaseScramble')
ARMS = tuple(f'{family}-{model}' for family in FAMILIES for model in MODELS)
SUMMARY_NAMES = ('log1p_mean_MW', 'q10_over_mean', 'sd_over_mean',
                 'normalized_calendar_trend', 'detrended_annual_cos', 'detrended_annual_sin')
FEATURE_NAMES = {family: prior.FEATURE_NAMES[family] for family in ('G5', 'Trend', 'Size')}
FEATURE_NAMES['LongTotal'] = FEATURE_NAMES['Size']+('history_span', 'history_maxgap')+tuple('total_'+x for x in SUMMARY_NAMES)
FEATURE_NAMES['JointLevel'] = FEATURE_NAMES['LongTotal']+tuple('thermal_'+x for x in SUMMARY_NAMES[:3])
FEATURE_NAMES['JointTrend'] = FEATURE_NAMES['JointLevel']+('thermal_'+SUMMARY_NAMES[3],)
FEATURE_NAMES['JointAmplitude'] = FEATURE_NAMES['JointTrend']+('thermal_detrended_annual_amplitude',)
FEATURE_NAMES['JointShape'] = FEATURE_NAMES['JointTrend']+tuple('thermal_'+x for x in SUMMARY_NAMES[4:])
FEATURE_NAMES['PhaseScramble'] = FEATURE_NAMES['JointTrend']+('thermal_rotated_annual_cos', 'thermal_rotated_annual_sin')


def history_summary(values: np.ndarray, month_age: np.ndarray, months: np.ndarray) -> dict[str, Any]:
    """Fixed six-coordinate summary; harmonic fit follows linear detrending.

    Calendar time increases towards the origin. Rank/condition describe H only;
    they are audit fields, never predictors or evidence of identified seasons.
    """
    v, age, month = np.asarray(values, float), np.asarray(month_age, float), np.asarray(months)
    if (v.shape != (12,) or age.shape != (12,) or month.shape != (12,) or
            not np.isfinite(v).all() or np.any(v < 0) or not np.isfinite(age).all() or
            np.any(age <= 0) or len(set(age)) != 12 or not np.isfinite(month).all() or
            np.any(month != np.floor(month)) or np.any((month < 1) | (month > 12))):
        raise ValueError('Twelve finite nonnegative volumes at distinct positive calendar ages required')
    phase = 2*np.pi*(month-1)/12
    h = np.column_stack([np.ones(12), np.cos(phase), np.sin(phase)])
    mean = v.mean()
    tc = -age+age.mean()
    if mean == 0:
        result = np.zeros(6)
        _, _, rank, singular = np.linalg.lstsq(h, np.zeros(12), rcond=None)
    else:
        z = v/mean-1
        beta = tc@z/(tc@tc)
        coef, _, rank, singular = np.linalg.lstsq(h, z-beta*tc, rcond=None)
        result = np.array([np.log1p(mean), np.quantile(v, .1, method='linear')/mean,
                           v.std(ddof=0)/mean, beta, coef[1], coef[2]])
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else float('inf')
    if not np.isfinite(result).all():
        raise ValueError('History signature must be finite')
    return dict(summary=result, harmonic_rank=int(rank), harmonic_condition=condition)


def phase_angle(region: str, origin: pd.Timestamp) -> float:
    """Per-case fixed phase control; no hidden data or outcome selects its angle."""
    message = 'history-signature-v32-phase/'+region+'/'+origin.strftime('%Y-%m')
    integer = int.from_bytes(hashlib.sha256(message.encode('utf-8')).digest()[:8], 'big')
    return 2*np.pi*(integer/2**64)


def _metadata(rawrows: Sequence[Mapping[str, str]]) -> dict[str, list[tuple]]:
    result: dict[str, list[tuple]] = {region: [] for region in SOURCES}
    seen = set()
    for row in rawrows:
        region = row['region']
        if region not in SOURCES:
            continue
        at = prior._utc(row['period_start'])
        if at < START:
            continue
        end, release = prior._utc(row['period_end']), prior._utc(row['available_at'])
        if at != at.normalize() or at.day != 1 or end != at+pd.offsets.MonthBegin(1) or release < end:
            raise ValueError('Whole calendar months with release at or after end required')
        key = prior._key(region, at)
        if key in seen:
            raise ValueError('Duplicate source region-month/vintage')
        seen.add(key)
        result[region].append((at, end, release, key, row))
    for values in result.values():
        values.sort(key=lambda item: item[0])
    return result


def _legal(records: list[tuple], origin: pd.Timestamp) -> list[tuple]:
    return [row for row in records if row[0] < origin and row[1] <= origin and row[2] <= origin][-12:][::-1]


def _contexts(histories: list[list[tuple]], dates: pd.DatetimeIndex,
              regions: Sequence[str]) -> dict[str, np.ndarray]:
    n = len(dates)
    if len(histories) != n or len(regions) != n or any(len(row) != 12 for row in histories):
        raise ValueError('Twelve legal past monthly contexts required for every case')
    g = np.array([[prior._visible(item[4]) for item in row] for row in histories], float).reshape(n, 12, 5)
    mw = np.array([[float(item[4]['total_mwh'])/((item[1]-item[0])/pd.Timedelta(hours=1))
                    for item in row] for row in histories], float).reshape(n, 12)
    if not np.isfinite(mw).all() or np.any(mw <= 0):
        raise ValueError('Finite strictly positive selected past total volumes required')
    thermal = mw*g[:, :, 0]
    start = np.array([[item[0].value for item in row] for row in histories], np.int64).reshape(n, 12)
    end = np.array([[item[1].value for item in row] for row in histories], np.int64).reshape(n, 12)
    release = np.array([[item[2].value for item in row] for row in histories], np.int64).reshape(n, 12)
    ages = np.array([[at.year*12+at.month-item[0].year*12-item[0].month
                      for item in row] for at, row in zip(dates, histories)], float).reshape(n, 12)
    months = np.array([[item[0].month for item in row] for row in histories], np.int64).reshape(n, 12)
    total_summaries = [history_summary(value, age, month) for value, age, month in zip(mw, ages, months)]
    thermal_summaries = [history_summary(value, age, month) for value, age, month in zip(thermal, ages, months)]
    origin_ns = dates.as_unit('ns').asi8.copy()
    return dict(origin_ns=origin_ns,
        context_keys=np.array([[item[3] for item in row] for row in histories], dtype=str).reshape(n, 12),
        context_start_ns=start, context_end_ns=end, context_release_ns=release,
        context_g=g, context_MW=mw, context_thermal_MW=thermal, context_month_age=ages,
        context_release_age_days=(origin_ns[:, None]-release)/(86400.*1e9),
        history_span=np.array([age.max()-age.min()+1 for age in ages]),
        history_maxgap=np.array([np.diff(np.sort(age)).max() for age in ages]),
        total_summary=np.array([item['summary'] for item in total_summaries], float).reshape(n, 6),
        thermal_summary=np.array([item['summary'] for item in thermal_summaries], float).reshape(n, 6),
        harmonic_rank=np.array([item['harmonic_rank'] for item in total_summaries], np.int64),
        harmonic_condition=np.array([item['harmonic_condition'] for item in total_summaries], float),
        phase_angle=np.array([phase_angle(region, at) for region, at in zip(regions, dates)]))


def _features(current: np.ndarray, context: Mapping[str, np.ndarray],
              dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    n = len(current)

    def add(base: np.ndarray, suffix: np.ndarray) -> np.ndarray:
        view = suffix.reshape((n,)+(1,)*(base.ndim-2)+(suffix.shape[-1],))
        return np.concatenate([base, np.broadcast_to(view, base.shape[:-1]+(suffix.shape[-1],))], axis=-1)

    phase = 2*np.pi*(dates.month.to_numpy()-1)/12
    logs = np.log(context['context_MW'][:, :3])
    suffix = np.column_stack([np.sin(phase), np.cos(phase), context['context_g'][:, :3].reshape(n, 15),
        context['context_month_age'][:, :3], context['context_release_age_days'][:, :3],
        logs[:, 0]-logs[:, 1], logs[:, 1]-logs[:, 2]])
    x = dict(G5=current.copy())
    x['Trend'] = add(x['G5'], suffix)
    x['Size'] = add(x['Trend'], logs[:, :1])
    x['LongTotal'] = add(x['Size'], np.column_stack([context['history_span'],context['history_maxgap'],context['total_summary']]))
    x['JointLevel'] = add(x['LongTotal'], context['thermal_summary'][:, :3])
    x['JointTrend'] = add(x['JointLevel'], context['thermal_summary'][:, 3:4])
    coefficients = context['thermal_summary'][:, 4:6]
    x['JointAmplitude'] = add(x['JointTrend'], np.sqrt((coefficients**2).sum(1))[:, None])
    x['JointShape'] = add(x['JointTrend'], coefficients)
    c, s = coefficients[:, 0], coefficients[:, 1]
    cosine, sine = np.cos(context['phase_angle']), np.sin(context['phase_angle'])
    rotated = np.column_stack([c*cosine-s*sine, c*sine+s*cosine])
    x['PhaseScramble'] = add(x['JointTrend'], rotated)
    return x


def build_cases(rawrows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """One matched L24 cohort for every family; current total is never a feature."""
    monthly = _metadata(rawrows)
    kept, histories, current, skips = [], [], [], []
    for region in SOURCES:
        for row in monthly[region]:
            at, end, release, key, raw = row
            if not (START <= at < END and end <= CUTOFF and release <= CUTOFF):
                continue
            g = prior._visible(raw)
            if g[0] <= 1e-12:
                skips.append(dict(key=key, reason='zero_POS'))
                continue
            history = _legal(monthly[region], at)
            if len(history) != 12:
                skips.append(dict(key=key, reason='missing_twelve_legal_contexts', legal_context_count=len(history)))
                continue
            kept.append((region, at, end, release, key));histories.append(history);current.append(g)
    n = len(kept)
    dates = pd.DatetimeIndex([row[1] for row in kept]).as_unit('ns')
    regions = np.array([row[0] for row in kept], dtype=str)
    context = _contexts(histories, dates, regions)
    g = np.array(current, float).reshape(n, 5)
    return dict(X_by_family=_features(g, context, dates), feature_names_by_family=FEATURE_NAMES.copy(),
        keys=[row[4] for row in kept], regions=regions, current_g=g,
        period_start_ns=context['origin_ns'].copy(), period_end_ns=np.array([row[2].value for row in kept], np.int64),
        available_ns=np.array([row[3].value for row in kept], np.int64), **context,
        audit=dict(case_count=n, skips=skips, context_rows=12, hidden_feature_fields_read=False,
            current_total_read=False, diagnostic='Source L24 current-group supervision; forward query uses supplied MG'))


def query_features(rawrows: Sequence[Mapping[str, str]], *, region: str,
                   origins: Sequence[Any], mg_draws: np.ndarray) -> dict[str, np.ndarray]:
    """Forecast-only context/features; origin and MG guards precede raw access."""
    dates = forward._dates(origins)
    draws = np.asarray(mg_draws, float)
    if region not in SOURCES or draws.shape != (len(dates),128,5):
        raise ValueError('Registered source and (origins,128,5) MG forecast particles required')
    forward._simplex_rows(draws, 5)
    monthly = _metadata(rawrows)
    histories = [_legal(monthly[region], at) for at in dates]
    context = _contexts(histories, dates, [region]*len(dates))
    return dict(MG_draws=draws.copy(), **context,
                **{'query_feature_'+name:value for name,value in _features(draws,context,dates).items()})


def fit_predict(rawrows: Sequence[Mapping[str, str]], cases: Mapping[str, Any], *,
                outer: str, origins: Sequence[Any], mg_draws: np.ndarray) -> dict[str, Any]:
    """Fit eighteen matched source heads; then project once per forecast particle."""
    query = query_features(rawrows,region=outer,origins=origins,mg_draws=mg_draws)
    regions = np.asarray(cases['regions'])
    keys = list(cases['keys'])
    if len(keys) != len(regions) or len(set(keys)) != len(keys) or not set(regions) <= set(SOURCES):
        raise ValueError('Unique source region-month training cases required')
    ids = np.flatnonzero(regions != outer)
    labels = regions[ids]
    if set(labels) != set(SOURCES)-{outer}:
        raise ValueError('All six non-outer source training regions required')
    training_keys = [keys[i] for i in ids]
    metadata = {}
    for field in ('origin_ns','period_end_ns','available_ns'):
        values = np.asarray(cases[field])
        if values.shape != (len(keys),) or not np.issubdtype(values.dtype,np.integer):
            raise ValueError('Integer nanosecond source case metadata required')
        metadata[field] = values[ids].copy()
    expected_metadata = {}
    for i, key in enumerate(training_keys):
        at = pd.Timestamp(int(metadata['origin_ns'][i]),unit='ns',tz='UTC')
        end = int(metadata['period_end_ns'][i]); available = int(metadata['available_ns'][i])
        if (pd.isna(at) or not START <= at < END or at.day != 1 or at != at.normalize() or
                end != (at+pd.offsets.MonthBegin(1)).value or not end <= available <= CUTOFF.value or
                key != prior._key(labels[i],at)):
            raise ValueError('Released L24 source cases excluding outer required')
        expected_metadata[key] = (at.value,end,available)
    wanted, responses = set(training_keys), {}
    for row in rawrows:
        region = row['region']
        if region == outer or region not in SOURCES:
            continue
        key = prior._key(region, prior._utc(row['period_start']))
        if key in wanted:
            if key in responses:
                raise ValueError('Duplicate training response')
            actual_metadata = tuple(prior._utc(row[field]).value for field in
                                    ('period_start','period_end','available_at'))
            if actual_metadata != expected_metadata[key]:
                raise ValueError('Training row metadata differ from matched source case')
            responses[key] = prior.response(row)
    if set(responses) != wanted:
        raise ValueError('Missing frozen-cohort source training responses')
    y = np.stack([responses[key] for key in training_keys])
    w = np.array([len(ids)/(6*np.count_nonzero(labels == region)) for region in labels])
    predictions, fits = [], []
    n = len(query['origin_ns'])
    for family in FAMILIES:
        x = np.asarray(cases['X_by_family'][family], float)
        if x.shape != (len(keys),len(FEATURE_NAMES[family])) or not np.isfinite(x).all():
            raise ValueError('Finite matched-cohort training feature matrices required')
        train = x[ids]
        future = query['query_feature_'+family].reshape(n*128,-1)
        for name in MODELS:
            mean, scale = np.zeros(train.shape[1]), np.ones(train.shape[1])
            parameters = forward.PARAMETERS[name].copy()
            if name == 'Ridge':
                mean = np.average(train, axis=0, weights=w)
                scale = np.sqrt(np.average((train-mean)**2, axis=0, weights=w))
                scale[scale <= 1e-12] = 1.
                model = Ridge(**parameters)
                fit_x, predict_x = (train-mean)/scale, (future-mean)/scale
            else:
                model = ExtraTreesRegressor(**parameters)
                fit_x, predict_x = train, future
            model.fit(fit_x, y, sample_weight=w)
            predictions.append(model.predict(predict_x).reshape(n,128,4))
            record = dict(arm=f'{family}-{name}', family=family, model=name, excluded_outer=outer,
                training_indices=ids.copy(), training_keys=training_keys.copy(), training_regions=labels.copy(),
                training_X=train.copy(), training_y=y.copy(), sample_weights=w.copy(),
                scaler_mean=mean.copy(), scaler_scale=scale.copy(), parameters=parameters,
                feature_names=FEATURE_NAMES[family],
                training_period_start_ns=np.asarray(cases['origin_ns'])[ids].copy(),
                training_period_end_ns=np.asarray(cases['period_end_ns'])[ids].copy(),
                training_available_ns=np.asarray(cases['available_ns'])[ids].copy(),
                audit=dict(train_only_preprocessing=True, outer_hidden_labels_read=False, common_cohort=True))
            if name == 'Ridge':
                record.update(coefficients=model.coef_.copy(), intercept=model.intercept_.copy())
            else:
                record['feature_importances'] = model.feature_importances_.copy()
            fits.append(record)
    raw = np.stack(predictions)
    ratios = prior.project(raw)
    ef = ratios@EF[POS]
    cif = query['MG_draws'][None,...,0]*ef
    return dict(ratios_raw=raw, ratios=ratios, EF_draws=ef, CIF_draws=cif,
        EF_point=ef.mean(-1), CIF_point=cif.mean(-1), projection_l1=np.abs(ratios-raw).sum(-1),
        **query, fits=fits, train_ids=ids,
        audit=dict(region=outer, arms=list(ARMS), fit_count=18, context_rows=12,
            oracle_current_group=False, forward_forecast=True, hourly_values_read=False,
            target_values_opened=False, unseen_values_opened=False, current_group_truth_read=False,
            current_total_read=False, raw_hidden_feature_fields_read=False, new_2025_training_labels_used=False,
            common_cohort=True, phase_control='Case-key SHA256 rotation preserves thermal harmonic amplitude',
            harmonic_scope='Sequential linear detrending then calendar harmonic fit; rank/condition are audit only'))
