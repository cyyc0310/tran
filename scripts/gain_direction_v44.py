"""Pure retrospective source-gain diagnostic; no I/O or production score access."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from scripts import coarse_context_v30 as v30
from scripts import fleet_information_v34 as v34
from scripts import monthly_transfer_v25 as law
from scripts.donor_inventory_v41 import ALL_SOURCES, LEGACY_SOURCES


CANDIDATES = ('SizeCapET', 'DirectEFET')
TARGETS = ('EF', 'CIF', 'Joint')
META_MODELS = ('Mean', 'HistoryForecast-Ridge', 'HistoryForecast-ET',
               'HistoryForecastCap-Ridge', 'HistoryForecastCap-ET')
ARMS = tuple(f'Gain-{c}-{t}-{model}' for c in CANDIDATES for t in TARGETS for model in META_MODELS)
META_NAMES = {'HistoryForecast': v34.FEATURE_NAMES['Size'][5:]+('MG_POS_mean', 'MG_POS_std'),
              'HistoryForecastCap': v34.FEATURE_NAMES['Size'][5:]+('MG_POS_mean', 'MG_POS_std')+v34.CAPACITY_FEATURE_NAMES}
EF = v30.EF[v30.POS]
PARAMETERS = {name: value.copy() for name, value in v34.PARAMETERS.items()}
SEED, SAMPLES = 17023, 128


def _real(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f'Real {name} required')
    try:
        result = np.asarray(raw, float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Real numeric {name} required') from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'Finite {name} with shape {shape} required')
    return result


def _simplex(value: np.ndarray, size: int) -> None:
    if np.any(value < 0) or not np.allclose(value.sum(-1), 1., atol=1e-12, rtol=0) or value.shape[-1] != size:
        raise ValueError('Nonnegative simplex rows required')


def _weights(regions: np.ndarray, mass: float) -> np.ndarray:
    return np.array([mass/(len(set(regions))*np.count_nonzero(regions == s)) for s in regions])


def source_mg(rawrows: Iterable[Mapping[str, str]], keys: Sequence[str]) -> dict[str, Any]:
    """Per-origin metadata filtering precedes all visible numeric conversion."""
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('Unique nonempty source-month keys required')
    queries = []
    for key in keys:
        try:
            region, month = key.split('/')
            at = pd.Timestamp(month+'-01', tz='UTC')
            if region not in ALL_SOURCES or key != v30._key(region, at) or not v30.START <= at < v30.END:
                raise ValueError('Unregistered source/month')
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError('Fixed source and canonical 2022–Jun24 month keys required') from exc
        queries.append((region, at, key))
    metadata = {s: [] for s in ALL_SOURCES}; seen = set()
    for row in rawrows:
        region = row['region']
        if region not in ALL_SOURCES:
            continue
        start = v30._utc(row['period_start'])
        if start < v30.START:
            continue
        end, release = v30._utc(row['period_end']), v30._utc(row['available_at'])
        if start != start.normalize() or start.day != 1 or end != start+pd.offsets.MonthBegin(1) or release < end:
            raise ValueError('Whole calendar month and legal release metadata required')
        key = v30._key(region, start)
        if key in seen:
            raise ValueError('Duplicate source monthly vintage')
        seen.add(key); metadata[region].append((start, end, release, key, row))
    points, draws, audits = [], [], []
    for region, origin, key in queries:
        legal = sorted((r for r in metadata[region] if r[0] < origin and r[1] <= origin and r[2] <= origin), key=lambda r:r[0])
        part = pd.DataFrame([dict(period_start=r[0], period_end=r[1], available_at=r[2],
                            **dict(zip(law.MG_COLUMNS, v30._visible(r[4])))) for r in legal],
                            columns=['period_start', 'period_end', 'available_at', *law.MG_COLUMNS])
        cache: dict[tuple, np.ndarray] = {}; counts = dict(harmonic_solve_count=0, fallback_count=0, prior_count=0)
        def point_at(history: pd.DataFrame, at: pd.Timestamp) -> np.ndarray:
            ident = (tuple(history.index), at.year, at.month)
            if ident not in cache:
                name = 'prior_count' if not len(history) else 'fallback_count' if len(history) < 12 else 'harmonic_solve_count'
                counts[name] += 1
                cache[ident] = law._point(history, at, law.MG_COLUMNS, np.array([.5, .125, .125, .125, .125]))
            return cache[ident]
        records, history_counts = [], []
        for _, label in part.tail(60).iterrows():
            pseudo = label.period_start+pd.Timedelta(days=min(origin.day, (label.period_end-label.period_start).days)-1)
            history = part.loc[part.period_end.le(pseudo) & part.available_at.le(pseudo)]
            horizon = law._ordinal(pseudo)-law._ordinal(history.period_start.iloc[-1]) if len(history) else 99
            records.append((label[list(law.MG_COLUMNS)].to_numpy(float)-point_at(history, pseudo), horizon))
            history_counts.append(len(history))
        records, history_counts = records[-24:], history_counts[-24:]
        point = point_at(part, origin)
        horizon = law._ordinal(origin)-law._ordinal(part.period_start.iloc[-1]) if len(part) else 99
        if records:
            age = np.arange(len(records)-1, -1, -1, dtype=float)
            gaps = np.array([abs(h-horizon) for _, h in records], float)
            weight = np.exp(-age/12-gaps/1.5); weight /= weight.sum()
            rng = np.random.default_rng(np.random.SeedSequence([SEED & 0xffffffff, (SEED >> 32) & 0xffffffff,
                                                                origin.year, origin.month, origin.day]))
            choice = rng.choice(len(records), size=SAMPLES, p=weight)
            active = part[list(law.MG_COLUMNS)].to_numpy(float).max(0) > 0
            draw = law._simplex_rows(point[None]+np.stack([r for r, _ in records])[choice], active)
        else:
            draw = np.repeat(point[None], SAMPLES, axis=0)
        _real(draw, (128, 5), 'source MG'); _simplex(draw, 5)
        points.append(point); draws.append(draw)
        audits.append(dict(key=key, region=region, origin_ns=origin.value, prefix_count=len(part),
            residual_count=len(records), prefix_keys=[r[3] for r in legal], horizon=horizon,
            residual_history_counts=history_counts, residual_zero_history_count=history_counts.count(0),
            residual_below12_count=sum(n < 12 for n in history_counts), seed=SEED, samples=SAMPLES, **counts))
    return dict(draws=np.stack(draws), points=np.stack(points), auditrows=audits)


def _fit(x: np.ndarray, y: np.ndarray, regions: np.ndarray, query: np.ndarray,
         model_name: str, mass: float) -> tuple[np.ndarray, dict[str, Any]]:
    weights = _weights(regions, mass)
    mean, scale = np.zeros(x.shape[1]), np.ones(x.shape[1])
    if model_name == 'Ridge':
        mean = np.average(x, axis=0, weights=weights)
        scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights)); scale[scale <= 1e-12] = 1.
        model = Ridge(**PARAMETERS['Ridge'])
    else:
        model = ExtraTreesRegressor(**PARAMETERS['ET'])
    _real(mean, (x.shape[1],), 'training scaler mean'); _real(scale, (x.shape[1],), 'training scaler scale')
    model.fit((x-mean)/scale, y, sample_weight=weights)
    raw = np.asarray(model.predict((query-mean)/scale), float)
    _real(raw, (len(query),) if y.ndim == 1 else (len(query), 4), 'model prediction')
    record = dict(model=model_name, parameters=PARAMETERS[model_name].copy(), sample_weights=weights,
                  scaler_mean=mean, scaler_scale=scale)
    if model_name == 'Ridge':
        record.update(coefficients=np.asarray(model.coef_).copy(), intercept=np.asarray(model.intercept_).copy())
    else:
        record['feature_importances'] = np.asarray(model.feature_importances_).copy()
    return raw, record


def fit_inner(X31: np.ndarray, Q: np.ndarray, Cpos: np.ndarray, y4: np.ndarray,
              regions: np.ndarray, sourceMG: np.ndarray, *, outer: str, inner: str) -> dict[str, Any]:
    regions = np.asarray(regions, dtype=str)
    if (regions.ndim != 1 or not len(regions) or outer not in LEGACY_SOURCES or inner not in ALL_SOURCES
            or inner == outer or any(s not in ALL_SOURCES for s in regions)):
        raise ValueError('Registered distinct outer/inner and source metadata required')
    held = np.flatnonzero(regions == inner)
    base_ids = np.flatnonzero(np.isin(regions, LEGACY_SOURCES) & (regions != outer) & (regions != inner))
    train_ids = np.flatnonzero((regions != outer) & (regions != inner))
    if (not len(held) or set(regions[base_ids]) != set(LEGACY_SOURCES)-{outer, inner}
            or len(set(regions[train_ids])) != 14):
        raise ValueError('Complete five/six legacy and fourteen expanded training sources required')
    n, h = len(regions), len(held)
    q = _real(Q, (n, 4), 'inventory shares'); _simplex(q, 4)
    capacity = _real(Cpos, (n,), 'POS capacity')
    if np.any(capacity <= 0): raise ValueError('Positive POS capacity required')
    mg = _real(sourceMG, (n,128,5), 'source MG'); _simplex(mg, 5)
    suffix = _real([X31[int(i)][5:] for i in held], (h,26), 'held legal history')
    size_query = np.concatenate([mg[held], np.broadcast_to(suffix[:,None], (h,128,26))], axis=-1)
    cap_query = v34._features(size_query, q[held], capacity[held])['SizeCap']
    model_ef, fits = [], []
    mass = 156 if outer == 'FPL' else 154
    for arm, family, name, ids in [('RidgePreRouter','Size','Ridge',base_ids),
            ('SizeCapET','SizeCap','ET',train_ids), ('DirectEFET','DirectEF','ET',train_ids)]:
        x = _real([X31[int(i)] for i in ids], (len(ids),31), 'selected training Size')
        y = _real([y4[int(i)] for i in ids], (len(ids),4), 'selected training response'); _simplex(y, 4)
        if family != 'Size': x = v34._features(x, q[ids], capacity[ids])['SizeCap']
        target = y@EF/1000 if family == 'DirectEF' else y
        query = size_query if family == 'Size' else cap_query
        raw, record = _fit(x, target, regions[ids], query.reshape(h*128,-1), name, mass)
        prediction = np.clip(raw.reshape(h,128), .4, 1.)*1000 if family == 'DirectEF' else v30.project(raw.reshape(h,128,4))@EF
        model_ef.append(prediction)
        fits.append(dict(arm=arm, family=family, training_indices=ids, training_source_count=len(set(regions[ids])), **record))
    model_ef = np.stack(model_ef)
    old_weights = np.any(q[held][:,[0,2]] > q[base_ids][:,[0,2]].max(0)+1e-12, axis=1)
    final = model_ef.copy(); final[0] = np.where(old_weights[:,None], (q[held]@EF)[:,None], model_ef[0])
    return dict(held_indices=held, model_EF=model_ef, EF_draws=final,
                CIF_draws=final*mg[held,:,0][None], old_coil_weights=old_weights,
                audit=dict(fits=fits, fit_count=3))


def meta_features(X31: np.ndarray, MG: np.ndarray, Q: np.ndarray, Cpos: np.ndarray) -> dict[str, np.ndarray]:
    n = len(X31)
    suffix = np.asarray(X31)[...,5:]  # Never convert or validate the five current-truth coordinates.
    if suffix.shape == (n,128,26):
        suffix = _real(suffix, (n,128,26), 'particle legal history')
        if not np.array_equal(suffix, np.broadcast_to(suffix[:,:1], suffix.shape)):
            raise ValueError('Particle-constant legal history required')
        suffix = suffix[:,0]
    suffix = _real(suffix, (n,26), 'legal history')
    mg = _real(MG, (n,128,5), 'MG'); _simplex(mg, 5)
    q = _real(Q, (n,4), 'inventory shares'); _simplex(q, 4)
    c = _real(Cpos, (n,), 'POS capacities')
    if np.any(c <= 0): raise ValueError('Positive POS capacities required')
    history = np.c_[suffix, mg[...,0].mean(1), mg[...,0].std(1, ddof=0)]
    with np.errstate(over='raise', divide='raise', invalid='raise'):
        try: pressure = np.exp(suffix[:,25])*suffix[:,2]/c
        except FloatingPointError as exc: raise ValueError('Finite untruncated past pressure required') from exc
    cap = _real(np.c_[history,q,pressure], (n,33), 'capacity meta features')
    return dict(HistoryForecast=history, HistoryForecastCap=cap)


def gain_targets(EF3: np.ndarray, CIF3: np.ndarray, trueef: np.ndarray, truecif: np.ndarray) -> np.ndarray:
    raw = np.asarray(EF3)
    if raw.ndim != 3 or raw.shape[0] != 3 or not raw.shape[1]: raise ValueError('Three endpoint arrays required')
    n, samples = raw.shape[1:]
    if not samples: raise ValueError('Nonempty teacher particles required')
    ef = _real(EF3, (3,n,samples), 'teacher EF').mean(2)
    cif = _real(CIF3, (3,n,samples), 'teacher CIF').mean(2)
    e = _real(trueef, (n,), 'source true EF'); ci = _real(truecif, (n,), 'source true CIF')
    de = (np.abs(ef[0]-e)[None]-np.abs(ef[1:]-e))/600
    dc = (np.abs(cif[0]-ci)[None]-np.abs(cif[1:]-ci))/1000
    return _real(np.stack([de,dc,.5*(de+dc)], axis=1), (2,3,n), 'normalized gains')


def fit_meta(features: Mapping[str,np.ndarray], gains: np.ndarray, regions: np.ndarray,
             query_features: Mapping[str,np.ndarray], *, outer: str) -> dict[str, Any]:
    regions = np.asarray(regions, dtype=str)
    if (regions.ndim != 1 or not len(regions) or outer not in LEGACY_SOURCES
            or any(s not in ALL_SOURCES or s == outer for s in regions)):
        raise ValueError('Allowed outer-excluded meta source metadata required')
    n = len(regions)
    gain = _real(gains, (2,3,n), 'source gain labels')
    train = {name:_real(features[name], (n,len(names)), name) for name,names in META_NAMES.items()}
    query = {name:_real(query_features[name], (6,len(names)), name+' query') for name,names in META_NAMES.items()}
    predictions, fits = [], []
    weights = _weights(regions, 6.)
    for ci, candidate in enumerate(CANDIDATES):
        for ti, target in enumerate(TARGETS):
            y = gain[ci,ti]
            for model in META_MODELS:
                arm = f'Gain-{candidate}-{target}-{model}'
                if model == 'Mean':
                    mean = float(np.average(y, weights=weights)); prediction = np.full(6,mean)
                    record = dict(model='Mean', mean=mean, sample_weights=weights.copy(), parameters={})
                else:
                    view, name = model.rsplit('-',1)
                    prediction, record = _fit(train[view], y, regions, query[view], name, 6.)
                    record.update(view=view, feature_names=META_NAMES[view])
                predictions.append(prediction); fits.append(dict(arm=arm,candidate=candidate,target=target,**record))
    result = np.stack(predictions)
    return dict(gain_prediction=result, actions=result > 0,
                audit=dict(outer=outer,fits=fits,sklearn_fit_count=24,mean_count=6,arms=ARMS))
