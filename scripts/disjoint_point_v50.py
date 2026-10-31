"""Frozen six-source point functions on explicitly registered query roles.

Pure rows and arrays. Training always uses the original six-source cohort; an
external query never becomes a training source. No I/O or label access.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from scripts import coarse_context_v30 as coarse
from scripts import forward_context_v31 as forward
from scripts import fleet_information_v34 as fleet
from scripts import conditional_law_v29 as scale_law
from scripts import group_regression_v27 as ratio_law
from scripts import monthly_transfer_v25 as monthly
from transcif.research.fuel_transport_v7 import FuelTrajectoryBank, recompose_paths

LEGACY7 = coarse.SOURCES
EXTERNAL10 = ('SWPP', 'AVA', 'BANC', 'JEA', 'SC', 'NWMT', 'SCL', 'WACM', 'SEC', 'SCEG')
BASES5 = ('Scale-anchor', 'B0-SizeET-ReplaceR', 'B0-SizeET-AlignedKeep',
          'B1-SizeCapRidge-ReplaceR', 'B1-SizeCapRidge-AlignedKeep')
BUDGETS = ('B0', 'B0', 'B0', 'B1', 'B1')
CACHE_NAMES = ('Scale-anchor', 'Recompose-Replace-Legacy-Size-ET',
               'Recompose-AlignedKeep-Legacy-Size-ET',
               'Recompose-Replace-Legacy-SizeCap-Ridge',
               'Recompose-AlignedKeep-Legacy-SizeCap-Ridge')
EF, POS, ZERO, SAMPLES = coarse.EF, coarse.POS, coarse.ZERO, 128
HEADS = ('Size-ET', 'SizeCap-Ridge')
DATE_FIELDS = ('training_period_start_ns', 'training_period_end_ns', 'training_available_ns')
EXTERNAL_DATES = pd.date_range('2024-07-01', '2024-12-24', tz='UTC').as_unit('ns')
PARITY_DATES = pd.date_range('2025-01-01', '2025-06-30', tz='UTC').as_unit('ns')
BANK_START, BANK_END = pd.Timestamp('2024-01-25', tz='UTC'), pd.Timestamp('2024-07-01', tz='UTC')


class FloatToken(str):
    """A JSON floating token, retained as text until its role is validated."""


class IntToken(str):
    """A JSON integer token, distinct from a floating parameter."""


def _real(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError('Real '+name+' required')
    try:
        result = np.asarray(raw, float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Numeric '+name+' required') from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('Finite matching '+name+' required')
    return result


def _dates(region: str, values: Sequence[Any]) -> tuple[str, pd.DatetimeIndex, int]:
    if region in EXTERNAL10:
        role, expected, year = 'external2024', EXTERNAL_DATES, 2022
    elif region in LEGACY7:
        role, expected, year = 'legacy2025_parity', PARITY_DATES, 2023
    else:
        raise ValueError('Registered external query or legacy parity region required')
    parsed = []
    for value in values:
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            at = pd.Timestamp(int(value), unit='ns', tz='UTC')
        else:
            at = coarse._utc(value)
        parsed.append(at)
    dates = pd.DatetimeIndex(parsed).as_unit('ns')
    if not dates.equals(expected):
        raise ValueError('Exact ordered UTC midnight dates for the registered role required')
    return role, dates, year


def _monthly_records(rows: Sequence[Mapping[str, str]], region: str) -> list[tuple]:
    records, seen = [], set()
    for row in rows:
        if row['region'] != region:
            continue
        start, end, release = (coarse._utc(row[k]) for k in ('period_start', 'period_end', 'available_at'))
        if (start != start.normalize() or start.day != 1 or end != start+pd.offsets.MonthBegin(1)
                or release < end):
            raise ValueError('Whole calendar month and legal release metadata required')
        key = coarse._key(region, start)
        if key in seen:
            raise ValueError('Unique query-region monthly vintage required')
        seen.add(key)
        records.append((start, end, release, key, row))
    return sorted(records, key=lambda r: r[0])


def _visible_frame(records: list[tuple], region: str, last: pd.Timestamp) -> pd.DataFrame:
    chosen = []
    for start, end, release, _, row in records:
        if start >= last or end > last or release > last:
            continue
        g = coarse._visible(row)
        chosen.append(dict(region=region, period_start=start, period_end=end, available_at=release,
                           **dict(zip(monthly.MG_COLUMNS, g))))
    if not chosen:
        raise ValueError('Released local coarse history required')
    return pd.DataFrame(chosen).sort_values(['period_end', 'available_at']).reset_index(drop=True)


def _mg_law(frame: pd.DataFrame, dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """V25 numerical law with explicit query-role validation at the public entry."""
    cols = list(monthly.MG_COLUMNS)
    monthly._valid_simplex(frame[cols].to_numpy(float), 5)
    prior, cache = np.array([.5, .125, .125, .125, .125]), {}

    def point(history: pd.DataFrame, at: pd.Timestamp) -> np.ndarray:
        key = (tuple(history.index), at.year, at.month)
        if key not in cache:
            cache[key] = monthly._point(history, at, monthly.MG_COLUMNS, prior)
        return cache[key]

    means, draws, metadata = [], [], []
    for at in dates:
        prefix = frame.loc[frame.period_end.le(at) & frame.available_at.le(at)]
        records = []
        for _, label in prefix.tail(60).iterrows():
            pseudo = label.period_start + pd.Timedelta(days=min(at.day, (label.period_end-label.period_start).days)-1)
            past = prefix.loc[prefix.period_end.le(pseudo) & prefix.available_at.le(pseudo)]
            horizon = monthly._ordinal(pseudo)-monthly._ordinal(past.period_start.iloc[-1]) if len(past) else 99
            records.append((label[cols].to_numpy(float)-point(past, pseudo), horizon))
        records = records[-24:]
        mean = point(prefix, at)
        horizon = monthly._ordinal(at)-monthly._ordinal(prefix.period_start.iloc[-1]) if len(prefix) else 99
        if records:
            age = np.arange(len(records)-1, -1, -1, dtype=float)
            gaps = np.array([abs(h-horizon) for _, h in records], float)
            weights = np.exp(-age/12-gaps/1.5); weights /= weights.sum()
            rng = np.random.default_rng(np.random.SeedSequence([17023, 0, at.year, at.month, at.day]))
            selected = rng.choice(len(records), size=SAMPLES, p=weights)
            active = prefix[cols].to_numpy(float).max(0) > 0
            particle = monthly._simplex_rows(mean[None]+np.stack([r for r, _ in records])[selected], active)
        else:
            particle = np.repeat(mean[None], SAMPLES, axis=0)
        release_age = float((at-prefix.available_at.max())/pd.Timedelta(days=1)) if len(prefix) else 999.
        means.append(mean); draws.append(particle); metadata.append([len(prefix), len(records), horizon, release_age])
    return dict(MG_mix=np.stack(means), MG_draws=np.stack(draws), MG_metadata=np.asarray(metadata, float))


def _context(records: list[tuple], dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    chosen = []
    for at in dates:
        legal = [r for r in records if r[0] >= coarse.START and r[0] < at and r[1] <= at and r[2] <= at]
        if len(legal) < 3:
            raise ValueError('Three fully released coarse contexts required at every query')
        chosen.append(legal[-3:][::-1])
    g = np.array([[coarse._visible(r[4]) for r in group] for group in chosen])
    mw = np.array([[float(r[4]['total_mwh'])/((r[1]-r[0])/pd.Timedelta(hours=1)) for r in group] for group in chosen])
    if not np.isfinite(mw).all() or np.any(mw <= 0):
        raise ValueError('Finite positive legal past monthly volume required')
    starts = np.array([[r[0].value for r in group] for group in chosen], np.int64)
    ends = np.array([[r[1].value for r in group] for group in chosen], np.int64)
    releases = np.array([[r[2].value for r in group] for group in chosen], np.int64)
    ages = np.array([[at.year*12+at.month-r[0].year*12-r[0].month for r in group]
                     for at, group in zip(dates, chosen)], float)
    return dict(origin_ns=dates.asi8.copy(), context_keys=np.array([[r[3] for r in group] for group in chosen]),
        context_start_ns=starts, context_end_ns=ends, context_release_ns=releases,
        context_g=g, context_MW=mw, context_month_age=ages,
        context_release_age_days=(dates.asi8[:, None]-releases)/(86400.*1e9))


def query(rows: Sequence[Mapping[str, str]], region: str, dates: Sequence[Any],
          q4: np.ndarray | None = None, cpos: float | None = None) -> dict[str, Any]:
    """Construct legal public inputs for one of the two fixed query roles."""
    role, at, inventory_year = _dates(region, dates)
    if (q4 is None) != (cpos is None):
        raise ValueError('Both capacity arguments must be supplied or both omitted')
    capacity = q = None
    if q4 is not None:
        q = _real(q4, (4,), 'capacity shares'); forward._simplex_rows(q, 4)
        capacity = float(_real(cpos, (), 'POS capacity'))
        if capacity <= 0:
            raise ValueError('Strictly positive inventory POS capacity required')
    records = _monthly_records(rows, region)
    laws = _mg_law(_visible_frame(records, region, at[-1]), at)
    context = _context(records, at)
    size = forward._features(laws['MG_draws'], context, at)['Size']
    features = dict(Size=size)
    if q is not None:
        features['SizeCap'] = fleet._features(size, np.tile(q, (len(at), 1)), np.full(len(at), capacity))['SizeCap']
    return dict(region=region, role=role, budget_mode='B0' if q is None else 'B1',
        origins=at.asi8.copy(), keys=np.array([f'{region}/{d:%Y-%m-%d}' for d in at]),
        **laws, features=features, **context,
        audit=dict(expected_inventory_year=None if q is None else inventory_year,
                   capacity_used=q is not None, local_hidden_fine_read=False,
                   current_total_read=False, current_group_truth_read=False, released_prefix_only=True,
                   actual_first_release_verified=False, authentic_2024_online_backtest=False))


def _identity(record: Mapping[str, Any], arm: str, outer: str, tokenized: bool) -> None:
    family, method = arm.split('-')
    if (record['arm'] != arm or record['family'] != family or record['model'] != method
            or record['excluded_outer'] != outer or tuple(record['feature_names']) != fleet.FEATURE_NAMES[family]):
        raise ValueError('Original fixed center identity and feature order required')
    params, expected = record['parameters'], forward.PARAMETERS[method]
    if set(params) != set(expected):
        raise ValueError('Original center parameter keys required')
    for name, value in expected.items():
        raw = params[name]
        if isinstance(value, bool):
            valid = type(raw) is bool and raw == value
        elif tokenized:
            kind = FloatToken if type(value) is float else IntToken
            valid = type(raw) is kind and type(value)(raw) == value
        else:
            valid = type(raw) is type(value) and raw == value
        if not valid:
            raise ValueError('Original center parameter value and token type required')


def decode_records(audits: Mapping[str, Mapping[str, Any]], outer: str) -> list[dict[str, Any]]:
    """Decode metadata only; leave every training response/feature token as text."""
    if outer not in LEGACY7 or set(audits) != {'size', 'cap'}:
        raise ValueError('Registered outer and two original audit roles required')
    chosen = []
    for name, arm in zip(('size', 'cap'), HEADS):
        audit = audits[name]
        if audit['region'] != outer or ('outer' in audit and audit['outer'] != outer):
            raise ValueError('Matching original outer audit required')
        found = [f for f in audit['fits'] if f['arm'] == arm]
        if len(found) != 1:
            raise ValueError('Unique frozen center record required')
        _identity(found[0], arm, outer, tokenized=True)
        chosen.append(dict(found[0]))
    for record in chosen:
        method = record['model']
        record['parameters'] = {k: v if isinstance(v, bool) else type(forward.PARAMETERS[method][k])(v)
                                for k, v in record['parameters'].items()}
        for name in DATE_FIELDS:
            if any(type(v) is not IntToken for v in record[name]):
                raise ValueError('Integer timestamp metadata tokens required')
            record[name] = [int(v) for v in record[name]]
    return chosen


def _record_metadata(record: Mapping[str, Any], arm: str, outer: str) -> None:
    _identity(record, arm, outer, tokenized=False)
    keys, regions = list(record['training_keys']), np.asarray(record['training_regions'], str)
    n = len(keys)
    if not n or len(set(keys)) != n or regions.shape != (n,) or set(regions) != set(LEGACY7)-{outer}:
        raise ValueError('Original six-source cohort required')
    dates = {}
    for name in DATE_FIELDS:
        values = np.asarray(record[name])
        if values.shape != (n,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError('Integer training date metadata required')
        dates[name] = values
    for i, raw in enumerate(dates[DATE_FIELDS[0]]):
        at = pd.Timestamp(int(raw), tz='UTC'); end = pd.Timestamp(int(dates[DATE_FIELDS[1]][i]), tz='UTC')
        if (not coarse.START <= at < coarse.END or at.day != 1 or at != at.normalize()
                or end != at+pd.offsets.MonthBegin(1) or keys[i] != coarse._key(regions[i], at)
                or not end.value <= dates[DATE_FIELDS[2]][i] <= coarse.CUTOFF.value):
            raise ValueError('Original canonical released training month required')


def _bank_metadata(bank: Mapping[str, np.ndarray], outer: str) -> None:
    expected = tuple(s for s in LEGACY7 if s != outer)
    if (outer not in LEGACY7 or bank.get('outer', outer) != outer
            or not np.array_equal(bank['source_regions'], expected) or set(bank['regions']) != set(expected)):
        raise ValueError('Original six-source ordered bank identity required')
    days, ids = np.asarray(bank['days']), np.asarray(bank['row_ids'])
    if (days.ndim != 1 or not len(days) or not np.issubdtype(days.dtype, np.integer)
            or not np.issubdtype(ids.dtype, np.integer) or not np.array_equal(ids, np.arange(len(days)))
            or int(bank['cutoff']) != BANK_END.value):
        raise ValueError('Original integer H1 bank provenance required')
    at = pd.DatetimeIndex(pd.to_datetime(days, unit='ns', utc=True)).as_unit('ns')
    if (at.hasnans or not at.equals(at.normalize()) or np.any(at < BANK_START) or np.any(at >= BANK_END)
            or pd.MultiIndex.from_arrays([bank['regions'], days]).has_duplicates
            or not np.array_equal(bank['months'], at.month)):
        raise ValueError('Unique original source H1 day/month metadata required')


def _center_training(records: Sequence[Mapping[str, Any]], outer: str) -> list[dict[str, Any]]:
    if len(records) != 2:
        raise ValueError('Two ordered original center records required')
    for record, arm in zip(records, HEADS):
        _record_metadata(record, arm, outer)
    for name in ('training_keys', 'training_regions', *DATE_FIELDS):
        if not np.array_equal(records[0][name], records[1][name]):
            raise ValueError('Identical original center cohorts required')
    tables = []
    for record, arm in zip(records, HEADS):
        n = len(record['training_keys']); d = len(record['feature_names'])
        x = _real(record['training_X'], (n, d), 'original training X')
        y = _real(record['training_y'], (n, 4), 'original training y'); forward._simplex_rows(y, 4)
        w = _real(record['sample_weights'], (n,), 'original sample weights')
        regions = np.asarray(record['training_regions'])
        expected = np.array([n/(6*np.count_nonzero(regions == s)) for s in regions])
        if np.any(w <= 0) or not np.allclose(w, expected, atol=1e-12, rtol=0):
            raise ValueError('Original weights and mass must remain unchanged')
        mean = _real(record['scaler_mean'], (d,), 'original scaler mean')
        sd = _real(record['scaler_scale'], (d,), 'original scaler scale')
        checked_mean, checked_sd = np.zeros(d), np.ones(d)
        if arm.endswith('Ridge'):
            checked_mean = np.average(x, axis=0, weights=w)
            checked_sd = np.sqrt(np.average((x-checked_mean)**2, axis=0, weights=w)); checked_sd[checked_sd <= 1e-12] = 1.
        if np.any(sd <= 0) or not np.allclose(mean, checked_mean, atol=1e-12, rtol=0) or not np.allclose(sd, checked_sd, atol=1e-12, rtol=0):
            raise ValueError('Original train-only scaler required')
        if arm.endswith('ET') and (np.any(mean != 0) or np.any(sd != 1)):
            raise ValueError('Original ET must use exact raw-feature zero/one scaler')
        tables.append(dict(X=x, y=y, weights=w, mean=mean, scale=sd))
    if not np.array_equal(tables[0]['X'], tables[1]['X'][:, :31]) or not np.array_equal(tables[0]['y'], tables[1]['y']):
        raise ValueError('Common original center Size features and responses required')
    return tables


def fit(bank6: Mapping[str, np.ndarray], monthly_rows: Sequence[Mapping[str, str]],
        records2: Sequence[Mapping[str, Any]], outer: str) -> dict[str, Any]:
    """Reconstruct seven Scale ETs and the two original six-source point heads."""
    _bank_metadata(bank6, outer)
    tables = _center_training(records2, outer)
    rows = list(monthly_rows)
    if any(row['region'] not in LEGACY7 for row in rows):
        raise ValueError('Only original monthly sources may enter training')
    ratio_law._bank(bank6, outer)
    scale = scale_law.fit_leaf_controls(bank6, rows, outer=outer)
    models, audits = [], []
    for record, table in zip(records2, tables):
        method = record['model']
        estimator = Ridge(**forward.PARAMETERS[method]) if method == 'Ridge' else ExtraTreesRegressor(**forward.PARAMETERS[method])
        estimator.fit((table['X']-table['mean'])/table['scale'], table['y'], sample_weight=table['weights'])
        models.append((estimator, table['mean'].copy(), table['scale'].copy()))
        reproduced = dict(coefficients=estimator.coef_.copy(), intercept=estimator.intercept_.copy()) if method == 'Ridge' else dict(feature_importances=estimator.feature_importances_.copy())
        audits.append(dict(arm=record['arm'], excluded_outer=outer, parameters=dict(record['parameters']),
            feature_names=record['feature_names'], training_keys=list(record['training_keys']),
            training_regions=np.asarray(record['training_regions']), training_X=table['X'].copy(),
            training_y=table['y'].copy(), sample_weights=table['weights'].copy(),
            scaler_mean=table['mean'].copy(), scaler_scale=table['scale'].copy(),
            **{k: np.asarray(record[k]).copy() for k in DATE_FIELDS}, **reproduced))
    return dict(outer=outer, bank=bank6, scale=scale, centers=models,
        audit=dict(outer=outer, fit_count=9, Scale_ET_fits=7, center_ET_fits=1, center_Ridge_fits=1,
            scale_audit=scale.audit, center_fits=audits, original_six_source_function=True,
            foreign_training_sources_used=False, models_shared_across_queries=True))


def _query_arrays(ctx: Mapping[str, Any], value: Mapping[str, Any]) -> tuple[pd.DatetimeIndex, np.ndarray]:
    role, dates, _ = _dates(value['region'], value['origins'])
    if role != value['role'] or (role == 'legacy2025_parity' and value['region'] != ctx['outer']):
        raise ValueError('Query role must be external or this context outer parity')
    n = len(dates)
    mg = _real(value['MG_draws'], (n, SAMPLES, 5), 'query MG'); forward._simplex_rows(mg, 5)
    means = _real(value['MG_mix'], (n, 5), 'query MG point'); forward._simplex_rows(means, 5)
    if not np.array_equal(value['keys'], [f"{value['region']}/{at:%Y-%m-%d}" for at in dates]):
        raise ValueError('Exact query key/role pairing required')
    if not np.array_equal(value['origin_ns'], dates.asi8):
        raise ValueError('Exact context/query nanosecond origins required')
    if value['budget_mode'] not in ('B0', 'B1'):
        raise ValueError('Explicit B0 or B1 query mode required')
    families = ('Size',) if value['budget_mode'] == 'B0' else ('Size', 'SizeCap')
    if set(value['features']) != set(families):
        raise ValueError('Only the requested budget feature families are permitted')
    for family in families:
        x = _real(value['features'][family], (n, SAMPLES, len(fleet.FEATURE_NAMES[family])), family)
        if not np.array_equal(x[..., :5], mg):
            raise ValueError('Literal MG feature coupling required')
    if value['budget_mode'] == 'B1' and not np.array_equal(value['features']['SizeCap'][..., :31], value['features']['Size']):
        raise ValueError('Aligned original center feature families required')
    if value['budget_mode'] == 'B1':
        forward._simplex_rows(value['features']['SizeCap'][..., 31:35], 4)
        if np.any(value['features']['SizeCap'][..., 35] < 0):
            raise ValueError('Nonnegative inventory pressure required')
    return dates, mg


def predict(ctx: Mapping[str, Any], query: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the same six-source function to external inputs or cached parity."""
    dates, mg = _query_arrays(ctx, query)
    if ctx['outer'] not in LEGACY7 or ctx['audit']['fit_count'] != 9:
        raise ValueError('Complete original fixed-function fit context required')
    if query['budget_mode'] == 'B1' and len(ctx['centers']) != 2:
        raise ValueError('Both fixed center models required for B1')
    n = len(dates); scale = ctx['scale']; bank = ctx['bank']
    if scale.outer != ctx['outer'] or scale.audit['fitted_head_models'] != 7:
        raise ValueError('Original seven-ET Scale context required')
    conditional = scale.final.conditional(mg.reshape(-1, 5))
    mean = conditional['mean'].reshape(n, SAMPLES, 4)
    std = conditional['scale'].reshape(n, SAMPLES)
    covariance = conditional['covariance3'].reshape(n, SAMPLES, 3, 3)
    outputs = {name: [] for name in ('A', 'R', 'fine', 'ratios', 'unprojected_ratios', 'projection_l1',
        'donor_ids', 'fallback_ids', 'effective_pos_ids', 'effective_month_ids', 'source_scale', 'scale_ratio',
        'zero_total_mask', 'unsupported_fraction')}
    for t, at in enumerate(dates):
        uniform = FuelTrajectoryBank._rng(17023, at).random((SAMPLES, 8))
        weights = monthly.donor_weights(bank, query['MG_mix'][t], at, budget='MG', conditional=True)
        ids = monthly._sample(weights, uniform[:, 0])
        original, original_q, fallback = monthly._fine_draws(bank, mg[t], ids, weights, uniform, 'MG')
        q, support, selected, donor_ratio = ratio_law._effective(bank, ids, fallback, mg[t, :, 0])
        if not np.array_equal(q, original_q) or not np.array_equal(original[:, POS], mg[t, :, :1]*donor_ratio):
            raise ValueError('Original sampler/effective donor pairing required')
        mapping = scale.table['bank_month_ids'][selected]
        source = scale.oof_scale[mapping]
        if not np.isfinite(source).all() or np.any(source <= 0):
            raise ValueError('Positive source OOF standard deviations required')
        multiplier = std[t]/source
        before = mean[t]+multiplier[:, None]*scale.oof_residual[mapping]
        ratio = ratio_law.project(before)
        fine = original.copy(); fine[:, POS] = mg[t, :, :1]*ratio
        r, zero = recompose_paths(fine, q, return_zero_total=True)
        a = bank['raw_cif'][ids]+((fine-bank['donor_mix'][ids])@EF)[:, None]
        fields = dict(A=a, R=r, fine=fine, ratios=ratio, unprojected_ratios=before,
            projection_l1=abs(ratio-before).sum(1), donor_ids=ids, fallback_ids=fallback,
            effective_pos_ids=selected, effective_month_ids=mapping, source_scale=source,
            scale_ratio=multiplier, zero_total_mask=zero,
            unsupported_fraction=ratio_law._unsupported(fine, q, support))
        for name, array in fields.items():
            if not np.isfinite(array).all():
                raise ValueError('Finite fixed Scale outputs required')
            outputs[name].append(array)
    arrays = {k: np.stack(v) for k, v in outputs.items()}
    u, p = arrays['fine']@EF, mg[..., 0]*(mean@EF[POS])
    raw_ratios, ratios = [], []
    families = ('Size',) if query['budget_mode'] == 'B0' else ('Size', 'SizeCap')
    for family, (estimator, center, sd) in zip(families, ctx['centers']):
        x = query['features'][family]
        value = estimator.predict((x.reshape(-1, x.shape[-1])-center)/sd).reshape(n, SAMPLES, 4)
        raw_ratios.append(value); ratios.append(coarse.project(value))
    ratios = np.stack(ratios); ef = ratios@EF[POS]; levels = ef*mg[None, ..., 0]
    a, r = arrays['A'], arrays['R']; raw = [a]
    for level in levels:
        raw.extend((r-u[..., None]+level[..., None],
                    (r-p[..., None]+level[..., None])+(a.mean(-1)-r.mean(-1))[..., None]))
    raw = np.stack(raw)
    if len(raw) != (3 if query['budget_mode'] == 'B0' else 5):
        raise ValueError('Complete requested-budget base registry required')
    if not np.isfinite(raw).all():
        raise ValueError('Finite fixed five-base paths required')
    return dict(raw=raw, **arrays, U=u, P=p, query_mean=mean, query_scale=std,
        query_covariance3=covariance, center_levels=levels, center_EF=ef,
        center_raw_ratios=np.stack(raw_ratios), center_ratios=ratios,
        MG_draws=mg.copy(), origins=dates.asi8.copy(), keys=np.asarray(query['keys']).copy(),
        arms=np.asarray(BASES5[:len(raw)]), budgets=np.asarray(BUDGETS[:len(raw)]),
        audit=dict(outer=ctx['outer'], query_region=query['region'], query_role=query['role'],
            query_budget=query['budget_mode'], capacity_used=query['budget_mode'] == 'B1',
            new_fits=0, models_shared_across_queries=True, local_hidden_fine_read=False,
            authentic_2024_online_backtest=False, actual_first_release_verified=False,
            source_scale_is_standard_deviation=True, method_innovation_claim=False))
