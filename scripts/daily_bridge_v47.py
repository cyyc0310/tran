"""Daily legal-context replay and fixed hourly level controls; no I/O or truth."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from scripts import coarse_context_v30 as v30
from scripts import forward_context_v31 as v31
from scripts import fleet_information_v34 as v34
from scripts import hourly_bridge_v46 as v46
from scripts.donor_inventory_v41 import ALL_SOURCES


SOURCES, POS, ZERO, EF = v31.SOURCES, v31.POS, v31.ZERO, v31.EF
FUELS = v30.FUELS
DAYS = pd.date_range('2025-01-01', '2025-06-30', freq='D', tz='UTC').as_unit('ns')
FEATURE_NAMES = {name: v34.FEATURE_NAMES[name] for name in ('Size', 'SizeCap', 'DirectEF')}
PARAMETERS = {name: value.copy() for name, value in v34.PARAMETERS.items()}
LEVEL_ARMS = v46.LEVEL_ARMS[:15]
MODES = ('Replace', 'Keep', 'CenterOnly', 'FlatKeep')
ARMS = v46.ARMS[:3] + tuple(f'{mode}-{name}' for mode in MODES for name in LEVEL_ARMS)
BUDGET_BY_ARM = {arm: ('M8' if arm == 'M8-reference' else 'B0' if arm in ARMS[:2]
                      or any(arm == f'{mode}-{scope}-Size-{method}'
                             for mode in MODES for scope in ('Legacy', 'Expanded')
                             for method in ('Ridge', 'ET')) else 'B1') for arm in ARMS}
_DATE_FIELDS = ('training_period_start_ns', 'training_period_end_ns', 'training_available_ns')


def daily_features(rows: Sequence[Mapping[str, str]], region: str, MG: np.ndarray,
                   q4: np.ndarray, Cpos: float) -> dict[str, Any]:
    """Use only the latest three fully released months at each daily UTC origin."""
    if region not in SOURCES:
        raise ValueError('Registered legacy query region required')
    mg = v46._real(MG, (181, 128, 5), 'daily MG')
    q = v46._real(q4, (4,), 'fixed capacity shares')
    capacity = float(v46._real(Cpos, (), 'POS capacity'))
    v31._simplex_rows(mg, 5); v31._simplex_rows(q, 4)
    if capacity <= 0:
        raise ValueError('Strictly positive fixed POS capacity required')
    context = v31._context(rows, region, DAYS)
    base = v31._features(mg, context, DAYS)
    cap = v34._features(base['Size'], np.tile(q, (181, 1)), np.full(181, capacity))
    features = dict(Size=base['Size'], SizeCap=cap['SizeCap'], DirectEF=cap['DirectEF'])
    return dict(region=region, MG_draws=mg.copy(), features=features, **context,
                audit=dict(hidden_fuel_fields_read=False, current_total_read=False,
                           current_group_truth_read=False, daily_calibration_claim=False,
                           monthly_supervision_replayed_daily=True))


def _metadata(fit: Mapping[str, Any], index: int, outer: str) -> dict[str, Any]:
    scope, family, method = LEVEL_ARMS[index].split('-')
    if (fit['arm'] != family+'-'+method or fit['family'] != family or fit['model'] != method
            or fit['excluded_outer'] != outer or fit['parameters'] != PARAMETERS[method]
            or tuple(fit['feature_names']) != FEATURE_NAMES[family]
            or (method == 'ET' and type(fit['parameters']['max_features']) is not float)):
        raise ValueError('Frozen head identity, parameters, outer and feature order required')
    kind = 'scalar_EF' if family == 'DirectEF' else 'simplex'
    if fit.get('projection_kind', kind) != kind:
        raise ValueError('Frozen head projection kind required')
    keys = list(fit['training_keys'])
    regions = np.asarray(fit['training_regions'], dtype=str)
    n = len(keys)
    old_sources = set(SOURCES)-{outer}
    if (not n or len(set(keys)) != n or regions.shape != (n,) or outer in regions
            or (scope == 'Legacy' and set(regions) != old_sources)
            or (scope == 'Expanded' and (not old_sources <= set(regions)
                                         or not set(regions) <= set(ALL_SOURCES)-{outer}))):
        raise ValueError('Unique excluded-outer registered source cohort required')
    dates = {}
    for name in _DATE_FIELDS:
        value = np.asarray(fit[name])
        if value.shape != (n,) or not np.issubdtype(value.dtype, np.integer):
            raise ValueError('Integer nanosecond training metadata required')
        dates[name] = value.copy()
    for i, value in enumerate(dates[_DATE_FIELDS[0]]):
        at = pd.Timestamp(int(value), unit='ns', tz='UTC')
        end = pd.Timestamp(int(dates[_DATE_FIELDS[1]][i]), unit='ns', tz='UTC')
        available = int(dates[_DATE_FIELDS[2]][i])
        if (pd.isna(at) or not v30.START <= at < v30.END or at != at.normalize() or at.day != 1
                or end != at+pd.offsets.MonthBegin(1) or not end.value <= available <= v31.CUTOFF.value
                or keys[i] != v30._key(regions[i], at)):
            raise ValueError('Canonical released pre-cutoff monthly training metadata required')
    return dict(training_keys=keys, training_regions=regions, **dates)


def _cohorts(tables: list[dict[str, Any]]) -> None:
    fields = ('training_keys', 'training_regions') + _DATE_FIELDS
    for start in (0, 6):
        for table in tables[start+1:start+6]:
            if any(not np.array_equal(table[name], tables[start][name]) for name in fields):
                raise ValueError('All six heads in each scope require identical cohort metadata')
    expanded = {key: i for i, key in enumerate(tables[6]['training_keys'])}
    for i, key in enumerate(tables[0]['training_keys']):
        if key not in expanded or any(tables[0][name][i] != tables[6][name][expanded[key]]
                                       for name in ('training_regions',)+_DATE_FIELDS):
            raise ValueError('Legacy cohort must be preserved inside expanded cohort')


def _query(query: Mapping[str, Any]) -> tuple[str, np.ndarray, dict[str, np.ndarray]]:
    outer = query['region']
    if outer not in SOURCES or not np.array_equal(query['origin_ns'], DAYS.asi8):
        raise ValueError('Fixed legacy outer and all 181 UTC daily origins required')
    mg = v46._real(query['MG_draws'], (181, 128, 5), 'daily MG')
    v31._simplex_rows(mg, 5)
    features = {name: v46._real(query['features'][name], (181, 128, len(names)), name)
                for name, names in FEATURE_NAMES.items()}
    if (not np.array_equal(features['Size'][..., :5], mg)
            or not np.array_equal(features['SizeCap'][..., :31], features['Size'])
            or not np.array_equal(features['DirectEF'], features['SizeCap'])
            or np.any(features['SizeCap'][..., -1] < 0)):
        raise ValueError('Aligned daily MG, Size and capacity feature families required')
    v31._simplex_rows(features['SizeCap'][..., 31:35], 4)
    return outer, mg, features


def replay_heads(records12: Sequence[Mapping[str, Any]], queryfeatures: Mapping[str, Any]) -> dict[str, Any]:
    """Replay six Ridge coefficients and reconstruct six fixed ET heads.

    Caller binds each record to its sealed parent and checks complete cohort
    identity and month-start raw/EF/projection equality. All twelve metadata
    checks precede conversion of any training numeric array.
    """
    try:
        outer, mg, features = _query(queryfeatures)
        if len(records12) != 12:
            raise ValueError('Exactly twelve ordered frozen head records required')
        tables = [_metadata(fit, i, outer) for i, fit in enumerate(records12)]
        _cohorts(tables)
        return _replay(records12, tables, features, outer, mg)
    except KeyError as exc:
        raise ValueError('Missing frozen replay field: '+str(exc)) from exc
    except FloatingPointError as exc:
        raise ValueError('Finite frozen replay arithmetic required') from exc


def _replay(records: Sequence[Mapping[str, Any]], tables: list[dict[str, Any]],
            features: dict[str, np.ndarray], outer: str, mg: np.ndarray) -> dict[str, Any]:
    numerics = []
    for i, (fit, table) in enumerate(zip(records, tables)):
        scope, family, method = LEVEL_ARMS[i].split('-')
        n, width = len(table['training_keys']), len(FEATURE_NAMES[family])
        x = v46._real(fit['training_X'], (n, width), 'frozen training X')
        y = v46._real(fit['training_y'], (n,) if family == 'DirectEF' else (n, 4), 'frozen training y')
        w = v46._real(fit['sample_weights'], (n,), 'frozen sample weights')
        regions = table['training_regions']
        mass = n if scope == 'Legacy' else (156 if outer == 'FPL' else 154)
        expected = np.array([mass/(len(set(regions))*np.count_nonzero(regions == s)) for s in regions])
        if np.any(w <= 0) or not np.allclose(w, expected, atol=1e-12, rtol=0):
            raise ValueError('Original source-balanced weights and mass required')
        if family == 'DirectEF':
            if np.any(y < .4-1e-12) or np.any(y > 1+1e-12):
                raise ValueError('Normalized DirectEF training responses required')
        else:
            v31._simplex_rows(y, 4)
        mean = v46._real(fit['scaler_mean'], (width,), 'frozen scaler mean')
        scale = v46._real(fit['scaler_scale'], (width,), 'frozen scaler scale')
        if np.any(scale <= 0):
            raise ValueError('Positive frozen scaler required')
        expected_mean, expected_scale = np.zeros(width), np.ones(width)
        if method == 'Ridge':
            expected_mean = np.average(x, axis=0, weights=w)
            expected_scale = np.sqrt(np.average((x-expected_mean)**2, axis=0, weights=w))
            expected_scale[expected_scale <= 1e-12] = 1
        if not (np.allclose(mean, expected_mean, atol=1e-12, rtol=0)
                and np.allclose(scale, expected_scale, atol=1e-12, rtol=0)):
            raise ValueError('Original weighted Ridge scaler or zero/one ET scaler required')
        if method == 'ET' and (np.any(mean != 0) or np.any(scale != 1)):
            raise ValueError('ET replay must use raw features and exact zero/one scaler')
        numerics.append((x, y, w, mean, scale))
    for start in (0, 6):
        x0, y0 = numerics[start][:2]
        for i in range(start, start+6):
            x, y = numerics[i][:2]
            target = y0 @ EF[POS]/1000 if records[i]['family'] == 'DirectEF' else y0
            if (not np.allclose(x[:, :31], x0, atol=1e-12, rtol=0)
                    or not np.allclose(y, target, atol=1e-12, rtol=0)):
                raise ValueError('Common scope training Size features and response identity required')
    raw_predictions, ef_raw, ef, projection, kinds, audits = [], [], [], [], [], []
    for i, (fit, (x, y, w, mean, scale)) in enumerate(zip(records, numerics)):
        family, method = fit['family'], fit['model']
        width = x.shape[1]
        query = features[family].reshape(-1, width)
        scalar = family == 'DirectEF'
        if method == 'Ridge':
            coef = v46._real(fit['coefficients'], (width,) if scalar else (4, width), 'Ridge coefficients')
            intercept = v46._real(fit['intercept'], () if scalar else (4,), 'Ridge intercept')
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                raw = ((query-mean)/scale) @ coef.T + intercept
            reproduced = dict(coefficients=coef.copy(), intercept=intercept.copy(), reproduction_max_abs=0.)
        else:
            expected = v46._real(fit['feature_importances'], (width,), 'sealed ET importance')
            fitted = ExtraTreesRegressor(**PARAMETERS['ET'])
            fitted.fit(x, y, sample_weight=w)
            error = float(abs(fitted.feature_importances_-expected).max())
            if not np.isfinite(error) or error > 1e-12:
                raise ValueError('Frozen ET feature-importance reconstruction failed')
            raw = fitted.predict(query)
            reproduced = dict(feature_importances=fitted.feature_importances_.copy(), reproduction_max_abs=error)
        raw = v46._real(raw, (181*128,) if scalar else (181*128, 4), 'raw prediction')
        raw = raw.reshape((181, 128) if scalar else (181, 128, 4))
        projected = np.clip(raw, .4, 1.) if scalar else v30.project(raw)
        raw_predictions.append(raw.copy())
        ef_raw.append(raw*1000 if scalar else raw @ EF[POS])
        ef.append(projected*1000 if scalar else projected @ EF[POS])
        projection.append(abs(raw-projected) if scalar else abs(raw-projected).sum(-1))
        kinds.append('scalar_EF' if scalar else 'simplex')
        audits.append(dict(arm=LEVEL_ARMS[i], source_arm=fit['arm'], fitted=method == 'ET',
            parameters=PARAMETERS[method].copy(), feature_names=FEATURE_NAMES[family],
            training_case_count=len(x), training_source_count=len(set(tables[i]['training_regions'])),
            sample_weights=w.copy(), scaler_mean=mean.copy(), scaler_scale=scale.copy(),
            **tables[i], **reproduced))
    ef = np.stack(ef)
    return dict(EF_raw=np.stack(ef_raw), EF_draws=ef, CIF_draws=ef*mg[None, ..., 0],
                projection_l1=np.stack(projection), projection_kind=np.array(kinds),
                raw_predictions=raw_predictions,
                audit=dict(region=outer, fit_count=6, Ridge_replays=6, fits=audits,
                           new_training_labels_used=False, daily_calibration_claim=False))


def build_paths(scale_raw: np.ndarray, scale_fine: np.ndarray, query_mean: np.ndarray,
                MG: np.ndarray, levels: np.ndarray, scale_recompose: np.ndarray,
                m8_raw: np.ndarray) -> dict[str, Any]:
    """Build63 un-clipped daily paths, filling one arm at a time to limit memory."""
    a = v46._real(scale_raw, (181, 128, 24), 'scale_raw')
    fine = v46._real(scale_fine, (181, 128, 8), 'scale_fine')
    mean = v46._real(query_mean, (181, 128, 4), 'query_mean')
    mg = v46._real(MG, (181, 128, 5), 'MG')
    s = v46._real(levels, (15, 181, 128), 'levels')
    recompose = v46._real(scale_recompose, (181, 128, 24), 'scale_recompose')
    m8 = v46._real(m8_raw, (181, 128, 24), 'm8_raw')
    for value, name in ((fine, 'fine'), (mean, 'query_mean'), (mg, 'MG')):
        v46._simplex(value, name)
    if (not np.allclose(fine[..., POS].sum(-1), mg[..., 0], atol=1e-12, rtol=0)
            or not np.allclose(fine[..., ZERO], mg[..., 1:], atol=1e-12, rtol=0)):
        raise ValueError('Fine POS and zero-carbon components must match MG')
    if np.any(s < 400*mg[None, ..., 0]-1e-9) or np.any(s > 1000*mg[None, ..., 0]+1e-9):
        raise ValueError('Monthly levels outside [400*MG_POS,1000*MG_POS]')
    try:
        with np.errstate(over='raise', invalid='raise'):
            return _paths(a, fine, mean, mg, s, recompose, m8)
    except FloatingPointError as exc:
        raise ValueError('Finite daily bridge arithmetic required') from exc


def _paths(a: np.ndarray, fine: np.ndarray, mean: np.ndarray, mg: np.ndarray,
           s: np.ndarray, recompose: np.ndarray, m8: np.ndarray) -> dict[str, Any]:
    u, p = fine @ EF, mg[..., 0]*(mean @ EF[POS])
    inherited, hour_minus_scale = u-p, a.mean(-1)-u
    zero = mg[..., 0] == 0
    raw = np.empty((63, 181, 128, 24))
    raw[:3] = a, recompose, m8
    names = ('raw_clip_fraction', 'zero_mg_pos_draw_count', 'zero_mg_pos_nonzero_raw_hour_count',
             'zero_mg_pos_positive_clipped_hour_count', 'new_level_mean', 'raw_hour_mean',
             'inherited_level_residual_mean', 'inherited_level_residual_std',
             'old_hour_minus_scale_mean', 'old_hour_minus_scale_std', 'raw_centered_shape_max_abs_change')
    diagnostics = {name: np.empty((60, 181), dtype=np.int64 if name.endswith('_count') else float)
                   for name in names}
    centered_a = a-a.mean(-1, keepdims=True)
    invariants = dict(replace_self_reconstruction=float(abs(a-u[..., None]+u[..., None]-a).max()),
        keep_self_reconstruction=float(abs(a-p[..., None]+p[..., None]-a).max()),
        keep_minus_replace=0., center_keep_ensemble_mean=0., flat_keep_particle_mean=0., shifted_centered_shape=0.)
    for mode_id, mode in enumerate(MODES):
        for i in range(15):
            j = mode_id*15+i
            path = raw[j+3]
            if mode == 'Replace': path[:] = a-u[..., None]+s[i, ..., None]
            elif mode == 'Keep': path[:] = a-p[..., None]+s[i, ..., None]
            elif mode == 'CenterOnly': path[:] = a+(s[i]-p).mean(-1)[:, None, None]
            else: path[:] = raw[18+i].mean(-1, keepdims=True)
            change = abs(path-path.mean(-1, keepdims=True)-centered_a).max((1, 2))
            values = ( ((path < 0) | (path > 1000)).mean((1, 2)), zero.sum(-1),
                ((abs(path) > 1e-9) & zero[..., None]).sum((1, 2)),
                ((path > 0) & zero[..., None]).sum((1, 2)), s[i].mean(-1), path.mean((1, 2)),
                inherited.mean(-1), inherited.std(-1), hour_minus_scale.mean(-1), hour_minus_scale.std(-1), change)
            for name, value in zip(names, values): diagnostics[name][j] = value
            key, error = 'shifted_centered_shape', float(change.max())
            if mode == 'Keep':
                invariants['keep_minus_replace'] = max(invariants['keep_minus_replace'],
                    float(abs(path-raw[3+i]-inherited[..., None]).max()))
            elif mode == 'CenterOnly':
                invariants['center_keep_ensemble_mean'] = max(invariants['center_keep_ensemble_mean'],
                    float(abs(path.mean(1)-raw[18+i].mean(1)).max()))
            elif mode == 'FlatKeep':
                key, error = 'flat_keep_particle_mean', float(abs(path.mean(-1)-raw[18+i].mean(-1)).max())
            invariants[key] = max(invariants[key], error)
    if (not np.isfinite(raw).all() or any(not np.isfinite(v).all() for v in diagnostics.values())
            or any(not np.isfinite(v) or v > 1e-9 for v in invariants.values())):
        raise ValueError('Finite paths/diagnostics and six invariants within 1e-9 required')
    return dict(raw=raw, old_scale_level=u, old_point_level=p, diagnostics=diagnostics,
                audit=dict(invariant_max_abs=invariants, diagnostic_arms=list(ARMS[3:]), raw_clipped=False,
                           scalar_level_does_not_identify_r4=True, monthly_energy_constraint=False,
                           risk_calibration_claim=False))
