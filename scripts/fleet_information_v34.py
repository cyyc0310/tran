"""Pure V34 B1 inventory information-value point baselines.

Inputs are caller-bound frozen arrays and registered-source capacity summaries.
No files, new labels, hourly series, unknown regions or physical output bounds.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from scripts import coarse_context_v30 as v30
from scripts import forward_context_v31 as v31
from scripts import fleet_inventory_v33 as inventory


SOURCES, EF, POS = v31.SOURCES, v31.EF, v31.POS
MODELS = ('Ridge', 'ET')
ARMS = ('Size-Ridge', 'Size-ET', 'Capacity-OP', 'Capacity-Operable', 'Utilization-OP',
        'SizeCap-Ridge', 'SizeCap-ET', 'IntensiveCap-Ridge', 'IntensiveCap-ET',
        'DirectEF-Ridge', 'DirectEF-ET')
FAMILIES = ('SizeCap', 'IntensiveCap', 'DirectEF')
POS_FUELS = tuple(v30.FUELS[i] for i in POS)
CAPACITY_FEATURE_NAMES = tuple('OP_capacity_POS_share_'+fuel for fuel in POS_FUELS) + ('lag0_POS_MW_per_OP_POS_capacity_MW',)
FEATURE_NAMES = {
    'Size': v31.FEATURE_NAMES['Size'],
    'SizeCap': v31.FEATURE_NAMES['Size'] + CAPACITY_FEATURE_NAMES,
    'IntensiveCap': v31.FEATURE_NAMES['Trend'] + CAPACITY_FEATURE_NAMES,
    'DirectEF': v31.FEATURE_NAMES['Size'] + CAPACITY_FEATURE_NAMES,
}
PARAMETERS = {name: v31.PARAMETERS[name].copy() for name in MODELS}
UTILIZATION_PARAMETERS = dict(x0=(0., 0., 0.), bounds=(-6., 6.), max_nfev=10000,
                              ftol=1e-10, xtol=1e-10, gtol=1e-10)
ORIGINS = pd.date_range('2025-01-01', periods=6, freq='MS', tz='UTC').as_unit('ns')
PROJECTION_KINDS = ('archived_simplex', 'archived_simplex', 'none', 'none', 'none',
                    'simplex', 'simplex', 'simplex', 'simplex', 'scalar_EF', 'scalar_EF')


def _training_metadata(fit: Mapping[str, Any], region: str) -> None:
    """Reject inadmissible label metadata before reading training numerics."""
    keys = list(fit['training_keys'])
    regions = np.asarray(fit['training_regions'], dtype=str)
    if (not keys or len(set(keys)) != len(keys) or regions.shape != (len(keys),)
            or set(regions) != set(SOURCES)-{region}):
        raise ValueError('Frozen six-source training metadata required')
    dates = {}
    for field in ('training_period_start_ns', 'training_period_end_ns', 'training_available_ns'):
        raw = np.asarray(fit[field])
        if raw.shape != (len(keys),) or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError('Integer nanosecond training metadata required')
        dates[field] = raw
    for i, value in enumerate(dates['training_period_start_ns']):
        at = pd.Timestamp(int(value), unit='ns', tz='UTC')
        end = pd.Timestamp(int(dates['training_period_end_ns'][i]), unit='ns', tz='UTC')
        release = int(dates['training_available_ns'][i])
        if (pd.isna(at) or not v30.START <= at < v30.END or at != at.normalize() or at.day != 1
                or end != at+pd.offsets.MonthBegin(1) or not end.value <= release <= v31.CUTOFF.value
                or keys[i] != v30._key(regions[i], at)):
            raise ValueError('Released pre-cutoff training metadata required')


def _parent_training(parent_audit: Mapping[str, Any], region: str) -> dict[str, Any]:
    if (parent_audit['region'] != region or parent_audit['outer'] != region
            or parent_audit['fit_count'] != 10):
        raise ValueError('Matching frozen V30 outer audit required')
    fits = parent_audit['fits']
    by_arm = {fit['arm']: fit for fit in fits}
    if len(fits) != 10 or len(by_arm) != 10 or set(by_arm) != set(v30.ARMS):
        raise ValueError('Complete frozen V30 head identities required')
    for model in MODELS:
        _training_metadata(by_arm['Size-'+model], region)
    tables = [v31._training(by_arm['Size-'+name], 'Size', name, region) for name in MODELS]
    for field in ('training_keys', 'training_regions', 'training_X', 'training_y', 'sample_weights',
                  'training_period_start_ns', 'training_period_end_ns', 'training_available_ns'):
        if not np.array_equal(tables[0][field], tables[1][field]):
            raise ValueError('Frozen Size heads require identical cohort, X, y, weights and dates')
    return tables[0]


def _inventory_metadata(regions: Iterable[str], dates: Iterable[pd.Timestamp]) -> list[dict[str, Any]]:
    result = []
    for region, at in zip(regions, dates):
        selected = inventory.inventory_for_origin(at)
        result.append(dict(key=v30._key(region, at), region=region, origin_ns=int(at.value), **selected))
    return result


def _capacity_metadata(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[int, str], Mapping[str, Any]]:
    indexed = {}
    for row in rows:
        region = row['region']
        if region not in SOURCES:
            continue  # Do not access any capacity or year of a non-source row.
        year_raw = row['year']
        if isinstance(year_raw, str):
            if not year_raw.isdecimal() or year_raw != year_raw.strip():
                raise ValueError('Exact integer inventory year metadata required')
            year = int(year_raw)
        elif isinstance(year_raw, (int, np.integer)) and not isinstance(year_raw, (bool, np.bool_)):
            year = int(year_raw)
        else:
            raise ValueError('Exact integer inventory year metadata required')
        if year not in inventory.SUPPORTED_YEARS:
            raise ValueError('Unsupported or future capacity year')
        key = (year, region)
        if key in indexed:
            raise ValueError('Duplicate capacity year/source key')
        indexed[key] = row
    expected = {(year, source) for year in inventory.SUPPORTED_YEARS for source in SOURCES}
    if set(indexed) != expected:
        raise ValueError('Complete exact 28 capacity year/source keys required; no fallback')
    return indexed


def _number(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('Finite nonnegative capacity required')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Finite nonnegative capacity required') from exc
    if not np.isfinite(number) or number < 0:
        raise ValueError('Finite nonnegative capacity required')
    return number


def _capacity_values(indexed: Mapping[tuple[int, str], Mapping[str, Any]]) -> dict[tuple[int, str], np.ndarray]:
    parsed = {}
    for key, row in indexed.items():
        capacities = np.array([_number(row[f'capacity_{fuel}_mw']) for fuel in v30.FUELS])
        pos, total = _number(row['capacity_pos_mw']), _number(row['capacity_total_mw'])
        with np.errstate(over='ignore'):
            expected_pos, expected_total = capacities[POS].sum(), capacities.sum()
        if (pos <= 0 or not np.isfinite(expected_total)
                or not np.isclose(pos, expected_pos, atol=1e-9, rtol=1e-12)
                or not np.isclose(total, expected_total, atol=1e-9, rtol=1e-12)):
            raise ValueError('Positive reconciled POS and total capacity required')
        parsed[key] = capacities
    return parsed


def _attach(capacities: Mapping[tuple[int, str], np.ndarray], metadata: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    selected = np.stack([capacities[(row['year'], row['region'])][POS] for row in metadata])
    totals = selected.sum(1)
    return selected/totals[:, None], totals


def _features(size: np.ndarray, shares: np.ndarray, pos_capacity: np.ndarray) -> dict[str, np.ndarray]:
    names = FEATURE_NAMES['Size']
    if (tuple(names) != tuple(v31.FEATURE_NAMES['Size']) or size.shape[-1] != len(names)
            or shares.shape != (len(size), 4) or pos_capacity.shape != (len(size),)):
        raise ValueError('Aligned frozen Size and capacity features required')
    suffix_shape = (len(size),) + (1,)*(size.ndim-2)
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            pressure = (np.exp(size[..., names.index('log_MW0')]) * size[..., names.index('lag0_POS')]
                        / pos_capacity.reshape(suffix_shape))
    except FloatingPointError as exc:
        raise ValueError('Finite untruncated inventory pressure required') from exc
    cap = np.broadcast_to(shares.reshape(suffix_shape+(4,)), size.shape[:-1]+(4,))
    extra = np.concatenate([cap, pressure[..., None]], axis=-1)
    intensive_ids = [names.index(name) for name in v31.FEATURE_NAMES['Trend']]
    result = dict(SizeCap=np.concatenate([size, extra], axis=-1),
                  IntensiveCap=np.concatenate([size[..., intensive_ids], extra], axis=-1))
    result['DirectEF'] = result['SizeCap'].copy()
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError('Finite capacity features required')
    return result


def _query(archived: Mapping[str, Any], dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    draws = np.asarray(archived['MG_draws'], float)
    size = np.asarray(archived['query_feature_Size'], float)
    if draws.shape != (6, 128, 5) or size.shape != (6, 128, 31) or not np.isfinite(size).all():
        raise ValueError('Finite archived MG (6,128,5) and Size (6,128,31) required')
    v31._simplex_rows(draws, 5)
    if (not np.array_equal(size[..., :5], draws)
            or not np.array_equal(size[..., 5:], np.broadcast_to(size[:, :1, 5:], size[..., 5:].shape))):
        raise ValueError('Archived Size features must match MG and particle-invariant history')
    names = FEATURE_NAMES['Size']
    for lag in range(3):
        ids = [names.index(f'lag{lag}_{name}') for name in v31.FEATURE_NAMES['G5']]
        v31._simplex_rows(size[..., ids], 5)
    phase = 2*np.pi*(dates.month.to_numpy()-1)/12
    if (not np.allclose(size[:, 0, names.index('month_sin')], np.sin(phase), atol=1e-12, rtol=0)
            or not np.allclose(size[:, 0, names.index('month_cos')], np.cos(phase), atol=1e-12, rtol=0)):
        raise ValueError('Archived Size feature names/calendar alignment changed')
    arms = np.asarray(archived['arms'])
    if arms.shape != (len(v31.ARMS),) or arms.tolist() != list(v31.ARMS):
        raise ValueError('Frozen V31 arm order required')
    indices = [v31.ARMS.index(arm) for arm in ARMS[:2]]
    old = {}
    for field in ('CIF_draws', 'EF_draws', 'projection_l1', 'CIF_point', 'EF_point'):
        raw = np.asarray(archived[field])
        shape = (6, 6) if field.endswith('_point') else (6, 6, 128)
        if raw.shape != shape:
            raise ValueError('Archived V31 prediction shape changed')
        values = np.asarray(raw[indices], float).copy()  # Only the two baseline heads are numeric inputs.
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Finite nonnegative archived baseline predictions required')
        old[field] = values
    for field in ('CIF', 'EF'):
        if not np.allclose(old[field+'_point'], old[field+'_draws'].mean(-1), atol=1e-10, rtol=0):
            raise ValueError('Archived particle-mean baseline points changed')
    if not np.allclose(old['CIF_draws'], old['EF_draws']*draws[None, ..., 0], atol=1e-10, rtol=0):
        raise ValueError('Archived baseline CIF/EF/MG correspondence changed')
    return size.copy(), draws.copy(), old


def _utilization_ratios(capacity_shares: np.ndarray, beta: np.ndarray) -> np.ndarray:
    mass = capacity_shares*np.exp(beta)
    return mass/mass.sum(-1, keepdims=True)


def _fit_utilization(shares: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit three relative multipliers; beta is not a physical utilization rate."""
    if (shares.ndim != 2 or shares.shape[1] != 4 or y.shape != shares.shape or weights.shape != (len(shares),)
            or not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError('Finite source-balanced utilization training arrays required')
    v31._simplex_rows(shares, 4); v31._simplex_rows(y, 4)

    def unpack(free: np.ndarray) -> np.ndarray:
        return np.array([free[0], 0., free[1], free[2]])

    def residual(free: np.ndarray) -> np.ndarray:
        return (np.sqrt(weights)[:, None]*(_utilization_ratios(shares, unpack(free))-y)).ravel()

    parameters = UTILIZATION_PARAMETERS.copy()
    initial = np.asarray(parameters.pop('x0'), float)
    result = least_squares(residual, x0=initial, **parameters)
    if not result.success:
        raise ValueError('Utilization optimization failed; no fallback')
    free, jacobian = np.asarray(result.x, float), np.asarray(result.jac, float)
    residuals = np.asarray(result.fun, float)
    if (free.shape != (3,) or jacobian.shape != (y.size, 3) or residuals.shape != (y.size,)
            or not np.isfinite(free).all() or np.any(np.abs(free) > 6)
            or not np.isfinite(jacobian).all() or not np.isfinite(residuals).all()
            or not np.isfinite(result.cost) or not np.isfinite(result.optimality)):
        raise ValueError('Finite successful utilization fit required')
    return unpack(free), dict(success=True, status=int(result.status), message=str(result.message),
        nfev=int(result.nfev), njev=None if result.njev is None else int(result.njev),
        cost=float(result.cost), optimality=float(result.optimality),
        active_mask=np.asarray(result.active_mask, int).copy(), jacobian_rank=int(np.linalg.matrix_rank(jacobian)),
        jacobian=jacobian.copy(), residuals=residuals.copy(),
        source_balanced_four_ratio_mse=float(np.sum(residuals**2)/(4*weights.sum())),
        rank_used_for_selection=False, physical_utilization_interpretation=False)


def _fit_record(reference: Mapping[str, Any], *, arm: str, family: str, model: str,
                x: np.ndarray, y: np.ndarray, mean: np.ndarray, scale: np.ndarray,
                region: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    record = {key: value.copy() if isinstance(value, np.ndarray) else list(value) if isinstance(value, list) else value
              for key, value in reference.items() if key not in ('training_X', 'training_y', 'scaler_mean', 'scaler_scale')}
    record.update(arm=arm, family=family, model=model, excluded_outer=region, parameters=dict(parameters),
                  training_X=x.copy(), training_y=y.copy(), scaler_mean=mean.copy(), scaler_scale=scale.copy(),
                  feature_names=CAPACITY_FEATURE_NAMES[:4] if family == 'Utilization' else FEATURE_NAMES[family])
    return record


def fit_predict(
    parent_audit: Mapping[str, Any], archived_query: Mapping[str, Any],
    capacity_op_rows: Iterable[Mapping[str, Any]], capacity_operable_rows: Iterable[Mapping[str, Any]],
    *, region: str,
) -> dict[str, Any]:
    """Copy two sealed baselines and fit exactly seven source-balanced B1 heads."""
    try:
        dates = v31._dates(archived_query['origins'])  # Before parent numbers, capacity rows or any fit.
        if not np.array_equal(dates.asi8, ORIGINS.asi8):
            raise ValueError('Complete fixed six V31 month-start origins required')
        if region not in SOURCES:
            raise ValueError('Registered outer source region required')
        reference = _parent_training(parent_audit, region)
        training_dates = pd.to_datetime(reference['training_period_start_ns'], unit='ns', utc=True)
        train_inventory = _inventory_metadata(reference['training_regions'], training_dates)
        query_inventory = _inventory_metadata([region]*len(dates), dates)
        # Complete metadata and release validation precedes any capacity conversion.
        op_metadata = _capacity_metadata(capacity_op_rows)
        operable_metadata = _capacity_metadata(capacity_operable_rows)
        op = _capacity_values(op_metadata)
        operable = _capacity_values(operable_metadata)
        for key in op:
            if np.any(op[key] > operable[key] + 1e-9):
                raise ValueError('Operable capacity cannot be smaller than OP capacity')
        train_shares, train_pos = _attach(op, train_inventory)
        query_shares, query_pos = _attach(op, query_inventory)
        sensitive_shares, _ = _attach(operable, query_inventory)
        size, draws, old = _query(archived_query, dates)
        train_features = _features(reference['training_X'], train_shares, train_pos)
        query_features = _features(size, query_shares, query_pos)
        n, particles = len(dates), draws.shape[1]
        ef_by_arm = {arm: old['EF_draws'][i] for i, arm in enumerate(ARMS[:2])}
        projection = {arm: old['projection_l1'][i] for i, arm in enumerate(ARMS[:2])}
        raw_ratios, ratios = {}, {}
        for arm, shares in (('Capacity-OP', query_shares), ('Capacity-Operable', sensitive_shares)):
            raw_ratios[arm] = np.broadcast_to(shares[:, None, :], (n, particles, 4)).copy()
            ratios[arm] = raw_ratios[arm].copy()
            ef_by_arm[arm] = ratios[arm]@EF[POS]
            projection[arm] = np.zeros((n, particles))
        y, weights = reference['training_y'], reference['sample_weights']
        beta, fit_result = _fit_utilization(train_shares, y, weights)
        utilization = _utilization_ratios(query_shares, beta)
        raw_ratios['Utilization-OP'] = np.broadcast_to(utilization[:, None, :], (n, particles, 4)).copy()
        ratios['Utilization-OP'] = raw_ratios['Utilization-OP'].copy()
        ef_by_arm['Utilization-OP'] = ratios['Utilization-OP']@EF[POS]
        projection['Utilization-OP'] = np.zeros((n, particles))
        utilization_record = _fit_record(reference, arm='Utilization-OP', family='Utilization', model='least_squares',
            x=train_shares, y=y, mean=np.zeros(4), scale=np.ones(4), region=region, parameters=UTILIZATION_PARAMETERS)
        utilization_record.update(beta=beta.copy(), fit_result=fit_result, gas_reference_beta=0., projection_kind='none')
        records = [utilization_record]
        for family in FAMILIES:
            x = train_features[family]
            query = query_features[family].reshape(-1, len(FEATURE_NAMES[family]))
            target = y@EF[POS]/1000 if family == 'DirectEF' else y
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
                raw = np.asarray(model.predict((query-mean)/scale), float)
                expected_shape = (len(query),) if family == 'DirectEF' else (len(query), 4)
                if raw.shape != expected_shape or not np.isfinite(raw).all():
                    raise ValueError('Finite correctly shaped B1 model predictions required')
                record = _fit_record(reference, arm=arm, family=family, model=name, x=x, y=target,
                                     mean=mean, scale=scale, region=region, parameters=PARAMETERS[name])
                if name == 'Ridge':
                    record.update(coefficients=np.asarray(model.coef_).copy(), intercept=np.asarray(model.intercept_).copy())
                else:
                    record['feature_importances'] = np.asarray(model.feature_importances_).copy()
                if family == 'DirectEF':
                    normalized = np.clip(raw, .4, 1.)
                    ef_by_arm[arm] = normalized.reshape(n, particles)*1000
                    projection[arm] = np.abs(raw-normalized).reshape(n, particles)
                    record.update(projection_kind='scalar_EF', scalar_clip=(.4, 1.),
                                  raw_query_prediction=raw.reshape(n, particles).copy())
                else:
                    raw_ratios[arm] = raw.reshape(n, particles, 4)
                    ratios[arm] = v30.project(raw_ratios[arm])
                    ef_by_arm[arm] = ratios[arm]@EF[POS]
                    projection[arm] = np.abs(ratios[arm]-raw_ratios[arm]).sum(-1)
                    record['projection_kind'] = 'simplex'
                records.append(record)
        ef = np.stack([ef_by_arm[arm] for arm in ARMS])
        cif = ef*draws[None, ..., 0]
        ef_points, cif_points = ef.mean(-1), cif.mean(-1)
        # Preserve archived points and draws literally instead of recomputing baselines.
        ef[:2], cif[:2] = old['EF_draws'], old['CIF_draws']
        ef_points[:2], cif_points[:2] = old['EF_point'], old['CIF_point']
        ratio_arms = tuple(arm for arm in ARMS if arm in ratios)
        return dict(EF_draws=ef, CIF_draws=cif, EF_point=ef_points, CIF_point=cif_points,
            projection_l1=np.stack([projection[arm] for arm in ARMS]), projection_kind=np.asarray(PROJECTION_KINDS),
            ratio_arms=np.asarray(ratio_arms), ratios_raw=np.stack([raw_ratios[a] for a in ratio_arms]),
            ratios=np.stack([ratios[a] for a in ratio_arms]), MG_draws=draws.copy(), origins=dates.asi8.copy(),
            query_feature_Size=size.copy(), **{'query_feature_'+k: v.copy() for k, v in query_features.items()},
            audit=dict(region=region, arms=list(ARMS), fit_count=len(records), fits=records,
                copied_baseline_arms=list(ARMS[:2]), query_feature_names=FEATURE_NAMES.copy(),
                training_inventory=train_inventory, query_inventory=query_inventory,
                capacity_scope='registered_ba_inventory_proxy', capacity_is_physical_output_bound=False,
                source_balanced_training=True, new_training_labels_used=False, hourly_values_read=False,
                target_values_opened=False, unseen_values_opened=False, current_group_truth_read=False,
                oracle_current_group=False, forward_forecast=True, information_budget='B1: B0 plus V33 capacity proxy',
                method_innovation_claim=False, beta_is_physical_utilization=False,
                point_definition='Mean of particle MG_POS times EF; EF point separately averages particle EF',
                projection_comparison='scalar_EF uses normalized scalar clipping; not comparable to simplex L1'))
    except KeyError as exc:
        raise ValueError('Missing frozen training/archive or capacity field: '+str(exc)) from exc
