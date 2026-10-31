"""Sharp structural bounds for the fixed-EF production proxy, not forecasts.

Only nonnegativity, observed row sums, fuel totals, and optional hourly group
sums are imposed. There are no dispatch/ramp/capacity or probability assumptions.
Individual hourly extrema need not be simultaneously attainable.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

FUELS = ('coal', 'gas', 'oil', 'nuclear', 'hydro', 'wind', 'solar', 'other')
EF = np.array([1000., 400., 800., 0., 0., 0., 0., 400.])
# A declared algebraic sensitivity, not estimated alternative emission factors.
EF_STRESS = np.array([1000., 500., 800., 0., 0., 0., 0., 600.])
ALL = (tuple(range(8)),)
GROUPS = {
    'none': ALL,
    'coal': ((0,), (1, 2, 3, 4, 5, 6, 7)),
    'gas': ((1,), (0, 2, 3, 4, 5, 6, 7)),
    'emitting': ((0, 1, 2, 7), (3, 4, 5, 6)),
    'coal_gas': ((0,), (1,), (2, 3, 4, 5, 6, 7)),
    'emitting_coal': ((0,), (1, 2, 7), (3, 4, 5, 6)),
}
BUDGETS = (
    ('month', 0, 'none'), ('day', 24, 'none'), ('12h', 12, 'none'),
    ('6h', 6, 'none'), ('3h', 3, 'none'), ('1h', 1, 'none'),
    ('day_coal', 24, 'coal'), ('day_gas', 24, 'gas'),
    ('day_emitting', 24, 'emitting'), ('day_coal_gas', 24, 'coal_gas'),
    ('day_emitting_coal', 24, 'emitting_coal'),
)
GROUP_ONLY = {'hour_coal_only': 'coal', 'hour_emitting_only': 'emitting',
              'hour_emitting_coal_only': 'emitting_coal'}


def group_only_bounds(generation: ArrayLike, groups: tuple[tuple[int, ...], ...],
                      ef: ArrayLike = EF) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Drop all fuel-total constraints; each hourly group can choose any member.

    Unlike the fuel-budget bounds, these endpoints are jointly attainable
    because no cross-hour constraints remain. This control isolates the value
    of knowing exact same-day fuel totals in addition to a sensor channel.
    """
    a, ef = np.asarray(generation, float), np.asarray(ef, float)
    if (a.ndim != 2 or not len(a) or len(a) % 24 or a.shape[1] != len(ef)
            or not np.isfinite(a).all() or np.any(a < 0) or np.any(a.sum(1) <= 0)
            or not np.isfinite(ef).all()
            or sorted(f for group in groups for f in group) != list(range(a.shape[1]))
            or any(not len(group) for group in groups)):
        raise ValueError('Complete nonnegative generation and disjoint fuel groups required')
    lo, hi = np.zeros(len(a)), np.zeros(len(a))
    for group in groups:
        fuel = np.asarray(group, int)
        mass = a[:, fuel].sum(1)
        lo += mass * ef[fuel].min()
        hi += mass * ef[fuel].max()
    lo /= a.sum(1)
    hi /= a.sum(1)
    return lo, hi, lo.reshape(-1, 24).mean(1), hi.reshape(-1, 24).mean(1)


def _transport_inputs(rows, columns, ef, coefficients=None):
    rows, columns, ef = (np.asarray(x, dtype=float) for x in (rows, columns, ef))
    if (rows.ndim != 1 or columns.ndim != 1 or not len(rows) or not len(columns)
            or ef.shape != columns.shape or not all(np.isfinite(x).all() for x in (rows, columns, ef))
            or np.any(rows < 0) or np.any(columns < 0)
            or not np.isclose(rows.sum(), columns.sum(), rtol=1e-11, atol=1e-9)):
        raise ValueError('Finite nonnegative balanced transportation margins required')
    if coefficients is not None:
        coefficients = np.asarray(coefficients, float)
        if coefficients.shape != rows.shape or not np.isfinite(coefficients).all():
            raise ValueError('One finite objective coefficient per hour required')
    return rows, columns, ef, coefficients


def marginal_emissions_bounds(rows: ArrayLike, columns: ArrayLike,
                              ef: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Sharp marginal numerator bounds by filling each hour from sorted fuels.

    Any nonnegative remainder can fill the other hours because there are no
    cell-specific capacities. This proves attainability of each endpoint.
    """
    rows, columns, ef, _ = _transport_inputs(rows, columns, ef)
    endpoints = []
    for order in (np.argsort(ef, kind='stable'), np.argsort(-ef, kind='stable')):
        supplies = columns[order]
        prefix = np.r_[0., np.cumsum(supplies)[:-1]]
        allocation = np.clip(rows[:, None]-prefix, 0., supplies)
        endpoints.append(allocation @ ef[order])
    return tuple(endpoints)


def _transport_extreme(rows, columns, ef, coefficients, maximize, witness):
    # Rank-one costs: the rearrangement inequality gives optimal monotone
    # pairing of unit generation mass, including negative coefficients.
    row_order = np.argsort(coefficients, kind='stable')
    fuel_order = np.argsort(ef if maximize else -ef, kind='stable')
    remaining_rows, remaining_fuels = rows.copy(), columns.copy()
    allocation = np.zeros((len(rows), len(columns))) if witness else None
    value, i, j = 0., 0, 0
    while i < len(rows) and j < len(columns):
        r, f = row_order[i], fuel_order[j]
        mass = min(remaining_rows[r], remaining_fuels[f])
        value += mass * coefficients[r] * ef[f]
        if witness:
            allocation[r, f] += mass
        remaining_rows[r] -= mass
        remaining_fuels[f] -= mass
        # At least one margin is exhausted exactly by the subtraction.
        if remaining_rows[r] <= 0:
            i += 1
        if remaining_fuels[f] <= 0:
            j += 1
    return allocation if witness else float(value)


def transport_witness(rows: ArrayLike, columns: ArrayLike, ef: ArrayLike,
                      coefficients: ArrayLike, maximize: bool = False) -> FloatArray:
    """Return an attaining generation table for one linear CIF functional."""
    args = _transport_inputs(rows, columns, ef, coefficients)
    return _transport_extreme(*args, maximize, True)


def functional_bounds(rows: ArrayLike, columns: ArrayLike, ef: ArrayLike,
                       coefficients: ArrayLike) -> tuple[float, float]:
    """Sharp bounds on sum_h coefficients[h] * emissions[h]."""
    args = _transport_inputs(rows, columns, ef, coefficients)
    return tuple(_transport_extreme(*args, flag, False) for flag in (False, True))


def oracle_bounds(generation: ArrayLike, block_ids: ArrayLike,
                  groups: tuple[tuple[int, ...], ...] = ALL,
                  ef: ArrayLike = EF) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Build constraints from retrospective truth to diagnose information value.

    Generation is aligned complete UTC days, grouped into disjoint temporal
    blocks. For each block fuel totals are known; for each hour the generation
    total of each fuel group is known. Singleton groups model extra sensors.
    """
    a, blocks, ef = np.asarray(generation, float), np.asarray(block_ids), np.asarray(ef, float)
    if (a.ndim != 2 or a.shape[1] != len(ef) or not len(a) or len(a) % 24
            or not np.isfinite(a).all() or np.any(a < 0) or np.any(a.sum(1) <= 0)
            or blocks.shape != (len(a),) or not np.isfinite(ef).all()):
        raise ValueError('Finite generation on aligned complete positive 24h days required')
    flat = [f for group in groups for f in group]
    if sorted(flat) != list(range(a.shape[1])) or any(not len(g) for g in groups):
        raise ValueError('Fuel groups must be a disjoint complete partition')
    total = a.sum(1)
    lo, hi = np.zeros(len(a)), np.zeros(len(a))
    ml, mh = np.zeros(len(a)//24), np.zeros(len(a)//24)
    day_ids = np.arange(len(a))//24
    for block in np.unique(blocks):
        indices = np.flatnonzero(blocks == block)
        for group in groups:
            fuel = np.asarray(group, int)
            sub = a[np.ix_(indices, fuel)]
            rows, cols, factors = sub.sum(1), sub.sum(0), ef[fuel]
            el, eh = marginal_emissions_bounds(rows, cols, factors)
            lo[indices] += el/total[indices]
            hi[indices] += eh/total[indices]
            for day in np.unique(day_ids[indices]):
                coefficient = (day_ids[indices] == day)/(24*total[indices])
                left, right = functional_bounds(rows, cols, factors, coefficient)
                ml[day] += left
                mh[day] += right
    return lo, hi, ml, mh


def read_evaluation(path: Path, region: str, year: int) -> tuple[pd.DatetimeIndex, FloatArray]:
    """Only parse requested known-region H1 numeric values, never other years."""
    if path.stem != f'{region}_hourly':
        raise ValueError('Region-specific hourly file required')
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        records = [row for row in reader if row['time'].startswith(f'{year}-')
                   and row['time'][:7] < f'{year}-07']
    if not records:
        raise ValueError('No H1 evaluation rows')
    frame = pd.DataFrame(records)
    times = pd.DatetimeIndex(pd.to_datetime(frame.time, utc=True, errors='raise'))
    if (times.hasnans or times.has_duplicates or not times.equals(times.floor('h'))
            or not frame.region.eq(region).all()):
        raise ValueError('Consistent region and unique UTC hours required')
    if not set(frame.valid) <= {'True', 'False'}:
        raise ValueError('Explicit boolean validity required')
    valid = frame.valid.eq('True').to_numpy()
    cols = [f'generation_{f}' for f in FUELS]
    # Invalid rows are intentionally not converted to float.
    values = np.full((len(frame), 8), np.nan)
    values[valid] = frame.loc[valid, cols].to_numpy(float)
    order = np.argsort(times.asi8)
    times, valid, values = times[order], valid[order], values[order]
    days, arrays = [], []
    for day in times.floor('D').unique():
        take = np.flatnonzero(times.floor('D') == day)
        if len(take) != 24 or not valid[take].all():
            continue
        table = values[take]
        if (not np.isfinite(table).all() or np.any(table < 0) or np.any(table.sum(1) <= 0)):
            raise ValueError('Invalid positive-generation evaluation truth')
        days.append(day)
        arrays.append(table)
    if not days:
        raise ValueError('No complete valid days')
    return pd.DatetimeIndex(days), np.concatenate(arrays)
