"""Source-only fixed-H1 native teachers for H2; no risk fitting or scoring."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scripts import run_weather_recomposition_v11 as native
from scripts.weather_coarse_v14 import legal_coarse_law
from transcif.research.coarse_forecast_v7 import COLUMNS
from transcif.research.fuel_transport_v7 import recompose_paths

SOURCES = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
SAMPLES, SEED = 128, 17023
START = pd.Timestamp('2024-01-25', tz='UTC')
CUTOFF = pd.Timestamp('2024-07-01', tz='UTC')
H1_DAYS = pd.date_range('2024-01-25', '2024-06-30', tz='UTC')
H2_DAYS = pd.date_range('2024-07-01', '2024-12-31', tz='UTC')
QUERY_DAYS = pd.date_range('2025-01-01', '2025-06-30', tz='UTC')
FUELS = ('coal', 'gas', 'oil', 'nuclear', 'hydro', 'wind', 'solar', 'other')
GENERATION = ['generation_'+f for f in FUELS]


def proper_frame(path: Path, region: str) -> pd.DataFrame:
    """Discard H2/future timestamps before reading any generation value numerically."""
    if region not in SOURCES or path.name != f'{region}_hourly.csv':
        raise ValueError('Registered source hourly filename required')
    with path.open(newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if '2024-01-25' <= row['time'] < '2024-07-01']
    if not rows:
        raise ValueError('Source has no proper rows')
    table = pd.DataFrame(rows)
    times = pd.DatetimeIndex(pd.to_datetime(table.time, utc=True, errors='raise'))
    if (times.has_duplicates or not times.equals(times.floor('h')) or np.any(times < START)
            or np.any(times >= CUTOFF) or not table.region.eq(region).all()
            or not set(table.valid) <= {'True', 'False'}):
        raise ValueError('Registered unique UTC proper-hour schema required')
    parts = []
    for day in sorted(times.floor('D').unique()):
        indices = np.flatnonzero(times.floor('D') == day)
        if len(indices) != 24 or not table.iloc[indices].valid.eq('True').all():
            continue
        indices = indices[np.argsort(times[indices].asi8)]
        if not np.array_equal(times[indices].asi8, pd.date_range(day, periods=24, freq='h').asi8):
            raise ValueError('Proper donors must be complete UTC days')
        values = table.iloc[indices][GENERATION].to_numpy(float)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values.sum(1) <= 0):
            raise ValueError('Finite nonnegative positive-total proper generation required')
        part = pd.DataFrame(values, columns=GENERATION)
        part['time'], part['region'], part['valid'] = times[indices], region, True
        parts.append(part)
    if not parts:
        raise ValueError('Source has no complete valid proper day')
    return pd.concat(parts, ignore_index=True)


def monthly_prefix(path: Path) -> pd.DataFrame:
    """Source coarse vintages only; numeric conversion follows latest-origin gate."""
    kept = []
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['region'] not in SOURCES:
                continue
            end, available = pd.Timestamp(row['period_end']), pd.Timestamp(row['available_at'])
            if end.tzinfo is None or available.tzinfo is None:
                raise ValueError('Explicit UTC coarse timestamps required')
            if end <= H2_DAYS[-1] and available <= H2_DAYS[-1]:
                kept.append({k: row[k] for k in ('region', 'period_start', 'period_end', 'available_at', *COLUMNS)})
    table = pd.DataFrame(kept, columns=['region', 'period_start', 'period_end', 'available_at', *COLUMNS])
    for key in ('period_start', 'period_end', 'available_at'):
        table[key] = pd.to_datetime(table[key], utc=True, errors='raise')
    table[list(COLUMNS)] = table[list(COLUMNS)].astype(float)
    return table


def exclusions(outer: str, teacher: str) -> tuple[str, ...]:
    if outer not in SOURCES or teacher not in SOURCES or outer == teacher:
        raise ValueError('Two different registered source exclusions required')
    return tuple(sorted((outer, teacher)))


def bank_arrays(bank: Any, excluded: tuple[str, ...]) -> dict[str, np.ndarray]:
    if len(excluded) != 2 or tuple(excluded) != exclusions(*excluded):
        raise ValueError('Two canonical distinct source exclusions required')
    expected = tuple(r for r in SOURCES if r not in excluded)
    if (len(excluded) != 2 or tuple(bank.source_regions_) != expected or bank.cutoff_ != CUTOFF
            or not len(bank.days_) or np.any(bank.days_ < START) or np.any(bank.days_ >= CUTOFF)
            or not set(bank.regions_) == set(expected) or not np.array_equal(bank.months_, bank.days_.month)):
        raise ValueError('Exact pair-excluded H1 donor bank required')
    arrays = dict(q=bank.q_, raw_cif=bank.raw_cif_, donor_mix=bank.donor_mix_,
        monthmean=bank.month_mean_generation_, support=bank.fuel_support_,
        regions=np.asarray(bank.regions_, dtype='U'), days=bank.days_.as_unit('ns').asi8,
        months=bank.months_, row_ids=np.arange(len(bank.days_)),
        excluded=np.asarray(excluded, dtype='U'), source_regions=np.asarray(expected, dtype='U'),
        cutoff=np.array(CUTOFF.value, dtype=np.int64))
    if any(not np.isfinite(arrays[k]).all() for k in ('q', 'raw_cif', 'donor_mix', 'monthmean')):
        raise ValueError('Nonfinite H1 bank arrays')
    return arrays


def split_days(split: str) -> pd.DatetimeIndex:
    if split not in ('H1_retrospective', 'H2_hindcast'):
        raise ValueError('Explicit registered H1 retrospective or H2 hindcast split required')
    return H1_DAYS if split == 'H1_retrospective' else H2_DAYS


def coarse_arrays(monthly: pd.DataFrame, region: str, split: str) -> dict[str, np.ndarray]:
    if region not in SOURCES:
        raise ValueError('Only registered source coarse queries allowed')
    days = split_days(split)
    law = legal_coarse_law(monthly, region, days)
    part = monthly[monthly.region.eq(region)]
    ends, available, counts = [], [], []
    for day in days:
        prefix = part[part.period_end.le(day) & part.available_at.le(day)]
        counts.append(len(prefix))
        ends.append(prefix.period_end.max().value if len(prefix) else -1)
        available.append(prefix.available_at.max().value if len(prefix) else -1)
    if not np.array_equal(np.asarray(counts), law['metadata'][:, 0]):
        raise ValueError('Coarse law released-prefix count mismatch')
    return dict(coarse_mix=law['mix'], coarse_draws=law['draws'], coarse_metadata=law['metadata'],
                coarse_max_period_end=np.array(ends, dtype=np.int64),
                coarse_max_available_at=np.array(available, dtype=np.int64))


def teachers(bank: Any, coarse: dict[str, np.ndarray], split: str) -> dict[str, np.ndarray]:
    days = split_days(split)
    mix, draws = coarse['coarse_mix'], coarse['coarse_draws']
    if (mix.shape != (len(days), 8) or draws.shape != (len(days), 128, 8)
            or not np.isfinite(mix).all() or not np.isfinite(draws).all()
            or np.any(mix < 0) or np.any(draws < 0)
            or not np.allclose(mix.sum(1), 1., atol=1e-12, rtol=0)
            or not np.allclose(draws.sum(2), 1., atol=1e-12, rtol=0)
            or bank.days_.min() < START or bank.days_.max() >= CUTOFF
            or (split == 'H2_hindcast' and bank.days_.max() >= days[0])):
        raise ValueError('Registered coarse simplex arrays and fixed-H1 bank required')
    ids, _ = native._indices(bank, mix, days, np.asarray(bank.source_regions_))
    raw = np.stack([recompose_paths(m, bank.q_[selected]) for m, selected in zip(draws, ids)])
    return dict(origins=days.asi8, raw=raw, center=raw.mean(1), donor_ids=ids,
        chronological_hindcast=np.array(split == 'H2_hindcast'),
        within_proper_future_donors=np.array(split == 'H1_retrospective'),
        future_donor_rows=np.array([np.count_nonzero(bank.days_ >= day) for day in days]),
        sampled_future_donors=(bank.days_.asi8[ids] >= days.asi8[:, None]).sum(1), **coarse)


def restore_bank(arrays: dict[str, np.ndarray], excluded: tuple[str, str]) -> Any:
    """Validate sealed H1 row provenance and restore only the native sampler state."""
    if len(excluded) != 2 or tuple(excluded) != exclusions(*excluded):
        raise ValueError('Two canonical distinct source exclusions required')
    expected = tuple(r for r in SOURCES if r not in excluded)
    q, regions, dates = arrays['q'], arrays['regions'], arrays['days']
    n = len(regions)
    if (q.shape != (n, 24, 8) or arrays['raw_cif'].shape != (n, 24)
            or arrays['monthmean'].shape != (n, 8) or arrays['donor_mix'].shape != (n, 8)
            or arrays['support'].shape != (n, 8) or arrays['support'].dtype != np.bool_
            or dates.shape != (n,) or not n or not np.issubdtype(dates.dtype, np.integer)
            or not np.array_equal(arrays['row_ids'], np.arange(n))
            or not np.array_equal(arrays['excluded'], excluded)
            or not np.array_equal(arrays['source_regions'], expected)
            or arrays['cutoff'].shape != () or arrays['cutoff'].item() != CUTOFF.value):
        raise ValueError('Invalid bank row schema or pair exclusions')
    days = pd.DatetimeIndex(pd.to_datetime(dates, utc=True))
    if (not days.equals(days.normalize()) or np.any(days < START) or np.any(days >= CUTOFF)
            or set(regions) != set(expected) or pd.MultiIndex.from_arrays([regions, dates]).has_duplicates
            or not np.array_equal(arrays['months'], days.month)):
        raise ValueError('Bank provenance must contain only unique pair-excluded H1 region-days')
    if (any(not np.isfinite(arrays[k]).all() for k in ('q', 'raw_cif', 'donor_mix', 'monthmean'))
            or np.any(q < 0) or np.any(arrays['monthmean'] < 0) or np.any(arrays['monthmean'].sum(1) <= 0)
            or not np.array_equal(arrays['support'], arrays['monthmean'] > 0)
            or not np.allclose(arrays['donor_mix'], arrays['monthmean']/arrays['monthmean'].sum(1)[:, None], atol=1e-12, rtol=0)
            or np.any(q[~np.broadcast_to(arrays['support'][:, None], q.shape)] != 1)):
        raise ValueError('Invalid H1 bank normalization or unsupported-fuel prior')
    for region in expected:
        for month in np.unique(days.month[regions == region]):
            take = (regions == region) & (days.month == month)
            if (not np.array_equal(arrays['monthmean'][take], np.broadcast_to(arrays['monthmean'][take][0], (take.sum(), 8)))
                    or not np.allclose(q[take].mean((0, 1)), 1., atol=1e-10, rtol=0)):
                raise ValueError('Bank normalization must use H1 region-month observed support')
    if not np.allclose(recompose_paths(arrays['donor_mix'], q), arrays['raw_cif'], atol=1e-9, rtol=0):
        raise ValueError('Bank monthly-share reconstruction failed')
    bank = native.FuelTrajectoryBank(expected)
    bank.q_, bank.donor_mix_, bank.days_, bank.regions_ = q, arrays['donor_mix'], days, regions
    bank.source_regions_, bank.months_, bank.cutoff_ = expected, days.month.to_numpy(), CUTOFF
    return bank


def query_rows(frames: dict[str, pd.DataFrame], held: str) -> dict[str, np.ndarray]:
    """Reproduce inherited six-source bank row ordering without refitting a query model."""
    regions, days = [], []
    for region in SOURCES:
        if region == held:
            continue
        ds = pd.DatetimeIndex(frames[region].time).floor('D').unique().sort_values()
        days.extend(ds.asi8.tolist()); regions.extend([region]*len(ds))
    return dict(bank_regions=np.asarray(regions, dtype='U'), bank_days=np.asarray(days, dtype=np.int64),
                bank_row_ids=np.arange(len(days)), bank_cutoff=np.array(CUTOFF.value, dtype=np.int64))
