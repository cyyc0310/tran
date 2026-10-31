"""Isolated retrospective raw-to-monthly exporter; no hourly output or predictor."""
from __future__ import annotations

from fractions import Fraction
import sys

import numpy as np
import pandas as pd

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from transcif.data.eia930 import FUEL_COLUMNS, MODERN_FUEL_COLUMNS  # noqa: E402
from transcif.research.eia_proxy import FUELS, _proxy_rows, MIN_MONTH_COVERAGE  # noqa: E402


def load_native_month_inputs(paths, regions):
    """Parse fuel/net values only after restricting the existing source roster."""
    frames, schemas = [], {}
    prefix = 'Net Generation (MW) from '
    for snapshot, path in enumerate(paths):
        header = pd.read_csv(path, nrows=0).columns
        mapping = MODERN_FUEL_COLUMNS if prefix + 'Hydropower Excluding Pumped Storage' in header else FUEL_COLUMNS
        names = {}
        for fuel, aliases in mapping.items():
            selected = [prefix + alias for alias in aliases if prefix + alias in header]
            if len(selected) != 1:
                raise ValueError('ambiguous native historical fuel schema')
            label = 'hydro_and_pumped_storage' if fuel == 'hydro' and mapping is FUEL_COLUMNS else fuel
            names[selected[0]] = f'generation_{label}_mw'
        unknown = [c for c in header if c.startswith(prefix) and c not in names and not c.endswith((' (Imputed)', ' (Adjusted)'))]
        if unknown:
            raise ValueError(f'unreviewed historical native fuel columns: {unknown}')
        columns = ['Balancing Authority', 'UTC Time at End of Hour', 'Net Generation (MW)', *names]
        raw = pd.read_csv(path, usecols=columns, dtype=str)
        raw = raw.loc[raw['Balancing Authority'].isin(regions)].copy()
        end = pd.to_datetime(raw.pop('UTC Time at End of Hour'), format='mixed', utc=True)
        if end.isna().any() or not end.eq(end.dt.floor('h')).all():
            raise ValueError('invalid historical UTC ending time')
        native = raw.rename(columns={'Balancing Authority': 'region', 'Net Generation (MW)': 'net_generation_mw', **names})
        native['valid_at'] = end - pd.Timedelta(hours=1)
        native = native.loc[native.valid_at.ge(pd.Timestamp('2022-01-01', tz='UTC'))
                            & native.valid_at.lt(pd.Timestamp('2025-01-01', tz='UTC'))].copy()
        for c in ('net_generation_mw', *names.values()):
            native[c] = pd.to_numeric(native[c], errors='raise')
        if np.isinf(native[['net_generation_mw', *names.values()]].to_numpy()).any():
            raise ValueError('infinite historical source generation')
        native['source_snapshot'] = snapshot
        schemas[str(snapshot)] = {'fuel_columns': list(names.values())}
        frames.append(native)
    combined = pd.concat(frames, ignore_index=True)
    duplicates = int(combined.duplicated(['region', 'valid_at']).sum())
    # Same earliest-file precedence as V144's source preparation.
    combined = combined.drop_duplicates(['region', 'valid_at'], keep='first').sort_values(['region', 'valid_at'])
    if set(combined.region) != set(regions):
        raise ValueError('historical source roster missing')
    return combined, schemas, duplicates


def annual_margins(combined, schemas, regions):
    records, audits = [], {}
    for region in regions:
        for year in (2022, 2023, 2024):
            rows = combined.loc[combined.region.eq(region) & combined.valid_at.dt.year.eq(year)].copy()
            rows.index = pd.DatetimeIndex(rows.valid_at)
            hourly, audit = _proxy_rows(rows, region, schemas)
            audits[f'{region}/{year}'] = audit
            hourly.index = pd.DatetimeIndex(hourly.time)
            for start, group in hourly.groupby(pd.Grouper(freq='MS')):
                end = start + pd.offsets.MonthBegin(1)
                expected = int((end - start) / pd.Timedelta(hours=1))
                generation = group.loc[group.valid, [f'generation_{f}' for f in FUELS]].to_numpy(float)
                count = len(generation)
                if count / expected < MIN_MONTH_COVERAGE or count == 0:
                    continue
                margins = [sum((Fraction(str(float(v))) for v in generation[:, j]), Fraction(0)) for j in range(8)]
                if sum(margins) <= 0:
                    continue
                records.append({'region': region, 'month': start.strftime('%Y-%m-01'), 'year': year,
                                'period_start': start.isoformat(), 'period_end': end.isoformat(),
                                'available_at': (end + pd.Timedelta(days=30)).isoformat(),
                                'observed_hours': count, 'calendar_hours': expected, 'coverage': count / expected,
                                **{f'margin_{f}': str(value) for f, value in zip(FUELS, margins)}})
        print(f'coarse margin export {region}', flush=True)
    return pd.DataFrame(records), audits
