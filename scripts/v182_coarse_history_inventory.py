"""Inventory only region/time metadata of potential 2022–2023 coarse histories."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import verify

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
OUT = ROOT / 'results/v182-coarse-history-inventory-r1'
PROTOCOL = ROOT / 'docs/V182_COARSE_HISTORY_INVENTORY_PROTOCOL.md'
META = ROOT / 'data/upstream-staging/eia-2022-2024/metadata.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    verify(BASE)
    metadata = json.loads(META.read_text())
    roster = pd.read_csv(BASE / 'source_sampling.csv')
    regions = sorted(roster.region.unique())
    if len(regions) != 29:
        raise ValueError('unexpected source roster')
    sources = [Path(v['input']) for v in metadata['native_schemas'].values()
               if any(year in Path(v['input']).name for year in ('2022', '2023'))]
    if len(sources) != 4 or len(set(sources)) != 4:
        raise ValueError('historical file inventory')
    hashes = {str(p): metadata['input_sha256'][str(p)] for p in sources}
    hashes |= {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), PROTOCOL, META, BASE / 'source_sampling.csv', BASE / 'summary.json')}
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('historical source snapshot changed')
    pieces = []
    for path in sources:
        frame = pd.read_csv(path, usecols=['Balancing Authority', 'UTC Time at End of Hour'])
        frame = frame.loc[frame['Balancing Authority'].isin(regions)]
        start = pd.to_datetime(frame['UTC Time at End of Hour'], format='%m/%d/%Y %I:%M:%S %p', utc=True) - pd.Timedelta(hours=1)
        keep = (start >= pd.Timestamp('2022-01-01', tz='UTC')) & (start < pd.Timestamp('2024-01-01', tz='UTC'))
        piece = pd.DataFrame({'region': frame.loc[keep, 'Balancing Authority'], 'unix_seconds': start.loc[keep].astype('int64') // 10**9})
        pieces.append(piece)
    frame = pd.concat(pieces, ignore_index=True)
    times = pd.to_datetime(frame.unix_seconds, unit='s', utc=True)
    frame['month'] = times.dt.strftime('%Y-%m')
    groups = {(r, m): g.unix_seconds.to_numpy() for (r, m), g in frame.groupby(['region', 'month'])}
    records = []
    for region in regions:
        for date in pd.date_range('2022-01-01', '2023-12-01', freq='MS', tz='UTC'):
            month = date.strftime('%Y-%m')
            observed = groups.get((region, month), np.array([], dtype=np.int64))
            unique = np.unique(observed)
            expected = np.arange(int(date.timestamp()), int((date + pd.offsets.MonthBegin(1)).timestamp()), 3600, dtype=np.int64)
            records.append({'region': region, 'month': month, 'rows': len(observed), 'unique_hours': len(unique),
                            'duplicate_hours': len(observed) - len(unique), 'calendar_hours': len(expected),
                            'full_utc_grid': bool(len(observed) == len(unique) and np.array_equal(unique, expected))})
    result = pd.DataFrame(records)
    if len(result) != 696 or result.duplicated(['region', 'month']).any():
        raise ValueError('source/month denominator')
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('inventory source changed during run')
    OUT.mkdir(parents=True)
    result.to_csv(OUT / 'month_inventory.csv', index=False)
    summary = {'assertion_failures': 0, 'source_regions': len(regions), 'source_months': len(result),
               'nonempty_months': int(result.rows.gt(0).sum()), 'full_utc_months': int(result.full_utc_grid.sum()),
               'regions_all_24_full_months': int(result.groupby('region').full_utc_grid.all().sum()),
               'duplicate_hours': int(result.duplicate_hours.sum()), 'generation_columns_parsed': False,
               'new_coarse_values_exported': False, 'hourly_outcomes_scored': False,
               'input_hashes': hashes, 'output_hashes': {'month_inventory.csv': sha(OUT / 'month_inventory.csv')},
               'scope': 'Region/time metadata only, 2022–2023 within the existing 29-source roster. No fuel completeness, public-vintage, coarse-supervision quality or performance claim.'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if 'hash' not in k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
