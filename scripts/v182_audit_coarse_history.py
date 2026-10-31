"""Rebuild all historical time grids using Python datetime rather than pandas units."""
from __future__ import annotations

import calendar
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pandas as pd

from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import verify

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v182-coarse-history-inventory-r2'
OUT = ROOT / 'results/v182-coarse-history-inventory-audit-r1'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = verify(SOURCE)
    paths = [Path(__file__), SOURCE / 'summary.json', SOURCE / 'month_inventory.csv']
    hashes = summary['input_hashes'] | {str(p.relative_to(ROOT)): sha(p) for p in paths}
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('time audit input changed')
    old = pd.read_csv(SOURCE / 'month_inventory.csv')
    regions = set(old.region)
    start, stop = datetime(2022, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 1, tzinfo=timezone.utc)
    groups, counts = {}, {}
    raw_paths = [Path(p) for p in hashes if Path(p).is_absolute() and Path(p).suffix == '.csv']
    if len(raw_paths) != 4:
        raise ValueError('time audit raw-file denominator')
    rows = 0
    for path in raw_paths:
        frame = pd.read_csv(path, usecols=['Balancing Authority', 'UTC Time at End of Hour'])
        frame = frame.loc[frame['Balancing Authority'].isin(regions)]
        for region, text in frame.itertuples(index=False, name=None):
            stamp = datetime.strptime(text, '%m/%d/%Y %I:%M:%S %p').replace(tzinfo=timezone.utc) - timedelta(hours=1)
            if not start <= stamp < stop:
                continue
            key = region, stamp.strftime('%Y-%m')
            groups.setdefault(key, set()).add(int(stamp.timestamp()))
            counts[key] = counts.get(key, 0) + 1
            rows += 1
    rebuilt = []
    for record in old.itertuples(index=False):
        year, month = map(int, record.month.split('-'))
        date = datetime(year, month, 1, tzinfo=timezone.utc)
        size = calendar.monthrange(year, month)[1] * 24
        grid = {int(date.timestamp()) + h * 3600 for h in range(size)}
        key = record.region, record.month
        actual = groups.get(key, set())
        count = counts.get(key, 0)
        fresh = {'region': record.region, 'month': record.month, 'rows': count, 'unique_hours': len(actual),
                 'duplicate_hours': count - len(actual), 'calendar_hours': size,
                 'full_utc_grid': bool(count == len(actual) and actual == grid)}
        for name, value in fresh.items():
            if value != getattr(record, name):
                raise ValueError(f'independent calendar reconstruction differs: {key} {name}')
        rebuilt.append(fresh)
    if len(rebuilt) != 696 or len(regions) != 29:
        raise ValueError('time audit complete denominator')
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('time audit input changed during run')
    OUT.mkdir(parents=True)
    pd.DataFrame(rebuilt).to_csv(OUT / 'reconstruction.csv', index=False)
    result = {'assertion_failures': 0, 'metadata_rows_parsed': rows, 'source_months_rebuilt': len(rebuilt),
              'full_utc_months': sum(r['full_utc_grid'] for r in rebuilt), 'generation_values_used': False,
              'input_hashes': hashes, 'output_hashes': {'reconstruction.csv': sha(OUT / 'reconstruction.csv')},
              'scope': 'All 696 time grids independently reconstructed with standard-library UTC datetime/calendar/sets, only region/time columns selected. No fuel validity or performance claim.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
