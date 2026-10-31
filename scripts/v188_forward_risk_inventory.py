"""Inventory legal pre-July labels and existing H2 queries; no hourly labels/models."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from extended_monthly_information import ExtendedMonthlyInformation
from v151_hourly_law import sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v188-forward-risk-inventory-r1'
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
CUTOFF = pd.Timestamp('2024-07-01')


def inventory():
    data = ExtendedMonthlyInformation(ROOT, ROOT / 'results/v183-coarse-history-export-r1/monthly_history.csv')
    samples = pd.read_csv(BASE / 'source_sampling.csv')
    samples = samples.loc[samples.budget.eq('asof_30d')].drop_duplicates('region')
    times_path = ROOT / 'results/v153-calendar-times-r1/timestamps.npz'
    with np.load(times_path, allow_pickle=False) as packed:
        seconds = packed['unix_seconds']
    canonical_index = pd.read_csv(ROOT / 'results/v144-canonical-profile-cache-r1/profile_index.csv')
    rows, query_rows, selections = [], [], {}
    for fold in range(6):
        block = BASE / f'fold_{fold}_asof_30d'
        proper = sorted(json.loads((block / 'summary.json').read_text())['source_regions'])
        queries = pd.read_csv(block / 'query_inputs.csv')
        queries = queries.loc[queries.role.eq('test') & queries.month.ge('2024-07-01')].copy()
        if set(queries.region) & set(proper):
            raise ValueError('region roles overlap')
        for rec in queries.itertuples(index=False):
            q = int(rec.profile_id)
            if not CUTOFF <= data.dates.iloc[q] < pd.Timestamp('2025-01-01'):
                raise ValueError('query time')
            query_rows.append({'fold': fold, 'profile_id': q, 'region': rec.region,
                               'month': rec.month, 'period': 'Q3' if data.dates.iloc[q].month < 10 else 'Q4'})
        for region in proper:
            original = np.asarray(json.loads(samples.loc[samples.region.eq(region), 'sampled_raw_hour_indices'].iloc[0]), int)
            profiles, selected = [], []
            for q in data.groups[region]:
                if q >= data.canonical_count or not data.index.month.iloc[q].startswith('2024'):
                    continue
                # Offsets are canonical metadata, not exposed by the monthly container.
                canonical = canonical_index.iloc[q]
                ids = original[(original >= canonical.offset) & (original < canonical.stop)]
                if data.release.iloc[q] <= CUTOFF:
                    selected.extend(ids.tolist()); profiles.extend([int(q)] * len(ids))
            ids = np.asarray(selected, int)
            if not np.all(seconds[ids] < int(CUTOFF.tz_localize('UTC').timestamp())):
                raise ValueError('empty or future raw training indices')
            monthly = [int(q) for q in data.groups[region] if data.release.iloc[q] <= CUTOFF
                       and any(data.release.iloc[p] <= data.dates.iloc[q] for p in data.groups[region])]
            rows.append({'fold': fold, 'region': region, 'original_sample_count': len(original),
                         'forward_sample_count': len(ids), 'fine_profile_count': len(set(profiles)),
                         'monthly_label_count': len(monthly), 'latest_month_label': max(data.index.month.iloc[monthly]),
                         'first_raw_utc': pd.to_datetime(int(seconds[ids].min()), unit='s', utc=True).isoformat() if len(ids) else None,
                         'last_raw_utc': pd.to_datetime(int(seconds[ids].max()), unit='s', utc=True).isoformat() if len(ids) else None})
            selections[f'{fold}:{region}'] = {'raw_ids': ids.tolist(), 'profiles': profiles, 'monthly_ids': monthly}
    return pd.DataFrame(rows), pd.DataFrame(query_rows), selections


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    paths = [Path(__file__), ROOT / 'docs/V188_FORWARD_RISK_PROTOCOL.md',
             ROOT / 'scripts/extended_monthly_information.py', ROOT / 'scripts/monthly_history_features.py',
             ROOT / 'results/v183-coarse-history-export-r1/monthly_history.csv',
             ROOT / 'results/v144-canonical-profile-cache-r1/profile_index.csv',
             ROOT / 'results/v153-calendar-times-r1/timestamps.npz', BASE / 'source_sampling.csv']
    paths += [BASE / f'fold_{f}_asof_30d' / name for f in range(6) for name in ('summary.json','query_inputs.csv')]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    rows, queries, selected = inventory()
    if queries.duplicated('profile_id').any() or not set(queries.region).issubset(set(rows.region)):
        raise ValueError('query identity/source roster')
    if any(sha(ROOT / k) != v for k,v in hashes.items()): raise ValueError('input changed')
    OUT.mkdir()
    rows.to_csv(OUT/'source_inventory.csv', index=False)
    queries.to_csv(OUT/'queries.csv', index=False)
    (OUT/'selections.json').write_text(json.dumps(selected, indent=2)+'\n')
    summary={'assertion_failures':0,'cutoff':str(CUTOFF),'query_count':len(queries),'query_regions':sorted(set(queries.region)),
             'proper_roster_regions_without_H2_test_query':sorted(set(rows.region)-set(queries.region)),
             'queries_by_period':queries.groupby('period').size().to_dict(),
             'queries_by_fold':queries.groupby('fold').size().to_dict(),
             'sampled_hour_count_by_fold':rows.groupby('fold').forward_sample_count.sum().to_dict(),
             'monthly_label_count_by_fold':rows.groupby('fold').monthly_label_count.sum().to_dict(),
             'hourly_generation_loaded':False,'models_fitted':False,'scores_computed':False,
             'input_hashes':hashes,'output_hashes':{p.name:sha(p) for p in OUT.iterdir()}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print({k:v for k,v in summary.items() if not k.endswith('hashes')},flush=True)
if __name__=='__main__':main()
