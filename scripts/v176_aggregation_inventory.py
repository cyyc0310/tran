"""Metadata-only inventory of fixed-budget synchronous source portfolios."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cif_pit_recalibration import source_samples
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import verify

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
CACHE = ROOT / 'results/v144-canonical-profile-cache-r1'
TIMES = ROOT / 'results/v153-calendar-times-r1/timestamps.npz'
OUT = ROOT / 'results/v176-aggregation-inventory-r1'
PROTOCOL = ROOT / 'docs/V176_AGGREGATION_INVENTORY_PROTOCOL.md'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    verify(BASE)
    paths = [Path(__file__), PROTOCOL, ROOT / 'scripts/cif_pit_recalibration.py',
             ROOT / 'scripts/v151_hourly_law.py', ROOT / 'scripts/v170_fuel_marginal_diagnosis.py',
             BASE / 'summary.json', BASE / 'source_sampling.csv', CACHE / 'profile_index.csv', TIMES]
    paths.extend(BASE / f'fold_{k}_asof_30d/summary.json' for k in range(6))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    index = pd.read_csv(CACHE / 'profile_index.csv')
    with np.load(TIMES, allow_pickle=False) as saved:
        seconds = saved['unix_seconds'].copy()
    dates = pd.to_datetime(index.month)
    release = dates + pd.offsets.MonthBegin(1) + pd.Timedelta(days=30)
    groups = {r: np.asarray(ids, int) for r, ids in index.groupby('region').indices.items()}
    complete = {}

    def full_month(q):
        q = int(q)
        if q not in complete:
            start = dates.iloc[q].tz_localize('UTC')
            end = start + pd.offsets.MonthBegin(1)
            expected = np.arange(int(start.timestamp()), int(end.timestamp()), 3600, dtype=np.int64)
            row = index.iloc[q]
            observed = seconds[int(row.offset):int(row.stop)]
            complete[q] = row.hours == len(expected) and np.array_equal(observed, expected)
        return bool(complete[q])

    def local(q):
        ids = groups[index.region.iloc[q]]
        legal = ids[(release.iloc[ids] <= dates.iloc[q]).to_numpy()]
        return int(legal[np.argmax(dates.iloc[legal].to_numpy())]) if len(legal) else None

    records = []
    for fold in range(6):
        fit = sorted(json.loads((BASE / f'fold_{fold}_asof_30d/summary.json').read_text())['source_regions'])
        samples = source_samples(BASE, fit)
        bank, sampled = {}, {}
        for region in fit:
            ids = samples[region]
            profiles = np.searchsorted(index.stop.to_numpy(), ids, side='right')
            if set(index.region.iloc[np.unique(profiles)]) != {region}:
                raise ValueError('source sample identity')
            bank[region] = {index.month.iloc[q]: int(q) for q in np.unique(profiles)}
            for q in np.unique(profiles):
                pick = ids[profiles == q]
                if not index.month.iloc[q].startswith('2024') or np.any(np.diff(seconds[pick]) <= 0):
                    raise ValueError('source time order/year')
                sampled[int(q)] = pick
        pairs = list(itertools.combinations(fit, 2))
        quads = sorted(itertools.combinations(fit, 4), key=lambda group:
                       hashlib.sha256(('v176/' + ','.join(group)).encode()).hexdigest())[:64]
        for parents in pairs + quads:
            months = sorted(set().union(*(set(bank[r]) for r in parents)))
            for month in months:
                qs = [bank[r].get(month) for r in parents]
                exists = all(q is not None for q in qs)
                locals_ = [local(q) if q is not None else None for q in qs]
                current_complete = exists and all(full_month(q) for q in qs)
                input_complete = all(q is not None and full_month(q) for q in locals_)
                aligned = input_complete and len(set(index.month.iloc[q] for q in locals_)) == 1
                shared = np.array([], dtype=seconds.dtype)
                parent_raw = np.empty((0, len(parents)), np.int64)
                if exists:
                    shared = seconds[sampled[qs[0]]]
                    for q in qs[1:]:
                        shared = np.intersect1d(shared, seconds[sampled[q]], assume_unique=True)
                    raw = []
                    for region, q in zip(parents, qs):
                        pick = sampled[q]
                        positions = np.searchsorted(seconds[pick], shared)
                        matched = pick[positions]
                        if not np.array_equal(seconds[matched], shared) or not np.isin(matched, samples[region]).all():
                            raise ValueError('portfolio used an unsampled source hour')
                        raw.append(matched)
                    parent_raw = np.column_stack(raw)
                records.append({'fold': fold, 'parents': json.dumps(parents), 'k': len(parents), 'month': month,
                                'profiles': json.dumps(qs), 'input_profiles': json.dumps(locals_),
                                'all_parents_sampled': exists, 'current_full_UTC_month': current_complete,
                                'input_full_UTC_month': input_complete, 'input_same_month': aligned,
                                'legal_aggregate_month': bool(current_complete and input_complete and aligned),
                                'parent_sampled_hours': json.dumps([len(sampled[q]) if q is not None else 0 for q in qs]),
                                'shared_sampled_hours': len(shared), 'clock_cells': len(np.unique((shared // 3600) % 24)),
                                'intersection_raw_sha256': hashlib.sha256(parent_raw.tobytes()).hexdigest()})
        print(f'portfolio metadata fold {fold}: {len(pairs)} pairs, {len(quads)} fixed quads', flush=True)
    frame = pd.DataFrame(records)
    if frame.duplicated(['fold', 'parents', 'month']).any():
        raise ValueError('duplicate portfolio metadata')
    summaries = []
    for k, g in frame.groupby('k'):
        legal = g.loc[g.legal_aggregate_month]
        summaries.append({'k': int(k), 'candidate_group_months': len(g), 'legal_group_months': len(legal),
                          'legal_nonempty': int((legal.shared_sampled_hours > 0).sum()),
                          'legal_all24clocks': int(legal.clock_cells.eq(24).sum()),
                          'median_shared_hours_legal': float(legal.shared_sampled_hours.median()) if len(legal) else None,
                          'min_shared_hours_legal': int(legal.shared_sampled_hours.min()) if len(legal) else None})
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('metadata inputs changed')
    OUT.mkdir(parents=True)
    frame.to_csv(OUT / 'portfolio_month_inventory.csv', index=False)
    result = {'assertion_failures': 0, 'inventory_rows': len(frame), 'by_size': summaries,
              'hourly_generation_values_read': False, 'CIF_scored': False, 'new_unique_source_hours': 0,
              'input_hashes': hashes,
              'output_hashes': {'portfolio_month_inventory.csv': sha(OUT / 'portfolio_month_inventory.csv')},
              'scope': 'Metadata feasibility only; parent-overlapping portfolios are not independent data or verified geographic unions.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2))


if __name__ == '__main__':
    main()
