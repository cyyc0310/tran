"""Count exact coarse/shape Cartesian supports without reading hourly generation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from monthly_history_features import MonthlyInformation
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import verify
from v179_increment_support_diagnostic import project_simplex

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
COARSE = ROOT / 'results/v178-joint-coarse-world-r1'
OUT = ROOT / 'results/v181-coarse-shape-inventory-r1'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    hashes = {}
    for archive in (BASE, COARSE):
        summary = verify(archive)
        hashes |= summary['input_hashes']
        for name in ('summary.json', *summary['output_hashes']):
            p = archive / name
            hashes[str(p.relative_to(ROOT))] = sha(p)
    paths = [Path(__file__), ROOT / 'docs/V181_COARSE_SHAPE_COUPLING_DESIGN.md',
             ROOT / 'scripts/v179_increment_support_diagnostic.py']
    hashes |= {str(p.relative_to(ROOT)): sha(p) for p in paths}
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('inventory input changed')
    monthly = MonthlyInformation(ROOT)
    with np.load(ROOT / 'results/v153-calendar-times-r1/timestamps.npz', allow_pickle=False) as packed:
        seconds = packed['unix_seconds']
    samples = pd.read_csv(BASE / 'source_sampling.csv')
    samples = samples.loc[samples.budget.eq('asof_30d')].set_index('region')
    records = []
    for fold in range(6):
        proper = sorted(json.loads((BASE / f'fold_{fold}_asof_30d/summary.json').read_text())['source_regions'])
        ids, sizes = [], np.zeros(24, int)
        for region in proper:
            raw = np.asarray(json.loads(samples.loc[region, 'sampled_raw_hour_indices']), int)
            if len(raw) > 4096 or len(np.unique(raw)) != len(raw):
                raise ValueError('source sample budget/identity')
            profiles = np.searchsorted(monthly.index.stop.to_numpy(), raw, side='right')
            if not monthly.index.region.iloc[profiles].eq(region).all() or not monthly.index.month.iloc[profiles].str.startswith('2024').all():
                raise ValueError('source sample role/year')
            for p in np.unique(profiles):
                ids.append(int(p))
                times = seconds[raw[profiles == p]]
                clock = (times // 3600) % 24
                for h in range(24):
                    n = int((clock == h).sum())
                    sizes[h] += n if n else len(times)
        ids = np.asarray(sorted(ids), int)
        inputs = np.asarray([monthly.local(int(p), 'asof_30d') for p in ids], int)
        delta = monthly.shares[ids] - monthly.shares[inputs]
        delta[abs(delta) <= 64 * np.finfo(float).eps] = 0
        frame = pd.read_csv(COARSE / f'fold_{fold}/world_probabilities.csv')
        frame = frame.loc[frame.method.eq('forest_history_scalar8')]
        for rec in frame.itertuples(index=False):
            q = int(rec.profile_id)
            if json.loads(rec.source_profiles) != ids.tolist() or monthly.index.region.iloc[q] in proper:
                raise ValueError('coarse source bank/query differs')
            values = project_simplex(monthly.shares[monthly.local(q, 'asof_30d')] + delta)
            n_unique = len(np.unique(values, axis=0))
            records.append({'fold': fold, 'profile_id': q, 'role': rec.role, 'region': rec.region,
                            'coarse_worlds': len(ids), 'unique_coarse_values': n_unique,
                            'shape_clock_atoms': int(sizes.sum()), 'max_shape_clock_atoms': int(max(sizes)),
                            'unmerged_cartesian_atoms': int(len(ids) * sizes.sum()),
                            'merged_cartesian_atoms': int(n_unique * sizes.sum()),
                            'max_merged_clock_atoms': int(n_unique * max(sizes))})
    frame = pd.DataFrame(records)
    if len(frame) != 482 or frame.duplicated(['fold', 'profile_id']).any():
        raise ValueError('complete query inventory')
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('inventory input changed during run')
    OUT.mkdir(parents=True)
    frame.to_csv(OUT / 'inventory.csv', index=False)
    result = {'assertion_failures': 0, 'query_months': len(frame),
              'unique_coarse_values_min': int(frame.unique_coarse_values.min()),
              'unique_coarse_values_max': int(frame.unique_coarse_values.max()),
              'cartesian_atoms_per_product_method': int(frame.merged_cartesian_atoms.sum()),
              'max_query_atoms': int(frame.merged_cartesian_atoms.max()),
              'max_clock_atoms': int(frame.max_merged_clock_atoms.max()),
              'hourly_generation_read': False, 'hourly_outcomes_scored': False,
              'input_hashes': hashes, 'output_hashes': {'inventory.csv': sha(OUT / 'inventory.csv')},
              'scope': 'Monthly inputs and original sampled-hour timestamp counts only. Atom counts are exact pre-CIF compression, not a measured memory bound or performance result.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
