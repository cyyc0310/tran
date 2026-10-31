"""Monthly-only preparation; no new hourly-risk fit, hourly labels, or scoring."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extended_monthly_information import ExtendedMonthlyInformation
from monthly_history_features import MonthlyInformation
from monthly_risk_features import feature_matrices
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import verify
from v184_coarse_history_value import ROOT, BASE, SOURCE, CONTEXT

OUT = ROOT / 'results/v186-monthly-risk-feature-preflight-r2'


def independent_volatility(data, prefix):
    """Separate pairwise-variance and scalar range/zero implementation."""
    rows = [data.shares[int(p)] for p in prefix[-12:]]
    count = len(rows)
    mean = [sum(row[f] for row in rows) / count for f in range(8)]
    std = [(sum((a[f] - b[f])**2 for a in rows for b in rows) / (2 * count**2))**.5 for f in range(8)]
    return np.asarray(mean + std + [min(row[f] for row in rows) for f in range(8)]
                      + [max(row[f] for row in rows) for f in range(8)]
                      + [sum(row[f] == 0 for row in rows) / count for f in range(8)] + [np.log1p(count)])


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    folder = ROOT / 'results/v185-coarse-representation-audit-r1'
    summary = verify(folder)
    hashes = dict(summary['input_hashes'])
    hashes |= {str((folder / p).relative_to(ROOT)): h for p, h in summary['output_hashes'].items()}
    paths = (folder / 'summary.json', Path(__file__), ROOT / 'scripts/monthly_risk_features.py',
             ROOT / 'docs/V186_MONTHLY_HISTORY_HOURLY_RISK_DESIGN.md')
    hashes |= {str(p.relative_to(ROOT)): sha(p) for p in paths}
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('upstream or preflight input changed')
    short = MonthlyInformation(ROOT)
    long = ExtendedMonthlyInformation(ROOT, ROOT / 'results/v183-coarse-history-export-r1/monthly_history.csv')
    if hasattr(short, 'raw') or hasattr(long, 'raw'):
        raise ValueError('unexpected hourly interface')
    inventory = pd.read_csv(CONTEXT / 'training_inventory.csv')
    rows, compact = [], {name: [] for name in ('base_state0', 'short_extra23', 'long_extra23', 'long_extra41')}
    maximum, guards = 0., 0
    for fold in range(6):
        proper = set(json.loads((BASE / f'fold_{fold}_asof_30d/summary.json').read_text())['source_regions'])
        source_ids = inventory.loc[inventory.fold.eq(fold) & inventory.budget.eq('short'), 'profile_id'].to_numpy(int)
        frame = pd.read_csv(SOURCE / f'fold_{fold}/world_probabilities.csv')
        query_ids = frame.loc[frame.role.eq('test'), 'profile_id'].drop_duplicates().to_numpy(int)
        for role, ids in (('source', source_ids), ('test', query_ids)):
            for q in ids:
                q = int(q)
                region = short.index.region.iloc[q]
                if ((region in proper) != (role == 'source') or not short.index.month.iloc[q].startswith('2024')):
                    raise ValueError('source/test role or year mismatch')
                matrices, contexts, extra = feature_matrices(short, long, q)
                for data, c in zip((short, long), contexts):
                    if (q in c['prefix'] or not (data.release.iloc[c['prefix']] <= data.dates.iloc[q]).all()
                            or any(data.index.region.iloc[p] != region for p in c['prefix'])):
                        raise ValueError('monthly risk prefix leakage')
                maximum = max(maximum, float(np.max(abs(extra - independent_volatility(long, contexts[1]['prefix'])))))
                rows.append({'fold': fold, 'role': role, 'profile_id': q, 'region': region, 'month': short.index.month.iloc[q],
                             'short_prefix': json.dumps(contexts[0]['prefix']), 'long_prefix': json.dumps(contexts[1]['prefix']),
                             **{name + '_sha256': hashlib.sha256(X.tobytes()).hexdigest() for name, X in matrices.items()}})
                for name, value in zip(compact, (matrices['direct20'][0], contexts[0]['history'][13:], contexts[1]['history'][13:], extra)):
                    compact[name].append(value)
        for q in query_ids[[0, -1]]:
            q = int(q)
            before = feature_matrices(short, long, q)[0]
            for data in (short, long):
                ids = data.groups[data.index.region.iloc[q]]
                ids = ids[(data.release.iloc[ids] > data.dates.iloc[q]).to_numpy()]
                saved = data.M[ids].copy(), data.shares[ids].copy()
                try:
                    changed = np.random.default_rng(q + 186).uniform(1, 1000, size=saved[0].shape)
                    data.M[ids] = changed; data.shares[ids] = changed / changed.sum(axis=1, keepdims=True)
                    after = feature_matrices(short, long, q)[0]
                finally:
                    data.M[ids], data.shares[ids] = saved
                if any(not np.array_equal(before[name], after[name]) for name in before):
                    raise ValueError('future monthly values changed risk features')
                guards += 1
        print(f'V186 feature preflight fold {fold} passed', flush=True)
    records = pd.DataFrame(rows)
    if (len(records) != 1205 or records.role.eq('source').sum() != 964 or records.role.eq('test').sum() != 241
            or records.duplicated(['fold', 'role', 'profile_id']).any() or maximum > 1e-12 or guards != 24):
        raise ValueError('preflight inventory or independent feature check')
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('preflight input changed')
    OUT.mkdir()
    records.to_csv(OUT / 'contexts.csv', index=False)
    np.savez_compressed(OUT / 'monthly_features.npz', **{k: np.asarray(v) for k, v in compact.items()})
    result = {'assertion_failures': 0, 'source_contexts': 964, 'test_contexts': 241, 'base_state_rows_checked': 1205 * 48,
              'independent_feature_rows': 1205, 'max_independent_volatility_error': maximum,
              'future_container_mutations': guards, 'future_feature_matrix_checks': guards * 4,
              'hourly_generation_loaded': False, 'new_hourly_risk_models_fitted': False, 'outcomes_scored': False,
              'frozen_monthly_context_includes_harmonic_Ridge_fitting': True,
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in OUT.iterdir()}}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
