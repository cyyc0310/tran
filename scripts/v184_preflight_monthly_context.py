"""Validate extended input contexts and coarse label pools without fitting models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from extended_monthly_information import ExtendedMonthlyInformation
from monthly_history_features import MonthlyInformation, monthly_context
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import verify

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v178-joint-coarse-world-r1'
HISTORY = ROOT / 'results/v183-coarse-history-export-r1'
OUT = ROOT / 'results/v184-monthly-context-preflight-r1'


def feature_mutation(data, q):
    before = monthly_context(data, q)
    ids = data.groups[data.index.region.iloc[q]]
    ids = ids[(data.release.iloc[ids] > data.dates.iloc[q]).to_numpy()]
    old = data.M[ids].copy(), data.shares[ids].copy()
    changed = np.random.default_rng(q + 184).uniform(1, 1000, size=old[0].shape)
    try:
        data.M[ids] = changed
        data.shares[ids] = changed / changed.sum(axis=1, keepdims=True)
        after = monthly_context(data, q)
    finally:
        data.M[ids], data.shares[ids] = old
    for key in ('base', 'history', 'prefix'):
        if not np.array_equal(before[key], after[key]):
            raise ValueError('future monthly values changed a context')


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    hashes = {}
    for archive in (SOURCE, HISTORY, ROOT / 'results/v183-coarse-history-export-audit-r1'):
        summary = verify(archive)
        for p, h in summary['input_hashes'].items():
            if p in hashes and hashes[p] != h:
                raise ValueError('preflight dependency conflict')
            hashes[p] = h
        for name in ('summary.json', *summary['output_hashes']):
            p = archive / name
            hashes[str(p.relative_to(ROOT))] = sha(p)
    for p in (Path(__file__), ROOT / 'scripts/extended_monthly_information.py', ROOT / 'docs/V184_COARSE_HISTORY_VALUE_DESIGN.md'):
        hashes[str(p.relative_to(ROOT))] = sha(p)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('preflight input changed')
    short = MonthlyInformation(ROOT)
    long = ExtendedMonthlyInformation(ROOT, HISTORY / 'monthly_history.csv')
    if hasattr(long, 'raw') or any(k in long.index for k in ('offset', 'stop')):
        raise ValueError('extended monthly container exposes hourly fields')
    contexts = {int(q): monthly_context(long, int(q)) for q in long.index.profile_id
                if long.index.month.iloc[q][:4] in ('2022', '2023', '2024')}
    training, comparison_count, mutations = [], 0, 0
    for fold in range(6):
        meta = json.loads((ROOT / f'results/v159-fuel-activity-source-pilot-r1/fold_{fold}_asof_30d/summary.json').read_text())
        proper = set(meta['source_regions'])
        frame = pd.read_csv(SOURCE / f'fold_{fold}/world_probabilities.csv')
        old_ids = json.loads(frame.loc[frame.method.eq('allmonth_hellinger')].iloc[0].source_profiles)
        all_ids = [q for q, context in contexts.items() if context is not None and long.index.region.iloc[q] in proper]
        for q in old_ids:
            a, b = monthly_context(short, q), contexts[q]
            if (not np.array_equal(a['base'], b['base']) or a['local'] != b['local']
                    or not np.array_equal(short.shares[q] - short.shares[a['local']], long.shares[q] - long.shares[b['local']])):
                raise ValueError('short/long_fixed base or delta negative control differs')
            comparison_count += 1
        for budget, ids in (('short', old_ids), ('long_fixed', old_ids), ('long_all', all_ids)):
            for q in ids:
                data = short if budget == 'short' else long
                context = monthly_context(data, q)
                if (context is None or data.index.region.iloc[q] not in proper
                        or not (data.release.iloc[context['prefix']] <= data.dates.iloc[q]).all()
                        or q in context['prefix']):
                    raise ValueError('training source/prefix role violation')
                training.append({'fold': fold, 'budget': budget, 'profile_id': q, 'region': data.index.region.iloc[q],
                                 'month': data.index.month.iloc[q], 'local': context['local'], 'prefix_months': len(context['prefix'])})
        queries = frame.loc[frame.role.eq('test')].profile_id.drop_duplicates().tolist()
        for q in queries:
            if not np.array_equal(monthly_context(short, q)['base'], monthly_context(long, q)['base']):
                raise ValueError('query base negative control differs')
        for q in (queries[0], queries[-1]):
            feature_mutation(short, q)
            feature_mutation(long, q)
            mutations += 2
        print(f'monthly context preflight fold {fold}: fixed={len(old_ids)}, all={len(all_ids)}', flush=True)
    if mutations != 24:
        raise ValueError('feature mutation denominator')
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('preflight inputs changed during run')
    frame = pd.DataFrame(training)
    OUT.mkdir(parents=True)
    frame.to_csv(OUT / 'training_inventory.csv', index=False)
    result = {'assertion_failures': 0, 'canonical_profile_ids_preserved': long.canonical_count,
              'historical_profiles_appended': len(long.historical_ids), 'training_base_delta_identity_checks': comparison_count,
              'query_base_identity_checks': 241, 'feature_future_mutation_cases': mutations,
              'models_fitted': False, 'coarse_scores_computed': False, 'hourly_values_used': False,
              'input_hashes': hashes, 'output_hashes': {'training_inventory.csv': sha(OUT / 'training_inventory.csv')},
              'scope': 'Monthly container/source-label readiness and feature-only guards, not model/law preflight or performance evidence.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
