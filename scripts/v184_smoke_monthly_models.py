"""Limited fold-zero monthly model smoke; no predictive outcomes are scored."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from extended_monthly_information import ExtendedMonthlyInformation
from monthly_history_features import MonthlyInformation
from monthly_history_value_models import MonthlyValueModels, METHODS
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import verify

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v184-monthly-model-smoke-r2'
CONTEXT = ROOT / 'results/v184-monthly-context-preflight-r1'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    prepared = verify(CONTEXT)
    paths = [Path(__file__), ROOT / 'scripts/monthly_history_value_models.py',
             ROOT / 'scripts/extended_monthly_information.py', CONTEXT / 'summary.json', CONTEXT / 'training_inventory.csv']
    hashes = prepared['input_hashes'] | {str(p.relative_to(ROOT)): sha(p) for p in paths}
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('smoke data/code changed')
    short = MonthlyInformation(ROOT)
    long = ExtendedMonthlyInformation(ROOT, ROOT / 'results/v183-coarse-history-export-r1/monthly_history.csv')
    inventory = pd.read_csv(CONTEXT / 'training_inventory.csv')
    meta = json.loads((ROOT / 'results/v159-fuel-activity-source-pilot-r1/fold_0_asof_30d/summary.json').read_text())
    frame = pd.read_csv(ROOT / 'results/v178-joint-coarse-world-r1/fold_0/world_probabilities.csv')
    queries = frame.loc[frame.role.eq('test')].profile_id.drop_duplicates().iloc[[0, -1]].tolist()
    models, checks = {}, 0
    with threadpool_limits(limits=2):
        for budget in ('short', 'long_fixed', 'long_all'):
            ids = inventory.loc[inventory.fold.eq(0) & inventory.budget.eq(budget), 'profile_id'].to_numpy(int)
            models[budget] = MonthlyValueModels(short if budget == 'short' else long, meta['source_regions'], ids)
            print(f'fitted smoke {budget}: {len(ids)} labels', flush=True)
        for q in queries:
            predictions = {b: m.build(q) for b, m in models.items()}
            for method in ('pooled_linear_residual', 'gaussian_increment', 'forest_base_scalar8', 'forest_base_delta8'):
                if any(not np.array_equal(a, b) for a, b in zip(predictions['short'][method], predictions['long_fixed'][method])):
                    raise ValueError(f'base negative control differs: {method}')
                checks += 1
            for budget, model in models.items():
                data = model.data
                ids = data.groups[data.index.region.iloc[q]]
                ids = ids[(data.release.iloc[ids] > data.dates.iloc[q]).to_numpy()]
                old = data.M[ids].copy(), data.shares[ids].copy()
                changed = np.random.default_rng(q).uniform(1, 1000, size=old[0].shape)
                try:
                    data.M[ids] = changed
                    data.shares[ids] = changed / changed.sum(axis=1, keepdims=True)
                    after = model.build(q)
                finally:
                    data.M[ids], data.shares[ids] = old
                for method in METHODS:
                    if any(not np.array_equal(a, b) for a, b in zip(predictions[budget][method], after[method])):
                        raise ValueError(f'future monthly leakage: {budget}/{method}')
            print(f'checked smoke query {q}', flush=True)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('smoke inputs changed during run')
    OUT.mkdir(parents=True)
    result = {'assertion_failures': 0, 'fold': 0, 'models_budget_groups_fitted': 3, 'query_profiles': queries,
              'methods_per_budget': len(METHODS), 'base_negative_control_laws': checks,
              'future_monthly_guard_laws': len(queries) * 3 * len(METHODS), 'coarse_or_hourly_outcomes_scored': False,
              'input_hashes': hashes, 'output_hashes': {},
              'scope': 'Limited fold0 smoke, strict previous-year seasonal rule. No performance scores or full model reconstruction; r1 is retired before any formal scoring.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
