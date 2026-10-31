"""Audit monthly-history legality, complete coarse predictions and paired value."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from monthly_history_features import MonthlyInformation, monthly_context
from v151_hourly_law import EF, sha

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v161-monthly-history-value-r1'
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
OUT = ROOT / 'results/v161-monthly-history-audit-r1'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = json.loads((SOURCE / 'summary.json').read_text())
    if summary['assertion_failures']:
        raise ValueError('upstream failure')
    for name, digest in summary['input_hashes'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('input changed')
    for name, digest in summary['output_hashes'].items():
        if sha(SOURCE / name) != digest:
            raise ValueError('output changed')
    data = MonthlyInformation(ROOT)
    contexts = pd.read_csv(SOURCE / 'query_contexts.csv')
    rows = pd.read_csv(SOURCE / 'predictions.csv')
    roles = pd.read_csv(BASE / 'region_roles.csv')
    expected_queries = set()
    for fold in range(6):
        queries = pd.read_csv(BASE / f'fold_{fold}_asof_30d/query_inputs.csv')
        expected_queries |= set(queries.loc[queries.role.eq('test'), 'profile_id'])
    if set(contexts.profile_id) != expected_queries or len(contexts) != len(expected_queries):
        raise ValueError('coarse query denominator')
    for row in contexts.itertuples(index=False):
        prefix = json.loads(row.history_profile_ids)
        q = int(row.profile_id)
        ids = data.groups[row.region]
        eligible = set(ids[(data.release.iloc[ids] <= data.dates.iloc[q]).to_numpy()])
        if set(prefix) != eligible or row.input_profile_id != data.local(q, 'asof_30d'):
            raise ValueError('history prefix/vintage mismatch')
        actual = monthly_context(data, q)
        if not np.array_equal(actual['base'], np.asarray(json.loads(row.base_features))):
            raise ValueError('base feature reproduction')
        if not np.array_equal(actual['history'], np.asarray(json.loads(row.history_features))):
            raise ValueError('history feature reproduction')
    backtests = pd.read_csv(SOURCE / 'coarse_backtests.csv')
    for row in backtests.itertuples(index=False):
        q, observed, past = row.query_profile_id, row.observed_profile_id, row.prediction_input_profile_id
        if len({data.index.region.iloc[i] for i in (q, observed, past)}) != 1:
            raise ValueError('cross-region local backtest')
        if data.release.iloc[observed] > data.dates.iloc[q] or data.release.iloc[past] > data.dates.iloc[observed]:
            raise ValueError('backtest time leakage')
        if abs(row.prediction - data.shares[past] @ EF) > 1e-9 or abs(row.actual - data.shares[observed] @ EF) > 1e-9:
            raise ValueError('backtest values differ')
    for fit in json.loads((SOURCE / 'fit_models.json').read_text()):
        proper = set(roles.loc[roles.fold.eq(fit['fold']) & roles.role.eq('fit'), 'region'])
        expected = set(contexts.loc[contexts.region.isin(proper), 'profile_id'])
        if set(fit['train_profiles']) != expected or set(fit['fit_regions']) != proper:
            raise ValueError('coarse fit role leakage')
        if fit['dimensions'] != (13 if fit['feature_budget'] == 'base' else 36):
            raise ValueError('coarse feature dimension')
    if rows.duplicated(['profile_id', 'method']).any() or len(rows) != len(expected_queries) * 9:
        raise ValueError('prediction completeness')
    for method, group in rows.groupby('method'):
        if set(group.profile_id) != expected_queries or group.region.nunique() != 29:
            raise ValueError('method query completeness')
    for row in rows.itertuples(index=False):
        if abs(row.actual_anchor - data.shares[row.profile_id] @ EF) > 1e-9:
            raise ValueError('anchor target mismatch')
        if abs(row.error - (row.prediction - row.actual_anchor)) > 1e-9:
            raise ValueError('saved error mismatch')
    macro = pd.read_csv(SOURCE / 'macro_metrics.csv').set_index('method')
    region = rows.groupby(['region', 'method']).error.agg(
        mae=lambda e: e.abs().mean(), rmse=lambda e: np.sqrt(np.mean(e**2)), bias='mean').reset_index()
    for method, group in region.groupby('method'):
        for metric in ('mae', 'rmse', 'bias'):
            if abs(group[metric].mean() - macro.loc[method, metric]) > 1e-9:
                raise ValueError('coarse macro mismatch')
    fold = rows.groupby(['fold', 'region', 'method']).error.apply(lambda e: e.abs().mean()).rename('mae').reset_index()
    fold = fold.groupby(['fold', 'method']).mae.mean().reset_index()
    comparisons = []
    paired = []
    for candidate in ('forest_history', 'ridge_history'):
        references = list(macro.index[~macro.index.str.endswith('_history')])
        for reference in references:
            a, b = macro.loc[candidate], macro.loc[reference]
            comparisons.append({'candidate': candidate, 'reference': reference,
                                'mae_reduction_percent': 100 * (1 - a.mae / b.mae),
                                'rmse_reduction_percent': 100 * (1 - a.rmse / b.rmse),
                                'worst_mae_change_percent': 100 * (a.worst_mae / b.worst_mae - 1)})
        best = macro.loc[references].sort_values(['mae']).index[0]
        for reference in dict.fromkeys([best, candidate.replace('_history', '_base'), candidate.replace('_history', '_noise')]):
            a = rows.loc[rows.method.eq(candidate)]
            b = rows.loc[rows.method.eq(reference)]
            if set(a.profile_id) != set(b.profile_id):
                raise ValueError('paired completeness')
            joined = a.merge(b, on=['profile_id', 'region', 'month'], suffixes=('_a', '_b'), validate='one_to_one')
            joined['difference'] = joined.error_a.abs() - joined.error_b.abs()
            months = sorted(joined.month.unique())
            matrix = np.zeros((29, len(months)))
            counts = np.zeros_like(matrix)
            for r, region_id in enumerate(sorted(joined.region.unique())):
                group = joined.loc[joined.region.eq(region_id)].set_index('month')
                for m, month in enumerate(months):
                    if month in group.index:
                        matrix[r, m] = group.loc[month, 'difference']
                        counts[r, m] = 1
            weights = np.random.default_rng(20261028).dirichlet(np.ones(len(months)), size=1000)
            distribution = ((matrix @ weights.T) / (counts @ weights.T)).mean(axis=0)
            source_delta = joined.groupby('region').difference.mean()
            fold_delta = joined.groupby(['fold_a', 'region']).difference.mean().groupby('fold_a').mean()
            paired.append({'candidate': candidate, 'reference': reference,
                           'candidate_minus_reference_mae': float(source_delta.mean()),
                           'descriptive_95_lower': float(np.quantile(distribution, .025)),
                           'descriptive_95_upper': float(np.quantile(distribution, .975)),
                           'improved_regions': int((source_delta < 0).sum()),
                           'improved_folds': int((fold_delta < 0).sum())})
    OUT.mkdir(parents=True)
    for name, frame in [('comparisons', pd.DataFrame(comparisons)), ('paired_month_mae', pd.DataFrame(paired)),
                        ('fold_mae', fold)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    result = {'assertion_failures': 0, 'verified_monthly_queries': len(contexts), 'verified_prediction_rows': len(rows),
              'verified_backtest_uses': len(backtests), 'source_summary_sha256': sha(SOURCE / 'summary.json'),
              'script_sha256': sha(Path(__file__)),
              'output_hashes': {p.name: sha(p) for p in sorted(OUT.iterdir())},
              'scope': 'Coarse anchor source-development value; no hourly prediction/coverage claim.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(pd.DataFrame(paired).to_string(index=False))


if __name__ == '__main__':
    main()
