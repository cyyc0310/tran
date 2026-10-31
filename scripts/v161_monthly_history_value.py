"""Coarse-only source-fold value of legally released monthly trajectories."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from monthly_history_features import MonthlyInformation, monthly_context
from v151_hourly_law import EF, sha
from v154_hourly_strong_baselines import balanced

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
OUT = ROOT / 'results/v161-monthly-history-value-r1'
PROTOCOL = ROOT / 'docs/V161_MONTHLY_HISTORY_VALUE_PROTOCOL.md'
METHODS = ['persistence', 'trend', 'harmonic', 'ridge_base', 'ridge_history', 'ridge_noise',
           'forest_base', 'forest_history', 'forest_noise']


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    dependencies = [Path(__file__), PROTOCOL, ROOT / 'scripts/monthly_history_features.py',
                    ROOT / 'scripts/v151_hourly_law.py', ROOT / 'scripts/v139_physical_reserve_calibration.py',
                    ROOT / 'scripts/v154_hourly_strong_baselines.py', BASE / 'summary.json', BASE / 'region_roles.csv',
                    ROOT / 'results/v144-canonical-profile-cache-r1/profiles.npz',
                    ROOT / 'results/v144-canonical-profile-cache-r1/profile_index.csv']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    upstream = json.loads((BASE / 'summary.json').read_text())
    if upstream['assertion_failures'] or upstream['completed_blocks'] != 12:
        raise ValueError('source role archive incomplete')
    for name, digest in upstream['output_hashes'].items():
        if sha(BASE / name) != digest:
            raise ValueError('source archive changed')
    data = MonthlyInformation(ROOT)
    roles = pd.read_csv(BASE / 'region_roles.csv')
    pool = set(roles.region)
    contexts, records, errors = {}, [], []
    for q, row in data.index.iterrows():
        if row.region not in pool or not row.month.startswith('2024'):
            continue
        context = monthly_context(data, int(q))
        if context is None:
            continue
        seed = int(hashlib.sha256(f'v161/noise/{row.region}/{row.month}'.encode()).hexdigest()[:16], 16)
        context['noise'] = np.r_[context['base'], np.random.default_rng(seed).normal(size=23)]
        contexts[int(q)] = context
        records.append({'profile_id': int(q), 'region': row.region, 'month': row.month,
                        'input_profile_id': context['local'], 'input_month': data.index.month.iloc[context['local']],
                        'history_profile_ids': json.dumps(context['prefix']),
                        'history_reports': len(context['prefix']), 'coarse_backtests': len(context['coarse_errors']),
                        'base_features': json.dumps(context['base'].tolist()),
                        'history_features': json.dumps(context['history'].tolist())})
        if any(data.release.iloc[p] > data.dates.iloc[q] for p in context['prefix']):
            raise ValueError('unreleased history feature')
        for e in context['coarse_errors']:
            observed = e['observed_profile_id']
            prediction_input = e['prediction_input_profile_id']
            if data.release.iloc[observed] > data.dates.iloc[q] or data.release.iloc[prediction_input] > data.dates.iloc[observed]:
                raise ValueError('coarse backtest timing leakage')
            errors.append({'query_profile_id': int(q)} | e)
    # Mutate every unavailable monthly value; all predictors must stay unchanged.
    mutation_count = 0
    candidates = sorted(contexts, key=lambda q: len(contexts[q]['prefix']))
    positions = np.rint(np.linspace(0, len(candidates) - 1, 12)).astype(int)
    for q in [candidates[p] for p in positions]:
        ids = data.groups[data.index.region.iloc[q]]
        unavailable = ids[(data.release.iloc[ids] > data.dates.iloc[q]).to_numpy()]
        old_m, old_s = data.M[unavailable].copy(), data.shares[unavailable].copy()
        changed = np.random.default_rng(q).uniform(1, 1000, size=old_m.shape)
        data.M[unavailable] = changed
        data.shares[unavailable] = changed / changed.sum(axis=1, keepdims=True)
        altered = monthly_context(data, q)
        data.M[unavailable], data.shares[unavailable] = old_m, old_s
        original = contexts[q]
        for key in ('base', 'history'):
            if not np.array_equal(original[key], altered[key]):
                raise ValueError('unavailable monthly value affected features')
        for key in ('persistence', 'trend', 'harmonic'):
            if original[key] != altered[key]:
                raise ValueError('unavailable value affected local baseline')
        mutation_count += 1
    predictions, fits = [], []
    with threadpool_limits(limits=2):
        for fold in range(6):
            roster = roles.loc[roles.fold.eq(fold)]
            fit = set(roster.loc[roster.role.eq('fit'), 'region'])
            test = set(roster.loc[roster.role.eq('test'), 'region'])
            train_ids = [q for q in contexts if data.index.region.iloc[q] in fit]
            test_ids = [q for q in contexts if data.index.region.iloc[q] in test]
            expected = pd.read_csv(BASE / f'fold_{fold}_asof_30d/query_inputs.csv')
            expected = set(expected.loc[expected.role.eq('test'), 'profile_id'])
            if set(test_ids) != expected:
                raise ValueError('monthly test denominator changed')
            target = np.asarray([data.shares[q] @ EF - contexts[q]['anchor'] for q in train_ids])
            weights = balanced(np.asarray([data.index.region.iloc[q] for q in train_ids]))
            anchor = np.asarray([contexts[q]['anchor'] for q in test_ids])
            forecast = {name: np.asarray([contexts[q][name] for q in test_ids])
                        for name in ('persistence', 'trend', 'harmonic')}
            for budget in ('base', 'history', 'noise'):
                X = np.asarray([contexts[q][budget] for q in train_ids])
                query = np.asarray([contexts[q][budget] for q in test_ids])
                mu = np.average(X, axis=0, weights=weights)
                sd = np.sqrt(np.average((X - mu)**2, axis=0, weights=weights))
                sd[sd <= 1e-12] = 1
                ridge = Ridge(alpha=1).fit((X - mu) / sd, target, sample_weight=weights)
                forest = ExtraTreesRegressor(n_estimators=32, min_samples_leaf=4, max_features=1.,
                                             random_state=20261028, n_jobs=2).fit(X, target, sample_weight=weights)
                forecast['ridge_' + budget] = np.clip(anchor + ridge.predict((query - mu) / sd), 0, 1000)
                forecast['forest_' + budget] = np.clip(anchor + forest.predict(query), 0, 1000)
                fits.append({'fold': fold, 'feature_budget': budget, 'dimensions': X.shape[1],
                             'train_profiles': train_ids, 'fit_regions': sorted(fit),
                             'ridge_mean': mu.tolist(), 'ridge_scale': sd.tolist()})
            if set(forecast) != set(METHODS):
                raise ValueError('missing prediction method')
            for j, q in enumerate(test_ids):
                row = data.index.iloc[q]
                actual = float(data.shares[q] @ EF)
                for method in METHODS:
                    value = float(forecast[method][j])
                    predictions.append({'fold': fold, 'region': row.region, 'month': row.month,
                                        'profile_id': q, 'method': method, 'actual_anchor': actual,
                                        'prediction': value, 'error': value - actual,
                                        'history_reports': len(contexts[q]['prefix'])})
    rows = pd.DataFrame(predictions)
    if rows.duplicated(['region', 'month', 'method']).any() or len(rows) != len(contexts) * len(METHODS):
        raise ValueError('coarse prediction denominator')
    regional = []
    for (region, method), group in rows.groupby(['region', 'method']):
        regional.append({'region': region, 'method': method, 'months': len(group),
                         'mae': float(group.error.abs().mean()),
                         'rmse': float(np.sqrt(np.mean(group.error**2))), 'bias': float(group.error.mean())})
    region_frame = pd.DataFrame(regional)
    macro = region_frame.groupby('method').agg(mae=('mae', 'mean'), rmse=('rmse', 'mean'), bias=('bias', 'mean'),
                                               worst_mae=('mae', 'max'), regions=('region', 'size')).reset_index()
    if not macro.regions.eq(29).all() or not np.isfinite(macro[['mae', 'rmse', 'bias']]).all().all():
        raise ValueError('coarse macro denominator/finite check')
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('coarse experiment inputs changed')
    OUT.mkdir(parents=True)
    for name, frame in [('predictions', rows), ('region_metrics', region_frame), ('macro_metrics', macro),
                        ('query_contexts', pd.DataFrame(records)), ('coarse_backtests', pd.DataFrame(errors))]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    (OUT / 'fit_models.json').write_text(json.dumps(fits, indent=2, allow_nan=False) + '\n')
    result = {'status': 'COARSE_ONLY_MONTHLY_HISTORY_VALUE', 'assertion_failures': 0,
              'source_regions': 29, 'test_months': len(contexts), 'prediction_rows': len(rows),
              'future_month_mutation_invariance_cases': mutation_count, 'input_hashes': hashes,
              'output_hashes': {p.name: sha(p) for p in sorted(OUT.iterdir())},
              'limitations': ['Coarse monthly energy-weighted EF anchor, not hourly CIF/coverage.',
                              'Only cached 2024 short released prefixes, not complete real-world multiyear histories.',
                              '2024 shared-time source development; synthetic release vintages and nominal EF proxy.']}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(macro.to_string(index=False))


if __name__ == '__main__':
    main()
