"""Refit every V171 forest, verify every saved probability, and rescore Brier."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from monthly_history_features import MonthlyInformation
from two_scale_zero import TwoScaleZero, METHODS, MONTH_METHODS, REFERENCES, NEW_METHODS, reference_probabilities
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import FUELS, verify
from v171_two_scale_zero import BASE, WORLD, COMPONENTS, METRICS, check_mutation

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v171-two-scale-zero-r1'
OUT = ROOT / 'results/v171-zero-audit-r1'


def independent_score(probability, observed):
    p, y = np.broadcast_arrays(np.asarray(probability, float), np.asarray(observed, float))
    # Separate the Brier terms instead of invoking the producer's score helper.
    brier = float(np.mean(p * p) - 2 * np.mean(p * y) + np.mean(y))
    return {'brier': brier, 'mean_forecast': float(p.mean()), 'observed_fraction': float(y.mean()),
            'calibration_bias': float(p.mean() - y.mean())}


def audit_fold(data, fold):
    frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
    monthly = MonthlyInformation(ROOT)
    predictor = TwoScaleZero(frozen, monthly)
    block = SOURCE / f'fold_{fold}'
    archived_queries = pd.read_csv(block / 'query_inputs.csv')
    pd.testing.assert_frame_equal(frozen.queries, archived_queries, check_exact=True)
    if predictor.provenance != json.loads((block / 'training_provenance.json').read_text()):
        raise ValueError('training provenance differs')
    contexts = {r['profile_id']: r for r in json.loads((block / 'contexts.json').read_text())}
    if len(contexts) != len(frozen.queries):
        raise ValueError('context denominator')
    with np.load(block / 'probabilities.npz', allow_pickle=False) as packed:
        arrays = {key: packed[key].copy() for key in packed.files}
    if not np.array_equal(arrays['profile_id'], frozen.queries.profile_id.to_numpy(int)):
        raise ValueError('probability query ordering')
    expected_shapes = {'hourly': (len(frozen.queries), len(METHODS), 48, 8),
                       'monthly': (len(frozen.queries), len(MONTH_METHODS), 8),
                       'components': (len(frozen.queries), len(COMPONENTS), 48, 8)}
    if set(arrays) != {'profile_id', *expected_shapes} or any(arrays[k].shape != v for k, v in expected_shapes.items()):
        raise ValueError('probability array inventory or overall dimensions')
    scores = {kind: pd.read_csv(block / f'{kind}_scores.csv').set_index(['profile_id', 'method', 'fuel'])
              for kind in ('hourly', 'monthly', 'components')}
    worlds = pd.read_csv(WORLD / f'fold_{fold}/world_probabilities.csv')
    maximum_probability = maximum_score = 0.
    checked = {kind: 0 for kind in scores}
    for q in frozen.queries.profile_id.iloc[[0, -1]]:
        check_mutation(predictor, data, monthly, int(q))
    for position, rec in enumerate(frozen.queries.itertuples(index=False)):
        q = int(rec.profile_id)
        hourly, month, components, context = predictor.predict(q)
        ref_hour, ref_month = reference_probabilities(frozen, worlds, q)
        hourly, month = ref_hour | hourly, ref_month | month
        expected = {'hourly': np.stack([hourly[m] for m in METHODS]),
                    'monthly': np.stack([month[m] for m in MONTH_METHODS]),
                    'components': np.stack([components[m] for m in COMPONENTS])}
        for kind, new in expected.items():
            old = arrays[kind][position]
            if new.shape != old.shape or not np.isfinite(old).all() or (old < 0).any() or (old > 1).any():
                raise ValueError('saved probability dimensions or support')
            maximum_probability = max(maximum_probability, float(np.max(abs(new - old))))
        stored_context = contexts[q]
        if (stored_context['local'] != context['local'] or stored_context['prefix'] != context['prefix']
                or not np.array_equal(stored_context['base'], context['base'])
                or not np.array_equal(stored_context['history'], context['history'])):
            raise ValueError('legal context reconstruction')
        if rec.role != 'test':
            continue
        row = data.index.iloc[q]
        states = data.states[int(row.offset):int(row.stop)]
        zero = data.raw[int(row.offset):int(row.stop)] == 0
        for kind, names in [('hourly', METHODS), ('monthly', MONTH_METHODS), ('components', COMPONENTS)]:
            for method_id, method in enumerate(names):
                for j, fuel in enumerate(FUELS):
                    if kind == 'components' and data.M[q, j] == 0:
                        continue
                    old = scores[kind].loc[(q, method, fuel)]
                    if (old.region != rec.region or old.month != rec.month or old.fold != fold or old.hours != len(states)
                            or old.input_fuel_positive != (data.M[context['local'], j] > 0)
                            or old.current_fuel_positive != (data.M[q, j] > 0)):
                        raise ValueError('score identity')
                    if kind == 'monthly':
                        p, y = arrays[kind][position, method_id, j], data.M[q, j] == 0
                        if old.is_world_proxy != (method in REFERENCES):
                            raise ValueError('world proxy label')
                    else:
                        p, y = arrays[kind][position, method_id, states, j], zero[:, j]
                    new = independent_score(p, y)
                    maximum_score = max(maximum_score, max(abs(new[key] - old[key]) for key in METRICS))
                    checked[kind] += 1
    if any(checked[k] != len(scores[k]) for k in scores) or max(maximum_probability, maximum_score) > 1e-10:
        raise ValueError('score/probability reconstruction')
    print(f'audited zero fold {fold}: {checked}', flush=True)
    return {'fold': fold, **{k + '_rows': v for k, v in checked.items()},
            'refit_forests': len(predictor.provenance), 'verified_query_forecasts': len(frozen.queries),
            'max_probability_error': maximum_probability, 'max_score_error': maximum_score}


def aggregate(rows):
    records = []
    for cohort, frame in [('all29', rows), ('input_fuel_positive', rows.loc[rows.input_fuel_positive])]:
        for key, group in frame.groupby(['region', 'method', 'fuel']):
            weights = group.hours.to_numpy(float)
            weights /= weights.sum()
            records.append({'cohort': cohort, **dict(zip(['region', 'method', 'fuel'], key)),
                            **{m: float(weights @ group[m].to_numpy()) for m in METRICS}})
    regional = pd.DataFrame(records)
    macro = pd.DataFrame([{'cohort': key[0], 'method': key[1], 'fuel': key[2], 'regions': len(g),
                          **{m: float(g[m].mean()) for m in METRICS}}
                         for key, g in regional.groupby(['cohort', 'method', 'fuel'])])
    return regional, macro


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = verify(SOURCE)
    hashes = summary['input_hashes'] | {str((SOURCE / 'summary.json').relative_to(ROOT)): sha(SOURCE / 'summary.json'),
                                      str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('input changed before audit')
    rebuilt = {}
    for kind, count in [('hourly', 15424), ('monthly', 9640), ('components', 2272)]:
        rows = pd.read_csv(SOURCE / f'{kind}_scores.csv')
        blocks = pd.concat([pd.read_csv(SOURCE / f'fold_{f}/{kind}_scores.csv') for f in range(6)], ignore_index=True)
        keys = ['profile_id', 'method', 'fuel']
        if len(rows) != count or rows.duplicated(keys).any() or not np.isfinite(rows[list(METRICS)]).all().all():
            raise ValueError('score denominator/finiteness')
        # Producer concatenates parsed block CSVs, then serializes that table.
        # Verify those exact bytes; another parse can move tiny values by one ULP.
        if (SOURCE / f'{kind}_scores.csv').read_text() != blocks.to_csv(index=False):
            raise ValueError('root CSV is not the exact serialization of parsed blocks')
        regional, macro = aggregate(rows)
        for label, new, identity in [('region_metrics', regional, ['cohort', 'region', 'method', 'fuel']),
                                     ('macro_metrics', macro, ['cohort', 'method', 'fuel'])]:
            old = pd.read_csv(SOURCE / f'{kind}_{label}.csv')
            pair = new.merge(old, on=identity, validate='one_to_one', suffixes=('_a', '_b'))
            metrics = list(METRICS) + (['regions'] if label == 'macro_metrics' else [])
            if (len(pair) != len(new) or len(pair) != len(old) or not np.isfinite(old[metrics]).all().all()
                    or any(max(abs(pair[m + '_a'] - pair[m + '_b'])) > 1e-10 for m in metrics)):
                raise ValueError('aggregate reconstruction')
        rebuilt[kind] = regional
    data = HourlyFuelData(ROOT)
    with threadpool_limits(limits=2):
        checks = [audit_fold(data, fold) for fold in range(6)]
    rng = np.random.default_rng(171)
    comparisons = []
    for kind, candidates, references in [
        ('hourly', NEW_METHODS, (*REFERENCES, 'direct20', 'direct43', 'month_persistence_conditional')),
        ('monthly', ('month_persistence', 'month_base', 'month_history'), (*REFERENCES, 'month_persistence', 'month_base')),
        ('components', ('conditional43',), ('clock',))]:
        regional = rebuilt[kind]
        for (cohort, fuel), group in regional.groupby(['cohort', 'fuel']):
            for method in candidates:
                a = group.loc[group.method.eq(method)].sort_values('region')
                for reference in references:
                    if method == reference:
                        continue
                    b = group.loc[group.method.eq(reference)].sort_values('region')
                    if not np.array_equal(a.region, b.region) or not len(a):
                        raise ValueError('paired source inventory')
                    delta = a.brier.to_numpy() - b.brier.to_numpy()
                    boot = delta[rng.integers(len(delta), size=(3000, len(delta)))].mean(axis=1)
                    comparisons.append({'kind': kind, 'cohort': cohort, 'fuel': fuel, 'method': method, 'reference': reference,
                                        'regions': len(a), 'brier': float(a.brier.mean()), 'reference_brier': float(b.brier.mean()),
                                        'paired_brier_difference': float(delta.mean()),
                                        'region_bootstrap_lo': float(np.quantile(boot, .025)),
                                        'region_bootstrap_hi': float(np.quantile(boot, .975)),
                                        'regions_better_brier': int((delta < -1e-12).sum())})
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('input changed during audit')
    verify(SOURCE)
    OUT.mkdir(parents=True)
    pd.DataFrame(checks).to_csv(OUT / 'reconstruction.csv', index=False)
    pd.DataFrame(comparisons).to_csv(OUT / 'comparisons.csv', index=False)
    result = {'assertion_failures': 0, 'refit_forests': sum(c['refit_forests'] for c in checks),
              'verified_query_forecasts': sum(c['verified_query_forecasts'] for c in checks),
              'rebuilt_hourly_rows': sum(c['hourly_rows'] for c in checks),
              'rebuilt_monthly_rows': sum(c['monthly_rows'] for c in checks),
              'rebuilt_component_rows': sum(c['components_rows'] for c in checks),
              'max_probability_error': max(c['max_probability_error'] for c in checks),
              'max_score_error': max(c['max_score_error'] for c in checks), 'comparisons': len(comparisons),
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in OUT.iterdir()},
              'scope': 'All models refit and probabilities verified with frozen model helper; all Brier scores independently expanded. Bootstrap is descriptive repeated source development; no CIF gate or novelty claim.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_hashes', 'output_hashes')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
