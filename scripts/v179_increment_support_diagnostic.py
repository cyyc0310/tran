"""Post-hoc coarse-only support diagnostic with frozen V178 probabilities."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from coarse_distribution_lifting import CoarseLawForecast
from joint_coarse_worlds import JointCoarsePrediction
from monthly_history_features import MonthlyInformation
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import verify
from v178_joint_coarse_world import aggregate_coarse, COARSE_METRICS, FUEL_METRICS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v178-joint-coarse-world-r1'
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
OUT = ROOT / 'results/v179-increment-support-r1'
PROTOCOL = ROOT / 'docs/V179_INCREMENT_SUPPORT_DIAGNOSTIC_PROTOCOL.md'


def project_simplex(values):
    values = np.asarray(values, float)
    if values.ndim != 2 or values.shape[1] != 8 or not np.isfinite(values).all():
        raise ValueError('invalid simplex projection input')
    ordered = np.sort(values, axis=1)[:, ::-1]
    partial = np.cumsum(ordered, axis=1) - 1
    ranks = np.arange(1, 9)
    active = (ordered - partial / ranks > 0).sum(axis=1) - 1
    theta = partial[np.arange(len(values)), active] / (active + 1)
    result = np.maximum(values - theta[:, None], 0)
    total = result.sum(axis=1)
    if max(abs(total - 1)) > 1e-12 or min(total) <= 0:
        raise ValueError('simplex projection not feasible')
    result[result <= 64 * np.finfo(float).eps] = 0
    result /= result.sum(axis=1, keepdims=True)
    for original, actual in zip(values, result):
        positive = actual > 0
        multiplier = float(np.mean(original[positive] - actual[positive]))
        if (max(abs(original[positive] - actual[positive] - multiplier)) > 1e-12
                or (original[~positive] - multiplier > 1e-12).any()):
            raise ValueError('cleaned projection violates KKT tolerance')
    return result


class IncrementScorer:
    def __init__(self, monthly, prior):
        self.monthly = monthly
        self.frozen = SimpleNamespace(prior=prior)
        self.cache = {}

    def prepare(self, q, details, local, proper):
        self.cache = {}
        S = self.monthly.shares[local]
        proxies = []
        for detail in details:
            profiles = np.asarray(json.loads(detail['source_profiles']), int)
            if (not set(self.monthly.index.region.iloc[profiles]).issubset(proper)
                    or not self.monthly.index.month.iloc[profiles].str.startswith('2024').all()):
                raise ValueError('diagnostic source world role/year')
            inputs = np.asarray([self.monthly.local(int(p), 'asof_30d') for p in profiles], int)
            if any(self.monthly.release.iloc[p] > self.monthly.dates.iloc[d] for d, p in zip(profiles, inputs)):
                raise ValueError('diagnostic source coarse input leakage')
            old = self.monthly.shares[inputs]
            current = self.monthly.shares[profiles]
            delta = current - old
            delta[abs(delta) <= 64 * np.finfo(float).eps] = 0
            key = tuple(profiles.tolist())
            if key not in self.cache:
                self.cache[key] = project_simplex(S + delta)
            w = np.asarray(json.loads(detail['weights']))
            if (w <= 0).any() or not np.isfinite(w).all() or abs(w.sum() - 1) > 1e-12:
                raise ValueError('frozen diagnostic probabilities invalid')
            absent = (old == 0) & (current == 0) & (S[None] > 0)
            missing_mass = absent @ S
            stable = np.max(abs(delta), axis=1) == 0
            proxies.append({'profile_id': q, 'method': detail['method'],
                            'input_positive_absent_source_mass': float(w @ missing_mass),
                            'zero_increment_with_absent_fuel_mass': float(w @ (stable & absent.any(axis=1)))})
        return proxies

    def support(self, profiles, B):
        # Every support was constructed before the score function reads q's label.
        return self.cache[tuple(map(int, profiles))]


def toys():
    query = np.zeros(8)
    query[:2] = [.8, .2]
    source = np.zeros(8)
    source[1] = 1
    prior = np.full(8, .125)
    multiplicative = (.99 * query + .01 * prior) * source / (.99 * source + .01 * prior)
    multiplicative /= multiplicative.sum()
    increment = project_simplex(query[None])[0]
    if multiplicative[0] != 0 or abs(increment[0] - .8) > 1e-14:
        raise ValueError('zero-increment semantic toy failed')
    values = np.asarray([[1.2, -.2, 0, 0, 0, 0, 0, 0], [.8, .8, -.6, 0, 0, 0, 0, 0]])
    projected = project_simplex(values)
    for original, actual in zip(values, projected):
        positive = actual > 1e-12
        theta = np.mean(original[positive] - actual[positive])
        if (max(abs((original[positive] - actual[positive]) - theta)) > 1e-12
                or (original[~positive] - theta > 1e-12).any()):
            raise ValueError('independent projection KKT toy failed')
    # Use an asymmetric example; symmetric inputs can coincide under both rules.
    asymmetric = np.asarray([[.9, .5, -.4, 0, 0, 0, 0, 0]])
    p = project_simplex(asymmetric)
    c = np.maximum(asymmetric, 0)
    c /= c.sum(axis=1, keepdims=True)
    if np.max(abs(p - c)) < 1e-3:
        raise ValueError('Euclidean projection accidentally replaced with clip-renorm')
    nearly = query.copy()
    nearly[0] = np.nextafter(nearly[0], 0)
    if (project_simplex(nearly[None])[0, 2:] != 0).any():
        raise ValueError('projection introduced numerical fuel presence')
    return {'query_coal_share': float(query[0]), 'multiplicative_coal_share': float(multiplicative[0]),
            'zero_increment_coal_share': float(increment[0]), 'projection_clip_difference': float(np.max(abs(p - c)))}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = verify(SOURCE)
    verify(BASE)
    paths = [Path(__file__), PROTOCOL, SOURCE / 'summary.json', BASE / 'summary.json',
             ROOT / 'scripts/joint_coarse_worlds.py', ROOT / 'scripts/monthly_history_features.py',
             ROOT / 'scripts/coarse_distribution_lifting.py', ROOT / 'scripts/v178_joint_coarse_world.py']
    paths.extend(SOURCE / name for name in summary['output_hashes'])
    paths.extend(BASE / f'fold_{k}_asof_30d/summary.json' for k in range(6))
    hashes = summary['input_hashes'] | {str(p.relative_to(ROOT)): sha(p) for p in paths}
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('diagnostic input changed')
    toy = toys()
    monthly = MonthlyInformation(ROOT)
    coarse, fuels, proxies = [], [], []
    mutation_cases = 0
    with threadpool_limits(limits=2):
        for fold in range(6):
            meta = json.loads((BASE / f'fold_{fold}_asof_30d/summary.json').read_text())
            proper = sorted(meta['source_regions'])
            point = CoarseLawForecast(monthly, proper)
            scorer = IncrementScorer(monthly, np.asarray(meta['carrier_prior']))
            frame = pd.read_csv(SOURCE / f'fold_{fold}/world_probabilities.csv')
            frame = frame.loc[frame.role.eq('test')]
            mutation_queries = set(frame.profile_id.drop_duplicates().iloc[[0, -1]])
            for q, group in frame.groupby('profile_id', sort=False):
                q = int(q)
                if monthly.index.region.iloc[q] in proper or len(group) != 14:
                    raise ValueError('diagnostic query role/inventory')
                forecast = point.predict(q)
                details = group.to_dict('records')
                proxy = scorer.prepare(q, details, forecast['local'], proper)
                if q in mutation_queries:
                    before = {key: values.copy() for key, values in scorer.cache.items()}
                    ids = monthly.groups[monthly.index.region.iloc[q]]
                    ids = ids[(monthly.release.iloc[ids] > monthly.dates.iloc[q]).to_numpy()]
                    original = monthly.M[ids].copy(), monthly.shares[ids].copy()
                    changed = np.random.default_rng(q).uniform(1, 1000, size=original[0].shape)
                    try:
                        monthly.M[ids] = changed
                        monthly.shares[ids] = changed / changed.sum(axis=1, keepdims=True)
                        scorer.prepare(q, details, forecast['local'], proper)
                        if set(before) != set(scorer.cache) or any(not np.array_equal(v, scorer.cache[key]) for key, v in before.items()):
                            raise ValueError('increment support used unreleased query monthly values')
                        if not np.array_equal(point.predict(q)['shares'], forecast['shares']):
                            raise ValueError('point reference used unreleased query monthly values')
                    finally:
                        monthly.M[ids], monthly.shares[ids] = original
                    scorer.prepare(q, details, forecast['local'], proper)
                    mutation_cases += 1
                # All increment supports exist now, before any query outcome score.
                records, records_fuel = JointCoarsePrediction.score_coarse(scorer, q, details, forecast)
                identity = {'fold': fold, 'profile_id': q, 'region': monthly.index.region.iloc[q],
                            'month': monthly.index.month.iloc[q]}
                for r in records:
                    if not r['method'].startswith('coarse_'):
                        r['method'] = 'increment_' + r['method']
                    coarse.append(identity | r)
                for r in records_fuel:
                    if not r['method'].startswith('coarse_'):
                        r['method'] = 'increment_' + r['method']
                    fuels.append(identity | r)
                proxies.extend(identity | r for r in proxy)
            print(f'increment support diagnostic fold {fold}', flush=True)
    coarse, fuels, proxies = pd.DataFrame(coarse), pd.DataFrame(fuels), pd.DataFrame(proxies)
    if (len(coarse) != 3856 or len(fuels) != 30848 or len(proxies) != 3374 or mutation_cases != 12
            or coarse.duplicated(['profile_id', 'method']).any() or fuels.duplicated(['profile_id', 'method', 'fuel_index']).any()
            or not np.isfinite(coarse[list(COARSE_METRICS)]).all().all() or not np.isfinite(fuels[list(FUEL_METRICS)]).all().all()):
        raise ValueError('diagnostic score denominator/finiteness')
    original = pd.read_csv(SOURCE / 'coarse_month_metrics.csv')
    points = coarse.loc[coarse.method.str.startswith('coarse_')].merge(
        original.loc[original.method.str.startswith('coarse_')], on=['profile_id', 'method'], validate='one_to_one', suffixes=('_a', '_b'))
    point_error = max(float(max(abs(points[m + '_a'] - points[m + '_b']))) for m in COARSE_METRICS)
    if len(points) != 482 or point_error > 1e-8:
        raise ValueError('unchanged coarse point references not reproduced')
    cr, cm, fr, fm = aggregate_coarse(coarse, fuels)
    proxy_region = proxies.groupby(['region', 'method'])[['input_positive_absent_source_mass', 'zero_increment_with_absent_fuel_mass']].mean().reset_index()
    proxy_macro = proxy_region.groupby('method')[['input_positive_absent_source_mass', 'zero_increment_with_absent_fuel_mass']].mean().reset_index()
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('diagnostic inputs changed during run')
    OUT.mkdir(parents=True)
    for name, frame in [('coarse_month_metrics', coarse), ('coarse_fuel_metrics', fuels), ('coarse_region_metrics', cr),
                        ('coarse_macro_metrics', cm), ('coarse_fuel_region_metrics', fr), ('coarse_fuel_macro_metrics', fm),
                        ('support_proxy_queries', proxies), ('support_proxy_regions', proxy_region), ('support_proxy_macro', proxy_macro)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    result = {'assertion_failures': 0, 'coarse_score_rows': len(coarse), 'fuel_score_rows': len(fuels),
              'proxy_rows': len(proxies), 'unchanged_point_records': len(points), 'max_point_reproduction_error': point_error,
              'future_month_mutation_cases': mutation_cases,
              'toy': toy, 'fine_laws_changed': False, 'hourly_outcomes_used': False,
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in OUT.iterdir()},
              'scope': 'Post-hoc coarse support semantics with fixed V178 weights; not a new fine forecast, causal attribution, optimized Gaussian comparison or independent confirmation.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(cm.loc[cm.cohort.eq('all29')].sort_values('energy_score').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
