"""Rebuild V179 coarse scores and independently check simplex geometry/losses."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from coarse_distribution_lifting import CoarseLawForecast
from joint_coarse_worlds import JointCoarsePrediction
from monthly_history_features import MonthlyInformation
from v151_hourly_law import EF, sha
from v170_fuel_marginal_diagnosis import verify
from v178_joint_coarse_world import aggregate_coarse, COARSE_METRICS, FUEL_METRICS
from v179_increment_support_diagnostic import IncrementScorer, SOURCE as ORIGINAL, BASE, OUT as SOURCE, ROOT

OUT = ROOT / 'results/v179-increment-support-audit-r1'


def bisection_projection(values):
    """Solve the simplex multiplier equation without sorting or active-set ranks."""
    low = np.min(values, axis=1) - 1
    high = np.max(values, axis=1)
    for _ in range(80):
        middle = (low + high) / 2
        mass = np.maximum(values - middle[:, None], 0).sum(axis=1)
        low = np.where(mass > 1, middle, low)
        high = np.where(mass > 1, high, middle)
    answer = np.maximum(values - ((low + high) / 2)[:, None], 0)
    answer[answer <= 64 * np.finfo(float).eps] = 0
    return answer / answer.sum(axis=1, keepdims=True)


def independent_losses(values, weight, truth):
    """Use pairwise absolute distances, without the production CDF/CRPS helpers."""
    differences = values[:, None] - values[None]
    marginal = weight @ abs(values - truth) - .5 * np.einsum('i,ijf,j->f', weight, abs(differences), weight)
    zero = weight @ (values == 0)
    projected, projected_truth = np.zeros_like(values), np.zeros_like(truth)
    for ids in ([3, 4, 5, 6], [1, 7]):
        projected[:, ids] = values[:, ids] - values[:, ids].mean(axis=1, keepdims=True)
        projected_truth[ids] = truth[ids] - truth[ids].mean()
    anchors, target = values @ EF, truth @ EF
    coarse = {
        'share_crps_mean': float(marginal.mean()),
        'energy_score': float(weight @ np.linalg.norm(values - truth, axis=1)
                              - .5 * weight @ np.linalg.norm(differences, axis=-1) @ weight),
        'same_EF_energy_score': float(weight @ np.linalg.norm(projected - projected_truth, axis=1)
                                      - .5 * weight @ np.linalg.norm(projected[:, None] - projected[None], axis=-1) @ weight),
        'anchor_crps': float(weight @ abs(anchors - target)
                             - .5 * weight @ abs(anchors[:, None] - anchors[None]) @ weight),
    }
    fuels = [{'crps': float(marginal[f]), 'predicted_zero': float(zero[f]),
              'zero_brier': float((zero[f] - (truth[f] == 0))**2)} for f in range(8)]
    return coarse, fuels


def compare_frames(new, old, keys):
    a, b = [frame.sort_values(keys).reset_index(drop=True) for frame in (new, old)]
    if len(a) != len(b) or list(a.columns) != list(b.columns) or a.duplicated(keys).any() or b.duplicated(keys).any():
        raise ValueError('audit table inventory/columns')
    error = 0.
    for column in a:
        if pd.api.types.is_numeric_dtype(a[column]) and not pd.api.types.is_bool_dtype(a[column]):
            difference = np.asarray(a[column], float) - np.asarray(b[column], float)
            if not np.isfinite(difference).all():
                raise ValueError('audit nonfinite table')
            error = max(error, float(np.max(abs(difference))))
        elif not np.array_equal(a[column], b[column]):
            raise ValueError(f'audit table identity: {column}')
    if error > 1e-8:
        raise ValueError(f'audit table numerical mismatch: {error}')
    return error


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = verify(SOURCE)
    verify(ORIGINAL)
    verify(ROOT / 'results/v178-joint-coarse-audit-r1')
    paths = [Path(__file__), SOURCE / 'summary.json', ORIGINAL / 'summary.json',
             ROOT / 'results/v178-joint-coarse-audit-r1/summary.json']
    paths.extend(SOURCE / name for name in summary['output_hashes'])
    hashes = summary['input_hashes'] | {str(p.relative_to(ROOT)): sha(p) for p in paths}
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('audit input changed')
    monthly = MonthlyInformation(ROOT)
    coarse, fuels, proxies = [], [], []
    projection_rows = mutation_cases = 0
    projection_error = loss_error = 0.
    with threadpool_limits(limits=2):
        for fold in range(6):
            meta = json.loads((BASE / f'fold_{fold}_asof_30d/summary.json').read_text())
            proper = sorted(meta['source_regions'])
            point = CoarseLawForecast(monthly, proper)
            scorer = IncrementScorer(monthly, np.asarray(meta['carrier_prior']))
            frame = pd.read_csv(ORIGINAL / f'fold_{fold}/world_probabilities.csv')
            frame = frame.loc[frame.role.eq('test')]
            mutation_queries = set(frame.profile_id.drop_duplicates().iloc[[0, -1]])
            for q, group in frame.groupby('profile_id', sort=False):
                q = int(q)
                forecast = point.predict(q)
                details = group.to_dict('records')
                fresh_proxies = scorer.prepare(q, details, forecast['local'], proper)
                S = monthly.shares[forecast['local']]
                checked = set()
                independent_supports = {}
                independent_proxies = []
                for detail in details:
                    profiles = np.asarray(json.loads(detail['source_profiles']), int)
                    key = tuple(profiles.tolist())
                    w = np.asarray(json.loads(detail['weights']))
                    inputs = np.asarray([monthly.local(int(p), 'asof_30d') for p in profiles])
                    old, current = monthly.shares[inputs], monthly.shares[profiles]
                    delta = current - old
                    delta[abs(delta) <= 64 * np.finfo(float).eps] = 0
                    if key not in checked:
                        projected = bisection_projection(S + delta)
                        actual = scorer.cache[key]
                        projection_error = max(projection_error, float(np.max(abs(projected - actual))))
                        if (projection_error > 1e-12 or (actual < 0).any()
                                or max(abs(actual.sum(axis=1) - 1)) > 1e-12):
                            raise ValueError('independent simplex projection mismatch')
                        independent_supports[key] = projected
                        projection_rows += len(profiles)
                        checked.add(key)
                    missing = np.asarray([sum(S[f] for f in range(8) if old[j, f] == 0 and current[j, f] == 0 and S[f] > 0)
                                          for j in range(len(profiles))])
                    stable = np.asarray([np.array_equal(d, np.zeros(8)) for d in delta])
                    independent_proxies.append({'profile_id': q, 'method': detail['method'],
                                                'input_positive_absent_source_mass': float(w @ missing),
                                                'zero_increment_with_absent_fuel_mass': float(w @ (stable & (missing > 0)))})
                compare_frames(pd.DataFrame(independent_proxies), pd.DataFrame(fresh_proxies), ['profile_id', 'method'])
                if q in mutation_queries:
                    before = {key: v.copy() for key, v in scorer.cache.items()}
                    ids = monthly.groups[monthly.index.region.iloc[q]]
                    ids = ids[(monthly.release.iloc[ids] > monthly.dates.iloc[q]).to_numpy()]
                    original = monthly.M[ids].copy(), monthly.shares[ids].copy()
                    changed = np.random.default_rng(q + 179).uniform(1, 5000, size=original[0].shape)
                    try:
                        monthly.M[ids] = changed
                        monthly.shares[ids] = changed / changed.sum(axis=1, keepdims=True)
                        scorer.prepare(q, details, forecast['local'], proper)
                        if (set(before) != set(scorer.cache)
                                or any(not np.array_equal(v, scorer.cache[k]) for k, v in before.items())
                                or not np.array_equal(point.predict(q)['shares'], forecast['shares'])):
                            raise ValueError('unreleased query M changed a forecast')
                    finally:
                        monthly.M[ids], monthly.shares[ids] = original
                    scorer.prepare(q, details, forecast['local'], proper)
                    mutation_cases += 1
                # Production support rebuilds exact quantile conventions. Bisection
                # supports independently check the continuous losses, not ULP ties.
                records, fuel_records = JointCoarsePrediction.score_coarse(scorer, q, details, forecast)
                lookup = {r['method']: r for r in records}
                fuel_lookup = {(r['method'], r['fuel_index']): r for r in fuel_records}
                laws = [(d['method'], independent_supports[tuple(json.loads(d['source_profiles']))], np.asarray(json.loads(d['weights']))) for d in details]
                laws += [('coarse_persistence', S[None], np.ones(1)), ('coarse_point_mix', forecast['shares'][None], np.ones(1))]
                for method, values, w in laws:
                    independent, independent_fuel = independent_losses(values, w, monthly.shares[q])
                    for name, value in independent.items():
                        loss_error = max(loss_error, abs(value - lookup[method][name]))
                    for f, record in enumerate(independent_fuel):
                        for name, value in record.items():
                            loss_error = max(loss_error, abs(value - fuel_lookup[(method, f)][name]))
                if loss_error > 1e-8:
                    raise ValueError('independent auxiliary score mismatch')
                identity = {'fold': fold, 'profile_id': q, 'region': monthly.index.region.iloc[q], 'month': monthly.index.month.iloc[q]}
                for r in records + fuel_records:
                    if not r['method'].startswith('coarse_'):
                        r['method'] = 'increment_' + r['method']
                coarse.extend(identity | r for r in records)
                fuels.extend(identity | r for r in fuel_records)
                proxies.extend(identity | r for r in fresh_proxies)
            print(f'audited increment support fold {fold}', flush=True)
    coarse, fuels, proxies = pd.DataFrame(coarse), pd.DataFrame(fuels), pd.DataFrame(proxies)
    if (len(coarse), len(fuels), len(proxies), mutation_cases) != (3856, 30848, 3374, 12):
        raise ValueError('audit complete denominator')
    cr, cm, fr, fm = aggregate_coarse(coarse, fuels)
    metrics = ['input_positive_absent_source_mass', 'zero_increment_with_absent_fuel_mass']
    pr = proxies.groupby(['region', 'method'])[metrics].mean().reset_index()
    pm = pr.groupby('method')[metrics].mean().reset_index()
    tables = [('coarse_month_metrics', coarse, ['profile_id', 'method']),
              ('coarse_fuel_metrics', fuels, ['profile_id', 'method', 'fuel_index']),
              ('coarse_region_metrics', cr, ['region', 'method']),
              ('coarse_macro_metrics', cm, ['cohort', 'method']),
              ('coarse_fuel_region_metrics', fr, ['region', 'method', 'fuel_index']),
              ('coarse_fuel_macro_metrics', fm, ['cohort', 'method', 'fuel_index']),
              ('support_proxy_queries', proxies, ['profile_id', 'method']),
              ('support_proxy_regions', pr, ['region', 'method']),
              ('support_proxy_macro', pm, ['method'])]
    table_error = max(compare_frames(fresh, pd.read_csv(SOURCE / f'{name}.csv'), keys) for name, fresh, keys in tables)
    old = pd.read_csv(ORIGINAL / 'coarse_region_metrics.csv')
    comparisons = []
    rng = np.random.default_rng(20261029)
    for cohort in ('all29', 'nonconstant24'):
        a = cr if cohort == 'all29' else cr.loc[~cr.region.isin(['CHPD', 'DOPD', 'GVL', 'SCL', 'SPA'])]
        b = old.loc[old.region.isin(a.region.unique())]
        pairs = []
        for method in sorted(a.method.unique()):
            if not method.startswith('increment_'):
                continue
            pairs += [(method, 'original', method.removeprefix('increment_')),
                      (method, 'original', 'world_gaussian'),
                      (method, 'original', 'allmonth_gaussian'),
                      (method, 'increment', 'increment_allmonth_gaussian')]
        for method, kind, reference in pairs:
            left = a.loc[a.method.eq(method)].sort_values('region')
            right = (b if kind == 'original' else a).loc[lambda x: x.method.eq(reference)].sort_values('region')
            if not np.array_equal(left.region, right.region):
                raise ValueError('paired diagnostic region mismatch')
            resample = rng.integers(0, len(left), size=(10000, len(left)))
            record = {'cohort': cohort, 'method': method, 'reference_kind': kind, 'reference': reference, 'regions': len(left)}
            for metric in ('energy_score', 'same_EF_energy_score', 'anchor_crps', 'share_crps_mean'):
                delta = left[metric].to_numpy() - right[metric].to_numpy()
                boot = delta[resample].mean(axis=1)
                record |= {metric + '_difference': float(delta.mean()), metric + '_bootstrap_lo': float(np.quantile(boot, .025)),
                           metric + '_bootstrap_hi': float(np.quantile(boot, .975)), metric + '_regions_better': int((delta < -1e-12).sum())}
            comparisons.append(record)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('audit inputs changed during run')
    OUT.mkdir(parents=True)
    pd.DataFrame(comparisons).to_csv(OUT / 'coarse_comparisons.csv', index=False)
    result = {'assertion_failures': 0, 'coarse_rows_rebuilt': len(coarse), 'fuel_rows_rebuilt': len(fuels),
              'proxy_rows_rebuilt': len(proxies), 'independent_projection_rows': projection_rows,
              'future_month_mutation_cases': mutation_cases, 'max_projection_error': projection_error,
              'max_independent_loss_error': loss_error, 'max_table_error': table_error, 'comparisons': len(comparisons),
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in OUT.iterdir()},
              'scope': 'All coarse/fuel/proxy records and nine tables rebuilt; independent bisection projection and pairwise losses. Shared exact quantile and aggregation helpers. Region bootstrap is descriptive post-hoc, no fine forecast or gate evaluation.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
