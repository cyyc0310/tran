"""Reconstruct V169 scores and fixed fuel margins; compare matched strong controls."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from calendar_cif_law import CalendarLaw
from coarse_distribution_lifting import CoarseLawForecast
from feasible_copula_prediction import FeasibleCopulaPrediction, METHODS, REFERENCES, SETTINGS
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from level_spread_diagnostics import score_laws
from monthly_history_features import MonthlyInformation
from v151_hourly_law import sha
from v159_fuel_activity_source_pilot import weighted_radius
from v169_feasible_copula import BASE, METRICS, aggregate, check_mutation

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v169-feasible-copula-r1'
OUT = ROOT / 'results/v169-feasible-copula-audit-r1'
NONCONSTANT = ['CHPD', 'DOPD', 'GVL', 'SCL', 'SPA']


def verify(path):
    summary = json.loads((path / 'summary.json').read_text())
    if summary['assertion_failures']:
        raise ValueError('failed upstream archive')
    for name, digest in summary['output_hashes'].items():
        if sha(path / name) != digest:
            raise ValueError(f'archive changed: {path / name}')
    return summary


def check_hourly_input(predictor, data, q):
    before = predictor.build(q)[0]
    ids = data.groups[data.index.region.iloc[q]]
    raw_ids = np.concatenate([np.arange(int(data.index.offset.iloc[q]), int(data.index.stop.iloc[q])) for q in ids])
    saved = data.raw[raw_ids].copy()
    try:
        data.raw[raw_ids] = np.random.default_rng(q).uniform(1, 500, size=saved.shape)
        after = predictor.build(q)[0]
    finally:
        data.raw[raw_ids] = saved
    for method in METHODS:
        for a, b in zip(before[method], after[method]):
            if not np.array_equal(a.x, b.x) or not np.array_equal(a.w, b.w):
                raise ValueError('held-out region hourly values affected prediction')


def rebuild(data, fold):
    frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
    monthly = MonthlyInformation(ROOT)
    predictor = FeasibleCopulaPrediction(frozen, CoarseLawForecast(monthly, frozen.fit))
    block = SOURCE / f'fold_{fold}'
    saved = pd.read_csv(block / 'test_month_metrics.csv').set_index(['profile_id', 'method', 'radius_policy', 'calibration'])
    saved_contexts = pd.read_csv(block / 'contexts.csv').set_index('profile_id')
    saved_worlds = pd.read_csv(block / 'world_probabilities.csv').set_index(['profile_id', 'source_profile_id'])
    saved_radii = pd.read_csv(block / 'calibration_radii.csv').set_index(['method', 'calibration'])
    if (set(saved_contexts.index) != set(frozen.queries.profile_id)
            or saved_contexts.index.duplicated().any() or saved_worlds.index.duplicated().any()):
        raise ValueError('context or world denominator')
    errors = {method: {} for method in METHODS}
    max_score = max_radius = max_context = 0.
    scored = worlds_checked = 0
    for q in frozen.queries.profile_id.iloc[[0, -1]]:
        check_mutation(predictor, data, monthly, int(q))
    check_hourly_input(predictor, data, int(frozen.queries.profile_id.iloc[0]))
    for role in ('calibration', 'test'):
        for rec in frozen.queries.loc[frozen.queries.role.eq(role)].itertuples(index=False):
            q = int(rec.profile_id)
            laws, forecast, worlds = predictor.build(q)
            context = saved_contexts.loc[q]
            if context.role != role or context.region != rec.region or context.month != rec.month or context.fold != fold:
                raise ValueError('context identity')
            for key in ('raw_anchor', 'anchor', 'past_anchor', 'sigma', 'local'):
                max_context = max(max_context, abs(float(forecast[key]) - float(context[key])))
            for world in worlds:
                old = saved_worlds.loc[(q, world['source_profile_id'])]
                if old.role != role or old.region != rec.region or old.month != rec.month or old.fold != fold:
                    raise ValueError('world identity')
                for key in ('conditional_weight', 'gaussian_weight', 'world_anchor'):
                    max_context = max(max_context, abs(float(world[key]) - float(old[key])))
                worlds_checked += 1
            actual = data.truth(q)
            row = data.index.iloc[q]
            clock = data.states[int(row.offset):int(row.stop)] % 24
            for method in METHODS:
                if role == 'calibration':
                    law = CalendarLaw(laws[method], clock)
                    distance = np.maximum.reduce([law.quantile(.05) - actual, actual - law.quantile(.95), np.zeros(len(actual))])
                    errors[method].setdefault(rec.region, []).append(distance)
                else:
                    for policy in (('recalibrated',) if method in REFERENCES else ('fixed', 'recalibrated')):
                        for calibration in ('pooled', 'max_region'):
                            ref = method if method in REFERENCES else SETTINGS[method][0]
                            radius = radius_map[(method if policy == 'recalibrated' else ref, calibration)]
                            old = saved.loc[(q, method, policy, calibration)]
                            if old.region != rec.region or old.month != rec.month or old.fold != fold:
                                raise ValueError('test score identity')
                            new = score_laws(laws[method], actual, clock, radius)
                            delta = np.asarray([new[key] - old[key] for key in new], float)
                            if not np.isfinite(delta).all():
                                raise ValueError('nonfinite reconstruction')
                            max_score = max(max_score, float(np.max(abs(delta))))
                            max_radius = max(max_radius, abs(radius - old.radius))
                            scored += 1
        if role == 'calibration':
            radius_map = {}
            for method in METHODS:
                for label, radius in zip(('pooled', 'max_region'), weighted_radius(errors[method])):
                    radius_map[(method, label)] = radius
                    max_radius = max(max_radius, abs(radius - saved_radii.loc[(method, label), 'radius']))
    archived = pd.read_csv(block / 'source_couplings.csv').set_index(['source_profile_id', 'hour', 'seed'])
    if set(archived.index) != set(predictor.source_records):
        raise ValueError('source coupling inventory')
    for (profile, hour, seed), metadata in predictor.source_records.items():
        old = archived.loc[(profile, hour, seed)]
        for key, value in metadata.items():
            if key in ('source_profile_id', 'hour', 'seed'):
                continue
            if isinstance(value, str):
                if old[key] != value:
                    raise ValueError('source permutation identity')
            elif not np.isclose(old[key], value, rtol=1e-12, atol=1e-12):
                raise ValueError('source coupling metadata')
        original = frozen.activity.donor(profile)[hour]
        new = predictor.shuffled(profile, hour, seed)
        repeated = np.tile(original, (metadata['copies'], 1))
        if not np.array_equal(np.sort(repeated, axis=0), np.sort(new, axis=0)) or (new.sum(axis=1) <= 0).any():
            raise ValueError('empirical fuel margin or positive support changed')
    if scored != len(saved) or worlds_checked != len(saved_worlds) or max(max_score, max_radius, max_context) > 1e-7:
        raise ValueError(f'reconstruction failed: {scored}, {max_score}, {max_radius}, {max_context}')
    print(f'audited fold {fold}: {scored} scores; {len(archived)} fuel-margin identities', flush=True)
    return {'fold': fold, 'rebuilt_score_rows': scored, 'source_coupling_cells': len(archived),
            'verified_worlds': worlds_checked, 'max_score_error': max_score,
            'max_radius_error': max_radius, 'max_context_error': max_context}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = verify(SOURCE)
    hashes = summary['input_hashes'] | {str((SOURCE / 'summary.json').relative_to(ROOT)): sha(SOURCE / 'summary.json'),
                                      str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    reference_sources = {'positive_recomposition': BASE, 'scalar_forest': BASE,
                         'activity_mmd_forest': ROOT / 'results/v160-carbon-cdf-source-r1',
                         'forecast_composition': ROOT / 'results/v164-scenario-reweighting-r1',
                         'world_entropy': ROOT / 'results/v164-scenario-reweighting-r1',
                         'hetero_history_barycentric': ROOT / 'results/v165-coarse-distribution-lifting-r1'}
    for directory in set(reference_sources.values()):
        verify(directory)
        for name in ('summary.json', 'test_month_metrics.csv'):
            hashes[str((directory / name).relative_to(ROOT))] = sha(directory / name)
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('input changed before audit')
    rows = pd.read_csv(SOURCE / 'test_month_metrics.csv')
    if (len(rows) != 8676 or rows.duplicated(['profile_id', 'method', 'radius_policy', 'calibration']).any()
            or not np.isfinite(rows[list(METRICS) + ['radius', 'hours']]).all().all()):
        raise ValueError('score denominator or finiteness')
    keys = ['profile_id', 'method', 'radius_policy', 'calibration']
    blocks = pd.concat([pd.read_csv(SOURCE / f'fold_{fold}/test_month_metrics.csv') for fold in range(6)], ignore_index=True)
    pd.testing.assert_frame_equal(rows.sort_values(keys).reset_index(drop=True),
                                  blocks.sort_values(keys).reset_index(drop=True),
                                  check_exact=True)
    rebuilt_regional, rebuilt_macro = aggregate(rows)
    for name, rebuilt, keys in [('region_metrics', rebuilt_regional, ['region', 'method', 'radius_policy', 'calibration']),
                                ('macro_metrics', rebuilt_macro, ['cohort', 'method', 'radius_policy', 'calibration'])]:
        saved = pd.read_csv(SOURCE / f'{name}.csv')
        pair = rebuilt.merge(saved, on=keys, suffixes=('_a', '_b'), validate='one_to_one')
        if len(pair) != len(saved) or any(max(abs(pair[m + '_a'] - pair[m + '_b'])) > 1e-8 for m in METRICS):
            raise ValueError('aggregation mismatch')
        if name == 'macro_metrics' and (not np.array_equal(pair.regions_a, pair.regions_b)
                or max(abs(pair.worst_coverage_a - pair.worst_coverage_b)) > 1e-10):
            raise ValueError('macro denominator/worst coverage')
    data = HourlyFuelData(ROOT)
    with threadpool_limits(limits=2):
        checks = [rebuild(data, fold) for fold in range(6)]
    reference_tables = [rebuilt_regional]
    cohort_keys = set(rows[['region', 'month', 'calibration', 'hours']].itertuples(index=False, name=None))
    for method, directory in reference_sources.items():
        old = pd.read_csv(directory / 'test_month_metrics.csv')
        old = old.loc[old.method.eq(method) & old.budget.eq('asof_30d')]
        if (set(old[['region', 'month', 'calibration', 'hours']].itertuples(index=False, name=None)) != cohort_keys
                or old.duplicated(['region', 'month', 'calibration']).any()
                or not np.isfinite(old[list(METRICS[:5])]).all().all()):
            raise ValueError('strong reference cohort or scores')
        rr = pd.DataFrame([{'region': key[0], 'method': method, 'calibration': key[1], 'radius_policy': 'recalibrated',
                            **{m: float(np.average(g[m], weights=g.hours)) for m in METRICS[:5]}}
                           for key, g in old.groupby(['region', 'calibration'])])
        reference_tables.append(rr)
    table = pd.concat(reference_tables, ignore_index=True)
    rng = np.random.default_rng(169)
    comparisons = []
    for rec in rebuilt_macro.loc[~rebuilt_macro.method.isin(REFERENCES)].itertuples(index=False):
        selected = table.loc[table.calibration.eq(rec.calibration)]
        if rec.cohort == 'nonconstant24':
            selected = selected.loc[~selected.region.isin(NONCONSTANT)]
        a = selected.loc[selected.method.eq(rec.method) & selected.radius_policy.eq(rec.radius_policy)].sort_values('region')
        for reference in (*REFERENCES, *reference_sources):
            b = selected.loc[selected.method.eq(reference) & selected.radius_policy.eq('recalibrated')].sort_values('region')
            if not np.array_equal(a.region, b.region) or len(a) != rec.regions:
                raise ValueError('paired strong reference inventory')
            delta = a.crps.to_numpy() - b.crps.to_numpy()
            boot = delta[rng.integers(len(delta), size=(3000, len(delta)))].mean(axis=1)
            comparisons.append({'cohort': rec.cohort, 'method': rec.method, 'radius_policy': rec.radius_policy,
                                'calibration': rec.calibration, 'reference': reference,
                                'paired_crps_difference': float(delta.mean()),
                                'region_bootstrap_lo': float(np.quantile(boot, .025)),
                                'region_bootstrap_hi': float(np.quantile(boot, .975)),
                                'regions_better_crps': int((delta < -1e-9).sum()),
                                'crps_reduction_percent': 100 * (1 - rec.crps / b.crps.mean()),
                                'coverage': rec.coverage, 'worst_coverage': rec.worst_coverage,
                                'width': rec.width, 'width_change_percent': 100 * (rec.width / b.width.mean() - 1),
                                'winkler': rec.winkler,
                                'numeric_threshold_only': bool(rec.coverage >= .9 and rec.worst_coverage >= .85
                                                               and rec.crps <= .95 * b.crps.mean()
                                                               and rec.width <= b.width.mean() + 1e-9
                                                               and rec.winkler < b.winkler.mean())})
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('dependency changed during audit')
    verify(SOURCE)
    OUT.mkdir(parents=True)
    pd.DataFrame(checks).to_csv(OUT / 'reconstruction.csv', index=False)
    pd.DataFrame(comparisons).to_csv(OUT / 'comparisons.csv', index=False)
    result = {'assertion_failures': 0, 'rebuilt_score_rows': sum(c['rebuilt_score_rows'] for c in checks),
              'verified_source_coupling_cells': sum(c['source_coupling_cells'] for c in checks),
              'future_month_mutation_cases': 12, 'hourly_input_mutation_cases': 6,
              'max_score_error': max(c['max_score_error'] for c in checks),
              'comparisons': len(comparisons), 'numeric_threshold_passes': sum(c['numeric_threshold_only'] for c in comparisons),
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in sorted(OUT.iterdir())},
              'scope': 'All scores reconstructed with frozen model/law helpers; not an independent scoring algorithm. Fuel column multisets verified directly. Region bootstrap is descriptive source-development evidence.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_hashes', 'output_hashes')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
