"""Check oracle moment identities, complete cohorts and honest width comparisons."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from hourly_fuel_data import HourlyFuelData
from frozen_recomposition import FrozenRecomposition
from level_spread_diagnostics import affine_laws, moments, score_laws
from v151_hourly_law import EF, sha

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v163-level-spread-diagnostic-r1'
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
PHASE = ROOT / 'results/v162-fixed-phase-diagnostic-r1'
OUT = ROOT / 'results/v163-level-spread-audit-r2'
KEYS = ['region', 'budget', 'month', 'method', 'calibration']
METRICS = ['crps', 'mae', 'coverage', 'tolerant_coverage', 'lower_miss', 'upper_miss',
           'lower_miss_ulp64', 'upper_miss_ulp64', 'width', 'winkler', 'support_below', 'support_above']


def ratio(tv, pv):
    if pv <= 1e-8:
        return 1., tv > 1e-8
    return (0., False) if tv <= 1e-8 else (float(np.sqrt(tv / pv)), False)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = json.loads((SOURCE / 'summary.json').read_text())
    if summary['assertion_failures']:
        raise ValueError('upstream failed')
    for name, digest in summary['input_hashes'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('input changed')
    for name, digest in summary['output_hashes'].items():
        if sha(SOURCE / name) != digest:
            raise ValueError('output changed')
    rows = pd.read_csv(SOURCE / 'month_metrics.csv')
    old = pd.read_csv(BASE / 'test_month_metrics.csv')
    old = old.loc[old.method.isin(['positive_recomposition', 'conditional_activity_recomposition'])]
    expected = set(old[KEYS].itertuples(index=False, name=None))
    expected_h2 = {key for key in expected if int(key[2][5:7]) >= 7}
    if len(rows) != 20960 or rows.duplicated(KEYS + ['case']).any():
        raise ValueError('oracle score denominator')
    for case, group in rows.groupby('case'):
        wanted = expected_h2 if case.startswith('H1_') or case == 'H2_baseline' else expected
        if set(group[KEYS].itertuples(index=False, name=None)) != wanted:
            raise ValueError('oracle cohort mismatch')
    match = rows.loc[rows.case.eq('baseline')].merge(old, on=KEYS, suffixes=('_new', '_old'), validate='one_to_one')
    baseline_errors = {m: float(np.max(np.abs(match[m + '_new'] - match[m + '_old'])))
                       for m in ('crps', 'mae', 'coverage', 'width', 'winkler')}
    if len(match) != 2272 or max(baseline_errors.values()) > 1e-7:
        raise ValueError('baseline reproduction')
    if not np.allclose(rows.coverage + rows.lower_miss + rows.upper_miss, 1, atol=1e-12):
        raise ValueError('strict coverage decomposition')
    if not np.allclose(rows.tolerant_coverage + rows.lower_miss_ulp64 + rows.upper_miss_ulp64, 1, atol=1e-12):
        raise ValueError('tolerant coverage decomposition')
    if (rows.tolerant_coverage + 1e-12 < rows.coverage).any():
        raise ValueError('tolerance decreased coverage')
    data = HourlyFuelData(ROOT)
    contexts = pd.read_csv(SOURCE / 'moment_contexts.csv')
    if len(contexts) != 1136 or contexts.duplicated(['profile_id', 'budget', 'method']).any():
        raise ValueError('moment context denominator')
    role_map = pd.read_csv(BASE / 'region_roles.csv')
    true_moment_error = 0.
    for q, group in contexts.groupby('profile_id'):
        row = data.index.iloc[q]
        y = data.truth(int(q))
        g = data.raw[int(row.offset):int(row.stop)].sum(axis=1)
        actual_anchor = float(data.shares[q] @ EF)
        actual_kappa = float(y.mean() - actual_anchor)
        covariance = float(np.mean((g - g.mean()) * (y - y.mean())))
        if abs(actual_kappa + covariance / g.mean()) > 1e-8:
            raise ValueError('actual energy/covariance identity')
        for name, value in [('true_mean', y.mean()), ('true_variance', y.var()),
                            ('true_energy_anchor', actual_anchor), ('true_kappa', actual_kappa)]:
            true_moment_error = max(true_moment_error, float(np.max(np.abs(group[name] - value))))
        for context in group.itertuples(index=False):
            if context.region != row.region or context.month != row.month or not row.month.startswith('2024'):
                raise ValueError('source moment identity/year')
            role = role_map.loc[role_map.fold.eq(context.fold) & role_map.region.eq(context.region), 'role']
            if len(role) != 1 or role.iloc[0] != 'test':
                raise ValueError('source moment role')
            if abs(context.delta_energy + context.delta_kappa - context.delta_location) > 1e-8:
                raise ValueError('level error components')
            scale, missing = ratio(context.true_variance, context.predicted_variance)
            if abs(scale - context.scale_ratio) > 1e-10 or missing != context.degenerate_unrecoverable:
                raise ValueError('spread parameter construction')
    if true_moment_error > 1e-7:
        raise ValueError('true moment mismatch')
    parameters = pd.read_csv(SOURCE / 'h1_parameters.csv')
    parameter_keys = ['region', 'budget', 'method']
    if parameters.duplicated(parameter_keys).any() or set(parameters[parameter_keys].itertuples(index=False, name=None)) != set(contexts[parameter_keys].itertuples(index=False, name=None)):
        raise ValueError('H1 parameter completeness')
    for parameter in parameters.itertuples(index=False):
        group = contexts.loc[contexts.region.eq(parameter.region) & contexts.budget.eq(parameter.budget)
                             & contexts.method.eq(parameter.method) & (pd.to_datetime(contexts.month).dt.month <= 6)]
        if len(group):
            delta = float(np.average(group.delta_location, weights=group.hours))
            tv = float(np.average(group.true_variance, weights=group.hours))
            pv = float(np.average(group.predicted_variance, weights=group.hours))
            scale, _ = ratio(tv, pv)
        else:
            delta, scale = 0., 1.
        if abs(delta - parameter.delta) > 1e-9 or abs(scale - parameter.scale) > 1e-9 or len(group) != parameter.h1_months:
            raise ValueError('H1 parameter used wrong cohort')
    expected_cases = {'baseline', 'energy_anchor_oracle', 'generation_covariance_oracle',
                      'monthly_location', 'monthly_scale', 'monthly_location_scale',
                      'phase_location_scale', 'H2_baseline', 'H1_location_H2',
                      'H1_scale_H2', 'H1_location_scale_H2'}
    if set(rows.case) != expected_cases:
        raise ValueError('diagnostic case identity')
    if not np.isfinite(rows[METRICS + ['delta', 'scale', 'post_mean', 'post_variance', 'clip_mass']]).all().all():
        raise ValueError('nonfinite diagnostic values')
    phases = pd.read_csv(PHASE / 'phase_policies.csv')
    phases = phases.loc[phases.case.eq('region_budget_method_oracle')].set_index(parameter_keys)
    parameter_map = parameters.set_index(parameter_keys)
    # Rebuild all generator moments and affine parameters. Re-score every case
    # for one maximum-variance context per fold/budget/method, selected without
    # consulting improvements, for a bounded numerical check of the saved CDFs.
    sample_keys = set(contexts.sort_values('predicted_variance', ascending=False).drop_duplicates(
        ['fold', 'budget', 'method'])[['profile_id', 'budget', 'method']].itertuples(index=False, name=None))
    row_groups = {key: group for key, group in rows.groupby(['profile_id', 'budget', 'method'])}
    if set(row_groups) != set(contexts[['profile_id', 'budget', 'method']].itertuples(index=False, name=None)):
        raise ValueError('score/context completeness')
    sampled_scores, parameter_rows, rebuilt_contexts = 0, 0, 0
    maximum_score_error, maximum_predicted_moment_error = 0., 0.
    dependency_hashes = dict(summary['input_hashes']) | {str(p.relative_to(ROOT)): sha(p) for p in (
        Path(__file__), ROOT / 'scripts/frozen_recomposition.py',
        ROOT / 'scripts/level_spread_diagnostics.py', SOURCE / 'summary.json',
        BASE / 'summary.json', PHASE / 'summary.json')}
    for archive in (BASE, PHASE):
        archived = json.loads((archive / 'summary.json').read_text())
        if archived['assertion_failures']:
            raise ValueError('upstream archive failure')
        for name, digest in archived['output_hashes'].items():
            if sha(archive / name) != digest:
                raise ValueError('upstream archive output changed')
    for (fold, budget), block_contexts in contexts.groupby(['fold', 'budget']):
        provider = FrozenRecomposition(data, BASE, int(fold), budget)
        queries = provider.queries.loc[provider.queries.role.eq('test')].set_index('profile_id')
        for q, query_contexts in block_contexts.groupby('profile_id'):
            if q not in queries.index:
                raise ValueError('context outside archived test queries')
            query = queries.loc[q]
            identity = data.index.iloc[q]
            hours = int(identity.stop - identity.offset)
            clock = data.states[int(identity.offset):int(identity.stop)] % 24
            actual = data.truth(int(q))
            laws, generated = provider.build(int(q))
            occupancy = np.bincount(clock, minlength=24) / hours
            if data.local(int(q), budget) != query.input_profile_id:
                raise ValueError('query input vintage identity')
            for context in query_contexts.itertuples(index=False):
                key = (int(q), budget, context.method)
                scored = row_groups[key]
                for column, value in [('region', identity.region), ('month', identity.month),
                                      ('fold', fold), ('hours', hours)]:
                    if not scored[column].eq(value).all() or getattr(context, column) != value:
                        raise ValueError('score/context query metadata mismatch')
                base_laws = laws[context.method]
                predicted_mean, predicted_variance = moments(base_laws, clock)
                generation, emissions = generated[context.method]
                pseudo_g = float(occupancy @ generation)
                energy = float(occupancy @ emissions / pseudo_g)
                predictions = {'predicted_mean': predicted_mean, 'predicted_variance': predicted_variance,
                               'predicted_energy_anchor': energy, 'predicted_kappa': predicted_mean - energy,
                               'mean_pseudo_generation': pseudo_g,
                               'delta_energy': context.true_energy_anchor - energy,
                               'delta_kappa': context.true_kappa - (predicted_mean - energy),
                               'delta_location': context.true_mean - predicted_mean}
                for name, value in predictions.items():
                    error = abs(getattr(context, name) - value)
                    maximum_predicted_moment_error = max(maximum_predicted_moment_error, error)
                    if not np.isclose(getattr(context, name), value, atol=1e-8, rtol=1e-12):
                        raise ValueError('predicted generator moment mismatch: ' + name)
                phase = int(phases.loc[(context.region, budget, context.method), 'phase'])
                shifted = [base_laws[(h + phase) % 24] for h in range(24)]
                shifted_mean, shifted_variance = moments(shifted, clock)
                shifted_scale, _ = ratio(context.true_variance, shifted_variance)
                hp = parameter_map.loc[(context.region, budget, context.method)]
                expected_parameters = {
                    'baseline': (0., 1., 0), 'H2_baseline': (0., 1., 0),
                    'energy_anchor_oracle': (predictions['delta_energy'], 1., 0),
                    'generation_covariance_oracle': (predictions['delta_kappa'], 1., 0),
                    'monthly_location': (predictions['delta_location'], 1., 0),
                    'monthly_scale': (0., context.scale_ratio, 0),
                    'monthly_location_scale': (predictions['delta_location'], context.scale_ratio, 0),
                    'phase_location_scale': (context.true_mean - shifted_mean, shifted_scale, phase),
                    'H1_location_H2': (hp.delta, 1., 0), 'H1_scale_H2': (0., hp.scale, 0),
                    'H1_location_scale_H2': (hp.delta, hp.scale, 0)}
                active = data.shares[q] > 0
                bounds = (float(EF[active].min()), float(EF[active].max())) if budget == 'same_month' else (0., 1000.)
                for score in scored.itertuples(index=False):
                    delta, scale, expected_phase = expected_parameters[score.case]
                    center = shifted_mean if expected_phase else predicted_mean
                    epsilon = 64 * np.finfo(float).eps
                    if abs(delta) <= epsilon * max(1., abs(center), abs(center + delta)):
                        delta = 0.
                    if abs(scale - 1) <= epsilon:
                        scale = 1.
                    if score.phase != expected_phase or not np.isclose(score.delta, delta, atol=1e-9, rtol=1e-12) or not np.isclose(score.scale, scale, atol=1e-10, rtol=1e-12):
                        raise ValueError('score affine parameter/policy mismatch')
                    radius = provider.radii[context.method, score.calibration]
                    parameter_rows += 1
                    if key in sample_keys:
                        transformed, clipping, _, _ = affine_laws(base_laws, clock, delta, scale, bounds, expected_phase)
                        verified = score_laws(transformed, actual, clock, radius)
                        verified.update(clip_mass=clipping, post_mean_error=verified['post_mean'] - context.true_mean)
                        for name, value in verified.items():
                            error = abs(getattr(score, name) - value)
                            maximum_score_error = max(maximum_score_error, error)
                            if not np.isclose(getattr(score, name), value, atol=1e-8, rtol=1e-12):
                                raise ValueError('rebuilt affine score mismatch: ' + name)
                        sampled_scores += 1
                rebuilt_contexts += 1
        print(f'bound generator/affine identities fold {fold}/{budget}', flush=True)
    region_records = []
    for key, group in rows.groupby(['case', 'region', 'budget', 'method', 'calibration']):
        region_records.append(dict(zip(['case', 'region', 'budget', 'method', 'calibration'], key))
                              | {m: float(np.sum(group[m] * group.hours) / group.hours.sum()) for m in METRICS})
    regional = pd.DataFrame(region_records)
    macro = pd.read_csv(SOURCE / 'macro_metrics.csv')
    for row in macro.itertuples(index=False):
        group = regional.loc[regional.case.eq(row.case) & regional.budget.eq(row.budget)
                             & regional.method.eq(row.method) & regional.calibration.eq(row.calibration)]
        if len(group) != row.regions:
            raise ValueError('macro source count')
        for metric in METRICS:
            if abs(group[metric].mean() - getattr(row, metric)) > 1e-8:
                raise ValueError('macro aggregate mismatch')
        if abs(group.coverage.min() - row.worst_coverage) > 1e-10:
            raise ValueError('worst coverage mismatch')
    comparisons = []
    for case in sorted(set(macro.case) - {'baseline', 'H2_baseline'}):
        reference = 'H2_baseline' if case.startswith('H1_') else 'baseline'
        joined = macro.loc[macro.case.eq(case)].merge(macro.loc[macro.case.eq(reference)],
            on=['budget', 'method', 'calibration'], suffixes=('_a', '_b'), validate='one_to_one')
        for row in joined.itertuples(index=False):
            if row.regions_a != row.regions_b:
                raise ValueError('comparison source cohort differs')
            comparisons.append({'case': case, 'reference': reference, 'budget': row.budget,
                                'method': row.method, 'calibration': row.calibration,
                                'crps_reduction_percent': 100 * (1 - row.crps_a / row.crps_b),
                                'coverage': row.coverage_a, 'worst_coverage': row.worst_coverage_a,
                                'width_change_percent': 100 * (row.width_a / row.width_b - 1),
                                'winkler_change_percent': 100 * (row.winkler_a / row.winkler_b - 1),
                                'oracle_joint_threshold': (row.coverage_a >= .9 and row.worst_coverage_a >= .85
                                                           and row.crps_a <= .95 * row.crps_b
                                                           and row.width_a <= row.width_b + 1e-9
                                                           and row.winkler_a < row.winkler_b)})
    if any(sha(ROOT / name) != digest for name, digest in dependency_hashes.items()):
        raise ValueError('audit dependency changed during computation')
    if any(sha(SOURCE / name) != digest for name, digest in summary['output_hashes'].items()):
        raise ValueError('source output changed during audit')
    OUT.mkdir(parents=True)
    pd.DataFrame(comparisons).to_csv(OUT / 'oracle_comparisons.csv', index=False)
    numerical = rows.loc[rows.tolerant_coverage - rows.coverage > 1e-12,
                         KEYS + ['case', 'coverage', 'tolerant_coverage']]
    numerical.to_csv(OUT / 'ulp_boundary_effects.csv', index=False)
    result = {'assertion_failures': 0, 'verified_rows': len(rows), 'baseline_records': len(match),
              'baseline_max_errors': baseline_errors, 'true_moment_max_error': true_moment_error,
              'verified_moment_contexts': len(contexts), 'verified_h1_parameters': len(parameters),
              'affine_parameter_rows_bound': parameter_rows,
              'frozen_radius_key_rows_bound': parameter_rows,
              'generator_moment_contexts_rebuilt': rebuilt_contexts,
              'generator_moment_max_error': maximum_predicted_moment_error,
              'sampled_cdf_contexts': len(sample_keys), 'sampled_cdf_score_rows': sampled_scores,
              'sampled_cdf_score_max_error': maximum_score_error,
              'cdf_sample_policy': 'Maximum predicted variance per fold/budget/method; all available cases and both calibrations.',
              'input_hashes': dependency_hashes,
              'source_summary_sha256': sha(SOURCE / 'summary.json'), 'script_sha256': sha(Path(__file__)),
              'output_hashes': {p.name: sha(p) for p in sorted(OUT.iterdir())},
              'scope': 'Label-oracle diagnosis, not a deployable model or a claim of improved forecast coverage.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(pd.DataFrame(comparisons).loc[lambda d: d.calibration.eq('pooled')].to_string(index=False))


if __name__ == '__main__':
    main()
