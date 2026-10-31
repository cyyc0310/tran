"""Source-label oracles identify level/spread headroom; none is a deployable method."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from level_spread_diagnostics import affine_laws, moments, score_laws, spread_ratio
from v151_hourly_law import EF, sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
PHASE = ROOT / 'results/v162-fixed-phase-diagnostic-r1'
OUT = ROOT / 'results/v163-level-spread-diagnostic-r1'
PROTOCOL = ROOT / 'docs/V163_LEVEL_SPREAD_DIAGNOSTIC_PROTOCOL.md'
METHODS = ('positive_recomposition', 'conditional_activity_recomposition')
METRICS = ('crps', 'mae', 'coverage', 'tolerant_coverage', 'lower_miss', 'upper_miss',
           'lower_miss_ulp64', 'upper_miss_ulp64', 'width', 'winkler', 'support_below', 'support_above')


def predicted_energy(generation_emissions, clock, phase=0):
    generation, emissions = generation_emissions
    occupancy = np.bincount(clock, minlength=24) / len(clock)
    assignment = (np.arange(24) + phase) % 24
    mean_generation = float(occupancy @ generation[assignment])
    if mean_generation <= 0:
        raise ValueError('nonpositive predicted generator mean')
    return float(occupancy @ emissions[assignment] / mean_generation), mean_generation


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    hashes = {}
    for archive in (BASE, PHASE):
        summary = json.loads((archive / 'summary.json').read_text())
        if summary['assertion_failures'] or summary['completed_blocks'] != 12:
            raise ValueError('incomplete upstream')
        for name, digest in summary['input_hashes'].items():
            if sha(ROOT / name) != digest:
                raise ValueError('upstream input changed')
        for name, digest in summary['output_hashes'].items():
            if sha(archive / name) != digest:
                raise ValueError('upstream output changed')
        hashes[str((archive / 'summary.json').relative_to(ROOT))] = sha(archive / 'summary.json')
        hashes.update(summary['input_hashes'])
    for path in (Path(__file__), PROTOCOL, ROOT / 'scripts/level_spread_diagnostics.py',
                 ROOT / 'scripts/frozen_recomposition.py'):
        hashes[str(path.relative_to(ROOT))] = sha(path)
    data = HourlyFuelData(ROOT)
    policies = pd.read_csv(PHASE / 'phase_policies.csv')
    policies = policies.loc[policies.case.eq('region_budget_method_oracle')].set_index(['region', 'budget', 'method'])
    outputs, contexts, parameters = [], [], []
    with threadpool_limits(limits=2):
        for fold in range(6):
            for budget in ('same_month', 'asof_30d'):
                provider = FrozenRecomposition(data, BASE, fold, budget)
                cache = {}
                for query in provider.queries.loc[provider.queries.role.eq('test')].itertuples(index=False):
                    q = int(query.profile_id)
                    row = data.index.iloc[q]
                    laws, generated = provider.build(q)
                    actual = data.truth(q)
                    g = data.raw[int(row.offset):int(row.stop)].sum(axis=1)
                    clock = data.states[int(row.offset):int(row.stop)] % 24
                    actual_mean, actual_variance = float(actual.mean()), float(actual.var())
                    true_anchor = float(data.shares[q] @ EF)
                    true_kappa = actual_mean - true_anchor
                    covariance = float(np.mean((g - g.mean()) * (actual - actual_mean)))
                    identity_error = abs(true_kappa + covariance / g.mean())
                    if identity_error > 1e-8:
                        raise ValueError('true energy/covariance identity failed')
                    for method in METHODS:
                        mean, variance = moments(laws[method], clock)
                        energy, pseudo_g = predicted_energy(generated[method], clock)
                        kappa = mean - energy
                        delta_energy, delta_kappa = true_anchor - energy, true_kappa - kappa
                        if abs(delta_energy + delta_kappa - (actual_mean - mean)) > 1e-8:
                            raise ValueError('level error decomposition failed')
                        scale, missing_spread = spread_ratio(actual_variance, variance)
                        context = {'fold': fold, 'profile_id': q, 'region': row.region, 'month': row.month,
                                   'budget': budget, 'method': method, 'hours': len(actual),
                                   'predicted_mean': mean, 'predicted_variance': variance,
                                   'predicted_energy_anchor': energy, 'predicted_kappa': kappa,
                                   'true_mean': actual_mean, 'true_variance': actual_variance,
                                   'true_energy_anchor': true_anchor, 'true_kappa': true_kappa,
                                   'delta_energy': delta_energy, 'delta_kappa': delta_kappa,
                                   'delta_location': actual_mean - mean, 'scale_ratio': scale,
                                   'degenerate_unrecoverable': missing_spread,
                                   'mean_pseudo_generation': pseudo_g,
                                   'true_generation_cv': float(g.std() / g.mean()),
                                   'true_covariance_identity_error': identity_error}
                        contexts.append(context)
                        cache[q, method] = (laws[method], generated[method], actual, clock, context)
                h1_parameters = {}
                for region in sorted(provider.queries.loc[provider.queries.role.eq('test'), 'region'].unique()):
                    for method in METHODS:
                        history = [entry[4] for (q, name), entry in cache.items()
                                   if name == method and entry[4]['region'] == region
                                   and int(entry[4]['month'][5:7]) <= 6]
                        if history:
                            weight = np.asarray([c['hours'] for c in history])
                            delta = float(np.average([c['delta_location'] for c in history], weights=weight))
                            true_variance = float(np.average([c['true_variance'] for c in history], weights=weight))
                            pred_variance = float(np.average([c['predicted_variance'] for c in history], weights=weight))
                            scale, failed = spread_ratio(true_variance, pred_variance)
                        else:
                            delta, scale, failed = 0., 1., False
                        h1_parameters[region, method] = delta, scale
                        parameters.append({'fold': fold, 'budget': budget, 'region': region, 'method': method,
                                           'h1_months': len(history), 'delta': delta, 'scale': scale,
                                           'no_h1_fallback': not history, 'degenerate_unrecoverable': failed})
                for (q, method), (laws, generated, actual, clock, context) in cache.items():
                    row = data.index.iloc[q]
                    if budget == 'same_month':
                        active = data.shares[q] > 0
                        bounds = float(EF[active].min()), float(EF[active].max())
                    else:
                        bounds = 0., 1000.
                    delta, scale = context['delta_location'], context['scale_ratio']
                    phase = int(policies.loc[(row.region, budget, method), 'phase'])
                    shifted = [laws[(h + phase) % 24] for h in range(24)]
                    shifted_mean, shifted_variance = moments(shifted, clock)
                    shifted_scale, _ = spread_ratio(context['true_variance'], shifted_variance)
                    cases = [('baseline', 0., 1., 0),
                             ('energy_anchor_oracle', context['delta_energy'], 1., 0),
                             ('generation_covariance_oracle', context['delta_kappa'], 1., 0),
                             ('monthly_location', delta, 1., 0),
                             ('monthly_scale', 0., scale, 0),
                             ('monthly_location_scale', delta, scale, 0),
                             ('phase_location_scale', context['true_mean'] - shifted_mean, shifted_scale, phase)]
                    if int(row.month[5:7]) >= 7:
                        h1_delta, h1_scale = h1_parameters[row.region, method]
                        cases.extend([('H2_baseline', 0., 1., 0), ('H1_location_H2', h1_delta, 1., 0),
                                      ('H1_scale_H2', 0., h1_scale, 0),
                                      ('H1_location_scale_H2', h1_delta, h1_scale, 0)])
                    for case, delta_case, scale_case, phase_case in cases:
                        transformed, clipping, used_delta, used_scale = affine_laws(
                            laws, clock, delta_case, scale_case, bounds, phase_case)
                        for calibration in ('pooled', 'max_region'):
                            radius = provider.radii[method, calibration]
                            scores = score_laws(transformed, actual, clock, radius)
                            outputs.append({'fold': fold, 'profile_id': q, 'region': row.region, 'month': row.month,
                                            'budget': budget, 'method': method, 'calibration': calibration,
                                            'case': case, 'phase': phase_case, 'hours': len(actual),
                                            'delta': used_delta, 'scale': used_scale, 'clip_mass': clipping,
                                            'post_mean_error': scores['post_mean'] - context['true_mean']} | scores)
                print(f'completed moment diagnostic {fold}/{budget}: {len(cache)} query/method contexts', flush=True)
    rows = pd.DataFrame(outputs)
    keys = ['region', 'budget', 'month', 'method', 'calibration', 'case']
    if rows.duplicated(keys).any() or len(rows) != 20960:
        raise ValueError('moment diagnostic denominator')
    old = pd.read_csv(BASE / 'test_month_metrics.csv')
    old = old.loc[old.method.isin(METHODS)]
    match_keys = ['region', 'budget', 'month', 'method', 'calibration']
    match = rows.loc[rows.case.eq('baseline')].merge(old, on=match_keys, suffixes=('_new', '_old'), validate='one_to_one')
    errors = {metric: float(np.max(np.abs(match[metric + '_new'] - match[metric + '_old'])))
              for metric in ('crps', 'mae', 'coverage', 'width', 'winkler')}
    if len(match) != 2272 or max(errors.values()) > 1e-7:
        raise ValueError('baseline moment reconstruction')
    region_records = []
    for key, group in rows.groupby(['case', 'region', 'budget', 'method', 'calibration']):
        region_records.append(dict(zip(['case', 'region', 'budget', 'method', 'calibration'], key))
                              | {m: float(np.average(group[m], weights=group.hours)) for m in METRICS})
    regional = pd.DataFrame(region_records)
    macro = regional.groupby(['case', 'budget', 'method', 'calibration']).agg(
        **{m: (m, 'mean') for m in METRICS}, worst_coverage=('coverage', 'min'),
        worst_tolerant_coverage=('tolerant_coverage', 'min'), regions=('region', 'size')).reset_index()
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('frozen moment inputs changed')
    OUT.mkdir(parents=True)
    for name, frame in [('month_metrics', rows), ('region_metrics', regional), ('macro_metrics', macro),
                        ('moment_contexts', pd.DataFrame(contexts)), ('h1_parameters', pd.DataFrame(parameters))]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    result = {'status': 'SOURCE_ONLY_LEVEL_SPREAD_ORACLE_DIAGNOSTIC', 'assertion_failures': 0,
              'scored_rows': len(rows), 'baseline_records_reproduced': len(match), 'baseline_max_errors': errors,
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in sorted(OUT.iterdir())},
              'limitations': ['All nonbaseline transformations use local labels/future coarse information; no method promotion.',
                              'Affine moments need not match after clipping; latent energy anchor is generator-specific.',
                              'H1/H2 is local-label temporal stability, not zero-hour local deployment.',
                              'Fixed-EF production proxy and 2024 shared-time source development.']}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(macro.loc[macro.calibration.eq('pooled')].to_string(index=False))


if __name__ == '__main__':
    main()
