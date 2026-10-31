"""Complete probability-split comparisons with source-only role and archive checks."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from v151_hourly_law import sha

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / 'results/v160-carbon-cdf-source-r1'
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
OUT = ROOT / 'results/v160-carbon-cdf-audit-r1'
METRICS = ['crps', 'mae', 'coverage', 'width', 'winkler']
KEYS = ['budget', 'region', 'month', 'calibration']
NEW_METHODS = ['scalar_cdf_forest', 'activity_mmd_forest', 'mean_smoothed_forest', 'carbon_cdf_forest']


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    archive_hashes = {}
    for archive in (NEW, BASE):
        summary = json.loads((archive / 'summary.json').read_text())
        if summary['assertion_failures'] or summary['completed_blocks'] != 12:
            raise ValueError('incomplete upstream archive')
        for name, digest in summary['input_hashes'].items():
            if sha(ROOT / name) != digest:
                raise ValueError('source input changed: ' + name)
        for name, digest in summary['output_hashes'].items():
            if sha(archive / name) != digest:
                raise ValueError('source output changed: ' + name)
        archive_hashes[str(archive.relative_to(ROOT))] = sha(archive / 'summary.json')
    rows = pd.read_csv(NEW / 'test_month_metrics.csv')
    old = pd.read_csv(BASE / 'test_month_metrics.csv')
    expected = set(old[KEYS].itertuples(index=False, name=None))
    if rows.duplicated(KEYS + ['method']).any() or len(rows) != len(expected) * 4:
        raise ValueError('score denominator')
    for method in NEW_METHODS:
        actual = rows.loc[rows.method.eq(method)]
        if set(actual[KEYS].itertuples(index=False, name=None)) != expected:
            raise ValueError('method query set changed')
    if not np.isfinite(rows[METRICS]).all().all():
        raise ValueError('nonfinite scores')
    for fold in range(6):
        for budget in ('same_month', 'asof_30d'):
            a = NEW / f'fold_{fold}_{budget}'
            b = BASE / f'fold_{fold}_{budget}'
            current = json.loads((a / 'summary.json').read_text())
            previous = json.loads((b / 'summary.json').read_text())
            for key, original_key in [('fit_regions', 'source_regions'),
                                      ('calibration_regions', 'calibration_regions'),
                                      ('test_regions', 'test_regions'),
                                      ('prototype_profile_ids', 'prototype_profile_ids')]:
                if current[key] != previous[original_key]:
                    raise ValueError('role/prototype identity changed')
            if not pd.read_csv(a / 'query_inputs.csv').equals(pd.read_csv(b / 'query_inputs.csv')):
                raise ValueError('query input vintage changed')
    reproduced = rows.loc[rows.budget.eq('asof_30d') & rows.method.eq('mean_smoothed_forest')].merge(
        old.loc[old.budget.eq('asof_30d') & old.method.eq('response_activity_forest')],
        on=KEYS, suffixes=('_new', '_old'), validate='one_to_one')
    maximum_errors = {m: float(np.max(np.abs(reproduced[m + '_new'] - reproduced[m + '_old']))) for m in METRICS}
    if len(reproduced) != 482 or max(maximum_errors.values()) > 1e-7:
        raise ValueError('asof mean-response reproduction')
    allrows = pd.concat([old, rows], ignore_index=True)
    regions = []
    for key, group in allrows.groupby(['region', 'budget', 'method', 'calibration']):
        regions.append(dict(zip(['region', 'budget', 'method', 'calibration'], key))
                       | {m: float(np.sum(group[m] * group.hours) / group.hours.sum()) for m in METRICS})
    regional = pd.DataFrame(regions)
    macro = regional.groupby(['budget', 'method', 'calibration']).agg(
        crps=('crps', 'mean'), mae=('mae', 'mean'), coverage=('coverage', 'mean'),
        worst_coverage=('coverage', 'min'), width=('width', 'mean'), winkler=('winkler', 'mean'),
        regions=('region', 'size')).reset_index()
    if not macro.regions.eq(29).all():
        raise ValueError('source denominator')
    saved_macro = pd.read_csv(NEW / 'macro_metrics.csv')
    match = saved_macro.merge(macro, on=['budget', 'method', 'calibration'], suffixes=('_saved', '_audit'), validate='one_to_one')
    if len(match) != 16:
        raise ValueError('macro method denominator')
    for metric in METRICS + ['worst_coverage']:
        if not np.allclose(match[metric + '_saved'], match[metric + '_audit'], rtol=0, atol=1e-9):
            raise ValueError('macro aggregation mismatch')
    contrasts = []
    for candidate in NEW_METHODS:
        for reference in sorted(set(macro.method) - {candidate}):
            a = macro.loc[macro.method.eq(candidate)]
            b = macro.loc[macro.method.eq(reference)]
            match = a.merge(b, on=['budget', 'calibration'], suffixes=('_a', '_b'), validate='one_to_one')
            for row in match.itertuples(index=False):
                contrasts.append({'candidate': candidate, 'reference': reference, 'budget': row.budget,
                                  'calibration': row.calibration,
                                  'crps_reduction_percent': 100 * (1 - row.crps_a / row.crps_b),
                                  'coverage': row.coverage_a, 'worst_coverage': row.worst_coverage_a,
                                  'width_change_percent': 100 * (row.width_a / row.width_b - 1),
                                  'winkler_change_percent': 100 * (row.winkler_a / row.winkler_b - 1),
                                  'passes_joint_gate': (row.coverage_a >= .9 and row.worst_coverage_a >= .85
                                                       and row.crps_a <= .95 * row.crps_b
                                                       and row.width_a <= row.width_b and row.winkler_a < row.winkler_b)})
    paired = []
    for budget in ('same_month', 'asof_30d'):
        candidates = macro.loc[macro.budget.eq(budget) & macro.calibration.eq('pooled')
                               & ~macro.method.eq('carbon_cdf_forest')]
        strongest = candidates.sort_values(['crps', 'method']).iloc[0].method
        references = list(dict.fromkeys([strongest, 'activity_mmd_forest', 'scalar_cdf_forest', 'mean_smoothed_forest']))
        a = allrows.loc[allrows.budget.eq(budget) & allrows.calibration.eq('pooled')
                        & allrows.method.eq('carbon_cdf_forest')]
        for reference in references:
            b = allrows.loc[allrows.budget.eq(budget) & allrows.calibration.eq('pooled') & allrows.method.eq(reference)]
            key = ['region', 'month']
            if set(a[key].itertuples(index=False, name=None)) != set(b[key].itertuples(index=False, name=None)):
                raise ValueError('paired query set changed')
            match = a.merge(b, on=key, suffixes=('_a', '_b'), validate='one_to_one')
            region_ids = sorted(match.region.unique())
            months = sorted(match.month.unique())
            if len(region_ids) != 29 or not match.hours_a.eq(match.hours_b).all():
                raise ValueError('paired source/hour denominator')
            matrix = np.zeros((29, len(months)))
            hours = np.zeros_like(matrix)
            for r, region in enumerate(region_ids):
                group = match.loc[match.region.eq(region)].set_index('month')
                for m, month in enumerate(months):
                    if month in group.index:
                        row = group.loc[month]
                        hours[r, m] = row.hours_a
                        matrix[r, m] = (row.crps_a - row.crps_b) * row.hours_a
            weights = np.random.default_rng(20261028).dirichlet(np.ones(len(months)), size=1000)
            distribution = ((matrix @ weights.T) / (hours @ weights.T)).mean(axis=0)
            paired.append({'budget': budget, 'reference': reference,
                           'carbon_cdf_minus_reference_crps': float(np.mean(matrix.sum(axis=1) / hours.sum(axis=1))),
                           'descriptive_95_lower': float(np.quantile(distribution, .025)),
                           'descriptive_95_upper': float(np.quantile(distribution, .975)),
                           'strongest_noncarbon_reference': strongest})
    role_map = pd.read_csv(BASE / 'region_roles.csv')
    fold_map = role_map.loc[role_map.role.eq('test'), ['region', 'fold']]
    fold_metrics = regional.merge(fold_map, on='region', validate='many_to_one').groupby(
        ['fold', 'budget', 'method', 'calibration']).agg(
            crps=('crps', 'mean'), coverage=('coverage', 'mean'), worst_coverage=('coverage', 'min'),
            width=('width', 'mean'), winkler=('winkler', 'mean'), regions=('region', 'size')).reset_index()
    OUT.mkdir(parents=True)
    for name, frame in [('all_macro_metrics', macro), ('all_region_metrics', regional),
                        ('comparisons', pd.DataFrame(contrasts)), ('paired_month_crps', pd.DataFrame(paired)),
                        ('fold_metrics', fold_metrics)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    result = {'assertion_failures': 0, 'scored_rows_verified': len(rows),
              'asof_mean_records_reproduced': len(reproduced), 'reproduction_max_errors': maximum_errors,
              'script_sha256': sha(Path(__file__)), 'archive_summary_hashes': archive_hashes,
              'output_hashes': {p.name: sha(p) for p in sorted(OUT.iterdir())},
              'scope': 'Source-only development and descriptive Dirichlet month reweighting, no blind confirmation.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(macro.loc[macro.calibration.eq('pooled')].to_string(index=False))


if __name__ == '__main__':
    main()
