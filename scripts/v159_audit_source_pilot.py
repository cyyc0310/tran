"""Independent archive/role/aggregation checks and complete source-pilot contrasts."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from hourly_fuel_data import HourlyFuelData
from v151_hourly_law import EF, sha

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
OUT = ROOT / 'results/v159-source-pilot-audit-r1'
METRICS = ['crps', 'mae', 'coverage', 'width', 'winkler']
PAIRS = [('positive_recomposition', 'legacy_recomposition'),
         ('activity_recomposition', 'positive_recomposition'),
         ('conditional_activity_recomposition', 'activity_recomposition'),
         ('raw_activity_forest', 'conditional_activity_recomposition'),
         ('projective_activity_forest', 'raw_activity_forest'),
         ('response_activity_forest', 'projective_activity_forest')]


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = json.loads((SOURCE / 'summary.json').read_text())
    if summary['assertion_failures'] or summary['completed_blocks'] != 12:
        raise ValueError('incomplete source run')
    for name, digest in summary['input_hashes'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('frozen source input changed')
    for name, digest in summary['output_hashes'].items():
        if sha(SOURCE / name) != digest:
            raise ValueError('source output changed')
    roles = pd.read_csv(SOURCE / 'region_roles.csv')
    if len(roles) != 174 or roles.duplicated(['fold', 'region']).any():
        raise ValueError('role denominator')
    if set(roles.region) & {'PACE', 'LGEE', 'SC'}:
        raise ValueError('development target in source pool')
    for role, count in [('test', 1), ('calibration', 1), ('fit', 4)]:
        if not roles.loc[roles.role.eq(role)].groupby('region').size().eq(count).all():
            raise ValueError('cross-fit role count')
    data = HourlyFuelData(ROOT)
    samples = pd.read_csv(SOURCE / 'source_sampling.csv')
    original = pd.read_csv(ROOT / 'results/v156-anchored-hourly-sampled-r1/source_sampling.csv')
    original = original.loc[original.target_region.eq('PACE') & original.seed.eq(41)
                            & original.region.isin(set(roles.region))]
    if samples.to_csv(index=False) != original.to_csv(index=False) or len(samples) != 58:
        raise ValueError('source sampling contract')
    query_count = 0
    methods = json.loads((SOURCE / 'run_manifest.json').read_text())['methods']
    for fold in range(6):
        roster = roles.loc[roles.fold.eq(fold)].set_index('region').role.to_dict()
        for budget in ('same_month', 'asof_30d'):
            block = SOURCE / f'fold_{fold}_{budget}'
            meta = json.loads((block / 'summary.json').read_text())
            for q in meta['prototype_profile_ids']:
                region = data.index.region.iloc[q]
                if roster[region] != 'fit' or not data.index.month.iloc[q].startswith('2024'):
                    raise ValueError('prototype identity leakage')
            queries = pd.read_csv(block / 'query_inputs.csv')
            expected = []
            for region, role in roster.items():
                if role == 'fit':
                    continue
                for q in data.groups[region]:
                    if data.index.month.iloc[q].startswith('2024') and data.local(int(q), budget) is not None:
                        expected.append(int(q))
            if set(queries.profile_id) != set(expected) or len(queries) != len(expected):
                raise ValueError('query denominator')
            for row in queries.itertuples(index=False):
                if roster[row.region] != row.role or data.local(row.profile_id, budget) != row.input_profile_id:
                    raise ValueError('query roles/input vintage')
            block_rows = pd.read_csv(block / 'test_month_metrics.csv')
            expected_keys = {(r.region, r.month, method, calibration)
                             for r in queries.loc[queries.role.eq('test')].itertuples(index=False)
                             for method in methods for calibration in ('pooled', 'max_region')}
            actual_keys = set(block_rows[['region', 'month', 'method', 'calibration']].itertuples(index=False, name=None))
            if actual_keys != expected_keys or len(block_rows) != len(expected_keys):
                raise ValueError('method/test-month completeness')
            query_count += len(queries)
    rows = pd.read_csv(SOURCE / 'test_month_metrics.csv')
    if rows.duplicated(['budget', 'region', 'month', 'method', 'calibration']).any():
        raise ValueError('duplicate source test')
    region_rows = []
    for key, group in rows.groupby(['region', 'budget', 'method', 'calibration']):
        region_rows.append(dict(zip(['region', 'budget', 'method', 'calibration'], key))
                           | {m: float(np.sum(group[m] * group.hours) / group.hours.sum()) for m in METRICS})
    regional = pd.DataFrame(region_rows)
    test_fold = roles.loc[roles.role.eq('test'), ['region', 'fold']]
    with_fold = regional.merge(test_fold, on='region', validate='many_to_one')
    fold_metrics = with_fold.groupby(['fold', 'budget', 'method', 'calibration']).agg(
        crps=('crps', 'mean'), coverage=('coverage', 'mean'), worst_coverage=('coverage', 'min'),
        width=('width', 'mean'), winkler=('winkler', 'mean'), regions=('region', 'size')).reset_index()
    macro = pd.read_csv(SOURCE / 'macro_metrics.csv')
    for row in macro.itertuples(index=False):
        group = regional.loc[regional.budget.eq(row.budget) & regional.method.eq(row.method)
                             & regional.calibration.eq(row.calibration)]
        if len(group) != 29:
            raise ValueError('macro source count')
        for metric in METRICS:
            if abs(group[metric].mean() - getattr(row, metric)) > 1e-9:
                raise ValueError('macro aggregation mismatch')
        if abs(group.coverage.min() - row.worst_coverage) > 1e-12:
            raise ValueError('worst source aggregation mismatch')
    comparisons = []
    pairs = PAIRS + [(candidate, reference)
                     for candidate in ('raw_activity_forest', 'projective_activity_forest', 'response_activity_forest')
                     for reference in ('legacy_recomposition', 'positive_recomposition',
                                       'activity_recomposition', 'conditional_activity_recomposition', 'scalar_forest')]
    pairs = list(dict.fromkeys(pairs))
    for candidate, reference in pairs:
        a = macro.loc[macro.method.eq(candidate)]
        b = macro.loc[macro.method.eq(reference)]
        match = a.merge(b, on=['budget', 'calibration'], suffixes=('_a', '_b'), validate='one_to_one')
        if len(match) != 4:
            raise ValueError('comparison denominator')
        for row in match.itertuples(index=False):
            comparisons.append({'candidate': candidate, 'reference': reference,
                                'budget': row.budget, 'calibration': row.calibration,
                                'crps_reduction_percent': 100 * (1 - row.crps_a / row.crps_b),
                                'coverage': row.coverage_a, 'worst_coverage': row.worst_coverage_a,
                                'width_change_percent': 100 * (row.width_a / row.width_b - 1),
                                'winkler_change_percent': 100 * (row.winkler_a / row.winkler_b - 1),
                                'passes_joint_gate': (row.coverage_a >= .9 and row.worst_coverage_a >= .85
                                                       and row.crps_a <= .95 * row.crps_b
                                                       and row.width_a <= row.width_b and row.winkler_a < row.winkler_b)})
    # Shared positive Dirichlet month weights preserve all sources when windows differ.
    paired = []
    for budget in ('same_month', 'asof_30d'):
        subset = macro.loc[macro.budget.eq(budget) & macro.calibration.eq('pooled')
                           & macro.method.isin(['legacy_recomposition', 'positive_recomposition', 'activity_recomposition',
                                                'conditional_activity_recomposition'])]
        strongest = subset.sort_values(['crps', 'method']).iloc[0].method
        for candidate in ('raw_activity_forest', 'projective_activity_forest', 'response_activity_forest'):
            a = rows.loc[rows.budget.eq(budget) & rows.calibration.eq('pooled') & rows.method.eq(candidate)]
            b = rows.loc[rows.budget.eq(budget) & rows.calibration.eq('pooled') & rows.method.eq(strongest)]
            a_keys = set(a[['region', 'month']].itertuples(index=False, name=None))
            b_keys = set(b[['region', 'month']].itertuples(index=False, name=None))
            if a_keys != b_keys or a.region.nunique() != 29:
                raise ValueError('paired source/month denominator')
            match = a.merge(b, on=['region', 'month'], suffixes=('_a', '_b'), validate='one_to_one')
            regions = sorted(match.region.unique())
            months = sorted(match.month.unique())
            matrix = np.zeros((len(regions), len(months)))
            hours = np.zeros_like(matrix)
            for r, region in enumerate(regions):
                group = match.loc[match.region.eq(region)].set_index('month')
                for m, month in enumerate(months):
                    if month in group.index:
                        row = group.loc[month]
                        if row.hours_a != row.hours_b:
                            raise ValueError('paired hours mismatch')
                        hours[r, m] = row.hours_a
                        matrix[r, m] = (row.crps_a - row.crps_b) * row.hours_a
            if np.any(hours.sum(axis=1) == 0):
                raise ValueError('empty paired source')
            rng = np.random.default_rng(20261028)
            month_weights = rng.dirichlet(np.ones(len(months)), size=1000)
            distribution = ((matrix @ month_weights.T) / (hours @ month_weights.T)).mean(axis=0)
            paired.append({'budget': budget, 'candidate': candidate, 'reference': strongest,
                           'candidate_minus_reference_crps': float(np.mean(matrix.sum(axis=1) / hours.sum(axis=1))),
                           'descriptive_95_lower': float(np.quantile(distribution, .025)),
                           'descriptive_95_upper': float(np.quantile(distribution, .975)),
                           'retained_replicates': len(distribution),
                           'description': 'Shared-calendar Dirichlet Bayesian bootstrap; development description only.'})
    # A reporting diagnostic only: constant input EF is identified for same-month,
    # whereas in asof it characterizes the past mix and is not a future guarantee.
    context = []
    for budget, region, month in rows[['budget', 'region', 'month']].drop_duplicates().itertuples(index=False, name=None):
        ids = data.groups[region]
        q = int(ids[data.index.month.iloc[ids].eq(month).to_numpy()][0])
        local = data.local(q, budget)
        active = data.shares[local] > 0
        span = float(np.ptp(EF[active]))
        context.append({'budget': budget, 'region': region, 'month': month,
                        'input_factor_span': span, 'input_constant_factor': span == 0})
    contexts = pd.DataFrame(context)
    annotated = rows.merge(contexts, on=['budget', 'region', 'month'], validate='many_to_one')
    strata = []
    for key, group in annotated.groupby(['budget', 'method', 'calibration', 'input_constant_factor']):
        per_region = [{m: float(np.average(g[m], weights=g.hours)) for m in METRICS}
                      for _, g in group.groupby('region')]
        frame = pd.DataFrame(per_region)
        strata.append(dict(zip(['budget', 'method', 'calibration', 'input_constant_factor'], key))
                      | {'regions': len(frame), 'months': len(group)}
                      | {m: float(frame[m].mean()) for m in METRICS})
    OUT.mkdir(parents=True)
    for name, frame in [('comparisons', pd.DataFrame(comparisons)), ('paired_month_crps', pd.DataFrame(paired)),
                        ('input_factor_contexts', contexts), ('factor_strata_metrics', pd.DataFrame(strata)),
                        ('fold_metrics', fold_metrics)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    result = {'assertion_failures': 0, 'verified_source_regions': 29, 'verified_role_rows': len(roles),
              'verified_queries': query_count, 'verified_test_scores': len(rows),
              'script_sha256': sha(Path(__file__)), 'source_summary_sha256': sha(SOURCE / 'summary.json'),
              'output_hashes': {p.name: sha(p) for p in sorted(OUT.iterdir())},
              'scope': 'Source-development description and archive/data-contract checks, not future-time validation.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(pd.DataFrame(comparisons).loc[lambda x: x.calibration.eq('pooled')].to_string(index=False))


if __name__ == '__main__':
    main()
