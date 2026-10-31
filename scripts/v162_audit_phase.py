"""Archive, oracle-selection and matched-cohort audit of fixed-phase headroom."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from v151_hourly_law import sha

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / 'results/v162-fixed-phase-diagnostic-r1'
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
OUT = ROOT / 'results/v162-fixed-phase-audit-r2'
KEYS = ['region', 'budget', 'month', 'method', 'calibration']
METRICS = ['crps', 'mae', 'coverage', 'width', 'winkler']


def best_phase(group):
    eligible = group.loc[group.crps <= group.crps.min() + 1e-10].copy()
    eligible['signed'] = np.where(eligible.phase <= 12, eligible.phase, eligible.phase - 24)
    eligible['distance'] = eligible.signed.abs()
    return int(eligible.sort_values(['distance', 'signed']).iloc[0].phase)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    input_summaries = {}
    for archive in (NEW, BASE):
        summary = json.loads((archive / 'summary.json').read_text())
        if summary['assertion_failures'] or summary['completed_blocks'] != 12:
            raise ValueError('upstream not complete')
        for name, digest in summary['input_hashes'].items():
            if sha(ROOT / name) != digest:
                raise ValueError('input changed')
        for name, digest in summary['output_hashes'].items():
            if sha(archive / name) != digest:
                raise ValueError('output changed')
        input_summaries[str(archive.relative_to(ROOT))] = sha(archive / 'summary.json')
    rows = pd.read_csv(NEW / 'phase_month_metrics.csv')
    old = pd.read_csv(BASE / 'test_month_metrics.csv')
    old = old.loc[old.method.isin(['positive_recomposition', 'conditional_activity_recomposition'])]
    if len(rows) != 54528 or rows.duplicated(KEYS + ['phase']).any():
        raise ValueError('phase denominator')
    expected = set(old[KEYS].itertuples(index=False, name=None))
    if len(expected) != 2272:
        raise ValueError('reference denominator')
    for phase, group in rows.groupby('phase'):
        if set(group[KEYS].itertuples(index=False, name=None)) != expected:
            raise ValueError('phase query completeness')
    match = rows.loc[rows.phase.eq(0)].merge(old, on=KEYS, suffixes=('_new', '_old'), validate='one_to_one')
    errors = {m: float(np.max(np.abs(match[m + '_new'] - match[m + '_old']))) for m in METRICS}
    if max(errors.values()) > 1e-7:
        raise ValueError('baseline reproduction')
    invariance = rows.groupby(KEYS).calendar_width.agg(lambda v: v.max() - v.min()).max()
    if invariance > 1e-9:
        raise ValueError('calendar width invariance')
    policies = pd.read_csv(NEW / 'phase_policies.csv')
    policy_keys = ['region', 'budget', 'method', 'case']
    if len(policies) != 348 or policies.duplicated(policy_keys).any():
        raise ValueError('phase policy denominator')
    phase_region = pd.read_csv(NEW / 'region_phase_metrics.csv')
    pooled = phase_region.loc[phase_region.calibration.eq('pooled')]
    for policy in policies.itertuples(index=False):
        group = pooled.loc[pooled.region.eq(policy.region) & pooled.budget.eq(policy.budget) & pooled.method.eq(policy.method)]
        if policy.case == 'region_budget_method_oracle':
            expected_phase = best_phase(group)
        elif policy.case == 'shared_region_oracle':
            shared = pooled.loc[pooled.region.eq(policy.region)
                                & pooled.method.eq('conditional_activity_recomposition')].groupby('phase').crps.mean().reset_index()
            expected_phase = best_phase(shared)
        elif policy.case == 'H1_phase_H2':
            first = rows.loc[rows.region.eq(policy.region) & rows.budget.eq(policy.budget)
                             & rows.method.eq(policy.method) & rows.calibration.eq('pooled')
                             & (pd.to_datetime(rows.month).dt.month <= 6)]
            if len(first):
                scores = first.groupby('phase').apply(lambda g: np.average(g.crps, weights=g.hours), include_groups=False).rename('crps').reset_index()
                expected_phase = best_phase(scores)
            else:
                expected_phase = 0
            if int(policy.h1_months) != first.month.nunique():
                raise ValueError('H1 month count')
        else:
            raise ValueError('unknown diagnostic policy')
        if policy.phase != expected_phase:
            raise ValueError('incorrect oracle/stability selection')
    selected = pd.read_csv(NEW / 'selected_month_metrics.csv')
    if selected.duplicated(KEYS + ['case']).any():
        raise ValueError('selected-case duplicates')
    h2keys = {key for key in expected if int(key[2][5:7]) >= 7}
    for case, group in selected.groupby('case'):
        target_keys = h2keys if case in ('H1_phase_H2', 'H2_unshifted') else expected
        if set(group[KEYS].itertuples(index=False, name=None)) != target_keys:
            raise ValueError('case query/cohort mismatch')
    oracle_rows = selected.loc[~selected.case.isin(['unshifted', 'H2_unshifted'])]
    bound = oracle_rows.merge(policies[['region', 'budget', 'method', 'case', 'phase']],
                              on=['region', 'budget', 'method', 'case'], suffixes=('_scored', '_policy'),
                              validate='many_to_one')
    if len(bound) != len(oracle_rows) or not bound.phase_scored.eq(bound.phase_policy).all():
        raise ValueError('selected results do not use selected policy')
    if not selected.loc[selected.case.isin(['unshifted', 'H2_unshifted']), 'phase'].eq(0).all():
        raise ValueError('unshifted case uses a nonzero phase')
    scores = METRICS + ['calendar_width', 'shuffle_crps', 'hours', 'fold', 'profile_id']
    bound = selected.merge(rows[KEYS + ['phase'] + scores], on=KEYS + ['phase'],
                           suffixes=('_selected', '_grid'), validate='many_to_one')
    if len(bound) != len(selected):
        raise ValueError('selected score grid binding denominator')
    for metric in scores:
        if not np.allclose(bound[metric + '_selected'], bound[metric + '_grid'], atol=1e-9, rtol=0):
            raise ValueError('selected score differs from phase grid')
    macro = pd.read_csv(NEW / 'macro_metrics.csv')
    recomputed = []
    for key, group in selected.groupby(['case', 'budget', 'method', 'calibration']):
        regional = []
        for _, source in group.groupby('region'):
            regional.append({m: float(np.sum(source[m] * source.hours) / source.hours.sum())
                             for m in METRICS + ['calendar_width', 'shuffle_crps']})
        frame = pd.DataFrame(regional)
        recomputed.append(dict(zip(['case', 'budget', 'method', 'calibration'], key))
                          | {m: float(frame[m].mean()) for m in frame.columns}
                          | {'worst_coverage': float(frame.coverage.min()), 'regions': len(frame)})
    joined_macro = macro.merge(pd.DataFrame(recomputed), on=['case', 'budget', 'method', 'calibration'],
                               suffixes=('_saved', '_audit'), validate='one_to_one')
    if len(joined_macro) != len(macro):
        raise ValueError('macro recomputation denominator')
    for metric in METRICS + ['calendar_width', 'shuffle_crps', 'worst_coverage', 'regions']:
        if not np.allclose(joined_macro[metric + '_saved'], joined_macro[metric + '_audit'], atol=1e-9, rtol=0):
            raise ValueError('macro recomputation mismatch')
    comparisons = []
    for case in ('region_budget_method_oracle', 'shared_region_oracle', 'H1_phase_H2'):
        reference = 'H2_unshifted' if case == 'H1_phase_H2' else 'unshifted'
        a = macro.loc[macro.case.eq(case)]
        b = macro.loc[macro.case.eq(reference)]
        joined = a.merge(b, on=['budget', 'method', 'calibration'], suffixes=('_a', '_b'), validate='one_to_one')
        for row in joined.itertuples(index=False):
            if row.regions_a != row.regions_b:
                raise ValueError('macro cohort source count mismatch')
            comparisons.append({'case': case, 'reference': reference, 'budget': row.budget,
                                'method': row.method, 'calibration': row.calibration, 'regions': row.regions_a,
                                'crps_reduction_percent': 100 * (1 - row.crps_a / row.crps_b),
                                'coverage': row.coverage_a, 'worst_coverage': row.worst_coverage_a,
                                'width_change_percent': 100 * (row.width_a / row.width_b - 1),
                                'winkler_change_percent': 100 * (row.winkler_a / row.winkler_b - 1),
                                'shuffle_crps_change': row.shuffle_crps_a - row.shuffle_crps_b,
                                'diagnostic_joint_threshold': (row.crps_a <= .95 * row.crps_b
                                                              and row.coverage_a >= .9 and row.worst_coverage_a >= .85
                                                              and row.width_a <= row.width_b + 1e-9
                                                              and row.winkler_a < row.winkler_b)})
    upper = pd.read_csv(NEW / 'coverage_upper_limits.csv')
    unresolved = upper.loc[upper.maximum_fixed_phase_coverage < .85].copy()
    OUT.mkdir(parents=True)
    pd.DataFrame(comparisons).to_csv(OUT / 'phase_comparisons.csv', index=False)
    unresolved.to_csv(OUT / 'phase_cannot_reach_85.csv', index=False)
    result = {'assertion_failures': 0, 'verified_phase_grid_rows': len(rows), 'verified_phase_zero_rows': len(match),
              'phase_zero_max_errors': errors, 'calendar_width_max_range': float(invariance),
              'verified_policies': len(policies), 'h2_source_count': int(selected.loc[selected.case.eq('H2_unshifted'), 'region'].nunique()),
              'source_summary_hashes': input_summaries, 'script_sha256': sha(Path(__file__)),
              'output_hashes': {p.name: sha(p) for p in sorted(OUT.iterdir())},
              'scope': 'Oracle/stability diagnostic integrity, not deployable performance or promotion.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(pd.DataFrame(comparisons).loc[lambda d: d.calibration.eq('pooled')].to_string(index=False))


if __name__ == '__main__':
    main()
