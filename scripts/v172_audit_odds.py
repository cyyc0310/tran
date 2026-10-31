"""Reconstruct complete V172 CIF laws, radii, scores, and same-budget comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from calendar_cif_law import CalendarLaw
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from level_spread_diagnostics import score_laws
from monthly_history_features import MonthlyInformation
from v151_hourly_law import sha
from v159_fuel_activity_source_pilot import weighted_radius
from v170_fuel_marginal_diagnosis import verify
from v172_zero_odds_cif import BASE, ZERO, METRICS, aggregate, check_mutation
from zero_odds_cif import ZeroOddsCIF, METHODS, REFERENCES, SETTINGS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v172-zero-odds-cif-r1'
OUT = ROOT / 'results/v172-zero-odds-audit-r1'
NONCONSTANT = ['CHPD', 'DOPD', 'GVL', 'SCL', 'SPA']


def audit_fold(data, fold):
    frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
    monthly = MonthlyInformation(ROOT)
    predictor = ZeroOddsCIF(frozen, monthly, ZERO / f'fold_{fold}')
    block = SOURCE / f'fold_{fold}'
    rows = pd.read_csv(block / 'test_month_metrics.csv').set_index(['profile_id', 'method', 'radius_policy', 'calibration'])
    saved_context = pd.read_csv(block / 'contexts.csv').set_index('profile_id')
    saved_updates = pd.read_csv(block / 'zero_updates.csv').set_index(['profile_id', 'method'])
    saved_radii = pd.read_csv(block / 'calibration_radii.csv').set_index(['method', 'calibration'])
    if (set(saved_context.index) != set(frozen.queries.profile_id) or saved_context.index.duplicated().any()
            or saved_updates.index.duplicated().any()
            or not np.isfinite(saved_context[['raw_anchor', 'anchor', 'past_anchor', 'sigma', 'local']]).all().all()
            or not np.isfinite(saved_radii.radius).all()):
        raise ValueError('diagnostic inventory')
    for q in frozen.queries.profile_id.iloc[[0, -1]]:
        check_mutation(predictor, data, monthly, int(q))
    errors = {name: {} for name in METHODS}
    max_score = max_radius = max_diagnostic = 0.
    checked = updated = physical_support = 0
    for role in ('calibration', 'test'):
        for rec in frozen.queries.loc[frozen.queries.role.eq(role)].itertuples(index=False):
            q = int(rec.profile_id)
            laws, forecast, updates = predictor.build(q)
            context = saved_context.loc[q]
            if context.role != role or context.region != rec.region or context.month != rec.month or context.fold != fold:
                raise ValueError('context identity')
            for key in ('raw_anchor', 'anchor', 'past_anchor', 'sigma', 'local'):
                max_diagnostic = max(max_diagnostic, abs(float(forecast[key]) - float(context[key])))
            for detail in updates:
                old = saved_updates.loc[(q, detail['method'])]
                if old.role != role or old.region != rec.region or old.month != rec.month or old.fold != fold:
                    raise ValueError('update identity')
                for key, value in detail.items():
                    if key == 'method':
                        continue
                    delta = (np.asarray(json.loads(value)) - np.asarray(json.loads(old[key]))) if isinstance(value, str) else np.asarray(value - old[key])
                    if not np.isfinite(delta).all():
                        raise ValueError('nonfinite update diagnostics')
                    max_diagnostic = max(max_diagnostic, float(np.max(abs(delta))))
                updated += 1
            for method, (reference, _, _) in SETTINGS.items():
                for new, old in zip(laws[method], laws[reference]):
                    if not np.array_equal(new.x, old.x) or not (new.w > 0).all():
                        raise ValueError('physical support or positive weights changed')
                    physical_support += 1
            actual = data.truth(q)
            row = data.index.iloc[q]
            clock = data.states[int(row.offset):int(row.stop)]
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
                            old = rows.loc[(q, method, policy, calibration)]
                            if old.region != rec.region or old.month != rec.month or old.fold != fold or old.hours != len(actual):
                                raise ValueError('score identity')
                            new = score_laws(laws[method], actual, clock, radius)
                            delta = np.asarray([new[key] - old[key] for key in new])
                            if not np.isfinite(delta).all():
                                raise ValueError('nonfinite score reconstruction')
                            max_score = max(max_score, float(np.max(abs(delta))))
                            max_radius = max(max_radius, abs(radius - old.radius))
                            checked += 1
        if role == 'calibration':
            radius_map = {}
            for method in METHODS:
                for label, radius in zip(('pooled', 'max_region'), weighted_radius(errors[method])):
                    radius_map[(method, label)] = radius
                    max_radius = max(max_radius, abs(radius - saved_radii.loc[(method, label), 'radius']))
    if (checked != len(rows) or updated != len(saved_updates) or max_score > 1e-7
            or max_radius > 1e-8 or max_diagnostic > 1e-8):
        raise ValueError('complete reconstruction failed')
    print(f'audited CIF odds fold {fold}: {checked} scores', flush=True)
    return {'fold': fold, 'rebuilt_score_rows': checked, 'verified_update_rows': updated,
            'physical_support_checks': physical_support, 'max_score_error': max_score,
            'max_radius_error': max_radius, 'max_diagnostic_error': max_diagnostic}


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
                         'hetero_history_barycentric': ROOT / 'results/v165-coarse-distribution-lifting-r1',
                         **{m: ROOT / 'results/v167-masked-monthly-growth-r1' for m in
                            ['mask_reweighted', 'neutral_share', 'neutral_energy', 'neutral_share_gaussian', 'neutral_energy_gaussian']}}
    for directory in set(reference_sources.values()):
        verify(directory)
        for name in ('summary.json', 'test_month_metrics.csv'):
            hashes[str((directory / name).relative_to(ROOT))] = sha(directory / name)
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('input changed before audit')
    rows = pd.read_csv(SOURCE / 'test_month_metrics.csv')
    if (len(rows) != 16388 or rows.duplicated(['profile_id', 'method', 'radius_policy', 'calibration']).any()
            or not np.isfinite(rows[list(METRICS) + ['radius', 'hours']]).all().all()):
        raise ValueError('score denominator/finiteness')
    blocks = pd.concat([pd.read_csv(SOURCE / f'fold_{f}/test_month_metrics.csv') for f in range(6)], ignore_index=True)
    if (SOURCE / 'test_month_metrics.csv').read_text() != blocks.to_csv(index=False):
        raise ValueError('root serialization does not match parsed blocks')
    regional, macro = aggregate(rows)
    for label, new, keys in [('region_metrics', regional, ['region', 'method', 'radius_policy', 'calibration']),
                             ('macro_metrics', macro, ['cohort', 'method', 'radius_policy', 'calibration'])]:
        old = pd.read_csv(SOURCE / f'{label}.csv')
        pair = new.merge(old, on=keys, validate='one_to_one', suffixes=('_a', '_b'))
        fields = list(METRICS) + (['worst_coverage', 'regions'] if label == 'macro_metrics' else [])
        if (len(pair) != len(old) or len(pair) != len(new) or not np.isfinite(old[fields]).all().all()
                or any(max(abs(pair[m + '_a'] - pair[m + '_b'])) > 1e-8 for m in fields)):
            raise ValueError('aggregate reconstruction')
    data = HourlyFuelData(ROOT)
    with threadpool_limits(limits=2):
        checks = [audit_fold(data, fold) for fold in range(6)]
    all_regions = [regional]
    cohort_keys = set(rows[['region', 'month', 'calibration', 'hours']].itertuples(index=False, name=None))
    for method, directory in reference_sources.items():
        frame = pd.read_csv(directory / 'test_month_metrics.csv')
        frame = frame.loc[frame.method.eq(method) & frame.budget.eq('asof_30d')]
        if 'radius_policy' in frame:
            frame = frame.loc[frame.radius_policy.eq('recalibrated')]
        if (set(frame[['region', 'month', 'calibration', 'hours']].itertuples(index=False, name=None)) != cohort_keys
                or frame.duplicated(['region', 'month', 'calibration']).any()
                or not np.isfinite(frame[list(METRICS[:5])]).all().all()):
            raise ValueError('strong reference cohort mismatch')
        rr = pd.DataFrame([{'region': key[0], 'method': method, 'calibration': key[1], 'radius_policy': 'recalibrated',
                            **{m: float(np.average(g[m], weights=g.hours)) for m in METRICS[:5]}}
                           for key, g in frame.groupby(['region', 'calibration'])])
        all_regions.append(rr)
    table = pd.concat(all_regions, ignore_index=True)
    rng = np.random.default_rng(172)
    comparisons = []
    for rec in macro.loc[~macro.method.isin(REFERENCES)].itertuples(index=False):
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
        raise ValueError('input changed during audit')
    verify(SOURCE)
    OUT.mkdir(parents=True)
    pd.DataFrame(checks).to_csv(OUT / 'reconstruction.csv', index=False)
    pd.DataFrame(comparisons).to_csv(OUT / 'comparisons.csv', index=False)
    result = {'assertion_failures': 0, 'rebuilt_score_rows': sum(c['rebuilt_score_rows'] for c in checks),
              'verified_update_rows': sum(c['verified_update_rows'] for c in checks),
              'physical_support_checks': sum(c['physical_support_checks'] for c in checks),
              'future_month_mutation_cases': 12, 'max_score_error': max(c['max_score_error'] for c in checks),
              'comparisons': len(comparisons), 'numeric_threshold_passes': sum(c['numeric_threshold_only'] for c in comparisons),
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in OUT.iterdir()},
              'scope': 'All CIF laws, diagnostics, calibration and scores reconstructed with frozen helpers; not an independent scoring algorithm. Exact physical supports verified. Region bootstrap is descriptive source-development evidence.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_hashes', 'output_hashes')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
