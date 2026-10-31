"""Reconstruct V180 laws, diagnostics, calibration and strong-reference comparisons."""
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
from rebased_coarse_lift import RebasedCoarseLift, METHODS, REFERENCES, SETTINGS, check_query_inputs
from v151_hourly_law import EF, sha
from v159_fuel_activity_source_pilot import weighted_radius
from v170_fuel_marginal_diagnosis import verify
from v173_moment_protected_risk import METRICS, aggregate
from v179_audit_increment_support import bisection_projection, compare_frames
from v180_rebased_coarse_lift import BASE, COARSE, OUT as SOURCE, ROOT

OUT = ROOT / 'results/v180-rebased-coarse-audit-r1'


def audit_fold(data, monthly, fold):
    frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
    predictor = RebasedCoarseLift(frozen, monthly, COARSE / f'fold_{fold}')
    block = SOURCE / f'fold_{fold}'
    rows = pd.read_csv(block / 'test_month_metrics.csv').set_index(['profile_id', 'method', 'radius_policy', 'calibration'])
    diagnostics = pd.read_csv(block / 'world_diagnostics.csv').set_index(['profile_id', 'method'])
    radii = pd.read_csv(block / 'calibration_radii.csv').set_index(['method', 'calibration'])
    if (len(rows) != int(frozen.queries.role.eq('test').sum()) * 40 or len(diagnostics) != len(frozen.queries) * 9
            or len(radii) != 22 or rows.index.duplicated().any() or diagnostics.index.duplicated().any() or radii.index.duplicated().any()):
        raise ValueError('fold inventory')
    source_cells = 0
    for shape, component, within, raw_ids in predictor.cells:
        profiles = predictor.profiles[component]
        means = data.M[profiles] / data.index.hours.to_numpy()[profiles, None]
        expected = np.ones_like(shape)
        np.divide(data.raw[raw_ids], means, out=expected, where=means > 0)
        if not np.array_equal(shape, expected):
            raise ValueError('independent source q differs')
        for j, p in enumerate(predictor.profiles):
            region = data.index.region.iloc[p]
            source = frozen.activity.sources[region]
            original = source['allowed'][source['profiles'] == p]
            take = raw_ids[component == j]
            if not set(take).issubset(set(original)) or (data.raw[take] < 0).any():
                raise ValueError('source hour budget/role')
            if max(abs(within[component == j] - 1 / len(take))) > 1e-14:
                raise ValueError('within-world probabilities')
            source_cells += 1
    # Verify exact clock selection, including the predeclared empty-clock fallback.
    for h, (_, component, _, raw_ids) in enumerate(predictor.cells):
        for j, p in enumerate(predictor.profiles):
            source = frozen.activity.sources[data.index.region.iloc[p]]
            original = source['allowed'][source['profiles'] == p]
            take = original[data.states[original] % 24 == h]
            if not len(take):
                take = original
            if not np.array_equal(raw_ids[component == j], take):
                raise ValueError('independent clock sample inventory differs')
    for q in frozen.queries.profile_id.iloc[[0, -1]]:
        check_query_inputs(predictor, int(q))
    errors = {m: {} for m in METHODS}
    checked = details_checked = physical = independent_crps_checks = 0
    score_error = diagnostic_error = projection_error = independent_crps_error = 0.
    exposure = []
    for role in ('calibration', 'test'):
        queries = frozen.queries.loc[frozen.queries.role.eq(role)]
        for rec in queries.itertuples(index=False):
            q = int(rec.profile_id)
            laws, forecast, details = predictor.build(q)
            local = monthly.local(q, 'asof_30d')
            S = monthly.shares[local]
            source_old = np.asarray([monthly.shares[monthly.local(int(p), 'asof_30d')] for p in predictor.profiles])
            delta = monthly.shares[predictor.profiles] - source_old
            delta[abs(delta) <= 64 * np.finfo(float).eps] = 0
            independent_support = bisection_projection(S + delta)
            for detail in details:
                method = detail['method']
                saved = diagnostics.loc[(q, method)]
                if (saved.fold, saved.role, saved.region, saved.month) != (fold, role, rec.region, rec.month):
                    raise ValueError('diagnostic identity')
                if json.loads(saved.source_profiles) != predictor.profiles.tolist():
                    raise ValueError('source profiles differ')
                values = np.asarray(json.loads(detail['coarse_support']))
                w = np.asarray(json.loads(detail['weights']))
                kind, key = SETTINGS[method]
                if kind == 'increment':
                    projection_error = max(projection_error, float(np.max(abs(values - independent_support))))
                elif not np.array_equal(values, np.broadcast_to(S if kind == 'past' else forecast['shares'], values.shape)):
                    raise ValueError('point support differs')
                if key == 'recomputed_gaussian':
                    p = np.asarray(json.loads(predictor.lookup.loc[(q, 'allmonth_hellinger')].weights))
                    logp = np.log(p) - .5 * ((values @ EF - forecast['anchor']) / forecast['sigma'])**2
                    expected_w = np.exp(np.maximum(logp - max(logp), -700))
                    expected_w /= expected_w.sum()
                else:
                    expected_w = np.asarray(json.loads(predictor.lookup.loc[(q, key)].weights))
                if max(abs(w - expected_w)) > 1e-12:
                    raise ValueError('independent probability differs')
                for name in ('weights', 'coarse_support', 'rejected_mass'):
                    a, b = np.asarray(json.loads(detail[name])), np.asarray(json.loads(saved[name]))
                    if a.shape != b.shape or not np.isfinite(b).all():
                        raise ValueError('diagnostic array dimensions')
                    diagnostic_error = max(diagnostic_error, float(np.max(abs(a - b))))
                missing = (monthly.shares[predictor.profiles] == 0) * values
                diagnostic_error = max(diagnostic_error, abs(float(w @ missing.sum(axis=1)) - saved.unknown_shape_energy_mass),
                                       abs(detail['max_rejected_mass'] - saved.max_rejected_mass))
                rejected = []
                for h, (shape, component, within, _) in enumerate(predictor.cells):
                    fuel = values[component] * shape
                    total = fuel.sum(axis=1)
                    probability = w[component] * within
                    good = total > 0
                    rejected.append(float(probability[~good].sum()))
                    x = fuel[good] @ EF / total[good]
                    order = np.argsort(x, kind='stable')
                    x, mass = x[order], probability[good][order]
                    cuts = np.r_[0, np.flatnonzero(np.diff(x)) + 1]
                    unique = x[cuts]
                    summed = np.add.reduceat(mass, cuts)
                    summed /= summed.sum()
                    law = laws[method][h]
                    if (not np.array_equal(unique, law.x) or max(abs(summed - law.w)) > 1e-12
                            or min(unique) < -1e-10 or max(unique) > 1000 + 1e-10):
                        raise ValueError('independent physical atom/mass differs')
                    # One deterministic synthetic outcome per cell; independently
                    # integrate the CDF on its sorted intervals to check CRPS.
                    target = (q * 17 + h * 31) % 1001
                    boundaries = np.unique(np.r_[unique, float(target)])
                    mid = (boundaries[:-1] + boundaries[1:]) / 2
                    cdf = np.r_[0., np.cumsum(summed)][np.searchsorted(unique, mid, side='right')]
                    independent = float(np.diff(boundaries) @ ((cdf - (mid >= target))**2))
                    independent_crps_error = max(independent_crps_error, abs(independent - law.crps(np.asarray([target]))[0]))
                    physical += 1
                    independent_crps_checks += 1
                diagnostic_error = max(diagnostic_error, float(max(abs(np.asarray(rejected) - np.asarray(json.loads(saved.rejected_mass))))))
                if role == 'test':
                    row = data.index.iloc[q]
                    clock = data.states[int(row.offset):int(row.stop)] % 24
                    exposure.append({'fold': fold, 'profile_id': q, 'region': rec.region, 'method': method,
                                     'hours': len(clock), 'rejected_mass': float(np.mean(np.asarray(rejected)[clock])),
                                     'max_rejected_mass': max(rejected), 'unknown_shape_energy_mass': saved.unknown_shape_energy_mass})
                details_checked += 1
            if projection_error > 1e-12 or diagnostic_error > 1e-8 or independent_crps_error > 1e-8:
                raise ValueError('support/diagnostic/independent CRPS error')
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
                            radius = radius_map[(method if policy == 'recalibrated' else 'world_gaussian', calibration)]
                            saved = rows.loc[(q, method, policy, calibration)]
                            if (saved.fold, saved.role, saved.region, saved.month, saved.hours) != (fold, role, rec.region, rec.month, len(actual)):
                                raise ValueError('score identity')
                            scores = score_laws(laws[method], actual, clock, radius)
                            score_error = max(score_error, abs(saved.radius - radius), max(abs(scores[m] - saved[m]) for m in scores))
                            checked += 1
        if role == 'calibration':
            radius_map = {}
            for method in METHODS:
                for label, radius in zip(('pooled', 'max_region'), weighted_radius(errors[method])):
                    if not np.isfinite(radii.loc[(method, label), 'radius']) or abs(radius - radii.loc[(method, label), 'radius']) > 1e-8:
                        raise ValueError('calibration radius differs')
                    radius_map[(method, label)] = radius
        print(f'audit rebased fold {fold} {role} complete', flush=True)
    if score_error > 1e-8 or checked != len(rows) or details_checked != len(diagnostics):
        raise ValueError('complete score reconstruction differs')
    return {'fold': fold, 'score_rows_rebuilt': checked, 'diagnostic_rows_rebuilt': details_checked,
            'independent_source_cells': source_cells, 'physical_cells': physical,
            'independent_crps_checks': independent_crps_checks, 'max_score_error': score_error,
            'max_diagnostic_error': diagnostic_error, 'max_projection_error': projection_error,
            'max_independent_crps_error': independent_crps_error}, exposure


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = verify(SOURCE)
    hashes = summary['input_hashes'] | {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                                      str((SOURCE / 'summary.json').relative_to(ROOT)): sha(SOURCE / 'summary.json')}
    for name in summary['output_hashes']:
        hashes[str((SOURCE / name).relative_to(ROOT))] = sha(SOURCE / name)
    external = {'positive_recomposition': BASE, 'scalar_forest': BASE, 'activity_mmd_forest': ROOT / 'results/v160-carbon-cdf-source-r1',
                'forecast_composition': ROOT / 'results/v164-scenario-reweighting-r1',
                'world_entropy': ROOT / 'results/v164-scenario-reweighting-r1',
                'hetero_history_barycentric': ROOT / 'results/v165-coarse-distribution-lifting-r1',
                'gaussian_pit_conditional43_a50': ROOT / 'results/v174-cif-recalibration-r1'}
    for directory in set(external.values()):
        verify(directory)
        for name in ('summary.json', 'test_month_metrics.csv'):
            hashes[str((directory / name).relative_to(ROOT))] = sha(directory / name)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('audit input changed')
    rows = pd.read_csv(SOURCE / 'test_month_metrics.csv')
    blocks = pd.concat([pd.read_csv(SOURCE / f'fold_{k}/test_month_metrics.csv') for k in range(6)], ignore_index=True)
    if (len(rows) != 9640 or rows.duplicated(['profile_id', 'method', 'radius_policy', 'calibration']).any()
            or not np.isfinite(rows[list(METRICS) + ['hours', 'radius']]).all().all()
            or (SOURCE / 'test_month_metrics.csv').read_text() != blocks.to_csv(index=False)):
        raise ValueError('complete root score inventory/serialization')
    regional, macro = aggregate(rows)
    compare_frames(regional, pd.read_csv(SOURCE / 'region_metrics.csv'), ['region', 'method', 'radius_policy', 'calibration'])
    compare_frames(macro, pd.read_csv(SOURCE / 'macro_metrics.csv'), ['cohort', 'method', 'radius_policy', 'calibration'])
    fixed = rows.loc[rows.radius_policy.eq('fixed')].merge(
        rows.loc[rows.method.eq('world_gaussian'), ['profile_id', 'calibration', 'radius']],
        on=['profile_id', 'calibration'], validate='many_to_one', suffixes=('_new', '_old'))
    if len(fixed) != 4338 or max(abs(fixed.radius_new - fixed.radius_old)) > 1e-8:
        raise ValueError('fixed radius control differs')
    data, monthly = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
    checks, exposure = [], []
    with threadpool_limits(limits=2):
        for fold in range(6):
            check, observed = audit_fold(data, monthly, fold)
            checks.append(check)
            exposure.extend(observed)
    tables = [regional]
    cohort_keys = set(rows[['region', 'month', 'calibration', 'hours']].itertuples(index=False, name=None))
    for method, directory in external.items():
        frame = pd.read_csv(directory / 'test_month_metrics.csv')
        frame = frame.loc[frame.method.eq(method) & frame.budget.eq('asof_30d')]
        if 'radius_policy' in frame:
            frame = frame.loc[frame.radius_policy.eq('recalibrated')]
        if (set(frame[['region', 'month', 'calibration', 'hours']].itertuples(index=False, name=None)) != cohort_keys
                or frame.duplicated(['region', 'month', 'calibration']).any()):
            raise ValueError('external reference cohort mismatch')
        tables.append(pd.DataFrame([{'region': region, 'method': method, 'calibration': calibration, 'radius_policy': 'recalibrated',
                                     **{m: float(np.average(group[m], weights=group.hours)) for m in METRICS[:5]}}
                                    for (region, calibration), group in frame.groupby(['region', 'calibration'])]))
    table = pd.concat(tables, ignore_index=True)
    comparisons = []
    rng = np.random.default_rng(180)
    for rec in macro.loc[~macro.method.isin(REFERENCES)].itertuples(index=False):
        selected = table.loc[table.calibration.eq(rec.calibration)]
        if rec.cohort == 'nonconstant24':
            selected = selected.loc[~selected.region.isin(['CHPD', 'DOPD', 'GVL', 'SCL', 'SPA'])]
        a = selected.loc[selected.method.eq(rec.method) & selected.radius_policy.eq(rec.radius_policy)].sort_values('region')
        peers = [(m, 'recalibrated', 'strong') for m in (*REFERENCES, *external)]
        peers += [(m, rec.radius_policy, 'same_architecture') for m in SETTINGS if m != rec.method]
        for method, policy, kind in peers:
            b = selected.loc[selected.method.eq(method) & selected.radius_policy.eq(policy)].sort_values('region')
            if not np.array_equal(a.region, b.region):
                raise ValueError('paired regions differ')
            delta = a.crps.to_numpy() - b.crps.to_numpy()
            boot = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
            comparisons.append({'cohort': rec.cohort, 'method': rec.method, 'radius_policy': rec.radius_policy, 'calibration': rec.calibration,
                                'reference': method, 'reference_radius_policy': policy, 'comparison': kind,
                                'paired_crps_difference': float(delta.mean()), 'region_bootstrap_lo': float(np.quantile(boot, .025)),
                                'region_bootstrap_hi': float(np.quantile(boot, .975)), 'regions_better_crps': int((delta < -1e-9).sum()),
                                'crps_reduction_percent': float(100 * (1 - rec.crps / b.crps.mean())),
                                'width_change_percent': float(100 * (rec.width / b.width.mean() - 1)),
                                'numeric_threshold_only': bool(rec.coverage >= .9 and rec.worst_coverage >= .85 and rec.crps <= .95 * b.crps.mean()
                                                               and rec.width <= b.width.mean() + 1e-9 and rec.winkler < b.winkler.mean())})
    exposures = pd.DataFrame(exposure)
    exposure_regions = pd.DataFrame([{'region': region, 'method': method,
                                     **{m: float(np.average(g[m], weights=g.hours)) for m in ('rejected_mass', 'unknown_shape_energy_mass')},
                                     'max_rejected_mass': float(g.max_rejected_mass.max())}
                                    for (region, method), g in exposures.groupby(['region', 'method'])])
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('audit inputs changed during run')
    OUT.mkdir(parents=True)
    for name, frame in [('reconstruction', pd.DataFrame(checks)), ('comparisons', pd.DataFrame(comparisons)),
                        ('exposure_queries', exposures), ('exposure_regions', exposure_regions)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    result = {'assertion_failures': 0, 'score_rows_rebuilt': sum(c['score_rows_rebuilt'] for c in checks),
              'diagnostic_rows_rebuilt': sum(c['diagnostic_rows_rebuilt'] for c in checks),
              'independent_source_cells': sum(c['independent_source_cells'] for c in checks),
              'physical_cells': sum(c['physical_cells'] for c in checks), 'independent_crps_checks': sum(c['independent_crps_checks'] for c in checks),
              'future_query_input_mutation_cases': 12, 'common_fixed_radius_rows': len(fixed),
              'max_score_error': max(c['max_score_error'] for c in checks), 'max_projection_error': max(c['max_projection_error'] for c in checks),
              'max_independent_crps_error': max(c['max_independent_crps_error'] for c in checks),
              'comparisons': len(comparisons), 'numeric_threshold_passes': sum(c['numeric_threshold_only'] for c in comparisons),
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in OUT.iterdir()},
              'scope': 'Complete laws/diagnostics/calibration/scores reconstruction; independent source q, additive projection, physical compression and synthetic-outcome CDF CRPS integration. Shared exact scoring/quantile helpers for formal metrics. Source-development comparisons only.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
