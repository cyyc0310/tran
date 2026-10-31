"""Reconstruct complete V173 CIF laws, radii, scores, and same-budget comparisons."""
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
from v151_hourly_law import EF, sha
from v159_fuel_activity_source_pilot import weighted_radius
from v170_fuel_marginal_diagnosis import verify
from v173_moment_protected_risk import BASE, ZERO, METRICS, aggregate, check_mutation, UNPROTECTED
from moment_protected_cif import MomentProtectedCIF, METHODS, REFERENCES, SETTINGS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v173-moment-protected-risk-r1'
OUT = ROOT / 'results/v173-moment-protected-audit-r1'
NONCONSTANT = ['CHPD', 'DOPD', 'GVL', 'SCL', 'SPA']


def unprotected_name(method):
    reference, zero, alpha, _ = SETTINGS[method]
    prefix = 'conditional' if reference == REFERENCES[0] else 'gaussian'
    return f'{prefix}_odds_{zero}_a{int(alpha * 100)}'


def audit_fold(data, fold):
    frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
    monthly = MonthlyInformation(ROOT)
    predictor = MomentProtectedCIF(frozen, monthly, ZERO / f'fold_{fold}')
    block = SOURCE / f'fold_{fold}'
    rows = pd.read_csv(block / 'test_month_metrics.csv').set_index(['profile_id', 'method', 'radius_policy', 'calibration'])
    saved_context = pd.read_csv(block / 'contexts.csv').set_index('profile_id')
    saved_updates = pd.read_csv(block / 'zero_updates.csv').set_index(['profile_id', 'method'])
    saved_radii = pd.read_csv(block / 'calibration_radii.csv').set_index(['method', 'calibration'])
    saved_certificates = pd.read_csv(block / 'projection_certificates.csv').set_index(['profile_id', 'method', 'state'])
    numeric = ['rank', 'iterations', 'moment_error', 'generation_error', 'emissions_error']
    success = ~saved_certificates.used_prior_fallback
    if (len(saved_certificates) != len(frozen.queries) * 768 or saved_certificates.index.duplicated().any()
            or not np.isfinite(saved_certificates[numeric]).all().all()
            or not np.isfinite(saved_certificates.loc[success, ['stationarity_error', 'pythagorean_error']]).all().all()
            or saved_certificates.moment_error.max() > 1e-10
            or saved_certificates[['generation_error', 'emissions_error']].to_numpy().max() > 2e-10
            or saved_certificates.loc[success, ['stationarity_error', 'pythagorean_error']].to_numpy().max() > 1e-8):
        raise ValueError('projection certificate inventory or thresholds')
    if (set(saved_context.index) != set(frozen.queries.profile_id) or saved_context.index.duplicated().any()
            or saved_updates.index.duplicated().any()
            or not np.isfinite(saved_context[['raw_anchor', 'anchor', 'past_anchor', 'sigma', 'local']]).all().all()
            or not np.isfinite(saved_radii.radius).all()):
        raise ValueError('diagnostic inventory')
    for q in frozen.queries.profile_id.iloc[[0, -1]]:
        check_mutation(predictor, data, monthly, int(q))
    errors = {name: {} for name in METHODS}
    max_score = max_radius = max_diagnostic = 0.
    checked = updated = physical_support = certified = 0
    for role in ('calibration', 'test'):
        for rec in frozen.queries.loc[frozen.queries.role.eq(role)].itertuples(index=False):
            q = int(rec.profile_id)
            laws, forecast, updates, certificates = predictor.build(q)
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
                before, after = [np.asarray(json.loads(detail[key])) for key in ('fuel_before', 'fuel_after')]
                denominator = before.sum()
                if (denominator <= 0 or abs(after.sum() - denominator) / denominator > 2e-10
                        or abs((after - before) @ EF / 1000) / denominator > 2e-10):
                    raise ValueError('independent aggregated GE moment arithmetic')
                if SETTINGS[detail['method']][3] == 'fuel8' and max(abs(after - before)) / denominator > 2e-10:
                    raise ValueError('independent aggregated fuel moment arithmetic')
            for certificate in certificates:
                old = saved_certificates.loc[(q, certificate['method'], certificate['state'])]
                if old.role != role or old.region != rec.region or old.month != rec.month or old.fold != fold:
                    raise ValueError('projection certificate identity')
                for key, value in certificate.items():
                    if key in ('state', 'method'):
                        continue
                    if value is None:
                        if not pd.isna(old[key]):
                            raise ValueError('fallback falsely reports an optimality certificate')
                    elif isinstance(value, (str, bool)):
                        if old[key] != value:
                            raise ValueError('projection status differs')
                    elif not np.isfinite(old[key]) or abs(value - old[key]) > 1e-12:
                        raise ValueError('projection numerical certificate differs')
                certified += 1
            for method, (reference, _, _, _) in SETTINGS.items():
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
    if (checked != len(rows) or updated != len(saved_updates) or certified != len(saved_certificates) or max_score > 1e-7
            or max_radius > 1e-8 or max_diagnostic > 1e-8):
        raise ValueError('complete reconstruction failed')
    print(f'audited protected CIF fold {fold}: {checked} scores', flush=True)
    return {'fold': fold, 'rebuilt_score_rows': checked, 'verified_update_rows': updated,
            'verified_projection_certificates': certified, 'projection_fallbacks': int(saved_certificates.used_prior_fallback.sum()),
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
    standard_reference_names = list(reference_sources)
    matched_reference_names = {unprotected_name(method) for method in SETTINGS}
    reference_sources.update({name: UNPROTECTED for name in matched_reference_names})
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
    old_unprotected = pd.read_csv(UNPROTECTED / 'test_month_metrics.csv').rename(columns={'method': 'reference_method'})
    matched_rows = rows.loc[~rows.method.isin(REFERENCES)].copy()
    matched_rows['reference_method'] = matched_rows.method.map(unprotected_name)
    matched = matched_rows.merge(old_unprotected,
        on=['profile_id', 'reference_method', 'radius_policy', 'calibration'],
        suffixes=('_new', '_old'), validate='many_to_one')
    fixed = matched.loc[matched.radius_policy.eq('fixed')]
    if (len(matched) != len(matched_rows) or max(abs(fixed.radius_new - fixed.radius_old)) > 1e-8
            or not np.array_equal(matched.region_new, matched.region_old)
            or not np.array_equal(matched.month_new, matched.month_old)
            or not np.array_equal(matched.hours_new, matched.hours_old)):
        raise ValueError('matched odds cohort or fixed-radius mismatch')
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
        if 'radius_policy' not in frame:
            frame = frame.assign(radius_policy='recalibrated')
        elif method not in matched_reference_names:
            frame = frame.loc[frame.radius_policy.eq('recalibrated')]
        if (set(frame[['region', 'month', 'calibration', 'hours']].itertuples(index=False, name=None)) != cohort_keys
                or frame.duplicated(['region', 'month', 'calibration', 'radius_policy']).any()
                or not np.isfinite(frame[list(METRICS[:5])]).all().all()):
            raise ValueError('strong reference cohort mismatch')
        rr = pd.DataFrame([{'region': key[0], 'method': method, 'calibration': key[1], 'radius_policy': key[2],
                            **{m: float(np.average(g[m], weights=g.hours)) for m in METRICS[:5]}}
                           for key, g in frame.groupby(['region', 'calibration', 'radius_policy'])])
        all_regions.append(rr)
    table = pd.concat(all_regions, ignore_index=True)
    rng = np.random.default_rng(173)
    comparisons = []
    for rec in macro.loc[~macro.method.isin(REFERENCES)].itertuples(index=False):
        selected = table.loc[table.calibration.eq(rec.calibration)]
        if rec.cohort == 'nonconstant24':
            selected = selected.loc[~selected.region.isin(NONCONSTANT)]
        a = selected.loc[selected.method.eq(rec.method) & selected.radius_policy.eq(rec.radius_policy)].sort_values('region')
        for reference in (*REFERENCES, *standard_reference_names, unprotected_name(rec.method)):
            reference_policy = rec.radius_policy if reference == unprotected_name(rec.method) else 'recalibrated'
            b = selected.loc[selected.method.eq(reference) & selected.radius_policy.eq(reference_policy)].sort_values('region')
            if not np.array_equal(a.region, b.region) or len(a) != rec.regions:
                raise ValueError('paired strong reference inventory')
            delta = a.crps.to_numpy() - b.crps.to_numpy()
            boot = delta[rng.integers(len(delta), size=(3000, len(delta)))].mean(axis=1)
            comparisons.append({'cohort': rec.cohort, 'method': rec.method, 'radius_policy': rec.radius_policy,
                                'calibration': rec.calibration, 'reference': reference, 'reference_radius_policy': reference_policy,
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
    protection_pairs = []
    for rec in macro.loc[macro.method.str.contains('_protect_GE_')].itertuples(index=False):
        peer = rec.method.replace('_protect_GE_', '_protect_fuel8_')
        selected = regional.loc[regional.calibration.eq(rec.calibration) & regional.radius_policy.eq(rec.radius_policy)]
        if rec.cohort == 'nonconstant24':
            selected = selected.loc[~selected.region.isin(NONCONSTANT)]
        a, b = [selected.loc[selected.method.eq(name)].sort_values('region') for name in (rec.method, peer)]
        if not np.array_equal(a.region, b.region):
            raise ValueError('GE/fuel8 paired region inventory')
        delta = a.crps.to_numpy() - b.crps.to_numpy()
        boot = delta[rng.integers(len(delta), size=(3000, len(delta)))].mean(axis=1)
        protection_pairs.append({'cohort': rec.cohort, 'calibration': rec.calibration, 'radius_policy': rec.radius_policy,
                                 'GE_method': rec.method, 'fuel8_method': peer, 'crps_GE_minus_fuel8': float(delta.mean()),
                                 'region_bootstrap_lo': float(np.quantile(boot, .025)), 'region_bootstrap_hi': float(np.quantile(boot, .975)),
                                 'width_GE_minus_fuel8': float(a.width.mean() - b.width.mean()),
                                 'worst_coverage_GE': float(a.coverage.min()), 'worst_coverage_fuel8': float(b.coverage.min())})
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('input changed during audit')
    verify(SOURCE)
    OUT.mkdir(parents=True)
    pd.DataFrame(checks).to_csv(OUT / 'reconstruction.csv', index=False)
    pd.DataFrame(comparisons).to_csv(OUT / 'comparisons.csv', index=False)
    pd.DataFrame(protection_pairs).to_csv(OUT / 'protection_pairs.csv', index=False)
    certs = pd.concat([pd.read_csv(SOURCE / f'fold_{f}/projection_certificates.csv') for f in range(6)], ignore_index=True)
    fallback = certs.groupby(['role', 'region', 'method']).agg(calls=('used_prior_fallback', 'size'),
                                                             prior_fallbacks=('used_prior_fallback', 'sum')).reset_index()
    fallback['fallback_fraction'] = fallback.prior_fallbacks / fallback.calls
    fallback.to_csv(OUT / 'fallback_by_region_method.csv', index=False)
    result = {'assertion_failures': 0, 'rebuilt_score_rows': sum(c['rebuilt_score_rows'] for c in checks),
              'verified_update_rows': sum(c['verified_update_rows'] for c in checks),
              'physical_support_checks': sum(c['physical_support_checks'] for c in checks),
              'verified_projection_certificates': sum(c['verified_projection_certificates'] for c in checks),
              'projection_prior_fallbacks': sum(c['projection_fallbacks'] for c in checks),
              'GE_fuel8_pairs': len(protection_pairs),
              'matched_fixed_radius_rows': len(fixed),
              'future_month_mutation_cases': 12, 'max_score_error': max(c['max_score_error'] for c in checks),
              'comparisons': len(comparisons), 'numeric_threshold_passes': sum(c['numeric_threshold_only'] for c in comparisons),
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in OUT.iterdir()},
              'scope': 'All protected CIF laws, certificates, diagnostics, calibration and scores reconstructed with frozen helpers; not an independent scoring algorithm. Exact physical supports verified. Region bootstrap is descriptive source-development evidence.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_hashes', 'output_hashes')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
