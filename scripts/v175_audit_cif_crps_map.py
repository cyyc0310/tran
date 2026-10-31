"""Refit direct CRPS maps, verify certificates and rebuild complete CIF comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from calendar_cif_law import CalendarLaw
from cif_crps_recalibration import CRPSRecalibration, METHODS, REFERENCES, SETTINGS, GRID, check_maps
from cif_crps_quadratic import unpack
from scipy.optimize import nnls
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from level_spread_diagnostics import score_laws
from monthly_history_features import MonthlyInformation
from v151_hourly_law import sha
from v159_fuel_activity_source_pilot import weighted_radius
from v170_fuel_marginal_diagnosis import verify
from v173_moment_protected_risk import METRICS, aggregate
from v175_cif_crps_map import BASE, OOF, STATS, OUT as SOURCE, ROOT, check_mutation

OUT = ROOT / 'results/v175-cif-crps-map-audit-r1'
NONCONSTANT = ['CHPD', 'DOPD', 'GVL', 'SCL', 'SPA']


def independent_certificate(packed, mapping, alpha, kind, certificate):
    """Check the returned map's first-order conditions without optimizing again."""
    mapping = check_maps(mapping)
    if certificate['used_identity_fallback']:
        if not np.array_equal(mapping, GRID) or certificate['kkt_error'] is not None or certificate['status'] == 'solved':
            raise ValueError('identity fallback falsely certified')
        return 0.
    H, b, constant = unpack(packed)
    lam = .95 * alpha
    D = (1 - lam) * GRID + lam * mapping
    Hmean = H.mean(axis=0) if kind == 'group_max' else H
    rho = 1e-6 * max(float(np.trace(Hmean)) / 17, 1e-8)
    if abs(rho - certificate['regularization']) > 1e-15:
        raise ValueError('regularization does not match fixed rule')
    gaps = np.diff(mapping)
    feasibility = max(float(max(np.maximum(-gaps, 0))), abs(float(gaps.sum()) - 1))
    before = np.einsum('i,...ij,j->...', GRID, H, GRID) - 2 * (b @ GRID) + constant
    after = np.einsum('i,...ij,j->...', D, H, D) - 2 * (b @ D) + constant
    gain = float(max(before) - max(after)) if kind == 'group_max' else float(before - after)
    penalty = rho * float(np.sum((mapping[1:-1] - GRID[1:-1])**2))
    if abs(gain - certificate['objective_gain']) > 1e-10 or gain - penalty < -1e-10:
        raise ValueError('returned map violates original regularized objective')
    if kind != 'group_max':
        A = lam**2 * H[1:-1, 1:-1] + rho * np.eye(17)
        fixed = (1 - lam) * GRID
        fixed[-1] += lam
        rhs = lam * (b[1:-1] - H[1:-1] @ fixed) + rho * GRID[1:-1]
        scale = max(float(np.trace(A)), float(max(abs(rhs))), 1e-16)
        if abs(scale - certificate['objective_scale']) > 1e-12:
            raise ValueError('mean objective scale differs')
        grad = 2 * lam * (H @ D - b)[1:-1] + 2 * rho * (mapping[1:-1] - GRID[1:-1])
        # Each gap affects its own and subsequent interior CDF values.
        gap_grad = np.r_[np.cumsum(grad[::-1])[::-1], 0.] / scale
        positive = gaps > 1e-10
        multiplier = gap_grad[positive].mean()
        kkt = max(float(max(abs(gap_grad[positive] - multiplier))),
                  float(max(np.maximum(multiplier - gap_grad[~positive], 0))) if (~positive).any() else 0.,
                  float(max(abs(gaps * (gap_grad - multiplier)))))
        if kkt > 1.01e-7 or feasibility > 1e-12:
            raise ValueError('independent mean QP certificate failed')
    else:
        risk_gradient = 2 * lam * (np.einsum('rij,j->ri', H, D) - b)[:, 1:-1]
        Jmono = np.zeros((18, 17))
        for k in range(18):
            if k < 17:
                Jmono[k, k] += 1
            if k > 0:
                Jmono[k, k - 1] -= 1
        J = np.vstack([np.column_stack([Jmono, np.zeros(18)]),
                       np.column_stack([-risk_gradient, np.ones(len(H))])])
        slack = np.r_[gaps, max(after) - after]
        active = slack <= 1e-7
        gradient = np.r_[2 * rho * (mapping[1:-1] - GRID[1:-1]), 1.]
        multipliers, _ = nnls(J[active].T, gradient, maxiter=1000)
        kkt = max(float(max(abs(gradient - J[active].T @ multipliers))),
                  float(max(abs(multipliers * slack[active]))))
        if kkt > 1.01e-6 or feasibility > 1e-9:
            raise ValueError('independent group minimax certificate failed')
    return kkt


def audit_fold(data, monthly, fold):
    frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
    predictor = CRPSRecalibration(frozen, monthly, OOF / f'fold_{fold}', STATS / f'fold_{fold}')
    block = SOURCE / f'fold_{fold}'
    rows = pd.read_csv(block / 'test_month_metrics.csv').set_index(['profile_id', 'method', 'radius_policy', 'calibration'])
    contexts = pd.read_csv(block / 'contexts.csv').set_index('profile_id')
    saved = pd.read_csv(block / 'recalibration_maps.csv').set_index(['profile_id', 'method'])
    radii = pd.read_csv(block / 'calibration_radii.csv').set_index(['method', 'calibration'])
    saved_cert = pd.read_csv(block / 'map_certificates.csv').set_index(['profile_id', 'method', 'state'])
    if (len(saved) != len(frozen.queries) * 16 or saved.index.duplicated().any()
            or set(contexts.index) != set(frozen.queries.profile_id) or contexts.index.duplicated().any()
            or len(radii) != 36 or radii.index.duplicated().any() or not np.isfinite(radii.radius).all()
            or len(saved_cert) != len(frozen.queries) * 768 or saved_cert.index.duplicated().any()):
        raise ValueError('complete diagnostic inventory')
    for q in frozen.queries.profile_id.iloc[[0, -1]]:
        check_mutation(predictor, data, monthly, int(q))
    errors = {name: {} for name in METHODS}
    max_score = max_radius = max_map = max_context = max_mass = 0.
    count = support_checks = map_records = 0
    certified = independent_checks = 0
    fixed_checked = set()
    max_independent_kkt = 0.
    for role in ('calibration', 'test'):
        for rec in frozen.queries.loc[frozen.queries.role.eq(role)].itertuples(index=False):
            q = int(rec.profile_id)
            laws, forecast, details, certificates = predictor.build(q)
            conditional_statistics, kernel_error = predictor.conditional_statistics(q)
            context = contexts.loc[q]
            if (context.role, context.region, context.month, context.fold) != (role, rec.region, rec.month, fold):
                raise ValueError('context identity')
            for key in ('raw_anchor', 'anchor', 'past_anchor', 'sigma', 'local'):
                delta = abs(float(forecast[key]) - float(context[key]))
                if not np.isfinite(delta):
                    raise ValueError('nonfinite forecast context')
                max_context = max(max_context, delta)
            for detail in details:
                old = saved.loc[(q, detail['method'])]
                if (old.role, old.region, old.month, old.fold) != (role, rec.region, rec.month, fold):
                    raise ValueError('map identity')
                for key in ('maps', 'state_mean_before', 'state_mean_after'):
                    a, b = np.asarray(json.loads(detail[key])), np.asarray(json.loads(old[key]))
                    if a.shape != b.shape or not np.isfinite(b).all():
                        raise ValueError('map/mean dimensions or finiteness')
                    max_map = max(max_map, float(np.max(abs(a - b))))
                if (not np.isfinite(old.kernel_pit_error) or old.kernel_pit_error > 1e-12
                        or abs(old.kernel_pit_error - kernel_error) > 1e-15):
                    raise ValueError('kernel PIT equivalence certificate differs')
                mappings = check_maps(json.loads(old['maps']))
                if mappings.shape != (48, len(GRID)):
                    raise ValueError('map dimensions')
                reference, _, alpha = SETTINGS[detail['method']]
                for state, (a, b) in enumerate(zip(laws[detail['method']], laws[reference])):
                    if not np.array_equal(a.x, b.x) or not (a.w > 0).all():
                        raise ValueError('physical support or positive mass changed')
                    # Independent scalar interpolation of the saved mapping.
                    cumulative = np.clip(np.r_[0., np.cumsum(b.w)], 0, 1)
                    cumulative[-1] = 1.
                    positions = np.searchsorted(GRID, cumulative, side='right') - 1
                    positions = np.clip(positions, 0, len(GRID) - 2)
                    fraction = (cumulative - GRID[positions]) / (GRID[positions + 1] - GRID[positions])
                    values = mappings[state, positions] * (1 - fraction) + mappings[state, positions + 1] * fraction
                    w = (1 - .95 * alpha) * b.w + .95 * alpha * np.maximum(np.diff(values), 0)
                    w /= w.sum()
                    max_mass = max(max_mass, float(max(abs(w - a.w))))
                    support_checks += 1
                map_records += 1
            for certificate in certificates:
                method, state = certificate['method'], certificate['state']
                old = saved_cert.loc[(q, method, state)]
                if (old.role, old.region, old.month, old.fold) != (role, rec.region, rec.month, fold):
                    raise ValueError('QP certificate identity')
                for key, v in certificate.items():
                    if key in ('method', 'state'):
                        continue
                    if v is None:
                        if not pd.isna(old[key]):
                            raise ValueError('fallback has a false KKT certificate')
                    elif isinstance(v, (str, bool)):
                        if old[key] != v:
                            raise ValueError('QP status differs')
                    elif not np.isfinite(old[key]) or abs(v - old[key]) > 1e-12:
                        raise ValueError('QP numerical certificate differs')
                reference, kind, alpha = SETTINGS[method]
                fixed = kind in ('global', 'group_max')
                if not fixed or method not in fixed_checked:
                    mappings = np.asarray(json.loads(saved.loc[(q, method), 'maps']))
                    packed = predictor.fixed_statistics[method] if fixed else conditional_statistics[(reference, kind)][state]
                    kkt = independent_certificate(packed, mappings[state], alpha, kind, certificate)
                    max_independent_kkt = max(max_independent_kkt, kkt)
                    independent_checks += 1
                    if fixed:
                        fixed_checked.add(method)
                certified += 1
            actual = data.truth(q)
            row = data.index.iloc[q]
            clock = data.states[int(row.offset):int(row.stop)]
            for method in METHODS:
                if role == 'calibration':
                    law = CalendarLaw(laws[method], clock)
                    error = np.maximum.reduce([law.quantile(.05) - actual, actual - law.quantile(.95), np.zeros(len(actual))])
                    errors[method].setdefault(rec.region, []).append(error)
                else:
                    for policy in (('recalibrated',) if method in REFERENCES else ('fixed', 'recalibrated')):
                        for calibration in ('pooled', 'max_region'):
                            reference = method if method in REFERENCES else SETTINGS[method][0]
                            radius = radius_map[(method if policy == 'recalibrated' else reference, calibration)]
                            old = rows.loc[(q, method, policy, calibration)]
                            if (old.region, old.month, old.fold, old.hours) != (rec.region, rec.month, fold, len(actual)):
                                raise ValueError('score identity')
                            scores = score_laws(laws[method], actual, clock, radius)
                            delta = np.asarray([scores[k] - old[k] for k in scores])
                            if not np.isfinite(delta).all() or not np.isfinite(old.radius):
                                raise ValueError('nonfinite score/radius')
                            max_score = max(max_score, float(max(abs(delta))))
                            max_radius = max(max_radius, abs(radius - old.radius))
                            count += 1
        if role == 'calibration':
            radius_map = {}
            for name in METHODS:
                if set(errors[name]) != set(frozen.queries.loc[frozen.queries.role.eq(role), 'region']):
                    raise ValueError('calibration region inventory')
                for label, radius in zip(('pooled', 'max_region'), weighted_radius(errors[name])):
                    radius_map[(name, label)] = radius
                    max_radius = max(max_radius, abs(radius - radii.loc[(name, label), 'radius']))
    if (count != len(rows) or map_records != len(saved) or certified != len(saved_cert) or max_score > 1e-7
            or max_radius > 1e-8 or max_map > 1e-10 or max_context > 1e-8 or max_mass > 1e-12):
        raise ValueError('complete reconstruction failed')
    print(f'CRPS map audit fold {fold}: {count} scores', flush=True)
    return {'fold': fold, 'rebuilt_score_rows': count, 'map_records': map_records,
            'certificate_applications_rebuilt': certified, 'independent_QP_checks': independent_checks,
            'max_independent_KKT': max_independent_kkt,
            'physical_support_checks': support_checks, 'max_score_error': max_score,
            'max_radius_error': max_radius, 'max_map_error': max_map,
            'max_context_error': max_context, 'max_independent_mass_error': max_mass}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = verify(SOURCE)
    hashes = summary['input_hashes'] | {str((SOURCE / 'summary.json').relative_to(ROOT)): sha(SOURCE / 'summary.json'),
                                      str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    external = {'positive_recomposition': BASE, 'scalar_forest': BASE,
                'activity_mmd_forest': ROOT / 'results/v160-carbon-cdf-source-r1',
                'forecast_composition': ROOT / 'results/v164-scenario-reweighting-r1',
                'world_entropy': ROOT / 'results/v164-scenario-reweighting-r1',
                'hetero_history_barycentric': ROOT / 'results/v165-coarse-distribution-lifting-r1',
                'gaussian_protect_fuel8_direct20_a50': ROOT / 'results/v173-moment-protected-risk-r1',
                'gaussian_pit_conditional43_a50': ROOT / 'results/v174-cif-recalibration-r1'}
    PIT = ROOT / 'results/v174-cif-recalibration-r1'
    for directory in set(external.values()) | {PIT}:
        verify(directory)
        for name in ('summary.json', 'test_month_metrics.csv'):
            hashes[str((directory / name).relative_to(ROOT))] = sha(directory / name)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('input changed before CIF audit')
    rows = pd.read_csv(SOURCE / 'test_month_metrics.csv')
    if (len(rows) != 16388 or rows.duplicated(['profile_id', 'method', 'radius_policy', 'calibration']).any()
            or not np.isfinite(rows[list(METRICS) + ['radius', 'hours']]).all().all()):
        raise ValueError('complete score count/finiteness')
    old_pit = pd.read_csv(PIT / 'test_month_metrics.csv').rename(columns={'method': 'peer'})
    mean_rows = rows.loc[rows.method.map(lambda name: name in SETTINGS and SETTINGS[name][1] != 'group_max')].copy()
    mean_rows['peer'] = mean_rows.method.str.replace('_crps_', '_pit_', regex=False)
    matched = mean_rows.merge(old_pit, on=['profile_id', 'peer', 'radius_policy', 'calibration'],
                              suffixes=('_new', '_old'), validate='one_to_one')
    fixed = matched.loc[matched.radius_policy.eq('fixed')]
    if (len(matched) != len(mean_rows) or len(fixed) != 5784 or max(abs(fixed.radius_new - fixed.radius_old)) > 1e-8
            or not np.array_equal(matched.region_new, matched.region_old)
            or not np.array_equal(matched.month_new, matched.month_old)
            or not np.array_equal(matched.hours_new, matched.hours_old)):
        raise ValueError('matched PIT row identity/fixed-radius mismatch')
    blocks = pd.concat([pd.read_csv(SOURCE / f'fold_{f}/test_month_metrics.csv') for f in range(6)], ignore_index=True)
    if (SOURCE / 'test_month_metrics.csv').read_text() != blocks.to_csv(index=False):
        raise ValueError('root producer serialization mismatch')
    regional, macro = aggregate(rows)
    for label, new, keys in [('region_metrics', regional, ['region', 'method', 'radius_policy', 'calibration']),
                             ('macro_metrics', macro, ['cohort', 'method', 'radius_policy', 'calibration'])]:
        old = pd.read_csv(SOURCE / f'{label}.csv')
        merged = new.merge(old, on=keys, validate='one_to_one', suffixes=('_a', '_b'))
        fields = list(METRICS) + (['worst_coverage', 'regions'] if label == 'macro_metrics' else [])
        if (len(merged) != len(new) or len(merged) != len(old) or not np.isfinite(old[fields]).all().all()
                or any(max(abs(merged[f + '_a'] - merged[f + '_b'])) > 1e-8 for f in fields)):
            raise ValueError('aggregate reconstruction failed')
    data, monthly = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
    with threadpool_limits(limits=2):
        checks = [audit_fold(data, monthly, fold) for fold in range(6)]
    tables = [regional]
    cohort_keys = set(rows[['region', 'month', 'calibration', 'hours']].itertuples(index=False, name=None))
    for name, directory in external.items():
        frame = pd.read_csv(directory / 'test_month_metrics.csv')
        frame = frame.loc[frame.method.eq(name) & frame.budget.eq('asof_30d')]
        if 'radius_policy' in frame:
            frame = frame.loc[frame.radius_policy.eq('recalibrated')]
        if (set(frame[['region', 'month', 'calibration', 'hours']].itertuples(index=False, name=None)) != cohort_keys
                or frame.duplicated(['region', 'month', 'calibration']).any()
                or not np.isfinite(frame[list(METRICS[:5])]).all().all()):
            raise ValueError('external strong reference cohort mismatch')
        tables.append(pd.DataFrame([{'region': key[0], 'method': name, 'calibration': key[1], 'radius_policy': 'recalibrated',
                                     **{m: float(np.average(group[m], weights=group.hours)) for m in METRICS[:5]}}
                                    for key, group in frame.groupby(['region', 'calibration'])]))
    pit_frame = pd.read_csv(PIT / 'test_month_metrics.csv')
    pit_frame = pit_frame.loc[~pit_frame.method.isin(REFERENCES)]
    # Retain both policies for every exact PIT peer. The standard best-PIT
    # reference already exists in tables, so do not duplicate that method.
    for (name, policy), frame in pit_frame.groupby(['method', 'radius_policy']):
        if name in external and policy == 'recalibrated':
            continue
        if (set(frame[['region', 'month', 'calibration', 'hours']].itertuples(index=False, name=None)) != cohort_keys
                or frame.duplicated(['region', 'month', 'calibration']).any()):
            raise ValueError('matched PIT cohort differs')
        tables.append(pd.DataFrame([{'region': key[0], 'method': name, 'calibration': key[1], 'radius_policy': policy,
                                     **{m: float(np.average(group[m], weights=group.hours)) for m in METRICS[:5]}}
                                    for key, group in frame.groupby(['region', 'calibration'])]))
    table = pd.concat(tables, ignore_index=True)
    rng = np.random.default_rng(175)
    comparisons = []
    for rec in macro.loc[~macro.method.isin(REFERENCES)].itertuples(index=False):
        selected = table.loc[table.calibration.eq(rec.calibration)]
        if rec.cohort == 'nonconstant24':
            selected = selected.loc[~selected.region.isin(NONCONSTANT)]
        a = selected.loc[selected.method.eq(rec.method) & selected.radius_policy.eq(rec.radius_policy)].sort_values('region')
        reference, kind, alpha = SETTINGS[rec.method]
        prefix = 'conditional' if reference == REFERENCES[0] else 'gaussian'
        peers = [(name, 'recalibrated', 'strong_reference') for name in (*REFERENCES, *external)]
        if kind != 'group_max':
            peers.append((rec.method.replace('_crps_', '_pit_'), rec.radius_policy, 'matched_PIT_objective'))
        if kind in ('conditional20', 'conditional43', 'group_max'):
            peers.append((f'{prefix}_crps_global_a{int(alpha * 100)}', rec.radius_policy,
                          'matched_global' if kind != 'group_max' else 'matched_mean_objective'))
        if kind == 'conditional43':
            peers.append((f'{prefix}_crps_conditional20_a{int(alpha * 100)}', rec.radius_policy, 'matched_no_history'))
        for name, policy, comparison in peers:
            b = selected.loc[selected.method.eq(name) & selected.radius_policy.eq(policy)].sort_values('region')
            if len(a) != rec.regions or not np.array_equal(a.region, b.region):
                raise ValueError('paired reference regions differ')
            difference = a.crps.to_numpy() - b.crps.to_numpy()
            boot = difference[rng.integers(len(a), size=(3000, len(a)))].mean(axis=1)
            comparisons.append({'cohort': rec.cohort, 'method': rec.method, 'radius_policy': rec.radius_policy,
                                'calibration': rec.calibration, 'comparison': comparison, 'reference': name,
                                'reference_radius_policy': policy, 'paired_crps_difference': float(difference.mean()),
                                'region_bootstrap_lo': float(np.quantile(boot, .025)),
                                'region_bootstrap_hi': float(np.quantile(boot, .975)),
                                'regions_better_crps': int((difference < -1e-9).sum()),
                                'crps_reduction_percent': float(100 * (1 - rec.crps / b.crps.mean())),
                                'coverage': rec.coverage, 'worst_coverage': rec.worst_coverage, 'width': rec.width,
                                'width_change_percent': float(100 * (rec.width / b.width.mean() - 1)), 'winkler': rec.winkler,
                                'numeric_threshold_only': bool(rec.coverage >= .9 and rec.worst_coverage >= .85
                                                               and rec.crps <= .95 * b.crps.mean()
                                                               and rec.width <= b.width.mean() + 1e-9
                                                               and rec.winkler < b.winkler.mean())})
    if len(comparisons) != 1504:
        raise ValueError('comparison inventory')
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('inputs changed during CIF audit')
    verify(SOURCE)
    OUT.mkdir(parents=True)
    pd.DataFrame(checks).to_csv(OUT / 'reconstruction.csv', index=False)
    pd.DataFrame(comparisons).to_csv(OUT / 'comparisons.csv', index=False)
    certificates = pd.concat([pd.read_csv(SOURCE / f'fold_{k}/map_certificates.csv') for k in range(6)], ignore_index=True)
    fallback = certificates.groupby(['role', 'region', 'method']).agg(
        applications=('used_identity_fallback', 'size'), identity_fallbacks=('used_identity_fallback', 'sum')).reset_index()
    fallback['fallback_fraction'] = fallback.identity_fallbacks / fallback.applications
    fallback.to_csv(OUT / 'fallback_by_region_method.csv', index=False)
    result = {'assertion_failures': 0, 'rebuilt_score_rows': sum(c['rebuilt_score_rows'] for c in checks),
              'map_records_rebuilt': sum(c['map_records'] for c in checks),
              'physical_support_checks': sum(c['physical_support_checks'] for c in checks),
              'certificate_applications_rebuilt': sum(c['certificate_applications_rebuilt'] for c in checks),
              'independent_QP_checks': sum(c['independent_QP_checks'] for c in checks),
              'identity_fallback_applications': int(certificates.used_identity_fallback.sum()),
              'matched_fixed_radius_rows': len(fixed),
              'future_month_mutation_cases': 12, 'max_score_error': max(c['max_score_error'] for c in checks),
              'max_independent_mass_error': max(c['max_independent_mass_error'] for c in checks),
              'comparisons': len(comparisons), 'numeric_threshold_passes': sum(c['numeric_threshold_only'] for c in comparisons),
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in OUT.iterdir()},
              'scope': 'All CRPS maps refitted; QP certificates independently checked; calibration/scores rebuilt with frozen helpers. Independent interpolation checked; bootstrap remains source-development description.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_hashes', 'output_hashes')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
