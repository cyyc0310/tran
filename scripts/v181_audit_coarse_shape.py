"""Rebuild exact product laws and audit their pre/post-conditioning marginals."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from calendar_cif_law import CalendarLaw
from coarse_shape_coupling import CoarseShapeCoupling, METHODS, REFERENCES, PRODUCT, DIAGONAL, check_inputs
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from level_spread_diagnostics import score_laws
from monthly_history_features import MonthlyInformation
from v151_hourly_law import EF, sha
from v154_hourly_strong_baselines import CompressedLaw
from v159_fuel_activity_source_pilot import weighted_radius
from v170_fuel_marginal_diagnosis import verify
from v173_moment_protected_risk import METRICS, aggregate
from v179_audit_increment_support import bisection_projection, compare_frames
from v181_coarse_shape_coupling import ROOT, BASE, COARSE, OUT as SOURCE, OLD, toys

OUT = ROOT / 'results/v181-coarse-shape-audit-r1'
PARTS = ROOT / 'results/v181-coarse-shape-audit-parts-r1'


def presence(values):
    return ((values > 0).astype(np.uint16) * (1 << np.arange(8, dtype=np.uint16))).sum(axis=1, dtype=np.uint16)


def independent_cell_loss(values, p, s, cell, laws):
    """Unmerged native coarse worlds, with independent sorted-interval integration."""
    shape, component, within, _ = cell
    fuel = (values[:, None] * shape[None]).reshape(-1, 8)
    total = fuel.sum(axis=1)
    mass = (p[:, None] * (s[component] * within)[None]).ravel()
    good = total > 0
    reference = CompressedLaw(fuel[good] @ EF / total[good], mass[good])
    error = 0.
    for target in np.arange(0, 1001, 100):
        boundaries = np.unique(np.r_[reference.x, float(target)])
        mid = (boundaries[:-1] + boundaries[1:]) / 2
        cdf = np.r_[0., np.cumsum(reference.w)][np.searchsorted(reference.x, mid, side='right')]
        score = float(np.diff(boundaries) @ (cdf - (mid >= target))**2)
        error = max(error, abs(score - laws.crps(np.asarray([target]))[0]))
    return error


def audit_fold(data, monthly, fold):
    frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
    predictor = CoarseShapeCoupling(frozen, monthly, COARSE / f'fold_{fold}')
    rebased = predictor.rebased
    block = SOURCE / f'fold_{fold}'
    scores = pd.read_csv(block / 'test_month_metrics.csv').set_index(['profile_id', 'method', 'radius_policy', 'calibration'])
    diagnostics = pd.read_csv(block / 'coupling_diagnostics.csv').set_index(['profile_id', 'method'])
    radii = pd.read_csv(block / 'calibration_radii.csv').set_index(['method', 'calibration'])
    if (len(scores) != int(frozen.queries.role.eq('test').sum()) * 14 or len(diagnostics) != len(frozen.queries) * 3
            or len(radii) != 8 or scores.index.duplicated().any() or diagnostics.index.duplicated().any() or radii.index.duplicated().any()):
        raise ValueError('complete fold inventory')
    source_cells = 0
    for h, (shape, component, within, raw_ids) in enumerate(rebased.cells):
        profiles = rebased.profiles[component]
        mean = data.M[profiles] / data.index.hours.to_numpy()[profiles, None]
        actual = np.divide(data.raw[raw_ids], mean, out=np.ones_like(shape), where=mean > 0)
        if not np.array_equal(actual, shape):
            raise ValueError('independent raw source q differs')
        for j, profile in enumerate(rebased.profiles):
            source = frozen.activity.sources[data.index.region.iloc[profile]]
            ids = source['allowed'][source['profiles'] == profile]
            chosen = ids[data.states[ids] % 24 == h]
            if not len(chosen):
                chosen = ids
            if (not np.array_equal(raw_ids[component == j], chosen)
                    or max(abs(within[component == j] - 1 / len(chosen))) > 1e-14):
                raise ValueError('source sample/clock/within identity differs')
            source_cells += 1
    selected_checks = set(map(int, frozen.queries.profile_id.iloc[[0, -1]]))
    for q in selected_checks:
        laws, details = check_inputs(predictor, q)
        del laws, details
    errors = {m: {} for m in METHODS}
    count = diagnostic_count = posterior_cells = independent_losses = 0
    score_error = diagnostic_error = projection_error = independent_loss_error = 0.
    exposure = []
    for role in ('calibration', 'test'):
        queries = frozen.queries.loc[frozen.queries.role.eq(role)]
        for number, rec in enumerate(queries.itertuples(index=False), start=1):
            q = int(rec.profile_id)
            laws, details = predictor.build(q)
            local = monthly.local(q, 'asof_30d')
            delta = monthly.shares[rebased.profiles] - monthly.shares[rebased.inputs]
            delta[abs(delta) <= 64 * np.finfo(float).eps] = 0
            projected = bisection_projection(monthly.shares[local] + delta)
            p = rebased.weights(q, 'forest_history_scalar8')
            pH = rebased.weights(q, 'allmonth_hellinger')
            for detail in details:
                method = detail['method']
                old = diagnostics.loc[(q, method)]
                if (old.fold, old.role, old.region, old.month) != (fold, role, rec.region, rec.month):
                    raise ValueError('diagnostic row identity')
                for scalar in ('unknown_shape_energy_mass', 'precondition_marginal_error', 'unique_coarse_values'):
                    if not np.isfinite(old[scalar]):
                        raise ValueError('nonfinite diagnostic scalar')
                    diagnostic_error = max(diagnostic_error, abs(detail[scalar] - old[scalar]))
                old_arrays = {}
                for key in ('source_profiles', 'coarse_support', 'coarse_weights', 'shape_weights', 'rejected_mass',
                            'effective_coarse_weights', 'effective_shape_weights', 'fuel_mean'):
                    a, b = np.asarray(json.loads(detail[key])), np.asarray(json.loads(old[key]))
                    if a.shape != b.shape or not np.isfinite(b).all():
                        raise ValueError('diagnostic array dimensions/finiteness')
                    old_arrays[key] = b
                    diagnostic_error = max(diagnostic_error, float(np.max(abs(a - b))))
                if json.loads(old.source_profiles) != rebased.profiles.tolist():
                    raise ValueError('coarse/shape world identity')
                values = np.asarray(json.loads(detail['coarse_support']))
                projection_error = max(projection_error, float(np.max(abs(values - projected))))
                s = pH if method == PRODUCT[1] else p
                if max(abs(np.asarray(json.loads(old.coarse_weights)) - p)) > 1e-12 or max(abs(np.asarray(json.loads(old.shape_weights)) - s)) > 1e-12:
                    raise ValueError('frozen probability identity')
                vc = presence(values)
                rejected, coarse_shift, shape_shift = [], [], []
                for h, (shape, component, within, _) in enumerate(rebased.cells):
                    qs = presence(shape)
                    mass = s[component] * within
                    if method == DIAGONAL:
                        good = (vc[component] & qs) > 0
                        joint_mass = p[component] * within
                        kept = float(joint_mass[good].sum())
                        posterior_coarse = np.bincount(component[good], weights=joint_mass[good], minlength=len(p)) / kept
                        posterior_shape = posterior_coarse
                        fuel = values[component] * shape
                        expected_fuel = joint_mass @ fuel / kept
                        rejection = float(joint_mass[~good].sum() / joint_mass.sum())
                    else:
                        # Exact support intersections from eight-bit presence masks,
                        # without the production float ratios or merged coarse rows.
                        good = (vc[:, None] & qs[None]) > 0
                        survival_c = good @ mass
                        survival_q = p @ good
                        kept = float(p @ survival_c)
                        posterior_coarse = p * survival_c / kept
                        posterior_shape = np.bincount(component, weights=mass * survival_q, minlength=len(p)) / kept
                        expected_fuel = (p @ values) * (mass @ shape) / kept
                        rejection = float(p @ ((~good) @ mass) / (p.sum() * mass.sum()))
                    saved_c = old_arrays['effective_coarse_weights'][h]
                    saved_s = old_arrays['effective_shape_weights'][h]
                    saved_mean = old_arrays['fuel_mean'][h]
                    saved_rejection = old_arrays['rejected_mass'][h]
                    diagnostic_error = max(diagnostic_error, float(max(abs(saved_c - posterior_coarse))),
                                           float(max(abs(saved_s - posterior_shape))), float(max(abs(saved_mean - expected_fuel))),
                                           abs(saved_rejection - rejection))
                    if (min(saved_c) < 0 or min(saved_s) < 0 or max(abs(np.asarray([saved_c.sum(), saved_s.sum()]) - 1)) > 1e-12
                            or rejection < 0 or rejection > 1 or min(saved_mean) < 0):
                        raise ValueError('post-conditioning domain')
                    rejected.append(rejection)
                    coarse_shift.append(float(sum(abs(posterior_coarse - p / p.sum()))))
                    shape_shift.append(float(sum(abs(posterior_shape - s / s.sum()))))
                    if q in selected_checks and h in {q % 24, (q + 11) % 24} and method in PRODUCT:
                        independent_loss_error = max(independent_loss_error, independent_cell_loss(values, p, s, rebased.cells[h], laws[method][h]))
                        independent_losses += 1
                    if min(laws[method][h].x) < -1e-10 or max(laws[method][h].x) > 1000 + 1e-10 or not (laws[method][h].w > 0).all():
                        raise ValueError('physical CIF support/mass')
                    posterior_cells += 1
                if role == 'test':
                    row = data.index.iloc[q]
                    clock = data.states[int(row.offset):int(row.stop)] % 24
                    exposure.append({'fold': fold, 'profile_id': q, 'region': rec.region, 'method': method, 'hours': len(clock),
                                     'rejected_mass': float(np.mean(np.asarray(rejected)[clock])), 'max_rejected_mass': max(rejected),
                                     'effective_coarse_l1_shift': float(np.mean(np.asarray(coarse_shift)[clock])),
                                     'effective_shape_l1_shift': float(np.mean(np.asarray(shape_shift)[clock])),
                                     'unknown_shape_energy_mass': old.unknown_shape_energy_mass})
                diagnostic_count += 1
            if projection_error > 1e-12 or diagnostic_error > 1e-8 or independent_loss_error > 1e-8:
                raise ValueError('independent product diagnostic/loss mismatch')
            actual = data.truth(q)
            row = data.index.iloc[q]
            clock = data.states[int(row.offset):int(row.stop)] % 24
            for method in METHODS:
                if role == 'calibration':
                    law = CalendarLaw(laws[method], clock)
                    distance = np.maximum.reduce([law.quantile(.05) - actual, actual - law.quantile(.95), np.zeros(len(actual))])
                    errors[method].setdefault(rec.region, []).append(distance)
                else:
                    for policy in (('recalibrated',) if method == 'world_gaussian' else ('fixed', 'recalibrated')):
                        for calibration in ('pooled', 'max_region'):
                            radius = radius_map[(method if policy == 'recalibrated' else 'world_gaussian', calibration)]
                            old = scores.loc[(q, method, policy, calibration)]
                            if (old.fold, old.role, old.region, old.month, old.hours) != (fold, role, rec.region, rec.month, len(actual)):
                                raise ValueError('formal score identity')
                            fresh = score_laws(laws[method], actual, clock, radius)
                            if not np.isfinite(old[list(METRICS) + ['radius']].to_numpy(float)).all():
                                raise ValueError('nonfinite formal scores')
                            score_error = max(score_error, abs(radius - old.radius), max(abs(fresh[m] - old[m]) for m in fresh))
                            count += 1
            del laws
            if number % 5 == 0 or number == len(queries):
                print(f'Cartesian audit fold {fold} {role} {number}/{len(queries)}', flush=True)
        if role == 'calibration':
            radius_map = {}
            for method in METHODS:
                if set(errors[method]) != set(queries.region):
                    raise ValueError('calibration inventory')
                for label, radius in zip(('pooled', 'max_region'), weighted_radius(errors[method])):
                    old = radii.loc[(method, label), 'radius']
                    if not np.isfinite(old) or abs(radius - old) > 1e-8:
                        raise ValueError('calibration radius differs')
                    radius_map[(method, label)] = radius
    if count != len(scores) or diagnostic_count != len(diagnostics) or score_error > 1e-8:
        raise ValueError('complete formal reconstruction')
    return {'fold': fold, 'score_rows_rebuilt': count, 'diagnostic_rows_rebuilt': diagnostic_count,
            'independent_source_cells': source_cells, 'independent_posterior_cells': posterior_cells,
            'independent_unmerged_loss_cells': independent_losses, 'max_score_error': score_error,
            'max_diagnostic_error': diagnostic_error, 'max_projection_error': projection_error,
            'max_independent_loss_error': independent_loss_error}, exposure


def audit_parts(producer_pid):
    """Audit immutable completed folds while the producer works on later folds."""
    manifest_path = SOURCE / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    data, monthly = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
    with threadpool_limits(limits=2):
        for fold in range(6):
            block = SOURCE / f'fold_{fold}'
            while not (block / 'summary.json').is_file():
                try:
                    os.kill(producer_pid, 0)
                except ProcessLookupError as exc:
                    raise RuntimeError(f'producer terminated before fold {fold}') from exc
                time.sleep(1)
            saved = verify(block)
            if saved['manifest_sha256'] != sha(manifest_path):
                raise ValueError('partial source manifest differs')
            hashes = manifest['input_hashes'] | {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                                               str(manifest_path.relative_to(ROOT)): sha(manifest_path),
                                               str((ROOT / 'scripts/v179_audit_increment_support.py').relative_to(ROOT)): sha(ROOT / 'scripts/v179_audit_increment_support.py')}
            for name in ('summary.json', *saved['output_hashes']):
                p = block / name
                hashes[str(p.relative_to(ROOT))] = sha(p)
            if any(sha(ROOT / p) != h for p, h in hashes.items()):
                raise ValueError('partial audit source changed')
            destination = PARTS / f'fold_{fold}'
            if destination.exists():
                result = verify(destination)
                if result['input_hashes'] != hashes:
                    raise ValueError('stale partial audit')
                continue
            check, exposure = audit_fold(data, monthly, fold)
            if any(sha(ROOT / p) != h for p, h in hashes.items()):
                raise ValueError('partial audit source changed during run')
            PARTS.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix='.fold-audit-', dir=PARTS))
            (staging / 'reconstruction.json').write_text(json.dumps(check, indent=2, allow_nan=False) + '\n')
            pd.DataFrame(exposure).to_csv(staging / 'exposure_queries.csv', index=False)
            result = {'assertion_failures': 0, 'fold': fold, 'input_hashes': hashes,
                      'output_hashes': {p.name: sha(p) for p in staging.iterdir()},
                      'scope': 'Full per-fold reconstruction, including all source/posterior cells and selected independent unmerged losses; root aggregates and comparisons await final audit.'}
            (staging / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            staging.rename(destination)
            print(f'atomic partial audit {fold} complete', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--follow-producer-pid', type=int)
    args = parser.parse_args()
    if args.follow_producer_pid is not None:
        audit_parts(args.follow_producer_pid)
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    summary = verify(SOURCE)
    hashes = summary['input_hashes'] | {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                                      str((SOURCE / 'summary.json').relative_to(ROOT)): sha(SOURCE / 'summary.json')}
    for name in summary['output_hashes']:
        hashes[str((SOURCE / name).relative_to(ROOT))] = sha(SOURCE / name)
    dependencies = [ROOT / 'scripts/v179_audit_increment_support.py']
    hashes |= {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    external = {'positive_recomposition': BASE, 'scalar_forest': BASE,
                'activity_mmd_forest': ROOT / 'results/v160-carbon-cdf-source-r1',
                'rebased_increment_gaussian': OLD,
                'hetero_history_barycentric': ROOT / 'results/v165-coarse-distribution-lifting-r1',
                'gaussian_pit_conditional43_a50': ROOT / 'results/v174-cif-recalibration-r1'}
    for directory in set(external.values()):
        verify(directory)
        for name in ('summary.json', 'test_month_metrics.csv'):
            p = directory / name
            hashes[str(p.relative_to(ROOT))] = sha(p)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('audit input changed')
    if toys() > 1e-10:
        raise ValueError('unmerged toy failed')
    rows = pd.read_csv(SOURCE / 'test_month_metrics.csv')
    blocks = pd.concat([pd.read_csv(SOURCE / f'fold_{k}/test_month_metrics.csv') for k in range(6)], ignore_index=True)
    if (len(rows) != 3374 or rows.duplicated(['profile_id', 'method', 'radius_policy', 'calibration']).any()
            or not np.isfinite(rows[list(METRICS) + ['hours', 'radius']]).all().all()
            or (SOURCE / 'test_month_metrics.csv').read_text() != blocks.to_csv(index=False)):
        raise ValueError('complete root inventory/serialization')
    regional, macro = aggregate(rows)
    compare_frames(regional, pd.read_csv(SOURCE / 'region_metrics.csv'), ['region', 'method', 'radius_policy', 'calibration'])
    compare_frames(macro, pd.read_csv(SOURCE / 'macro_metrics.csv'), ['cohort', 'method', 'radius_policy', 'calibration'])
    fixed = rows.loc[rows.radius_policy.eq('fixed')].merge(rows.loc[rows.method.eq('world_gaussian'), ['profile_id', 'calibration', 'radius']],
                                                        on=['profile_id', 'calibration'], validate='many_to_one', suffixes=('_new', '_old'))
    if len(fixed) != 1446 or max(abs(fixed.radius_new - fixed.radius_old)) > 1e-8:
        raise ValueError('common fixed radius differs')
    data, monthly = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
    checks, exposure = [], []
    with threadpool_limits(limits=2):
        for fold in range(6):
            part = PARTS / f'fold_{fold}'
            if part.exists():
                summary_part = verify(part)
                if any(sha(ROOT / p) != h for p, h in summary_part['input_hashes'].items()):
                    raise ValueError('stale partial reconstruction')
                for name in ('summary.json', *summary_part['output_hashes']):
                    p = part / name
                    hashes[str(p.relative_to(ROOT))] = sha(p)
                fresh = json.loads((part / 'reconstruction.json').read_text())
                records = pd.read_csv(part / 'exposure_queries.csv').to_dict('records')
            else:
                fresh, records = audit_fold(data, monthly, fold)
            checks.append(fresh)
            exposure.extend(records)
    if (sum(c['independent_posterior_cells'] for c in checks) != 34704
            or sum(c['independent_unmerged_loss_cells'] for c in checks) != 48
            or sum(c['score_rows_rebuilt'] for c in checks) != 3374
            or sum(c['diagnostic_rows_rebuilt'] for c in checks) != 1446):
        raise ValueError('complete audit scope denominator')
    table = [regional]
    keys = set(rows[['region', 'month', 'calibration', 'hours']].itertuples(index=False, name=None))
    for method, directory in external.items():
        frame = pd.read_csv(directory / 'test_month_metrics.csv')
        frame = frame.loc[frame.method.eq(method) & frame.budget.eq('asof_30d')]
        if 'radius_policy' in frame:
            frame = frame.loc[frame.radius_policy.eq('recalibrated')]
        if set(frame[['region', 'month', 'calibration', 'hours']].itertuples(index=False, name=None)) != keys or frame.duplicated(['region', 'month', 'calibration']).any():
            raise ValueError('external reference cohort mismatch')
        table.append(pd.DataFrame([{'region': region, 'method': method, 'calibration': calibration, 'radius_policy': 'recalibrated',
                                   **{m: float(np.average(g[m], weights=g.hours)) for m in METRICS[:5]}}
                                  for (region, calibration), g in frame.groupby(['region', 'calibration'])]))
    table = pd.concat(table, ignore_index=True)
    rng = np.random.default_rng(181)
    comparisons = []
    for rec in macro.loc[macro.method.isin(PRODUCT)].itertuples(index=False):
        selected = table.loc[table.calibration.eq(rec.calibration)]
        if rec.cohort == 'nonconstant24':
            selected = selected.loc[~selected.region.isin(['CHPD', 'DOPD', 'GVL', 'SCL', 'SPA'])]
        a = selected.loc[selected.method.eq(rec.method) & selected.radius_policy.eq(rec.radius_policy)].sort_values('region')
        peers = [('world_gaussian', 'recalibrated', 'strong'), (DIAGONAL, rec.radius_policy, 'matched_pairing')]
        peers += [(method, 'recalibrated', 'strong') for method in external]
        peers += [(method, rec.radius_policy, 'shape_marginal_control') for method in PRODUCT if method != rec.method]
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
    if len(comparisons) != 144:
        raise ValueError('comparison denominator')
    exposure = pd.DataFrame(exposure)
    metrics = ['rejected_mass', 'effective_coarse_l1_shift', 'effective_shape_l1_shift', 'unknown_shape_energy_mass']
    exposure_regions = pd.DataFrame([{'region': r, 'method': m, **{key: float(np.average(g[key], weights=g.hours)) for key in metrics},
                                     'max_rejected_mass': float(g.max_rejected_mass.max())}
                                    for (r, m), g in exposure.groupby(['region', 'method'])])
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('audit inputs changed during run')
    OUT.mkdir(parents=True)
    for name, frame in [('reconstruction', pd.DataFrame(checks)), ('comparisons', pd.DataFrame(comparisons)),
                        ('exposure_queries', exposure), ('exposure_regions', exposure_regions)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    result = {'assertion_failures': 0, 'score_rows_rebuilt': sum(c['score_rows_rebuilt'] for c in checks),
              'diagnostic_rows_rebuilt': sum(c['diagnostic_rows_rebuilt'] for c in checks),
              'independent_source_cells': sum(c['independent_source_cells'] for c in checks),
              'independent_posterior_cells': sum(c['independent_posterior_cells'] for c in checks),
              'independent_unmerged_loss_cells': sum(c['independent_unmerged_loss_cells'] for c in checks),
              'future_query_input_mutation_cases': 12, 'common_fixed_radius_rows': len(fixed),
              'max_score_error': max(c['max_score_error'] for c in checks),
              'max_independent_loss_error': max(c['max_independent_loss_error'] for c in checks),
              'max_diagnostic_error': max(c['max_diagnostic_error'] for c in checks),
              'comparisons': len(comparisons), 'numeric_threshold_passes': sum(c['numeric_threshold_only'] for c in comparisons),
              'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in OUT.iterdir()},
              'scope': 'All laws/diagnostics/calibration/formal scores rebuilt; independent raw q and bit-mask posterior marginals for all cells. Native unmerged Cartesian plus independent CDF integration on 48 selected real cells and all toy cases; shared formal scoring/quantile helpers. Exploratory source development.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
