"""Evaluate exact product coarse/shape laws under the frozen source budget."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import tempfile
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from calendar_cif_law import CalendarLaw
from coarse_shape_coupling import CoarseShapeCoupling, METHODS, REFERENCES, PRODUCT, DIAGONAL, product_laws, check_inputs
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from level_spread_diagnostics import score_laws
from monthly_history_features import MonthlyInformation
from v151_hourly_law import EF, sha
from v154_hourly_strong_baselines import CompressedLaw
from v159_fuel_activity_source_pilot import weighted_radius
from v170_fuel_marginal_diagnosis import verify
from v173_moment_protected_risk import METRICS, aggregate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
COARSE = ROOT / 'results/v178-joint-coarse-world-r1'
OLD = ROOT / 'results/v180-rebased-coarse-lift-r1'
OUT = ROOT / 'results/v181-coarse-shape-coupling-r1'
PRE = ROOT / 'results/v181-coarse-shape-preflight-r1'
PROTOCOL = ROOT / 'docs/V181_COARSE_SHAPE_COUPLING_PROTOCOL.md'


def toys():
    values = np.zeros((3, 8)); values[:, [0, 3]] = [[.6, .4], [.6, .4], [0, 1]]
    shape = np.ones((6, 8)); shape[:, [0, 3]] = [[1, 0], [0, 1], [2, .2], [.1, 1], [0, 0], [1, 2]]
    component = np.repeat(np.arange(3), 2)
    within = np.full(6, .5)
    p = np.asarray([.2, .3, .5]); s = np.asarray([.4, .1, .5])
    cells = [(shape, component, within, np.arange(6))] * 24
    max_error = 0.
    for coarses in (values, np.repeat(values[:1], 3, axis=0)):
        laws, details = product_laws(coarses, p, {'same': p, 'different': s}, cells)
        for name, shape_p in (('same', p), ('different', s)):
            fuel = coarses[:, None] * shape[None]
            total = fuel.sum(axis=-1)
            joint = p[:, None] * (shape_p[component] * within)[None]
            good = total > 0
            mass = joint[good].sum()
            reference = CompressedLaw(fuel[good] @ EF / total[good], joint[good])
            coarse_post = (joint * good).sum(axis=1) / mass
            shape_post = np.bincount(component, weights=(joint * good).sum(axis=0), minlength=3) / mass
            mean = np.einsum('ij,ijf->f', joint, fuel) / mass
            d = details[name]
            max_error = max(max_error, max(abs(coarse_post - d['effective_coarse_weights'][0])),
                            max(abs(shape_post - d['effective_shape_weights'][0])), max(abs(mean - d['fuel_mean'][0])),
                            abs(float(joint[~good].sum()) - d['rejected_mass'][0]))
            for law in laws[name]:
                if not np.array_equal(reference.x, law.x):
                    raise ValueError('Cartesian toy support differs')
                max_error = max(max_error, float(max(abs(reference.w - law.w))), abs(reference.mean - law.mean),
                                float(max(abs(reference.crps(np.arange(0, 1001, 11)) - law.crps(np.arange(0, 1001, 11))))))
    if max_error > 1e-10:
        raise ValueError('Cartesian toy numerical mismatch')
    return max_error


def preflight(data, monthly):
    toy_error = toys()
    durations = []
    for fold in range(6):
        frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
        predictor = CoarseShapeCoupling(frozen, monthly, COARSE / f'fold_{fold}')
        for q in frozen.queries.profile_id.iloc[[0, -1]]:
            start = time.monotonic()
            laws, _ = check_inputs(predictor, int(q))
            if set(laws) != set(METHODS):
                raise ValueError('preflight method inventory')
            durations.append(time.monotonic() - start)
            del laws
        print(f'Cartesian preflight fold {fold}: {durations[-2:]} seconds', flush=True)
    return {'future_query_input_mutation_cases': len(durations), 'toy_max_error': toy_error,
            'full_guard_seconds': durations, 'peak_rss_native_units': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'rss_units_note': 'macOS bytes; platform-native ru_maxrss recorded, not a portable byte assertion',
            'hourly_outcomes_scored': False}


def run_fold(data, monthly, fold, block, manifest_sha):
    frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
    predictor = CoarseShapeCoupling(frozen, monthly, COARSE / f'fold_{fold}')
    errors = {m: {} for m in METHODS}
    scores, details, radii = [], [], []
    for role in ('calibration', 'test'):
        queries = frozen.queries.loc[frozen.queries.role.eq(role)]
        for number, rec in enumerate(queries.itertuples(index=False), start=1):
            q = int(rec.profile_id)
            laws, records = predictor.build(q)
            identity = {'fold': fold, 'role': role, 'profile_id': q, 'region': rec.region, 'month': rec.month}
            details.extend(identity | r for r in records)
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
                            scores.append(identity | {'budget': 'asof_30d', 'hours': len(actual), 'method': method,
                                                      'radius_policy': policy, 'calibration': calibration, 'radius': radius,
                                                      **score_laws(laws[method], actual, clock, radius)})
            del laws
            if number % 5 == 0 or number == len(queries):
                print(f'Cartesian fold {fold} {role} {number}/{len(queries)}', flush=True)
        if role == 'calibration':
            for method in METHODS:
                if set(errors[method]) != set(queries.region):
                    raise ValueError('calibration roster incomplete')
                for label, radius in zip(('pooled', 'max_region'), weighted_radius(errors[method])):
                    radii.append({'fold': fold, 'method': method, 'calibration': label, 'radius': radius})
            radius_map = {(r['method'], r['calibration']): r['radius'] for r in radii}
    scores, details = pd.DataFrame(scores), pd.DataFrame(details)
    if len(scores) != int(frozen.queries.role.eq('test').sum()) * 14 or len(details) != len(frozen.queries) * 3:
        raise ValueError('fold score/diagnostic denominator')
    if not np.isfinite(scores[list(METRICS) + ['radius']]).all().all():
        raise ValueError('nonfinite formal score')
    old = pd.read_csv(OLD / f'fold_{fold}/test_month_metrics.csv')
    old = old.loc[old.method.isin(REFERENCES)]
    matched = scores.loc[scores.method.isin(REFERENCES)].merge(old, on=['profile_id', 'method', 'calibration', 'radius_policy'],
                                                            validate='one_to_one', suffixes=('_new', '_old'))
    error = max(float(max(abs(matched[m + '_new'] - matched[m + '_old']))) for m in METRICS)
    if len(matched) != int(frozen.queries.role.eq('test').sum()) * 6 or error > 1e-7:
        raise ValueError('V180 diagonal/Gaussian reference mismatch')
    staging = Path(tempfile.mkdtemp(prefix='.v181-cartesian-', dir=OUT.parent))
    for name, frame in [('test_month_metrics', scores), ('coupling_diagnostics', details), ('calibration_radii', pd.DataFrame(radii))]:
        frame.to_csv(staging / f'{name}.csv', index=False)
    result = {'assertion_failures': 0, 'fold': fold, 'proper_regions': frozen.fit, 'manifest_sha256': manifest_sha,
              'score_rows': len(scores), 'diagnostic_rows': len(details), 'reference_rows': len(matched), 'reference_max_error': error,
              'output_hashes': {p.name: sha(p) for p in staging.iterdir()}}
    (staging / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    staging.rename(block)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    destination = PRE if args.preflight else OUT
    if destination.exists() and (args.preflight or not args.resume):
        raise FileExistsError(destination)
    hashes = {}
    for archive in (OLD, ROOT / 'results/v180-rebased-coarse-audit-r1', ROOT / 'results/v181-coarse-shape-inventory-r1'):
        summary = verify(archive)
        for p, h in summary['input_hashes'].items():
            if p in hashes and hashes[p] != h:
                raise ValueError('upstream hash disagreement')
            hashes[p] = h
        for name in ('summary.json', *summary['output_hashes']):
            p = archive / name
            hashes[str(p.relative_to(ROOT))] = sha(p)
    for p in (PROTOCOL, Path(__file__), ROOT / 'scripts/coarse_shape_coupling.py'):
        hashes[str(p.relative_to(ROOT))] = sha(p)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('frozen input changed')
    data, monthly = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
    if args.preflight:
        with threadpool_limits(limits=2):
            result = preflight(data, monthly)
        if any(sha(ROOT / p) != h for p, h in hashes.items()):
            raise ValueError('preflight inputs changed')
        destination.mkdir(parents=True)
        result |= {'assertion_failures': 0, 'input_hashes': hashes, 'output_hashes': {}}
        (destination / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)
        return
    if verify(PRE)['input_hashes'] != hashes:
        raise ValueError('preflight code/data differs')
    hashes[str((PRE / 'summary.json').relative_to(ROOT))] = sha(PRE / 'summary.json')
    manifest = json.dumps({'input_hashes': hashes, 'methods': METHODS}, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if destination.exists():
        if (destination / 'run_manifest.json').read_text() != manifest or (destination / 'summary.json').exists():
            raise ValueError('invalid resume')
    else:
        destination.mkdir(parents=True)
        (destination / 'run_manifest.json').write_text(manifest)
    with threadpool_limits(limits=2):
        for fold in range(6):
            block = destination / f'fold_{fold}'
            if block.exists():
                if verify(block)['manifest_sha256'] != sha(destination / 'run_manifest.json'):
                    raise ValueError('saved block manifest differs')
            else:
                run_fold(data, monthly, fold, block, sha(destination / 'run_manifest.json'))
    scores = pd.concat([pd.read_csv(destination / f'fold_{k}/test_month_metrics.csv') for k in range(6)], ignore_index=True)
    if len(scores) != 3374 or scores.duplicated(['profile_id', 'method', 'radius_policy', 'calibration']).any():
        raise ValueError('complete score inventory')
    regional, macro = aggregate(scores)
    for name, frame in [('test_month_metrics', scores), ('region_metrics', regional), ('macro_metrics', macro)]:
        frame.to_csv(destination / f'{name}.csv', index=False)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('input changed during experiment')
    result = {'assertion_failures': 0, 'completed_blocks': 6, 'score_rows': len(scores), 'input_hashes': hashes,
              'output_hashes': {str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob('*')) if p.is_file()},
              'scope': 'Exact Cartesian coarse/whole-shape independence controls. No new model or target-hour information; exploratory source development, not an original-method claim or independent confirmation.'}
    (destination / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(macro.loc[macro.cohort.eq('all29')].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
