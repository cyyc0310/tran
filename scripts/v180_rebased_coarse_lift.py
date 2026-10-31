"""Evaluate ordinary query-base coarse composition and hourly shape recomposition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from calendar_cif_law import CalendarLaw
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from level_spread_diagnostics import score_laws
from monthly_history_features import MonthlyInformation
from rebased_coarse_lift import RebasedCoarseLift, METHODS, REFERENCES, SETTINGS, check_query_inputs
from v151_hourly_law import sha, EF
from v159_fuel_activity_source_pilot import weighted_radius
from v170_fuel_marginal_diagnosis import verify
from v173_moment_protected_risk import METRICS, aggregate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
COARSE = ROOT / 'results/v178-joint-coarse-world-r1'
OUT = ROOT / 'results/v180-rebased-coarse-lift-r1'
PRE = ROOT / 'results/v180-rebased-coarse-preflight-r1'
PROTOCOL = ROOT / 'docs/V180_REBASED_COARSE_LIFT_PROTOCOL.md'


def preflight(data, monthly):
    cases, sample_rows, scale_error = 0, 0, 0.
    for fold in range(6):
        frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
        predictor = RebasedCoarseLift(frozen, monthly, COARSE / f'fold_{fold}')
        sample_rows += predictor.sample_checks
        for q in frozen.queries.profile_id.iloc[[0, -1]]:
            check_query_inputs(predictor, int(q))
            cases += 1
        for shape, component, _, raw_ids in predictor.cells:
            profiles = predictor.profiles[component]
            mean = monthly.M[profiles] / data.index.hours.to_numpy()[profiles, None]
            for multiplier in (2., 4.):
                other = np.divide(multiplier * data.raw[raw_ids], multiplier * mean,
                                  out=np.ones_like(shape), where=mean > 0)
                scale_error = max(scale_error, float(np.max(abs(shape - other))))
        print(f'preflight rebased lift fold {fold}', flush=True)
    if scale_error > 1e-12 or cases != 12:
        raise ValueError('source scaling/input preflight failed')
    return {'query_input_mutation_cases': cases, 'source_sample_rows_across_folds': sample_rows,
            'max_shape_scale_error': scale_error, 'hourly_outcomes_scored': False}


def run_fold(data, monthly, fold, destination, manifest_sha):
    frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
    predictor = RebasedCoarseLift(frozen, monthly, COARSE / f'fold_{fold}')
    errors = {m: {} for m in METHODS}
    scores, details, radii = [], [], []
    for role in ('calibration', 'test'):
        queries = frozen.queries.loc[frozen.queries.role.eq(role)]
        for number, rec in enumerate(queries.itertuples(index=False), start=1):
            q = int(rec.profile_id)
            laws, _, diagnostic = predictor.build(q)
            identity = {'fold': fold, 'role': role, 'profile_id': q, 'region': rec.region, 'month': rec.month}
            details.extend(identity | d for d in diagnostic)
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
                            reference = method if policy == 'recalibrated' else 'world_gaussian'
                            radius = radius_map[(reference, calibration)]
                            scores.append(identity | {'budget': 'asof_30d', 'hours': len(actual), 'method': method,
                                                      'radius_policy': policy, 'calibration': calibration, 'radius': radius,
                                                      **score_laws(laws[method], actual, clock, radius)})
            if number % 10 == 0 or number == len(queries):
                print(f'rebased fold {fold} {role}: {number}/{len(queries)}', flush=True)
        if role == 'calibration':
            for method in METHODS:
                if set(errors[method]) != set(queries.region):
                    raise ValueError('calibration region inventory')
                for label, radius in zip(('pooled', 'max_region'), weighted_radius(errors[method])):
                    radii.append({'fold': fold, 'method': method, 'calibration': label, 'radius': radius})
            radius_map = {(r['method'], r['calibration']): r['radius'] for r in radii}
    scores, details = pd.DataFrame(scores), pd.DataFrame(details)
    ntest = int(frozen.queries.role.eq('test').sum())
    if len(scores) != 40 * ntest or len(details) != 9 * len(frozen.queries) or not np.isfinite(scores[list(METRICS)]).all().all():
        raise ValueError('fold score/diagnostic denominator')
    old = pd.read_csv(COARSE / f'fold_{fold}/test_month_metrics.csv')
    old = old.loc[old.method.isin(REFERENCES)]
    paired = scores.loc[scores.method.isin(REFERENCES)].merge(old, on=['profile_id', 'method', 'calibration'], validate='one_to_one', suffixes=('_new', '_old'))
    error = max(float(max(abs(paired[m + '_new'] - paired[m + '_old']))) for m in METRICS)
    if len(paired) != ntest * 4 or error > 1e-7:
        raise ValueError('original reference reproduction')
    staging = Path(tempfile.mkdtemp(prefix='.v180-lift-', dir=OUT.parent))
    for name, frame in [('test_month_metrics', scores), ('world_diagnostics', details), ('calibration_radii', pd.DataFrame(radii))]:
        frame.to_csv(staging / f'{name}.csv', index=False)
    summary = {'assertion_failures': 0, 'fold': fold, 'proper_regions': frozen.fit,
               'manifest_sha256': manifest_sha, 'score_rows': len(scores), 'diagnostic_rows': len(details),
               'reference_records': len(paired), 'reference_max_error': error,
               'output_hashes': {p.name: sha(p) for p in staging.iterdir()}}
    (staging / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    staging.rename(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    destination = PRE if args.preflight else OUT
    if destination.exists() and (args.preflight or not args.resume):
        raise FileExistsError(destination)
    hashes = {}
    for directory in (BASE, COARSE, ROOT / 'results/v179-increment-support-r1', ROOT / 'results/v179-increment-support-audit-r1'):
        summary = verify(directory)
        for p, digest in summary.get('input_hashes', {}).items():
            if p in hashes and hashes[p] != digest:
                raise ValueError('upstream hash disagreement')
            hashes[p] = digest
        for name in ['summary.json', *summary['output_hashes']]:
            p = directory / name
            hashes[str(p.relative_to(ROOT))] = sha(p)
    for p in (PROTOCOL, Path(__file__), ROOT / 'scripts/rebased_coarse_lift.py'):
        hashes[str(p.relative_to(ROOT))] = sha(p)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('input changed')
    data, monthly = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
    if args.preflight:
        with threadpool_limits(limits=2):
            result = preflight(data, monthly)
        if any(sha(ROOT / p) != h for p, h in hashes.items()):
            raise ValueError('preflight input changed')
        destination.mkdir(parents=True)
        result |= {'assertion_failures': 0, 'input_hashes': hashes, 'output_hashes': {}}
        (destination / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)
        return
    prepared = verify(PRE)
    if prepared['input_hashes'] != hashes:
        raise ValueError('preflight source/code differs')
    hashes[str((PRE / 'summary.json').relative_to(ROOT))] = sha(PRE / 'summary.json')
    manifest = json.dumps({'input_hashes': hashes, 'methods': METHODS, 'settings': SETTINGS}, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if destination.exists():
        if (destination / 'run_manifest.json').read_text() != manifest or (destination / 'summary.json').exists():
            raise ValueError('invalid resume manifest')
    else:
        destination.mkdir(parents=True)
        (destination / 'run_manifest.json').write_text(manifest)
    with threadpool_limits(limits=2):
        for fold in range(6):
            block = destination / f'fold_{fold}'
            if block.exists():
                if verify(block)['manifest_sha256'] != sha(destination / 'run_manifest.json'):
                    raise ValueError('saved block manifest mismatch')
            else:
                run_fold(data, monthly, fold, block, sha(destination / 'run_manifest.json'))
    rows = pd.concat([pd.read_csv(destination / f'fold_{k}/test_month_metrics.csv') for k in range(6)], ignore_index=True)
    if len(rows) != 9640 or rows.duplicated(['profile_id', 'method', 'radius_policy', 'calibration']).any():
        raise ValueError('complete score inventory')
    regional, macro = aggregate(rows)
    for name, frame in [('test_month_metrics', rows), ('region_metrics', regional), ('macro_metrics', macro)]:
        frame.to_csv(destination / f'{name}.csv', index=False)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('input changed during run')
    result = {'assertion_failures': 0, 'completed_blocks': 6, 'score_rows': len(rows), 'input_hashes': hashes,
              'output_hashes': {str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob('*')) if p.is_file()},
              'scope': 'Ordinary additive coarse support and sampled within-fuel shape recomposition; fixed V178 coarse kernels; source development, no original-method or new-region confirmation.'}
    (destination / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(macro.loc[macro.cohort.eq('all29') & macro.calibration.eq('pooled')].sort_values('crps').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
