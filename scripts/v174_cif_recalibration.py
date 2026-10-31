"""Compare ordinary PIT recalibration against frozen hourly CIF strong baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from calendar_cif_law import CalendarLaw
from cif_pit_recalibration import PITRecalibration, METHODS, REFERENCES, SETTINGS
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from level_spread_diagnostics import score_laws
from monthly_history_features import MonthlyInformation
from v151_hourly_law import sha
from v159_fuel_activity_source_pilot import weighted_radius
from v170_fuel_marginal_diagnosis import verify
from v173_moment_protected_risk import METRICS, aggregate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
OLD = ROOT / 'results/v165-coarse-distribution-lifting-r1'
OOF = ROOT / 'results/v174-nested-pit-source-r1'
OUT = ROOT / 'results/v174-cif-recalibration-r1'
PROTOCOL = ROOT / 'docs/V174_CIF_RECALIBRATION_PROTOCOL.md'


def check_mutation(predictor, data, monthly, q):
    before = predictor.build(q)[0]
    ids = data.groups[data.index.region.iloc[q]]
    ids = ids[(data.release.iloc[ids] > data.dates.iloc[q]).to_numpy()]
    saved = [(d.M[ids].copy(), d.shares[ids].copy()) for d in (data, monthly)]
    changed = np.random.default_rng(q).uniform(1, 1000, size=saved[0][0].shape)
    try:
        for d in (data, monthly):
            d.M[ids], d.shares[ids] = changed, changed / changed.sum(axis=1, keepdims=True)
        after = predictor.build(q)[0]
    finally:
        for d, (M, S) in zip((data, monthly), saved):
            d.M[ids], d.shares[ids] = M, S
    for name in METHODS:
        for a, b in zip(before[name], after[name]):
            if not np.array_equal(a.x, b.x) or not np.array_equal(a.w, b.w):
                raise ValueError('unreleased coarse input changed a prediction')


def run_fold(data, monthly, fold, destination, manifest_hash):
    frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
    predictor = PITRecalibration(frozen, monthly, OOF / f'fold_{fold}')
    for q in frozen.queries.profile_id.iloc[[0, -1]]:
        check_mutation(predictor, data, monthly, int(q))
    errors = {name: {} for name in METHODS}
    contexts, updates, radii, scores = [], [], [], []
    for role in ('calibration', 'test'):
        records = frozen.queries.loc[frozen.queries.role.eq(role)]
        for number, rec in enumerate(records.itertuples(index=False), start=1):
            q = int(rec.profile_id)
            laws, forecast, details = predictor.build(q)
            identity = {'fold': fold, 'role': role, 'profile_id': q, 'region': rec.region, 'month': rec.month}
            contexts.append(identity | {k: v for k, v in forecast.items() if k != 'shares'})
            updates.extend(identity | detail for detail in details)
            actual = data.truth(q)
            row = data.index.iloc[q]
            clock = data.states[int(row.offset):int(row.stop)]
            for method in METHODS:
                if role == 'calibration':
                    law = CalendarLaw(laws[method], clock)
                    error = np.maximum.reduce([law.quantile(.05) - actual, actual - law.quantile(.95), np.zeros(len(actual))])
                    errors[method].setdefault(rec.region, []).append(error)
                else:
                    policies = ('recalibrated',) if method in REFERENCES else ('fixed', 'recalibrated')
                    for policy in policies:
                        for calibration in ('pooled', 'max_region'):
                            reference = method if method in REFERENCES else SETTINGS[method][0]
                            radius = radius_map[(method if policy == 'recalibrated' else reference, calibration)]
                            scores.append(identity | {'budget': 'asof_30d', 'hours': len(actual), 'method': method,
                                                      'radius_policy': policy, 'calibration': calibration, 'radius': radius,
                                                      **score_laws(laws[method], actual, clock, radius)})
            if number % 10 == 0 or number == len(records):
                print(f'fold {fold} {role}: {number}/{len(records)}', flush=True)
        if role == 'calibration':
            for name in METHODS:
                if set(errors[name]) != set(records.region):
                    raise ValueError('calibration roster incomplete')
                for label, radius in zip(('pooled', 'max_region'), weighted_radius(errors[name])):
                    radii.append({'fold': fold, 'method': name, 'calibration': label, 'radius': radius})
            radius_map = {(r['method'], r['calibration']): r['radius'] for r in radii}
    rows = pd.DataFrame(scores)
    if len(rows) != frozen.queries.role.eq('test').sum() * 52 or not np.isfinite(rows[list(METRICS)]).all().all():
        raise ValueError('score count or finiteness')
    old = pd.read_csv(OLD / f'fold_{fold}/test_month_metrics.csv')
    old = old.loc[old.method.isin(REFERENCES)]
    reproduced = rows.loc[rows.method.isin(REFERENCES)].merge(old, on=['profile_id', 'method', 'calibration'],
                                                           suffixes=('_new', '_old'), validate='one_to_one')
    max_error = max(float(max(abs(reproduced[m + '_new'] - reproduced[m + '_old']))) for m in METRICS)
    if len(reproduced) != frozen.queries.role.eq('test').sum() * 4 or max_error > 1e-7:
        raise ValueError('original references not reproduced')
    staging = Path(tempfile.mkdtemp(prefix='.v174-cif-', dir=OUT.parent))
    for name, frame in [('test_month_metrics', rows), ('contexts', pd.DataFrame(contexts)),
                        ('recalibration_maps', pd.DataFrame(updates)), ('calibration_radii', pd.DataFrame(radii))]:
        frame.to_csv(staging / f'{name}.csv', index=False)
    summary = {'assertion_failures': 0, 'fold': fold, 'proper_regions': frozen.fit,
               'manifest_sha256': manifest_hash, 'score_rows': len(rows), 'map_records': len(updates),
               'baseline_records_reproduced': len(reproduced), 'baseline_max_error': max_error,
               'future_month_mutation_cases': 2,
               'output_hashes': {p.name: sha(p) for p in staging.iterdir()}}
    (staging / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    staging.rename(destination)
    print(f'completed CIF fold {fold}: {len(rows)} scores', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if OUT.exists() and not args.resume:
        raise FileExistsError(OUT)
    archives = [BASE, OLD, OOF, ROOT / 'results/v174-pit-preflight-r1',
                ROOT / 'results/v174-nested-pit-audit-r1']
    hashes = {}
    for archive in archives:
        summary = verify(archive)
        hashes |= summary.get('input_hashes', {})
        for name in ['summary.json', *summary['output_hashes']]:
            hashes[str((archive / name).relative_to(ROOT))] = sha(archive / name)
    dependencies = [PROTOCOL, Path(__file__), ROOT / 'scripts/cif_pit_recalibration.py',
                    ROOT / 'scripts/v174_nested_pit_archive.py', ROOT / 'scripts/v173_moment_protected_risk.py',
                    ROOT / 'scripts/coarse_distribution_lifting.py', ROOT / 'scripts/scenario_reweighting.py',
                    ROOT / 'scripts/frozen_recomposition.py', ROOT / 'scripts/monthly_history_features.py',
                    ROOT / 'scripts/level_spread_diagnostics.py', ROOT / 'scripts/v170_fuel_marginal_diagnosis.py']
    hashes |= {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    # Overlapping archives must bind identical current code and data.
    for archive in archives:
        prior = json.loads((archive / 'summary.json').read_text()).get('input_hashes', {})
        if any(hashes[name] != digest for name, digest in prior.items()):
            raise ValueError('upstream dependency disagreement')
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('input changed')
    manifest = json.dumps({'input_hashes': hashes, 'methods': METHODS, 'settings': SETTINGS,
                           'budget': 'asof_30d', 'source_year': 2024}, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if OUT.exists():
        if (OUT / 'run_manifest.json').read_text() != manifest or (OUT / 'summary.json').exists():
            raise ValueError('invalid resume')
    else:
        OUT.mkdir(parents=True)
        (OUT / 'run_manifest.json').write_text(manifest)
    data, monthly = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
    with threadpool_limits(limits=2):
        for fold in range(6):
            destination = OUT / f'fold_{fold}'
            if destination.exists():
                saved = verify(destination)
                if saved['manifest_sha256'] != sha(OUT / 'run_manifest.json'):
                    raise ValueError('saved block manifest mismatch')
            else:
                run_fold(data, monthly, fold, destination, sha(OUT / 'run_manifest.json'))
    rows = pd.concat([pd.read_csv(OUT / f'fold_{k}/test_month_metrics.csv') for k in range(6)], ignore_index=True)
    if len(rows) != 12532 or rows.duplicated(['profile_id', 'method', 'radius_policy', 'calibration']).any():
        raise ValueError('complete score inventory')
    regional, macro = aggregate(rows)
    for name, frame in [('test_month_metrics', rows), ('region_metrics', regional), ('macro_metrics', macro)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('inputs changed during experiment')
    result = {'assertion_failures': 0, 'completed_blocks': 6, 'score_rows': len(rows), 'input_hashes': hashes,
              'output_hashes': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
              'scope': 'Ordinary empirical PIT-CDF regression baseline, not a novel method or exact CRPS-map optimizer; shared-2024 source development only.'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(macro.loc[macro.cohort.eq('all29') & macro.calibration.eq('pooled')].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
