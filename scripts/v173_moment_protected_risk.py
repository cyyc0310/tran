"""Source-only moment-protected zero-event odds updates of physical CIF forecasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from calendar_cif_law import CalendarLaw
from moment_protected_cif import MomentProtectedCIF, METHODS, REFERENCES, SETTINGS
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from level_spread_diagnostics import score_laws
from monthly_history_features import MonthlyInformation
from v151_hourly_law import sha
from v159_fuel_activity_source_pilot import weighted_radius

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
OLD = ROOT / 'results/v165-coarse-distribution-lifting-r1'
ZERO = ROOT / 'results/v171-two-scale-zero-r1'
ZERO_AUDIT = ROOT / 'results/v171-zero-audit-r1'
UNPROTECTED = ROOT / 'results/v172-zero-odds-cif-r1'
UNPROTECTED_AUDIT = ROOT / 'results/v172-zero-odds-audit-r1'
OUT = ROOT / 'results/v173-moment-protected-risk-r1'
PROTOCOL = ROOT / 'docs/V173_MOMENT_PROTECTED_RISK_PROTOCOL.md'
METRICS = ('crps', 'mae', 'coverage', 'width', 'winkler', 'tolerant_coverage',
           'lower_miss', 'upper_miss', 'support_below', 'support_above')


def check_mutation(predictor, data, monthly, q):
    """Unreleased query monthly statistics cannot change any predictive atom."""
    before = predictor.build(q)[0]
    ids = data.groups[data.index.region.iloc[q]]
    ids = ids[(data.release.iloc[ids] > data.dates.iloc[q]).to_numpy()]
    saved = [(d.M[ids].copy(), d.shares[ids].copy()) for d in (data, monthly)]
    changed = np.random.default_rng(q).uniform(1, 1000, size=saved[0][0].shape)
    try:
        for d in (data, monthly):
            d.M[ids] = changed
            d.shares[ids] = changed / changed.sum(axis=1, keepdims=True)
        after = predictor.build(q)[0]
    finally:
        for d, (M, S) in zip((data, monthly), saved):
            d.M[ids], d.shares[ids] = M, S
    for method in METHODS:
        for a, b in zip(before[method], after[method]):
            if not np.array_equal(a.x, b.x) or not np.array_equal(a.w, b.w):
                raise ValueError('unreleased query monthly input affected a predictive law')


def run_block(data, monthly, fold, destination, manifest_hash):
    frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
    predictor = MomentProtectedCIF(frozen, monthly, ZERO / f'fold_{fold}')
    for q in frozen.queries.profile_id.iloc[[0, -1]]:
        check_mutation(predictor, data, monthly, int(q))
    errors = {method: {} for method in METHODS}
    contexts, updates, outputs, radii, certificates = [], [], [], [], []
    for role in ('calibration', 'test'):
        records = frozen.queries.loc[frozen.queries.role.eq(role)]
        for number, rec in enumerate(records.itertuples(index=False), start=1):
            q = int(rec.profile_id)
            laws, forecast, details, cert = predictor.build(q)
            identity = {'fold': fold, 'role': role, 'profile_id': q, 'region': rec.region, 'month': rec.month}
            contexts.append(identity | {k: v for k, v in forecast.items() if k != 'shares'})
            updates.extend(identity | detail for detail in details)
            certificates.extend(identity | record for record in cert)
            # All legal predictions precede the first read of this query's labels.
            actual = data.truth(q)
            row = data.index.iloc[q]
            clock = data.states[int(row.offset):int(row.stop)]
            for method in METHODS:
                cells = laws[method]
                if role == 'calibration':
                    law = CalendarLaw(cells, clock)
                    distance = np.maximum.reduce([law.quantile(.05) - actual,
                                                  actual - law.quantile(.95), np.zeros(len(actual))])
                    errors[method].setdefault(rec.region, []).append(distance)
                else:
                    for policy in (('recalibrated',) if method in REFERENCES else ('fixed', 'recalibrated')):
                        for calibration in ('pooled', 'max_region'):
                            reference = method if method in REFERENCES else SETTINGS[method][0]
                            radius = radius_map[(method if policy == 'recalibrated' else reference, calibration)]
                            outputs.append({**identity, 'budget': 'asof_30d', 'hours': len(actual),
                                            'method': method, 'radius_policy': policy, 'calibration': calibration,
                                            'radius': radius, **score_laws(cells, actual, clock, radius)})
            if number % 10 == 0 or number == len(records):
                print(f'fold {fold} {role}: {number}/{len(records)}', flush=True)
        if role == 'calibration':
            wanted = set(records.region)
            for method in METHODS:
                if set(errors[method]) != wanted:
                    raise ValueError('calibration region denominator')
                for label, radius in zip(('pooled', 'max_region'), weighted_radius(errors[method])):
                    radii.append({'fold': fold, 'method': method, 'calibration': label, 'radius': radius})
            radius_map = {(r['method'], r['calibration']): r['radius'] for r in radii}
    rows = pd.DataFrame(outputs)
    ntest = frozen.queries.role.eq('test').sum()
    if len(rows) != ntest * 68 or not np.isfinite(rows[list(METRICS)]).all().all():
        raise ValueError('score denominator/finiteness')
    old = pd.read_csv(OLD / f'fold_{fold}/test_month_metrics.csv')
    old = old.loc[old.method.isin(REFERENCES)]
    keys = ['profile_id', 'method', 'calibration']
    baseline = rows.loc[rows.method.isin(REFERENCES)].merge(old, on=keys, suffixes=('_new', '_old'), validate='one_to_one')
    reproduction = {m: float(np.max(abs(baseline[m + '_new'] - baseline[m + '_old']))) for m in METRICS}
    if len(baseline) != ntest * 4 or max(reproduction.values()) > 1e-7:
        raise ValueError(f'frozen references not reproduced: {reproduction}')
    staging = Path(tempfile.mkdtemp(prefix='.v173-', dir=OUT.parent))
    for name, frame in [('test_month_metrics', rows), ('contexts', pd.DataFrame(contexts)),
                        ('zero_updates', pd.DataFrame(updates)),
                        ('projection_certificates', pd.DataFrame(certificates)),
                        ('calibration_radii', pd.DataFrame(radii))]:
        frame.to_csv(staging / f'{name}.csv', index=False)
    summary = {'assertion_failures': 0, 'fold': fold, 'proper_regions': frozen.fit,
               'manifest_sha256': manifest_hash, 'score_rows': len(rows),
               'baseline_records_reproduced': len(baseline), 'baseline_max_errors': reproduction,
               'future_month_mutation_cases': 2, 'zero_update_records': len(updates),
               'projection_calls': len(certificates),
               'projection_fallbacks': sum(c['used_prior_fallback'] for c in certificates),
               'max_moment_error': max(c['moment_error'] for c in certificates),
               'output_hashes': {p.name: sha(p) for p in sorted(staging.iterdir())}}
    (staging / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    staging.rename(destination)
    print(f'completed fold {fold}: {len(rows)} scores, references max error {max(reproduction.values()):.3g}', flush=True)


def aggregate(rows):
    keys = ['region', 'method', 'radius_policy', 'calibration']
    regional = pd.DataFrame([dict(zip(keys, key)) | {m: float(np.average(g[m], weights=g.hours)) for m in METRICS}
                             for key, g in rows.groupby(keys)])
    macro = []
    for cohort, table in [('all29', regional), ('nonconstant24', regional.loc[~regional.region.isin(['CHPD', 'DOPD', 'GVL', 'SCL', 'SPA'])])]:
        for key, group in table.groupby(keys[1:]):
            macro.append({'cohort': cohort, **dict(zip(keys[1:], key)), 'regions': len(group),
                          **{m: float(group[m].mean()) for m in METRICS}, 'worst_coverage': float(group.coverage.min())})
    return regional, pd.DataFrame(macro)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if OUT.exists() and not args.resume:
        raise FileExistsError(OUT)
    upstream = json.loads((BASE / 'summary.json').read_text())
    if upstream['assertion_failures']:
        raise ValueError('upstream failed')
    dependencies = [PROTOCOL, Path(__file__), ROOT / 'scripts/zero_odds_cif.py', ROOT / 'scripts/moment_protected_cif.py',
                    ROOT / 'scripts/moment_protection.py', ROOT / 'scripts/v173_preflight_moments.py',
                    ROOT / 'scripts/two_scale_zero.py', ROOT / 'scripts/frozen_recomposition.py',
                    ROOT / 'scripts/scenario_reweighting.py', ROOT / 'scripts/coarse_distribution_lifting.py',
                    ROOT / 'scripts/monthly_history_features.py', ROOT / 'scripts/fuel_activity_forest.py',
                    ROOT / 'scripts/level_spread_diagnostics.py', ROOT / 'scripts/calendar_cif_law.py',
                    BASE / 'summary.json', OLD / 'summary.json', ZERO / 'summary.json', ZERO_AUDIT / 'summary.json',
                    UNPROTECTED / 'summary.json', UNPROTECTED_AUDIT / 'summary.json',
                    ROOT / 'results/v173-moment-preflight-r4/summary.json']
    for archive in (BASE, OLD, ZERO, ZERO_AUDIT, UNPROTECTED, UNPROTECTED_AUDIT, ROOT / 'results/v173-moment-preflight-r4'):
        summary = json.loads((archive / 'summary.json').read_text())
        if summary['assertion_failures']:
            raise ValueError('failed reference archive')
        for name, digest in summary['output_hashes'].items():
            if sha(archive / name) != digest:
                raise ValueError('reference output changed')
            dependencies.append(archive / name)
    hashes = upstream['input_hashes'] | json.loads((ZERO_AUDIT / 'summary.json').read_text())['input_hashes'] | {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('input changed')
    text = json.dumps({'input_hashes': hashes, 'methods': METHODS, 'settings': SETTINGS,
                       'budget': 'asof_30d', 'source_year': 2024}, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if OUT.exists():
        if (OUT / 'run_manifest.json').read_text() != text or (OUT / 'summary.json').exists():
            raise ValueError('invalid resume')
    else:
        OUT.mkdir(parents=True)
        (OUT / 'run_manifest.json').write_text(text)
    data, monthly = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
    with threadpool_limits(limits=2):
        for fold in range(6):
            block = OUT / f'fold_{fold}'
            if block.exists():
                saved = json.loads((block / 'summary.json').read_text())
                if saved['assertion_failures'] or saved['manifest_sha256'] != sha(OUT / 'run_manifest.json'):
                    raise ValueError('invalid saved block')
                for name, digest in saved['output_hashes'].items():
                    if sha(block / name) != digest:
                        raise ValueError('saved block output changed')
            else:
                run_block(data, monthly, fold, block, sha(OUT / 'run_manifest.json'))
    rows = pd.concat([pd.read_csv(OUT / f'fold_{k}/test_month_metrics.csv') for k in range(6)], ignore_index=True)
    if len(rows) != 16388 or rows.duplicated(['profile_id', 'method', 'radius_policy', 'calibration']).any():
        raise ValueError('aggregate score denominator')
    regional, macro = aggregate(rows)
    for name, frame in [('test_month_metrics', rows), ('region_metrics', regional), ('macro_metrics', macro)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('inputs changed during experiment')
    result = {'status': 'SOURCE_ONLY_MOMENT_PROTECTED_RISK', 'assertion_failures': 0,
              'completed_blocks': 6, 'scored_rows': len(rows), 'input_hashes': hashes,
              'output_hashes': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
              'limitations': ['Standard KL moment projection of frozen odds updates; not a new method claim.',
                              'Physical supports and prior generation/emissions expectations preserved to numerical tolerance; explicit source-prior solver fallbacks recorded.',
                              '2024 shared-time source development, synthetic release dates, nominal fixed-EF production proxy.',
                              'No unopened target regions evaluated; no distribution-free target coverage guarantee.']}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(macro.loc[macro.cohort.eq('all29') & macro.calibration.eq('pooled')].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
