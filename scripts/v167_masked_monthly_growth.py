"""Source-only ordinary controls for missing monthly fuel-growth responses."""
from pathlib import Path
import json
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from calendar_cif_law import CalendarLaw
from masked_monthly_growth import MaskedMonthlyGrowth, METHODS, REFERENCE
from monthly_history_features import MonthlyInformation
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from level_spread_diagnostics import score_laws
from v151_hourly_law import sha
from v159_fuel_activity_source_pilot import weighted_radius

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
OUT = ROOT / 'results/v167-masked-monthly-growth-r1'
PROTOCOL = ROOT / 'docs/V167_MASKED_GROWTH_PROTOCOL.md'
METRICS = ['crps', 'mae', 'coverage', 'width', 'winkler', 'tolerant_coverage',
           'lower_miss', 'upper_miss', 'support_below', 'support_above']


def check_mutation(predictor, data, monthly, q):
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
                raise ValueError('unreleased target statistics changed predictive law')


def run_block(data, fold, destination, manifest_hash):
    frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
    monthly = MonthlyInformation(ROOT)
    diagnostic = MaskedMonthlyGrowth(frozen, monthly)
    for q in frozen.queries.profile_id.iloc[[0, -1]]:
        check_mutation(diagnostic, data, monthly, int(q))
    contexts, worlds, sensitivity, outputs = [], [], [], []
    errors = {name: {} for name in METHODS}
    for role in ['calibration', 'test']:
        for rec in frozen.queries.loc[frozen.queries.role.eq(role)].itertuples(index=False):
            q = int(rec.profile_id)
            laws, context, donor_records, sensitivity_records = diagnostic.build(q)
            identity = {'fold': fold, 'role': role, 'profile_id': q, 'region': rec.region, 'month': rec.month}
            contexts.append(identity | context)
            worlds.extend(identity | record for record in donor_records)
            sensitivity.extend(identity | record for record in sensitivity_records)
            # All forecasts are constructed before reading held-out hourly labels.
            actual = data.truth(q)
            row = data.index.iloc[q]
            clock = data.states[int(row.offset):int(row.stop)] % 24
            for method in METHODS:
                cells = laws[method]
                if role == 'calibration':
                    law = CalendarLaw(cells, clock)
                    distance = np.maximum.reduce([law.quantile(.05) - actual,
                                                  actual - law.quantile(.95), np.zeros(len(actual))])
                    errors[method].setdefault(rec.region, []).append(distance)
                else:
                    policies = ['recalibrated'] if method in METHODS[:3] else ['fixed', 'recalibrated']
                    for policy in policies:
                        for calibration in ['pooled', 'max_region']:
                            radius = own_radii[(method, calibration)] if policy == 'recalibrated' else frozen.radii[(REFERENCE[method], calibration)]
                            outputs.append({'fold': fold, 'budget': 'asof_30d', 'profile_id': q,
                                            'region': rec.region, 'month': rec.month, 'hours': len(actual),
                                            'method': method, 'is_oracle': method.startswith('oracle_'),
                                            'radius_policy': policy, 'calibration': calibration, 'radius': radius,
                                            **score_laws(cells, actual, clock, radius)})
        if role == 'calibration':
            radii = []
            wanted = set(frozen.queries.loc[frozen.queries.role.eq(role), 'region'])
            for method in METHODS:
                if set(errors[method]) != wanted:
                    raise ValueError('calibration cohort')
                for label, radius in zip(['pooled', 'max_region'], weighted_radius(errors[method])):
                    radii.append({'fold': fold, 'method': method, 'calibration': label, 'radius': radius})
            own_radii = {(r['method'], r['calibration']): r['radius'] for r in radii}
    rows = pd.DataFrame(outputs)
    if not np.isfinite(rows[METRICS]).all().all() or len(rows) != frozen.queries.role.eq('test').sum() * 26:
        raise ValueError('score denominator/finite')
    old = pd.read_csv(BASE / f'fold_{fold}_asof_30d/test_month_metrics.csv')
    keys = ['region', 'month', 'method', 'calibration']
    baseline = rows.loc[rows.method.isin(METHODS[:2])].merge(old, on=keys, suffixes=('_new', '_old'), validate='one_to_one')
    reproduction = {m: float(np.max(abs(baseline[m + '_new'] - baseline[m + '_old']))) for m in METRICS[:5]}
    if len(baseline) != frozen.queries.role.eq('test').sum() * 4 or max(reproduction.values()) > 1e-7:
        raise ValueError('baseline reproduction')
    gaussian_old = pd.read_csv(ROOT / 'results/v164-scenario-reweighting-r1/test_month_metrics.csv')
    gaussian_old = gaussian_old.loc[gaussian_old.fold.eq(fold) & gaussian_old.method.eq('world_gaussian')]
    gaussian = rows.loc[rows.method.eq('world_gaussian')].merge(gaussian_old, on=keys, suffixes=('_new', '_old'), validate='one_to_one')
    gaussian_error = max(float(np.max(abs(gaussian[m + '_new'] - gaussian[m + '_old']))) for m in METRICS[:5])
    if len(gaussian) != len(gaussian_old) or gaussian_error > 1e-7:
        raise ValueError('V164 Gaussian not reproduced')
    staging = Path(tempfile.mkdtemp(prefix='.v167-', dir=OUT.parent))
    for name, frame in [('test_month_metrics', rows), ('contexts', pd.DataFrame(contexts)),
                        ('donor_exposures', pd.DataFrame(worlds)), ('carrier_sensitivity', pd.DataFrame(sensitivity)),
                        ('calibration_radii', pd.DataFrame(radii))]:
        frame.to_csv(staging / f'{name}.csv', index=False)
    summary = {'assertion_failures': 0, 'fold': fold, 'proper_regions': frozen.fit,
               'manifest_sha256': manifest_hash, 'score_rows': len(rows),
               'baseline_records_reproduced': len(baseline), 'baseline_max_errors': reproduction,
               'gaussian_records_reproduced': len(gaussian), 'gaussian_max_error': gaussian_error,
               'future_month_mutation_cases': 2,
               'output_hashes': {p.name: sha(p) for p in sorted(staging.iterdir())}}
    (staging / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    staging.rename(destination)
    print(f'completed diagnostic fold {fold}: {len(rows)} scores', flush=True)


def aggregate(rows):
    records = []
    keys = ['region', 'method', 'radius_policy', 'calibration']
    for key, group in rows.groupby(keys):
        records.append(dict(zip(keys, key)) | {m: float(np.average(group[m], weights=group.hours)) for m in METRICS})
    regional = pd.DataFrame(records)
    macro = []
    for cohort, table in [('all29', regional), ('nonconstant24', regional.loc[~regional.region.isin(['CHPD', 'DOPD', 'GVL', 'SCL', 'SPA'])])]:
        for key, group in table.groupby(keys[1:]):
            macro.append({'cohort': cohort, **dict(zip(keys[1:], key)), 'regions': len(group),
                          **{m: float(group[m].mean()) for m in METRICS}, 'worst_coverage': float(group.coverage.min())})
    return regional, pd.DataFrame(macro)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    upstream = json.loads((BASE / 'summary.json').read_text())
    if upstream['assertion_failures']:
        raise ValueError('upstream failed')
    for name, digest in upstream['output_hashes'].items():
        if sha(BASE / name) != digest:
            raise ValueError('upstream archive changed')
    dependencies = [PROTOCOL, Path(__file__), ROOT / 'scripts/coarse_oracle_zero_audit.py', ROOT / 'scripts/masked_monthly_growth.py',
                    ROOT / 'results/v164-scenario-reweighting-r1/summary.json',
                    ROOT / 'results/v164-scenario-reweighting-r1/test_month_metrics.csv',
                    ROOT / 'scripts/frozen_recomposition.py', ROOT / 'scripts/level_spread_diagnostics.py',
                    ROOT / 'scripts/scenario_reweighting.py', ROOT / 'scripts/coarse_distribution_lifting.py',
                    ROOT / 'scripts/monthly_history_features.py', BASE / 'summary.json']
    hashes = upstream['input_hashes'] | {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('input changed')
    OUT.mkdir(parents=True)
    manifest = {'input_hashes': hashes, 'methods': METHODS, 'source_year': 2024,
                'information': 'Only released query monthly prefixes; hourly labels used only for calibration/evaluation.'}
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    data = HourlyFuelData(ROOT)
    with threadpool_limits(limits=2):
        for fold in range(6):
            run_block(data, fold, OUT / f'fold_{fold}', sha(OUT / 'run_manifest.json'))
    rows = pd.concat([pd.read_csv(OUT / f'fold_{k}/test_month_metrics.csv') for k in range(6)], ignore_index=True)
    if len(rows) != 6266 or rows.duplicated(['profile_id', 'method', 'radius_policy', 'calibration']).any():
        raise ValueError('aggregate cohort')
    regional, macro = aggregate(rows)
    for name, frame in [('test_month_metrics', rows), ('region_metrics', regional), ('macro_metrics', macro)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('dependency changed during experiment')
    result = {'status': 'SOURCE_ONLY_MASKED_MONTHLY_GROWTH_CONTROLS', 'assertion_failures': 0,
              'scored_rows': len(rows), 'completed_blocks': 6, 'input_hashes': hashes,
              'output_hashes': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
              'limitations': ['Neutral growth and donor weighting are ordinary controls, not a novelty claim.',
                              'Missing fuel-hour shapes still use q=1; no conditional shape completion yet.',
                              'Shared-time 2024 source development, synthetic release dates and nominal fixed-EF production proxy.']}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(macro.loc[macro.cohort.eq('all29') & macro.calibration.eq('pooled')].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
