"""Frozen source-only comparison of coarse forecast and scenario updates."""
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
from scenario_reweighting import CoarseForecast, METHODS, ScenarioPrediction
from v151_hourly_law import sha
from v159_fuel_activity_source_pilot import weighted_radius

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
OUT = ROOT / 'results/v164-scenario-reweighting-r1'
PROTOCOL = ROOT / 'docs/V164_SCENARIO_REWEIGHTING_PROTOCOL.md'
METRICS = ('crps', 'mae', 'coverage', 'width', 'winkler', 'tolerant_coverage',
           'lower_miss', 'upper_miss', 'support_below', 'support_above')


def check_mutation(predictor, data, coarse, q):
    """Perturb query-region unreleased months, leaving every legal prefix intact."""
    before = predictor.inputs(q)
    region = data.index.region.iloc[q]
    ids = data.groups[region]
    ids = ids[(data.release.iloc[ids] > data.dates.iloc[q]).to_numpy()]
    saved = [(d.M[ids].copy(), d.shares[ids].copy()) for d in (data, coarse)]
    changed = np.random.default_rng(q).uniform(1, 1000, size=saved[0][0].shape)
    try:
        for d in (data, coarse):
            d.M[ids] = changed
            d.shares[ids] = changed / changed.sum(axis=1, keepdims=True)
        after = predictor.inputs(q)
    finally:
        for d, (monthly, shares) in zip((data, coarse), saved):
            d.M[ids], d.shares[ids] = monthly, shares
    for key in before[0]:
        if not np.array_equal(before[0][key], after[0][key]):
            raise ValueError('unreleased query coarse value affected forecast')
    if any(not np.array_equal(a, b) for a, b in zip(before[1:], after[1:])):
        raise ValueError('unreleased query coarse value affected world prior')


def run_block(data, monthly, fold, block, manifest_hash):
    frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
    coarse = CoarseForecast(monthly, frozen.fit)
    predictor = ScenarioPrediction(frozen, coarse)
    query = frozen.queries
    for q in query.profile_id.iloc[[0, -1]]:
        check_mutation(predictor, data, monthly, int(q))
    errors = {name: {} for name in METHODS}
    forecasts, diagnostics, outputs, radii = [], [], [], []
    for role in ('calibration', 'test'):
        for record in query.loc[query.role.eq(role)].itertuples(index=False):
            q = int(record.profile_id)
            laws, forecast, details = predictor.build(q)
            # Read held-out hourly labels only after constructing all forecasts.
            actual = data.truth(q)
            row = data.index.iloc[q]
            clock = data.states[int(row.offset):int(row.stop)] % 24
            forecasts.append({'fold': fold, 'role': role, 'profile_id': q, 'region': row.region,
                              'month': row.month, 'input_profile_id': forecast['local'],
                              **{k: v for k, v in forecast.items() if k not in ('shares', 'local')},
                              'forecast_shares': json.dumps(forecast['shares'].tolist())})
            for detail in details:
                diagnostics.append({'fold': fold, 'role': role, 'profile_id': q, **detail})
            for method, cells in laws.items():
                if role == 'calibration':
                    law = CalendarLaw(cells, clock)
                    error = np.maximum.reduce([law.quantile(.05) - actual,
                                               actual - law.quantile(.95), np.zeros(len(actual))])
                    errors[method].setdefault(row.region, []).append(error)
                else:
                    for calibration in ('pooled', 'max_region'):
                        radius = radius_map[(method, calibration)]
                        outputs.append({'fold': fold, 'budget': 'asof_30d', 'profile_id': q,
                                        'region': row.region, 'month': row.month, 'hours': len(actual),
                                        'method': method, 'calibration': calibration, 'radius': radius,
                                        **score_laws(cells, actual, clock, radius)})
        if role == 'calibration':
            wanted = set(query.loc[query.role.eq(role), 'region'])
            for method in METHODS:
                if set(errors[method]) != wanted:
                    raise ValueError('calibration cohort mismatch')
                pooled, maximum = weighted_radius(errors[method])
                for name, radius in [('pooled', pooled), ('max_region', maximum)]:
                    radii.append({'fold': fold, 'method': method, 'calibration': name, 'radius': radius})
            radius_map = {(r['method'], r['calibration']): r['radius'] for r in radii}
    rows = pd.DataFrame(outputs)
    expected_count = query.role.eq('test').sum() * len(METHODS) * 2
    if len(rows) != expected_count or not np.isfinite(rows[list(METRICS)]).all().all():
        raise ValueError('test score denominator or finiteness')
    old = pd.read_csv(BASE / f'fold_{fold}_asof_30d/test_month_metrics.csv')
    keys = ['region', 'month', 'method', 'calibration']
    joined = rows.loc[rows.method.isin(METHODS[:2])].merge(old, on=keys, suffixes=('_new', '_old'), validate='one_to_one')
    reproduction = {m: float(np.max(np.abs(joined[m + '_new'] - joined[m + '_old'])))
                    for m in METRICS[:5]}
    if max(reproduction.values()) > 1e-7 or len(joined) != query.role.eq('test').sum() * 4:
        raise ValueError('baseline not reproduced')
    forecasts = pd.DataFrame(forecasts)
    old_forecast = pd.read_csv(ROOT / 'results/v161-monthly-history-value-r1/predictions.csv')
    old_forecast = old_forecast.loc[old_forecast.fold.eq(fold) & old_forecast.method.eq('forest_base')]
    matched = forecasts.loc[forecasts.role.eq('test')].merge(old_forecast, on='profile_id', validate='one_to_one')
    forecast_error = float(np.max(np.abs(matched.raw_anchor - matched.prediction)))
    if len(matched) != len(old_forecast) or forecast_error > 1e-8:
        raise ValueError('coarse forest not reproduced')
    staging = Path(tempfile.mkdtemp(prefix='.v164-', dir=OUT.parent))
    for name, frame in [('test_month_metrics', rows), ('forecasts', forecasts),
                        ('scenario_updates', pd.DataFrame(diagnostics)),
                        ('calibration_radii', pd.DataFrame(radii)), ('coarse_oof', pd.DataFrame(coarse.oof))]:
        frame.to_csv(staging / f'{name}.csv', index=False)
    summary = {'fold': fold, 'manifest_sha256': manifest_hash, 'assertion_failures': 0,
               'proper_regions': frozen.fit, 'coarse_train_profiles': coarse.ids.tolist(),
               'mutation_invariance_cases': 2, 'baseline_records_reproduced': len(joined),
               'baseline_max_errors': reproduction, 'coarse_forecast_max_error': forecast_error,
               'coarse_forecast_records': len(matched), 'score_rows': len(rows),
               'output_hashes': {p.name: sha(p) for p in sorted(staging.iterdir())}}
    (staging / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    staging.rename(block)
    print(f'completed fold {fold}: {len(rows)} scores; baseline max error {max(reproduction.values()):.3g}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if OUT.exists() and not args.resume:
        raise FileExistsError(OUT)
    base = json.loads((BASE / 'summary.json').read_text())
    if base['assertion_failures']:
        raise ValueError('upstream failed')
    hashes = dict(base['input_hashes'])
    for name, digest in base['output_hashes'].items():
        if sha(BASE / name) != digest:
            raise ValueError('source archive changed')
    dependencies = [Path(__file__), PROTOCOL, ROOT / 'scripts/scenario_reweighting.py',
                    ROOT / 'scripts/frozen_recomposition.py', ROOT / 'scripts/level_spread_diagnostics.py',
                    ROOT / 'scripts/monthly_history_features.py', ROOT / 'scripts/v159_fuel_activity_source_pilot.py',
                    BASE / 'summary.json', ROOT / 'results/v161-monthly-history-value-r1/summary.json',
                    ROOT / 'results/v161-monthly-history-value-r1/predictions.csv']
    hashes.update({str(p.relative_to(ROOT)): sha(p) for p in dependencies})
    for name, digest in hashes.items():
        if sha(ROOT / name) != digest:
            raise ValueError('input changed')
    text = json.dumps({'input_hashes': hashes, 'methods': METHODS, 'budget': 'asof_30d'},
                      sort_keys=True, indent=2) + '\n'
    if OUT.exists():
        if (OUT / 'run_manifest.json').read_text() != text or (OUT / 'summary.json').exists():
            raise ValueError('invalid resume')
    else:
        OUT.mkdir(parents=True)
        (OUT / 'run_manifest.json').write_text(text)
    data, monthly = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
    blocks = []
    with threadpool_limits(limits=2):
        for fold in range(6):
            block = OUT / f'fold_{fold}'
            if block.exists():
                saved = json.loads((block / 'summary.json').read_text())
                if saved['manifest_sha256'] != sha(OUT / 'run_manifest.json') or saved['assertion_failures']:
                    raise ValueError('block identity changed')
                for name, digest in saved['output_hashes'].items():
                    if sha(block / name) != digest:
                        raise ValueError('block output changed')
            else:
                run_block(data, monthly, fold, block, sha(OUT / 'run_manifest.json'))
            blocks.append(block)
    rows = pd.concat([pd.read_csv(b / 'test_month_metrics.csv') for b in blocks], ignore_index=True)
    if len(rows) != 241 * len(METHODS) * 2 or rows.duplicated(['profile_id', 'method', 'calibration']).any():
        raise ValueError('aggregate cohort changed')
    regions = []
    for key, group in rows.groupby(['region', 'method', 'calibration']):
        regions.append(dict(zip(['region', 'method', 'calibration'], key))
                       | {m: float(np.average(group[m], weights=group.hours)) for m in METRICS})
    regional = pd.DataFrame(regions)
    macro = []
    for cohort, subset in [('all29', regional), ('nonconstant24', regional.loc[~regional.region.isin(['CHPD', 'DOPD', 'GVL', 'SCL', 'SPA'])])]:
        for key, group in subset.groupby(['method', 'calibration']):
            macro.append({'cohort': cohort, 'method': key[0], 'calibration': key[1], 'regions': len(group),
                          **{m: float(group[m].mean()) for m in METRICS},
                          'worst_coverage': float(group.coverage.min())})
    macro = pd.DataFrame(macro)
    rows.to_csv(OUT / 'test_month_metrics.csv', index=False)
    regional.to_csv(OUT / 'region_metrics.csv', index=False)
    macro.to_csv(OUT / 'macro_metrics.csv', index=False)
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('inputs changed during run')
    summary = {'status': 'SOURCE_ONLY_COARSE_SCENARIO_REWEIGHTING', 'assertion_failures': 0,
               'completed_blocks': len(blocks), 'scored_rows': len(rows), 'input_hashes': hashes,
               'output_hashes': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
               'limitations': ['2024 shared-time source development; not time-forward or untouched target confirmation.',
                               'Synthetic monthly release dates and nominal fixed-EF production proxy.',
                               'Standard forecast/entropy controls, not a novelty claim or a distribution-free target guarantee.']}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(macro.loc[macro.calibration.eq('pooled')].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
