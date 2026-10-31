"""Source-only direct, persistence, and ordinary two-level zero-event baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from monthly_history_features import MonthlyInformation
from two_scale_zero import TwoScaleZero, METHODS, MONTH_METHODS, REFERENCES, reference_probabilities
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import FUELS, verify

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
WORLD = ROOT / 'results/v169-feasible-copula-r1'
OLD = ROOT / 'results/v170-fuel-marginal-diagnosis-r1'
OUT = ROOT / 'results/v171-two-scale-zero-r1'
PROTOCOL = ROOT / 'docs/V171_TWO_SCALE_ZERO_PROTOCOL.md'
COMPONENTS = ('clock', 'conditional43')
METRICS = ('brier', 'mean_forecast', 'observed_fraction', 'calibration_bias')


def score_probability(probability, observed):
    probability, observed = np.broadcast_arrays(np.asarray(probability, float), np.asarray(observed, float))
    return {'brier': float(np.mean((probability - observed)**2)), 'mean_forecast': float(probability.mean()),
            'observed_fraction': float(observed.mean()), 'calibration_bias': float(probability.mean() - observed.mean())}


def check_mutation(predictor, data, monthly, q):
    before = predictor.predict(q)[:3]
    ids = data.groups[data.index.region.iloc[q]]
    unreleased = ids[(data.release.iloc[ids] > data.dates.iloc[q]).to_numpy()]
    saved = [(d.M[unreleased].copy(), d.shares[unreleased].copy()) for d in (data, monthly)]
    changed = np.random.default_rng(q).uniform(1, 1000, size=saved[0][0].shape)
    try:
        for d in (data, monthly):
            d.M[unreleased] = changed
            d.shares[unreleased] = changed / changed.sum(axis=1, keepdims=True)
        after_month = predictor.predict(q)[:3]
    finally:
        for d, (M, S) in zip((data, monthly), saved):
            d.M[unreleased], d.shares[unreleased] = M, S
    raw_ids = np.concatenate([np.arange(int(data.index.offset.iloc[p]), int(data.index.stop.iloc[p])) for p in ids])
    raw_saved = data.raw[raw_ids].copy()
    try:
        data.raw[raw_ids] = np.random.default_rng(q).uniform(1, 500, size=raw_saved.shape)
        after_hour = predictor.predict(q)[:3]
    finally:
        data.raw[raw_ids] = raw_saved
    for before_dict, month_dict, hour_dict in zip(before, after_month, after_hour):
        for name, value in before_dict.items():
            if not np.array_equal(value, month_dict[name]) or not np.array_equal(value, hour_dict[name]):
                raise ValueError('unreleased monthly or held hourly values affected probability')


def score_query(data, rec, hourly, monthly, components):
    q = int(rec.profile_id)
    row = data.index.iloc[q]
    states = data.states[int(row.offset):int(row.stop)]
    actual = data.raw[int(row.offset):int(row.stop)] == 0
    local = data.local(q, 'asof_30d')
    outputs = {key: [] for key in ('hourly', 'monthly', 'components')}
    for j, fuel in enumerate(FUELS):
        identity = {'profile_id': q, 'region': rec.region, 'month': rec.month, 'hours': len(states), 'fuel': fuel,
                    'input_fuel_positive': bool(data.M[local, j] > 0), 'current_fuel_positive': bool(data.M[q, j] > 0)}
        for name, probability in hourly.items():
            outputs['hourly'].append(identity | {'method': name, **score_probability(probability[states, j], actual[:, j])})
        for name, probability in monthly.items():
            outputs['monthly'].append(identity | {'method': name, 'is_world_proxy': name in REFERENCES,
                                                  **score_probability(probability[j], data.M[q, j] == 0)})
        if data.M[q, j] > 0:
            for name, probability in components.items():
                outputs['components'].append(identity | {'method': name, **score_probability(probability[states, j], actual[:, j])})
    return outputs


def run_block(data, monthly, fold, destination, manifest_hash):
    frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
    predictor = TwoScaleZero(frozen, monthly)
    worlds = pd.read_csv(WORLD / f'fold_{fold}/world_probabilities.csv')
    for q in frozen.queries.profile_id.iloc[[0, -1]]:
        check_mutation(predictor, data, monthly, int(q))
    saved_hourly, saved_monthly, saved_components, contexts = [], [], [], []
    outputs = {key: [] for key in ('hourly', 'monthly', 'components')}
    for number, rec in enumerate(frozen.queries.itertuples(index=False), start=1):
        q = int(rec.profile_id)
        hourly, month_probability, components, context = predictor.predict(q)
        reference_hourly, reference_monthly = reference_probabilities(frozen, worlds, q)
        hourly = reference_hourly | hourly
        month_probability = reference_monthly | month_probability
        saved_hourly.append(np.stack([hourly[m] for m in METHODS]))
        saved_monthly.append(np.stack([month_probability[m] for m in MONTH_METHODS]))
        saved_components.append(np.stack([components[m] for m in COMPONENTS]))
        contexts.append({'profile_id': q, 'local': context['local'], 'base': context['base'].tolist(),
                         'history': context['history'].tolist(), 'prefix': context['prefix']})
        # Future hourly and monthly labels are read only after legal forecasts exist.
        if rec.role == 'test':
            scores = score_query(data, rec, hourly, month_probability, components)
            for key in outputs:
                outputs[key].extend({'fold': fold} | r for r in scores[key])
        if number % 20 == 0 or number == len(frozen.queries):
            print(f'fold {fold}: {number}/{len(frozen.queries)} query forecasts', flush=True)
    frames = {key: pd.DataFrame(value) for key, value in outputs.items()}
    ntest = int(frozen.queries.role.eq('test').sum())
    if len(frames['hourly']) != ntest * 64 or len(frames['monthly']) != ntest * 40:
        raise ValueError('block score denominator')
    old = pd.read_csv(OLD / 'test_month_metrics.csv')
    old = old.loc[old.fold.eq(fold) & old.arm.eq('legal')]
    pair = frames['hourly'].loc[frames['hourly'].method.isin(REFERENCES)].merge(
        old, on=['profile_id', 'method', 'fuel'], suffixes=('_new', '_old'), validate='one_to_one')
    reproduction = {'brier': float(max(abs(pair.brier - pair.zero_brier))),
                    'mean_forecast': float(max(abs(pair.mean_forecast - pair.predicted_zero_probability))),
                    'observed_fraction': float(max(abs(pair.observed_fraction - pair.actual_zero_fraction)))}
    if len(pair) != ntest * 16 or max(reproduction.values()) > 1e-12:
        raise ValueError('old zero probabilities not reproduced')
    if any(not np.isfinite(frame[list(METRICS)]).all().all() for frame in frames.values()):
        raise ValueError('nonfinite scores')
    staging = Path(tempfile.mkdtemp(prefix='.v171-', dir=OUT.parent))
    for key, frame in frames.items():
        frame.to_csv(staging / f'{key}_scores.csv', index=False)
    frozen.queries.to_csv(staging / 'query_inputs.csv', index=False)
    np.savez_compressed(staging / 'probabilities.npz', profile_id=frozen.queries.profile_id.to_numpy(int),
                        hourly=np.asarray(saved_hourly), monthly=np.asarray(saved_monthly), components=np.asarray(saved_components))
    (staging / 'contexts.json').write_text(json.dumps(contexts, indent=2, allow_nan=False) + '\n')
    (staging / 'training_provenance.json').write_text(json.dumps(predictor.provenance, indent=2, allow_nan=False) + '\n')
    summary = {'assertion_failures': 0, 'fold': fold, 'proper_regions': frozen.fit, 'manifest_sha256': manifest_hash,
               'hourly_rows': len(frames['hourly']), 'monthly_rows': len(frames['monthly']),
               'component_rows': len(frames['components']), 'forests': len(predictor.provenance),
               'reference_rows_reproduced': len(pair), 'reference_max_errors': reproduction,
               'future_month_mutation_cases': 2, 'hourly_input_mutation_cases': 2,
               'empty_clock_cells': predictor.empty_clock_cells,
               'output_hashes': {p.name: sha(p) for p in sorted(staging.iterdir())}}
    (staging / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    staging.rename(destination)
    print(f'completed zero-event fold {fold}: {len(frames["hourly"])} hourly score rows', flush=True)


def aggregate(rows):
    regional = []
    for cohort, table in [('all29', rows), ('input_fuel_positive', rows.loc[rows.input_fuel_positive])]:
        for key, group in table.groupby(['region', 'method', 'fuel']):
            regional.append({'cohort': cohort, **dict(zip(['region', 'method', 'fuel'], key)),
                             **{m: float(np.average(group[m], weights=group.hours)) for m in METRICS}})
    regional = pd.DataFrame(regional)
    macro = pd.DataFrame([{'cohort': key[0], 'method': key[1], 'fuel': key[2], 'regions': len(group),
                          **{m: float(group[m].mean()) for m in METRICS}}
                         for key, group in regional.groupby(['cohort', 'method', 'fuel'])])
    return regional, macro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if OUT.exists() and not args.resume:
        raise FileExistsError(OUT)
    base, world, old = verify(BASE), verify(WORLD), verify(OLD)
    dependencies = [Path(__file__), PROTOCOL, ROOT / 'scripts/two_scale_zero.py',
                    ROOT / 'scripts/frozen_recomposition.py', ROOT / 'scripts/monthly_history_features.py',
                    ROOT / 'scripts/hourly_fuel_data.py', ROOT / 'scripts/fuel_activity_forest.py',
                    ROOT / 'scripts/v154_hourly_strong_baselines.py', ROOT / 'scripts/v170_fuel_marginal_diagnosis.py',
                    BASE / 'summary.json', WORLD / 'summary.json', OLD / 'summary.json']
    dependencies.extend(WORLD / name for name in world['output_hashes'])
    dependencies.extend(OLD / name for name in old['output_hashes'])
    dependencies.extend(BASE / name for name in base['output_hashes'])
    hashes = base['input_hashes'] | {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('input changed')
    text = json.dumps({'input_hashes': hashes, 'hourly_methods': METHODS, 'monthly_methods': MONTH_METHODS,
                       'components': COMPONENTS, 'budget': 'asof_30d'}, indent=2, sort_keys=True, allow_nan=False) + '\n'
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
                        raise ValueError('saved output changed')
            else:
                run_block(data, monthly, fold, block, sha(OUT / 'run_manifest.json'))
    frames = {}
    for key, count in [('hourly', 15424), ('monthly', 9640), ('components', 2272)]:
        rows = pd.concat([pd.read_csv(OUT / f'fold_{f}/{key}_scores.csv') for f in range(6)], ignore_index=True)
        if len(rows) != count or rows.duplicated(['profile_id', 'method', 'fuel']).any():
            raise ValueError('aggregate denominator')
        regional, macro = aggregate(rows)
        for label, frame in [('scores', rows), ('region_metrics', regional), ('macro_metrics', macro)]:
            frame.to_csv(OUT / f'{key}_{label}.csv', index=False)
        frames[key] = macro
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('dependency changed during experiment')
    summary = {'assertion_failures': 0, 'status': 'SOURCE_ONLY_ORDINARY_ZERO_EVENT_CONTROLS',
               'completed_folds': 6, 'hourly_rows': 15424, 'monthly_rows': 9640, 'component_rows': 2272,
               'input_hashes': hashes, 'output_hashes': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
               'limitations': ['Zero-event Brier only; no new CIF forecast or coverage gate.',
                               'Ordinary direct and hurdle forests are controls, not a novelty claim.',
                               '2024 shared-time source development, synthetic releases, nonnegative fixed-EF production proxy.']}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(frames['hourly'].loc[frames['hourly'].cohort.eq('input_fuel_positive')].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
