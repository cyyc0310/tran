"""Source-only carbon-distribution split probe and general distributional controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from calendar_cif_law import CalendarLaw
from fuel_activity_forest import IndexedLeafMixture, activity_law, carrier, response_values, source_activity
from hourly_fuel_data import HourlyFuelData
from v151_hourly_law import EF, sha
from v154_hourly_strong_baselines import CompressedLaw, balanced
from v159_fuel_activity_source_pilot import weighted_radius

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
OUT = ROOT / 'results/v160-carbon-cdf-source-r1'
PROTOCOL = ROOT / 'docs/V160_CARBON_CDF_SOURCE_PROTOCOL.md'
METHODS = ('scalar_cdf_forest', 'activity_mmd_forest', 'mean_smoothed_forest', 'carbon_cdf_forest')
METRICS = ('crps', 'mae', 'coverage', 'width', 'winkler')


def split_labels(data, activity: np.ndarray, scalar: np.ndarray, weights: np.ndarray,
                 prototypes: list[int], prior: np.ndarray, fold: int, budget: str):
    quotient = activity / activity.sum(axis=1, keepdims=True)
    seed = int(hashlib.sha256(f'v160/{fold}/{budget}'.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    pair_ids = rng.choice(len(activity), size=(2, 1024), replace=True, p=weights / weights.sum())
    distances = np.linalg.norm(quotient[pair_ids[0]] - quotient[pair_ids[1]], axis=1)
    positive_distance = distances[distances > 0]
    bandwidth = float(np.median(positive_distance)) if len(positive_distance) else 1.
    omega = rng.normal(size=(8, 80)) / bandwidth
    phase = rng.uniform(0, 2 * np.pi, 80)
    mmd = np.sqrt(2 / 80) * np.cos(quotient @ omega + phase)
    responses = []
    for q in prototypes:
        base = .99 * data.shares[q] + .01 * prior
        values, positive = response_values(base, activity)
        if not positive.all():
            raise ValueError('smoothed prototype lacks positive support')
        responses.append(values)
    response = np.column_stack(responses)
    mu = np.average(response, axis=0, weights=weights)
    sd = np.sqrt(np.average((response - mu)**2, axis=0, weights=weights))
    active = sd > 1e-8
    mean = (response[:, active] - mu[active]) / sd[active] if active.any() else np.zeros((len(activity), 1))
    carbon_thresholds = (np.arange(16) + .5) * 1000 / 16
    residual_thresholds = -1000 + (np.arange(80) + .5) * 2000 / 80
    carbon = (response[:, :, None] <= carbon_thresholds).reshape(len(activity), 80).astype(float)
    scalar_cdf = (scalar[:, None] <= residual_thresholds).astype(float)
    labels = {'scalar_cdf_forest': scalar_cdf, 'activity_mmd_forest': mmd,
              'mean_smoothed_forest': mean, 'carbon_cdf_forest': carbon}
    metadata = {'rff_seed': seed, 'gaussian_bandwidth': bandwidth,
                'positive_bandwidth_pairs': len(positive_distance),
                'carbon_thresholds': carbon_thresholds.tolist(),
                'residual_thresholds': residual_thresholds.tolist(),
                'active_mean_dimensions': int(active.sum()),
                'varying_carbon_cdf_columns': int(np.sum(np.ptp(carbon, axis=0) > 0)),
                'varying_scalar_cdf_columns': int(np.sum(np.ptp(scalar_cdf, axis=0) > 0))}
    return labels, metadata


def run_block(data, fold: int, budget: str, sampling: pd.DataFrame, block: Path, run_digest: str):
    old_block = BASE / f'fold_{fold}_{budget}'
    old_meta = json.loads((old_block / 'summary.json').read_text())
    fit = sorted(old_meta['source_regions'])
    cal = set(old_meta['calibration_regions'])
    test = set(old_meta['test_regions'])
    if set(fit) & (cal | test) or cal & test:
        raise ValueError('source role leakage')
    draw = sampling.loc[sampling.budget.eq(budget)].set_index('region')
    allowed = {r: np.asarray(json.loads(draw.loc[r, 'sampled_raw_hour_indices']), int) for r in fit}
    proper = {r: data.source(r, budget, allowed[r]) for r in fit}
    prior = np.mean([data.shares[proper[r][3]].mean(axis=0) for r in fit], axis=0) + 1e-6
    prior /= prior.sum()
    if not np.array_equal(prior, np.asarray(old_meta['carrier_prior'])):
        raise ValueError('source prior changed')
    sources = {r: source_activity(data, r, budget, allowed[r], prior) for r in fit}
    X = np.concatenate([sources[r]['X'] for r in fit])
    scalar = np.concatenate([sources[r]['y'] for r in fit])
    activity = np.concatenate([sources[r]['u'] for r in fit])
    weights = balanced(np.concatenate([np.full(len(sources[r]['y']), r) for r in fit]))
    prototypes = old_meta['prototype_profile_ids']
    if not set(data.index.region.iloc[prototypes]).issubset(fit):
        raise ValueError('prototype identity leakage')
    labels, metadata = split_labels(data, activity, scalar, weights, prototypes, prior, fold, budget)
    models = {name: IndexedLeafMixture(X, label, weights) for name, label in labels.items()}
    queries = pd.read_csv(old_block / 'query_inputs.csv')
    errors = {name: {} for name in METHODS}
    targets, rejected = {}, []
    for query in queries.itertuples(index=False):
        q = int(query.profile_id)
        if data.index.region.iloc[q] != query.region or not data.index.month.iloc[q].startswith('2024'):
            raise ValueError('source query identity/year')
        if query.role not in ('calibration', 'test') or query.region not in (cal if query.role == 'calibration' else test):
            raise ValueError('source query role')
        local = data.local(q, budget)
        if local != query.input_profile_id:
            raise ValueError('local statistic vintage changed')
        S = data.shares[local]
        B = carrier(S, budget, prior)
        qx = data.features(q, local, np.arange(48))
        row = data.index.iloc[q]
        requested = data.states[int(row.offset):int(row.stop)]
        laws = {}
        for name, model in models.items():
            leaves = model.forest.apply(qx)
            cells, mass = [], []
            for state in range(48):
                ids, probability = model.atoms(leaves[state])
                if name == 'scalar_cdf_forest':
                    law = CompressedLaw(np.clip(scalar[ids] + S @ EF, 0, 1000), probability)
                    excluded = 0.
                else:
                    law, excluded = activity_law(B, activity[ids], probability)
                cells.append(law)
                mass.append(excluded)
            laws[name] = CalendarLaw(cells, requested)
            rejected.append({'profile_id': q, 'method': name,
                             'mean_cell_rejected_mass': float(np.mean(mass)),
                             'max_cell_rejected_mass': float(np.max(mass))})
        if query.role == 'test':
            targets[q] = laws
        else:
            actual = data.truth(q)
            for name, law in laws.items():
                distance = np.maximum.reduce([law.quantile(.05) - actual, actual - law.quantile(.95), np.zeros(len(actual))])
                errors[name].setdefault(query.region, []).append(distance)
    outputs, radii = [], []
    for name in METHODS:
        if set(errors[name]) != cal:
            raise ValueError('calibration region denominator')
        pooled, maximum = weighted_radius(errors[name])
        for calibration, radius in (('pooled', pooled), ('max_region', maximum)):
            radii.append({'fold': fold, 'budget': budget, 'method': name,
                          'calibration': calibration, 'radius': radius})
            for q, laws in targets.items():
                row = data.index.iloc[q]
                actual = data.truth(q)
                law = laws[name]
                lo = np.maximum(0, law.quantile(.05) - radius)
                hi = np.minimum(1000, law.quantile(.95) + radius)
                if np.any(lo > hi) or not np.isfinite(lo).all():
                    raise ValueError('invalid interval')
                outputs.append({'fold': fold, 'budget': budget, 'region': row.region, 'month': row.month,
                                'method': name, 'calibration': calibration, 'hours': len(actual),
                                'crps': float(law.crps(actual).mean()), 'mae': float(np.abs(actual - law.mean).mean()),
                                'coverage': float(((actual >= lo) & (actual <= hi)).mean()),
                                'width': float(np.mean(hi - lo)),
                                'winkler': float(np.mean(hi - lo + 20 * np.maximum(lo - actual, 0)
                                                        + 20 * np.maximum(actual - hi, 0)))})
    rows = pd.DataFrame(outputs)
    if len(rows) != len(targets) * len(METHODS) * 2 or set(rows.region) != test:
        raise ValueError('test denominator')
    if not np.isfinite(rows[list(METRICS)]).all().all():
        raise ValueError('nonfinite metrics')
    reproduction = 0
    if budget == 'asof_30d':
        old = pd.read_csv(old_block / 'test_month_metrics.csv')
        old = old.loc[old.method.eq('response_activity_forest')]
        keys = ['region', 'month', 'calibration']
        match = rows.loc[rows.method.eq('mean_smoothed_forest')].merge(old, on=keys, suffixes=('_new', '_old'), validate='one_to_one')
        if len(match) != len(targets) * 2:
            raise ValueError('reproduction denominator')
        for metric in METRICS:
            if not np.allclose(match[metric + '_new'], match[metric + '_old'], atol=1e-7, rtol=0):
                raise ValueError('asof mean-response reproduction failed: ' + metric)
        reproduction = len(match)
    staging = Path(tempfile.mkdtemp(prefix='.v160-block-', dir=OUT.parent))
    for name, frame in [('test_month_metrics', rows), ('calibration_radii', pd.DataFrame(radii)),
                        ('rejected_atom_mass', pd.DataFrame(rejected)), ('query_inputs', queries)]:
        frame.to_csv(staging / f'{name}.csv', index=False)
    metadata.update({'fold': fold, 'budget': budget, 'fit_regions': fit,
                     'calibration_regions': sorted(cal), 'test_regions': sorted(test),
                     'prototype_profile_ids': prototypes, 'fit_rows': len(scalar),
                     'asof_mean_records_reproduced': reproduction, 'score_rows': len(rows),
                     'assertion_failures': 0, 'run_manifest_sha256': run_digest,
                     'output_hashes': {p.name: sha(p) for p in sorted(staging.iterdir())}})
    (staging / 'summary.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    staging.rename(block)
    print(f'completed CDF source fold {fold}/{budget}: {len(rows)} scores; reproduced={reproduction}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if OUT.exists() and not args.resume:
        raise FileExistsError(OUT)
    old = json.loads((BASE / 'summary.json').read_text())
    if old['assertion_failures'] or old['completed_blocks'] != 12:
        raise ValueError('incomplete source pilot')
    for name, digest in old['input_hashes'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('upstream source input changed')
    for name, digest in old['output_hashes'].items():
        if sha(BASE / name) != digest:
            raise ValueError('upstream source output changed')
    paths = [Path(__file__), PROTOCOL, BASE / 'summary.json', ROOT / 'scripts/fuel_activity_forest.py',
             ROOT / 'scripts/hourly_fuel_data.py', ROOT / 'scripts/v151_hourly_law.py',
             ROOT / 'scripts/v139_physical_reserve_calibration.py', ROOT / 'scripts/calendar_cif_law.py',
             ROOT / 'scripts/v154_hourly_strong_baselines.py', ROOT / 'scripts/v159_fuel_activity_source_pilot.py']
    hashes = dict(old['input_hashes']) | {str(p.relative_to(ROOT)): sha(p) for p in paths}
    manifest_text = json.dumps({'input_hashes': hashes, 'methods': list(METHODS),
                                'source_manifest_sha256': sha(BASE / 'run_manifest.json')},
                               indent=2, sort_keys=True, allow_nan=False) + '\n'
    if OUT.exists():
        if (OUT / 'run_manifest.json').read_text() != manifest_text:
            raise ValueError('resume frozen inputs differ')
        if (OUT / 'summary.json').exists():
            raise RuntimeError('CDF pilot already complete')
    else:
        OUT.mkdir(parents=True)
        (OUT / 'run_manifest.json').write_text(manifest_text)
    run_digest = sha(OUT / 'run_manifest.json')
    data = HourlyFuelData(ROOT)
    samples = pd.read_csv(BASE / 'source_sampling.csv')
    blocks = []
    with threadpool_limits(limits=2):
        for fold in range(6):
            for budget in ('same_month', 'asof_30d'):
                block = OUT / f'fold_{fold}_{budget}'
                if block.exists():
                    saved = json.loads((block / 'summary.json').read_text())
                    if saved['run_manifest_sha256'] != run_digest or saved['assertion_failures']:
                        raise ValueError('checkpoint identity')
                    for name, digest in saved['output_hashes'].items():
                        if sha(block / name) != digest:
                            raise ValueError('checkpoint changed')
                else:
                    run_block(data, fold, budget, samples, block, run_digest)
                blocks.append(block)
    rows = pd.concat([pd.read_csv(b / 'test_month_metrics.csv') for b in blocks], ignore_index=True)
    if rows.duplicated(['budget', 'region', 'month', 'method', 'calibration']).any():
        raise ValueError('duplicate source scores')
    regions = []
    for key, group in rows.groupby(['region', 'budget', 'method', 'calibration']):
        regions.append(dict(zip(['region', 'budget', 'method', 'calibration'], key))
                       | {m: float(np.average(group[m], weights=group.hours)) for m in METRICS})
    regional = pd.DataFrame(regions)
    macro = regional.groupby(['budget', 'method', 'calibration']).agg(
        crps=('crps', 'mean'), mae=('mae', 'mean'), coverage=('coverage', 'mean'),
        worst_coverage=('coverage', 'min'), width=('width', 'mean'), winkler=('winkler', 'mean'),
        regions=('region', 'size')).reset_index()
    if not macro.regions.eq(29).all():
        raise ValueError('source macro denominator')
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('frozen inputs changed')
    for name, frame in [('test_month_metrics', rows), ('region_metrics', regional), ('macro_metrics', macro)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    result = {'status': 'SOURCE_ONLY_CARBON_CDF_SPLIT_PILOT', 'completed_blocks': len(blocks),
              'scored_rows': len(rows), 'source_regions': 29, 'assertion_failures': 0,
              'asof_mean_records_reproduced': sum(json.loads((b / 'summary.json').read_text())['asof_mean_records_reproduced'] for b in blocks),
              'input_hashes': hashes,
              'output_hashes': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
              'limitations': ['2024 shared-time source-region development, no independent target confirmation.',
                              'Finite-feature MMD/CDF forests are approximate distributional controls, not original honest DRF.',
                              'No CDF-label column standardization; uniform thresholds approximate an integrated response metric.']}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(macro.to_string(index=False))


if __name__ == '__main__':
    main()
