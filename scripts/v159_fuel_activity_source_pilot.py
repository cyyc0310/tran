"""Cross-region source-only fuel activity pilot, with immutable block checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import tempfile

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from calendar_cif_law import CalendarLaw
from fuel_activity_forest import (
    ActivityDonors, IndexedLeafMixture, activity_law, carrier,
    response_values, source_activity)
from hourly_fuel_data import HourlyFuelData
from sampled_calendar_source import SampledCalendarSource
from v151_hourly_law import EF, Law, sha
from v154_hourly_strong_baselines import CompressedLaw, balanced

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v156-anchored-hourly-sampled-r1'
OUT = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
PROTOCOL = ROOT / 'docs/V159_FUEL_ACTIVITY_SOURCE_PILOT_PROTOCOL.md'
METHODS = ('legacy_recomposition', 'positive_recomposition', 'activity_recomposition',
           'conditional_activity_recomposition', 'scalar_forest', 'raw_activity_forest',
           'projective_activity_forest', 'response_activity_forest')
SCORES = ('crps', 'mae', 'coverage', 'width', 'winkler')


def weighted_radius(errors: dict[str, list[np.ndarray]]) -> tuple[float, float]:
    values = {r: np.concatenate(v) for r, v in errors.items()}
    pooled = Law(np.concatenate(list(values.values())), np.concatenate([
        np.full(len(v), 1 / (len(values) * len(v))) for v in values.values()])).quantile(.9)
    maximum = max(Law(v, np.ones(len(v))).quantile(.9) for v in values.values())
    return pooled, maximum


def fit_models(data, sources: dict, prior: np.ndarray, budget: str):
    fit = sorted(sources)
    X = np.concatenate([sources[r]['X'] for r in fit])
    y = np.concatenate([sources[r]['y'] for r in fit])
    u = np.concatenate([sources[r]['u'] for r in fit])
    groups = np.concatenate([np.full(len(sources[r]['y']), r) for r in fit])
    weights = balanced(groups)
    inputs = np.unique(np.concatenate([sources[r]['inputs'] for r in fit]))
    if not set(data.index.region.iloc[inputs]).issubset(fit):
        raise ValueError('prototype region leakage')
    ordered = inputs[np.lexsort((inputs, data.shares[inputs, 0]))]
    prototypes = ordered[np.rint(np.linspace(0, len(ordered) - 1, 5)).astype(int)]
    responses, invalid_counts = [], []
    for q in prototypes:
        values, positive = response_values(carrier(data.shares[q], budget, prior), u)
        responses.extend([positive.astype(float), values])
        invalid_counts.append(int((~positive).sum()))
    response = np.column_stack(responses)
    mu = np.average(response, axis=0, weights=weights)
    sd = np.sqrt(np.average((response - mu)**2, axis=0, weights=weights))
    active = sd > 1e-8
    embedding = (response[:, active] - mu[active]) / sd[active]
    if not active.any():
        embedding = np.zeros((len(y), 1))
    labels = {'scalar_forest': y, 'raw_activity_forest': u,
              'projective_activity_forest': u / u.sum(axis=1, keepdims=True),
              'response_activity_forest': embedding}
    models = {name: IndexedLeafMixture(X, label, weights) for name, label in labels.items()}
    metadata = {'prototype_profile_ids': prototypes.tolist(),
                'active_embedding_dimensions': int(active.sum()),
                'undefined_prototype_atoms': invalid_counts, 'fit_rows': len(y),
                'source_regions': fit, 'carrier_prior': prior.tolist()}
    return models, y, u, metadata


def run_block(data, fold: int, budget: str, roles: pd.DataFrame,
              sampling: pd.DataFrame, unix: np.ndarray, block: Path, run_digest: str):
    roster = roles.loc[roles.fold.eq(fold)]
    fit = sorted(roster.loc[roster.role.eq('fit'), 'region'])
    cal = sorted(roster.loc[roster.role.eq('calibration'), 'region'])
    test = sorted(roster.loc[roster.role.eq('test'), 'region'])
    if set(fit) & set(cal + test) or set(cal) & set(test):
        raise ValueError('fold role leakage')
    draw = sampling.loc[sampling.budget.eq(budget)].set_index('region')
    allowed = {r: np.asarray(json.loads(draw.loc[r, 'sampled_raw_hour_indices']), int) for r in fit}
    proper = {r: data.source(r, budget, allowed[r]) for r in fit}
    prior = np.mean([data.shares[proper[r][3]].mean(axis=0) for r in fit], axis=0) + 1e-6
    prior /= prior.sum()
    sources = {r: source_activity(data, r, budget, allowed[r], prior) for r in fit}
    models, scalar, activity, metadata = fit_models(data, sources, prior, budget)
    allowed_all = np.concatenate([allowed[r] for r in fit])
    legacy = SampledCalendarSource(data.index, data.raw, data.M, unix, allowed_all)
    shape_bank = ActivityDonors(data, sources, 'shape')
    activity_bank = ActivityDonors(data, sources)
    donor_bank = {r: np.unique(sources[r]['profiles']) for r in fit}
    month_number = pd.to_datetime(data.index.month).dt.month.to_numpy()
    query_records, rejection_records, outputs, radii = [], [], [], []
    errors = {name: {} for name in METHODS}
    targets = {}
    for region in cal + test:
        for q in data.groups[region]:
            row = data.index.iloc[q]
            if not row.month.startswith('2024'):
                continue
            local = data.local(int(q), budget)
            if local is None:
                continue
            S = data.shares[local]
            B = carrier(S, budget, prior)
            query_records.append({'fold': fold, 'budget': budget, 'profile_id': int(q),
                                  'region': region, 'month': row.month, 'input_profile_id': local,
                                  'input_month': data.index.month.iloc[local],
                                  'role': 'calibration' if region in cal else 'test'})
            donors = []
            for r in fit:
                ids = donor_bank[r]
                distance = np.abs(month_number[ids] - month_number[q])
                distance = np.minimum(distance, 12 - distance)
                if len(ids) < 2:
                    raise ValueError('insufficient donor months')
                donors.extend(ids[np.lexsort((ids, distance))[:2]])
            donors = np.asarray(donors, int)
            if region in set(data.index.region.iloc[donors]):
                raise ValueError('self donor leakage')
            current_distance = .5 * ((np.sqrt(data.shares[donors]) - np.sqrt(S))**2).sum(axis=1)
            weights = np.exp(-current_distance / .3**2)
            weights /= weights.sum()
            old_law = legacy.build(int(q), donors, weights, local, False, False)
            cell_laws, shape_rejected = shape_bank.laws(donors, weights, S)
            cells24 = data.states[int(row.offset):int(row.stop)] % 24
            laws = {'legacy_recomposition': old_law,
                    'positive_recomposition': CalendarLaw(cell_laws, cells24)}
            rejection_records.append({'fold': fold, 'budget': budget, 'profile_id': int(q),
                                      'method': 'positive_recomposition',
                                      'mean_cell_rejected_mass': float(np.mean(shape_rejected)),
                                      'max_cell_rejected_mass': float(np.max(shape_rejected))})
            if budget == 'same_month':
                laws['activity_recomposition'] = laws['positive_recomposition']
                laws['conditional_activity_recomposition'] = laws['positive_recomposition']
            else:
                cell_laws, rejected = activity_bank.laws(donors, weights, B)
                laws['activity_recomposition'] = CalendarLaw(cell_laws, cells24)
                input_ids = np.asarray([data.local(int(d), budget) for d in donors], int)
                input_distance = .5 * ((np.sqrt(data.shares[input_ids]) - np.sqrt(S))**2).sum(axis=1)
                conditional_weights = np.exp(-input_distance / .3**2)
                conditional_weights /= conditional_weights.sum()
                conditional_laws, conditional_rejected = activity_bank.laws(donors, conditional_weights, B)
                laws['conditional_activity_recomposition'] = CalendarLaw(conditional_laws, cells24)
                if max(rejected + conditional_rejected) != 0:
                    raise ValueError('positive asof carrier unexpectedly rejected atoms')
            qx = data.features(int(q), local, np.arange(48))
            cells48 = data.states[int(row.offset):int(row.stop)]
            for name, model in models.items():
                leaves = model.forest.apply(qx)
                cell_laws, rejected = [], []
                for state in range(48):
                    ids, probability = model.atoms(leaves[state])
                    if name == 'scalar_forest':
                        law = CompressedLaw(np.clip(scalar[ids] + S @ EF, 0, 1000), probability)
                        mass = 0.
                    else:
                        law, mass = activity_law(B, activity[ids], probability)
                    cell_laws.append(law)
                    rejected.append(mass)
                laws[name] = CalendarLaw(cell_laws, cells48)
                rejection_records.append({'fold': fold, 'budget': budget, 'profile_id': int(q),
                                          'method': name, 'mean_cell_rejected_mass': float(np.mean(rejected)),
                                          'max_cell_rejected_mass': float(np.max(rejected))})
            actual = data.truth(int(q))
            if region in test:
                targets[int(q)] = laws
            else:
                for name, law in laws.items():
                    distance = np.maximum.reduce([
                        law.quantile(.05) - actual, actual - law.quantile(.95), np.zeros(len(actual))])
                    errors[name].setdefault(region, []).append(distance)
    for name in METHODS:
        if set(errors[name]) != set(cal):
            raise ValueError('calibration region denominator')
        pooled, maximum = weighted_radius(errors[name])
        for calibration, radius in (('pooled', pooled), ('max_region', maximum)):
            radii.append({'fold': fold, 'budget': budget, 'method': name,
                          'calibration': calibration, 'radius': radius})
            for q, laws in targets.items():
                law = laws[name]
                actual = data.truth(q)
                row = data.index.iloc[q]
                lo = np.maximum(0, law.quantile(.05) - radius)
                hi = np.minimum(1000, law.quantile(.95) + radius)
                if not np.isfinite(lo).all() or np.any(lo > hi):
                    raise ValueError('invalid interval')
                outputs.append({'fold': fold, 'budget': budget, 'region': row.region,
                                'month': row.month, 'method': name, 'calibration': calibration,
                                'hours': len(actual), 'crps': float(law.crps(actual).mean()),
                                'mae': float(np.abs(actual - law.mean).mean()),
                                'coverage': float(((actual >= lo) & (actual <= hi)).mean()),
                                'width': float(np.mean(hi - lo)),
                                'winkler': float(np.mean(hi - lo + 20 * np.maximum(lo - actual, 0)
                                                        + 20 * np.maximum(actual - hi, 0)))})
    rows = pd.DataFrame(outputs)
    expected = len(targets) * len(METHODS) * 2
    if len(rows) != expected or set(rows.region) != set(test):
        raise ValueError('test denominator')
    if not np.isfinite(rows[list(SCORES)]).all().all():
        raise ValueError('nonfinite score')
    if budget == 'same_month':
        for method in ('activity_recomposition', 'conditional_activity_recomposition'):
            a = rows.loc[rows.method.eq('positive_recomposition')].set_index(['region', 'month', 'calibration'])
            b = rows.loc[rows.method.eq(method)].set_index(['region', 'month', 'calibration'])
            if not np.array_equal(a[list(SCORES)], b.loc[a.index, list(SCORES)]):
                raise ValueError('same-month activity duplicate control')
    staging = Path(tempfile.mkdtemp(prefix='.v159-block-', dir=OUT.parent))
    for name, frame in [('test_month_metrics', rows), ('query_inputs', pd.DataFrame(query_records)),
                        ('rejected_atom_mass', pd.DataFrame(rejection_records)),
                        ('calibration_radii', pd.DataFrame(radii))]:
        frame.to_csv(staging / f'{name}.csv', index=False)
    metadata.update({'fold': fold, 'budget': budget, 'calibration_regions': cal, 'test_regions': test,
                     'test_profiles': len(targets), 'score_rows': len(rows),
                     'legacy_empty_cells': legacy.empty_cell_fallbacks,
                     'activity_empty_cells': activity_bank.empty_calendar_cells,
                     'shape_empty_cells': shape_bank.empty_calendar_cells,
                     'run_manifest_sha256': run_digest, 'assertion_failures': 0,
                     'output_hashes': {p.name: sha(p) for p in sorted(staging.iterdir())}})
    (staging / 'summary.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    staging.rename(block)
    print(f'completed source fold {fold}/{budget}: {len(targets)} test months, {len(rows)} scores', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if OUT.exists() and not args.resume:
        raise FileExistsError(OUT)
    dependencies = [PROTOCOL, Path(__file__), ROOT / 'scripts/fuel_activity_forest.py',
                    ROOT / 'scripts/hourly_fuel_data.py', ROOT / 'scripts/sampled_calendar_source.py',
                    ROOT / 'scripts/calendar_cif_law.py', ROOT / 'scripts/v151_hourly_law.py',
                    ROOT / 'scripts/v139_physical_reserve_calibration.py',
                    ROOT / 'scripts/v154_hourly_strong_baselines.py', BASE / 'summary.json',
                    BASE / 'source_sampling.csv', BASE / 'region_roles.csv',
                    ROOT / 'results/v144-canonical-profile-cache-r1/profiles.npz',
                    ROOT / 'results/v144-canonical-profile-cache-r1/profile_index.csv',
                    ROOT / 'results/v153-calendar-times-r1/timestamps.npz']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    upstream = json.loads((BASE / 'summary.json').read_text())
    for name, digest in upstream['output_hashes'].items():
        if sha(BASE / name) != digest:
            raise ValueError('V156 archive changed')
    original_roles = pd.read_csv(BASE / 'region_roles.csv')
    pool = set(original_roles.loc[original_roles.target_region.eq('PACE')
                                 & original_roles.seed.eq(41) & original_roles.role.eq('fit'), 'region'])
    pool -= {'PACE', 'LGEE', 'SC'}
    ordered = sorted(pool, key=lambda r: hashlib.sha256(('v159/' + r).encode()).hexdigest())
    group = {r: i % 6 for i, r in enumerate(ordered)}
    if len(group) != 29:
        raise ValueError('frozen pilot source pool')
    roles = pd.DataFrame([{'fold': k, 'region': r,
                           'role': 'test' if g == k else 'calibration' if g == (k + 1) % 6 else 'fit'}
                          for k in range(6) for r, g in sorted(group.items())])
    samples = pd.read_csv(BASE / 'source_sampling.csv')
    samples = samples.loc[samples.target_region.eq('PACE') & samples.seed.eq(41) & samples.region.isin(pool)]
    manifest = {'input_hashes': hashes, 'pool': sorted(pool), 'fold_group': group,
                'methods': list(METHODS), 'carrier_smoothing': .01, 'source_year': 2024,
                'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                'pandas': pd.__version__, 'sklearn': sklearn.__version__}}
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if OUT.exists():
        if (OUT / 'run_manifest.json').read_text() != manifest_text:
            raise ValueError('resume frozen inputs differ')
        if (OUT / 'summary.json').exists():
            raise RuntimeError('experiment already complete')
        if (OUT / 'region_roles.csv').read_text() != roles.to_csv(index=False):
            raise ValueError('resume role table changed')
        if (OUT / 'source_sampling.csv').read_text() != samples.to_csv(index=False):
            raise ValueError('resume sampling table changed')
    else:
        OUT.mkdir(parents=True)
        (OUT / 'run_manifest.json').write_text(manifest_text)
        roles.to_csv(OUT / 'region_roles.csv', index=False)
        samples.to_csv(OUT / 'source_sampling.csv', index=False)
    run_digest = sha(OUT / 'run_manifest.json')
    data = HourlyFuelData(ROOT)
    unix = np.load(ROOT / 'results/v153-calendar-times-r1/timestamps.npz', allow_pickle=False)['unix_seconds']
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
                            raise ValueError('checkpoint output changed')
                else:
                    run_block(data, fold, budget, roles, samples, unix, block, run_digest)
                blocks.append(block)
    rows = pd.concat([pd.read_csv(b / 'test_month_metrics.csv') for b in blocks], ignore_index=True)
    if rows.duplicated(['budget', 'region', 'month', 'method', 'calibration']).any():
        raise ValueError('source region repeated as test')
    region_rows = []
    for key, g in rows.groupby(['region', 'budget', 'method', 'calibration']):
        region_rows.append(dict(zip(['region', 'budget', 'method', 'calibration'], key))
                           | {m: float(np.average(g[m], weights=g.hours)) for m in SCORES})
    regional = pd.DataFrame(region_rows)
    macro = regional.groupby(['budget', 'method', 'calibration']).agg(
        crps=('crps', 'mean'), mae=('mae', 'mean'), coverage=('coverage', 'mean'),
        worst_coverage=('coverage', 'min'), width=('width', 'mean'),
        winkler=('winkler', 'mean'), regions=('region', 'size')).reset_index()
    if not macro.regions.eq(29).all():
        raise ValueError('source macro denominator')
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise ValueError('frozen pilot input changed')
    for name, frame in [('test_month_metrics', rows), ('region_metrics', regional), ('macro_metrics', macro)]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    summary = {'status': 'SOURCE_ONLY_ACTIVITY_COORDINATE_PILOT', 'assertion_failures': 0,
               'completed_blocks': len(blocks), 'source_regions': len(pool), 'scored_rows': len(rows),
               'input_hashes': hashes,
               'output_hashes': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
               'limitations': ['Source-only 2024 cross-region development, not blind confirmation.',
                               'Fixed-EF production proxy, retrospective quality and simulated release times.',
                               'Finite response embedding and positive-domain conditioning are not established novelty.',
                               'Heteroscedastic/CQR/original DRF comparisons remain necessary before promotion.']}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(macro.to_string(index=False))


if __name__ == '__main__':
    main()
