"""Formal monthly information-value experiment; no hourly data are loaded."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from extended_monthly_information import ExtendedMonthlyInformation
from monthly_history_features import MonthlyInformation, monthly_context
from monthly_history_value_models import MonthlyValueModels, METHODS
from monthly_distribution_scoring import score, aggregate, COARSE_METRICS, FUEL_METRICS
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import verify

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/v178-joint-coarse-world-r1'
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
CONTEXT = ROOT / 'results/v184-monthly-context-preflight-r1'
OUT = ROOT / 'results/v184-coarse-history-value-r1'
PRE = ROOT / 'results/v184-coarse-history-model-preflight-r2'
PROTOCOL = ROOT / 'docs/V184_COARSE_HISTORY_VALUE_PROTOCOL.md'
BUDGETS = ('short', 'long_fixed', 'long_all')
BASE_CONTROL = ('pooled_linear_residual', 'gaussian_increment', 'forest_base_scalar8', 'forest_base_delta8')


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def fingerprint(model):
    forests = dict(model.models) | {'anchor': model.anchor_model}
    trees = {name: [{key: array_hash(getattr(tree.tree_, key)) for key in
                     ('children_left', 'children_right', 'feature', 'threshold', 'value', 'weighted_n_node_samples')}
                    for tree in fit.estimators_] for name, fit in forests.items()}
    mean, scale, fit = model.linear
    return model.metadata | {'tree_arrays': trees,
                             'linear': {name: array_hash(value) for name, value in
                                        [('mean', mean), ('scale', scale), ('coef', fit.coef_), ('intercept', fit.intercept_)]}}


def source_matrices(model):
    mean, scale, fit = model.linear
    return {'profiles': model.ids, 'inputs': model.inputs, 'X_base': model.X['base'], 'X_history': model.X['history'],
            'previous': model.previous, 'current': model.current, 'delta': model.delta, 'prior': model.prior,
            'linear_mean': mean, 'linear_scale': scale, 'linear_coef': fit.coef_, 'linear_intercept': fit.intercept_,
            **{'residual_' + key: value for key, value in model.residuals.items()}}


def source_prefix_rows(model):
    return [{'profile_id': int(q), 'region': model.data.index.region.iloc[q], 'month': model.data.index.month.iloc[q],
             'local': model.contexts[int(q)]['local'],
             'prefix': json.dumps([int(p) for p in model.contexts[int(q)]['prefix']])} for q in model.ids]


def fit_fold(short, long, inventory, fold):
    meta = json.loads((BASE / f'fold_{fold}_asof_30d/summary.json').read_text())
    frame = pd.read_csv(SOURCE / f'fold_{fold}/world_probabilities.csv')
    queries = frame.loc[frame.role.eq('test'), ['profile_id', 'region', 'month']].drop_duplicates().reset_index(drop=True)
    models = {}
    for budget in BUDGETS:
        entries = inventory.loc[inventory.fold.eq(fold) & inventory.budget.eq(budget)]
        data = short if budget == 'short' else long
        models[budget] = MonthlyValueModels(data, meta['source_regions'], entries.profile_id.to_numpy(int))
        if set(queries.region) & models[budget].proper:
            raise ValueError('query/proper source overlap')
        for q, p in zip(models[budget].ids, models[budget].inputs):
            context = models[budget].contexts[int(q)]
            if q in context['prefix'] or not (data.release.iloc[context['prefix']] <= data.dates.iloc[q]).all():
                raise ValueError('illegal source training prefix')
            if data.index.region.iloc[q] not in models[budget].proper or p != data.local(int(q), 'asof_30d'):
                raise ValueError('source role/input mismatch')
        print(f'fitted monthly fold {fold} {budget}: {len(entries)} labels', flush=True)
    for name in BASE_CONTROL:
        if name.startswith('forest_') and fingerprint(models['short'])['tree_arrays'][name] != fingerprint(models['long_fixed'])['tree_arrays'][name]:
            raise ValueError('base forest fit negative control differs')
    return models, queries


def build_all(models, short, q):
    laws = {}
    by_budget = {budget: model.build(q) for budget, model in models.items()}
    for method in BASE_CONTROL:
        if any(not np.array_equal(a, b) for a, b in zip(by_budget['short'][method], by_budget['long_fixed'][method])):
            raise ValueError('base law negative control differs')
    if any(not np.array_equal(a, b) for a, b in zip(by_budget['short']['persistence_residual'], by_budget['short']['seasonal_residual'])):
        raise ValueError('short seasonal fallback differs from persistence')
    for budget, distributions in by_budget.items():
        laws.update({f'{budget}__{method}': law for method, law in distributions.items()})
    local = short.local(q, 'asof_30d')
    laws['coarse_persistence'] = short.shares[local][None].copy(), np.ones(1)
    if len(laws) != 28:
        raise ValueError('coarse method inventory')
    return laws


def mutation(models, short, q):
    before = build_all(models, short, q)
    for budget, model in models.items():
        data = model.data
        ids = data.groups[data.index.region.iloc[q]]
        ids = ids[(data.release.iloc[ids] > data.dates.iloc[q]).to_numpy()]
        saved = data.M[ids].copy(), data.shares[ids].copy()
        changed = np.random.default_rng(q + 184).uniform(1, 1000, size=saved[0].shape)
        try:
            data.M[ids] = changed
            data.shares[ids] = changed / changed.sum(axis=1, keepdims=True)
            after = model.build(q)
        finally:
            data.M[ids], data.shares[ids] = saved
        for method in METHODS:
            if any(not np.array_equal(a, b) for a, b in zip(before[f'{budget}__{method}'], after[method])):
                raise ValueError('unreleased query M changed a coarse law')
    return before


def toy():
    values = np.zeros((2, 8)); values[0, 0] = 1; values[1, 1] = 1
    target = values.mean(axis=0)
    coarse, fuels = score(values, np.asarray([.25, .75]), target, {})
    if abs(coarse['energy_score'] - (.5 - .1875) * np.sqrt(2)) > 1e-12 or abs(coarse['anchor_crps'] - 187.5) > 1e-10 or abs(coarse['share_crps_mean'] - .078125) > 1e-12:
        raise ValueError('independent two-atom score toy differs')
    if len(fuels) != 8 or coarse['anchor_coverage'] != 1 or coarse['anchor_width'] != 600:
        raise ValueError('coarse interval toy differs')


def preflight(short, long, inventory):
    toy()
    models_meta, cases, serialized_rows = {}, 0, 0
    for fold in range(6):
        models, queries = fit_fold(short, long, inventory, fold)
        models_meta[str(fold)] = {b: fingerprint(m) for b, m in models.items()}
        for model in models.values():
            rows = pd.DataFrame(source_prefix_rows(model))
            restored = pd.read_csv(io.StringIO(rows.to_csv(index=False)))
            if not np.array_equal(rows.profile_id, restored.profile_id):
                raise ValueError('source prefix identity serialization differs')
            for a, b in zip(rows.prefix, restored.prefix):
                if json.loads(a) != json.loads(b):
                    raise ValueError('source prefix serialization differs')
            serialized_rows += len(rows)
        for q in queries.profile_id.iloc[[0, -1]]:
            mutation(models, short, int(q))
            cases += 1
        print(f'monthly model preflight fold {fold} passed', flush=True)
    if cases != 12:
        raise ValueError('model preflight query denominator')
    return {'query_future_mutation_cases': cases, 'budget_guard_groups': cases * 3,
            'method_law_guards': cases * 3 * 9, 'source_prefix_roundtrip_rows': serialized_rows,
            'held_outcomes_scored': False}, models_meta


def run_fold(short, long, inventory, fold, destination, manifest_sha):
    models, queries = fit_fold(short, long, inventory, fold)
    metadata = {b: fingerprint(m) for b, m in models.items()}
    expected = json.loads((PRE / 'model_fingerprints.json').read_text())[str(fold)]
    if metadata != expected:
        raise ValueError('preflight model refit fingerprint differs')
    coarse, fuels, prediction_rows, arrays, weights = [], [], [], [], []
    offset = 0
    for number, rec in enumerate(queries.itertuples(index=False), start=1):
        q = int(rec.profile_id)
        laws = build_all(models, short, q)
        # First evaluation access to the query's current coarse composition.
        actual = short.shares[q].copy()
        cache = {}
        identity = {'fold': fold, 'profile_id': q, 'region': rec.region, 'month': rec.month}
        for method, (values, probability) in laws.items():
            metrics, fuel_metrics = score(values, probability, actual, cache)
            coarse.append(identity | {'method': method} | metrics)
            fuels.extend(identity | {'method': method} | r for r in fuel_metrics)
            prediction_rows.append(identity | {'method': method, 'coarse_offset': offset, 'coarse_stop': offset + len(values)})
            arrays.append(values); weights.append(probability)
            offset += len(values)
        if number % 10 == 0 or number == len(queries):
            print(f'monthly scoring fold {fold}: {number}/{len(queries)}', flush=True)
    coarse, fuels = pd.DataFrame(coarse), pd.DataFrame(fuels)
    if len(coarse) != len(queries) * 28 or len(fuels) != len(queries) * 224 or not np.isfinite(coarse[list(COARSE_METRICS)]).all().all() or not np.isfinite(fuels[list(FUEL_METRICS)]).all().all():
        raise ValueError('fold score denominator/finiteness')
    old = pd.read_csv(SOURCE / f'fold_{fold}/coarse_month_metrics.csv')
    old = old.loc[old.method.eq('coarse_persistence')]
    paired = coarse.loc[coarse.method.eq('coarse_persistence')].merge(old, on=['profile_id', 'method'], validate='one_to_one', suffixes=('_new', '_old'))
    errors = [float(max(abs(paired[m + '_new'] - paired[m + '_old']))) for m in COARSE_METRICS[:-1]]
    old_fuels = pd.read_csv(SOURCE / f'fold_{fold}/coarse_fuel_metrics.csv')
    old_fuels = old_fuels.loc[old_fuels.method.eq('coarse_persistence')]
    paired_fuel = fuels.loc[fuels.method.eq('coarse_persistence')].merge(old_fuels, on=['profile_id', 'method', 'fuel_index'], validate='one_to_one', suffixes=('_new', '_old'))
    errors.extend(float(max(abs(paired_fuel[m + '_new'].astype(float) - paired_fuel[m + '_old'].astype(float)))) for m in FUEL_METRICS[:-1])
    if len(paired) != len(queries) or len(paired_fuel) != len(queries) * 8 or max(errors) > 1e-8:
        raise ValueError('point persistence reference not reproduced')
    staging = Path(tempfile.mkdtemp(prefix='.v184-coarse-', dir=OUT.parent))
    for name, table in [('coarse_month_metrics', coarse), ('coarse_fuel_metrics', fuels), ('prediction_index', pd.DataFrame(prediction_rows))]:
        table.to_csv(staging / f'{name}.csv', index=False)
    np.savez_compressed(staging / 'coarse_predictions.npz', values=np.concatenate(arrays), probabilities=np.concatenate(weights))
    (staging / 'model_fingerprints.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    for budget, model in models.items():
        np.savez_compressed(staging / f'{budget}_training_matrices.npz', **source_matrices(model))
        pd.DataFrame(source_prefix_rows(model)).to_csv(staging / f'{budget}_training_prefix.csv', index=False)
    result = {'assertion_failures': 0, 'fold': fold, 'manifest_sha256': manifest_sha,
              'coarse_score_rows': len(coarse), 'fuel_score_rows': len(fuels), 'prediction_rows': len(prediction_rows),
              'packed_coarse_atoms': offset, 'point_reference_rows': len(paired), 'point_fuel_reference_rows': len(paired_fuel),
              'max_point_reference_error': max(errors), 'base_negative_control_laws': len(queries) * len(BASE_CONTROL),
              'output_hashes': {p.name: sha(p) for p in staging.iterdir()}}
    (staging / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
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
    for archive in (CONTEXT, ROOT / 'results/v184-monthly-model-smoke-r2', ROOT / 'results/v183-coarse-history-export-audit-r1'):
        summary = verify(archive)
        for p, h in summary['input_hashes'].items():
            if p in hashes and hashes[p] != h:
                raise ValueError('upstream input conflict')
            hashes[p] = h
        for name in ('summary.json', *summary['output_hashes']):
            p = archive / name
            hashes[str(p.relative_to(ROOT))] = sha(p)
    paths = [Path(__file__), PROTOCOL, ROOT / 'docs/V184_PREFLIGHT_SERIALIZER_ADDENDUM.md', ROOT / 'scripts/monthly_distribution_scoring.py',
             ROOT / 'scripts/monthly_history_value_models.py', ROOT / 'scripts/extended_monthly_information.py']
    hashes |= {str(p.relative_to(ROOT)): sha(p) for p in paths}
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('input changed')
    short = MonthlyInformation(ROOT)
    long = ExtendedMonthlyInformation(ROOT, ROOT / 'results/v183-coarse-history-export-r1/monthly_history.csv')
    inventory = pd.read_csv(CONTEXT / 'training_inventory.csv')
    if args.preflight:
        with threadpool_limits(limits=2):
            result, metadata = preflight(short, long, inventory)
        if any(sha(ROOT / p) != h for p, h in hashes.items()):
            raise ValueError('preflight inputs changed')
        destination.mkdir(parents=True)
        (destination / 'model_fingerprints.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        result |= {'assertion_failures': 0, 'input_hashes': hashes,
                   'output_hashes': {'model_fingerprints.json': sha(destination / 'model_fingerprints.json')}}
        (destination / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)
        return
    if verify(PRE)['input_hashes'] != hashes:
        raise ValueError('preflight source or code differs')
    hashes[str((PRE / 'summary.json').relative_to(ROOT))] = sha(PRE / 'summary.json')
    hashes[str((PRE / 'model_fingerprints.json').relative_to(ROOT))] = sha(PRE / 'model_fingerprints.json')
    manifest = json.dumps({'input_hashes': hashes, 'budgets': BUDGETS, 'methods': METHODS}, sort_keys=True, indent=2, allow_nan=False) + '\n'
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
                    raise ValueError('saved fold manifest differs')
            else:
                run_fold(short, long, inventory, fold, block, sha(destination / 'run_manifest.json'))
    coarse = pd.concat([pd.read_csv(destination / f'fold_{k}/coarse_month_metrics.csv') for k in range(6)], ignore_index=True)
    fuels = pd.concat([pd.read_csv(destination / f'fold_{k}/coarse_fuel_metrics.csv') for k in range(6)], ignore_index=True)
    if len(coarse) != 6748 or len(fuels) != 53984 or coarse.duplicated(['profile_id', 'method']).any() or fuels.duplicated(['profile_id', 'method', 'fuel_index']).any():
        raise ValueError('complete score inventory')
    regions, macro, fuel_regions, fuel_macro = aggregate(coarse, fuels)
    for name, table in [('coarse_month_metrics', coarse), ('coarse_fuel_metrics', fuels), ('coarse_region_metrics', regions),
                        ('coarse_macro_metrics', macro), ('coarse_fuel_region_metrics', fuel_regions), ('coarse_fuel_macro_metrics', fuel_macro)]:
        table.to_csv(destination / f'{name}.csv', index=False)
    if any(sha(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('inputs changed during run')
    result = {'assertion_failures': 0, 'completed_blocks': 6, 'coarse_score_rows': len(coarse), 'fuel_score_rows': len(fuels),
              'input_hashes': hashes, 'output_hashes': {str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob('*')) if p.is_file()},
              'scope': 'Ordinary coarse information-value controls. No hourly scores, no original-method claim, shared-2024 source development; actual China/public vintages unverified.'}
    (destination / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(macro.loc[macro.cohort.eq('all29')].sort_values('energy_score').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
