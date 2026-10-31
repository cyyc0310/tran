"""Preflight and frozen coarse-only response/support experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from coarse_representation_controls import RepresentationControls, NAMES, SUPPORTS
from extended_monthly_information import ExtendedMonthlyInformation
from monthly_history_value_models import FORESTS
from monthly_distribution_scoring import score, aggregate, COARSE_METRICS, FUEL_METRICS
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import verify
from v184_coarse_history_value import ROOT, BASE, SOURCE, CONTEXT, OUT as V184, PRE as V184_PRE, source_matrices, source_prefix_rows

OUT = ROOT / 'results/v185-coarse-representation-r1'
PRE = ROOT / 'results/v185-coarse-representation-preflight-r1'
PROTOCOL = ROOT / 'docs/V185_COARSE_REPRESENTATION_PROTOCOL.md'


def fit_fold(data, inventory, fold):
    """Use exactly V184's long_all proper labels and query roster."""
    proper = json.loads((BASE / f'fold_{fold}_asof_30d/summary.json').read_text())['source_regions']
    ids = inventory.loc[inventory.fold.eq(fold) & inventory.budget.eq('long_all'), 'profile_id'].to_numpy(int)
    model = RepresentationControls(data, proper, ids)
    expected = json.loads((V184_PRE / 'model_fingerprints.json').read_text())[str(fold)]['long_all']
    if model.metadata()['base'] != expected:
        raise ValueError('V184 long_all fit fingerprint changed')
    frame = pd.read_csv(SOURCE / f'fold_{fold}/world_probabilities.csv')
    queries = frame.loc[frame.role.eq('test'), ['profile_id', 'region', 'month']].drop_duplicates().reset_index(drop=True)
    for q, c in model.base.contexts.items():
        if (q in c['prefix'] or not (data.release.iloc[c['prefix']] <= data.dates.iloc[q]).all()
                or data.index.region.iloc[q] not in model.base.proper):
            raise ValueError('training source/prefix leakage')
    if set(queries.region) & model.base.proper:
        raise ValueError('query/proper source overlap')
    print(f'V185 fit fold {fold}: {len(ids)} monthly labels', flush=True)
    return model, queries


def checked_hashes():
    """Verify current upstream provenance, including producer inputs."""
    hashes = {}
    for folder in (V184, ROOT / 'results/v184-coarse-history-value-audit-r1'):
        summary = verify(folder)
        for name, h in summary['input_hashes'].items():
            if name in hashes and hashes[name] != h:
                raise ValueError('upstream hash conflict')
            hashes[name] = h
        hashes[str((folder / 'summary.json').relative_to(ROOT))] = sha(folder / 'summary.json')
        for name, h in summary['output_hashes'].items():
            hashes[str((folder / name).relative_to(ROOT))] = h
    for p in (Path(__file__), PROTOCOL, ROOT / 'scripts/coarse_representation_controls.py'):
        hashes[str(p.relative_to(ROOT))] = sha(p)
    if any(sha(ROOT / name) != h for name, h in hashes.items()):
        raise ValueError('changed experiment input')
    return hashes


def compare_old(coarse, fuels, fold):
    """Require exact original metric reproduction up to CSV roundoff."""
    maximum = 0.
    for table, filename, metrics, keys in (
        (coarse, 'coarse_month_metrics.csv', COARSE_METRICS, ['profile_id']),
        (fuels, 'coarse_fuel_metrics.csv', FUEL_METRICS, ['profile_id', 'fuel_index']),
    ):
        old = pd.read_csv(V184 / f'fold_{fold}' / filename)
        for name in (*FORESTS, 'gaussian'):
            old_name = 'gaussian_increment' if name == 'gaussian' else name
            a = table.loc[table.method.eq('increment__' + name)].sort_values(keys)
            b = old.loc[old.method.eq('long_all__' + old_name)].sort_values(keys)
            if len(a) != len(b) or not np.array_equal(a[keys].to_numpy(), b[keys].to_numpy()):
                raise ValueError('old row identities differ')
            maximum = max(maximum, float(np.max(abs(a[list(metrics)].to_numpy(float) - b[list(metrics)].to_numpy(float)))))
    if maximum > 1e-8:
        raise ValueError('V184 reference scores differ')
    return maximum


def run_fold(data, inventory, fold, destination, manifest_sha):
    """Build each full factorial before accessing its held current shares."""
    model, queries = fit_fold(data, inventory, fold)
    expected = json.loads((PRE / 'model_fingerprints.json').read_text())[str(fold)]
    if model.metadata() != expected:
        raise ValueError('V185 preflight fit fingerprint changed')
    coarse, fuels, rows, arrays, weights = [], [], [], [], []
    offset = 0
    for number, rec in enumerate(queries.itertuples(index=False), start=1):
        q = int(rec.profile_id)
        laws = model.build(q)
        truth = data.shares[q].copy()
        identity = {'fold': fold, 'profile_id': q, 'region': rec.region, 'month': rec.month}
        cache = {}
        for name, (v, p) in laws.items():
            c, f = score(v, p, truth, cache)
            coarse.append(identity | {'method': name} | c)
            fuels.extend(identity | {'method': name} | item for item in f)
            rows.append(identity | {'method': name, 'coarse_offset': offset, 'coarse_stop': offset + len(v)})
            arrays.append(v); weights.append(p); offset += len(v)
        if number % 10 == 0 or number == len(queries):
            print(f'V185 fold {fold}: {number}/{len(queries)}', flush=True)
    coarse, fuels = pd.DataFrame(coarse), pd.DataFrame(fuels)
    error = compare_old(coarse, fuels, fold)
    staging = Path(tempfile.mkdtemp(prefix=f'.fold_{fold}_', dir=destination.parent))
    for name, frame in [('coarse_month_metrics', coarse), ('coarse_fuel_metrics', fuels), ('prediction_index', pd.DataFrame(rows)),
                        ('training_prefix', pd.DataFrame(source_prefix_rows(model.base)))]:
        frame.to_csv(staging / f'{name}.csv', index=False)
    np.savez_compressed(staging / 'coarse_predictions.npz', values=np.concatenate(arrays), probabilities=np.concatenate(weights))
    np.savez_compressed(staging / 'training_matrices.npz', **source_matrices(model.base), level_target=model.target,
                        level_mean=model.mean, level_scale=model.scale)
    (staging / 'model_fingerprints.json').write_text(json.dumps(model.metadata(), indent=2, allow_nan=False) + '\n')
    summary = {'assertion_failures': 0, 'manifest_sha256': manifest_sha, 'coarse_score_rows': len(coarse), 'fuel_score_rows': len(fuels),
               'max_reference_error': error, 'output_hashes': {p.name: sha(p) for p in staging.iterdir()}}
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
    hashes = checked_hashes()
    data = ExtendedMonthlyInformation(ROOT, ROOT / 'results/v183-coarse-history-export-r1/monthly_history.csv')
    inventory = pd.read_csv(CONTEXT / 'training_inventory.csv')
    if args.preflight:
        metadata, cases = {}, 0
        with threadpool_limits(limits=2):
            for fold in range(6):
                model, queries = fit_fold(data, inventory, fold)
                metadata[str(fold)] = model.metadata()
                for q in queries.profile_id.iloc[[0, -1]]:
                    model.future_guard(int(q)); cases += 1
        if cases != 12 or checked_hashes() != hashes:
            raise ValueError('preflight inventory or provenance changed')
        destination.mkdir()
        (destination / 'model_fingerprints.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        result = {'assertion_failures': 0, 'model_fold_groups': 6, 'future_query_cases': cases, 'future_method_guards': cases * 16,
                  'held_outcomes_scored': False, 'input_hashes': hashes,
                  'output_hashes': {'model_fingerprints.json': sha(destination / 'model_fingerprints.json')}}
    else:
        if verify(PRE)['input_hashes'] != hashes:
            raise ValueError('preflight inputs differ')
        hashes[str((PRE / 'summary.json').relative_to(ROOT))] = sha(PRE / 'summary.json')
        hashes[str((PRE / 'model_fingerprints.json').relative_to(ROOT))] = sha(PRE / 'model_fingerprints.json')
        manifest = json.dumps({'input_hashes': hashes, 'names': NAMES, 'supports': SUPPORTS}, sort_keys=True, indent=2, allow_nan=False) + '\n'
        if destination.exists():
            if (destination / 'run_manifest.json').read_text() != manifest or (destination / 'summary.json').exists():
                raise ValueError('invalid resume')
        else:
            destination.mkdir(); (destination / 'run_manifest.json').write_text(manifest)
        with threadpool_limits(limits=2):
            for fold in range(6):
                folder = destination / f'fold_{fold}'
                if folder.exists():
                    if verify(folder)['manifest_sha256'] != sha(destination / 'run_manifest.json'):
                        raise ValueError('fold manifest differs')
                else:
                    run_fold(data, inventory, fold, folder, sha(destination / 'run_manifest.json'))
        coarse = pd.concat([pd.read_csv(destination / f'fold_{k}/coarse_month_metrics.csv') for k in range(6)], ignore_index=True)
        fuels = pd.concat([pd.read_csv(destination / f'fold_{k}/coarse_fuel_metrics.csv') for k in range(6)], ignore_index=True)
        if (len(coarse) != 3856 or len(fuels) != 30848 or coarse.duplicated(['profile_id', 'method']).any()
                or fuels.duplicated(['profile_id', 'method', 'fuel_index']).any()):
            raise ValueError('formal score inventory')
        tables = dict(zip(('coarse_region_metrics', 'coarse_macro_metrics', 'coarse_fuel_region_metrics', 'coarse_fuel_macro_metrics'), aggregate(coarse, fuels)))
        for name, table in {'coarse_month_metrics': coarse, 'coarse_fuel_metrics': fuels, **tables}.items():
            table.to_csv(destination / f'{name}.csv', index=False)
        if any(sha(ROOT / p) != h for p, h in hashes.items()):
            raise ValueError('inputs changed during experiment')
        result = {'assertion_failures': 0, 'completed_blocks': 6, 'coarse_score_rows': len(coarse), 'fuel_score_rows': len(fuels),
                  'input_hashes': hashes, 'output_hashes': {str(p.relative_to(destination)): sha(p) for p in destination.rglob('*') if p.is_file()},
                  'scope': 'Ordinary coarse response/support controls; no hourly or original-method result.'}
    (destination / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
