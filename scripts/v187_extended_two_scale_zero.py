"""Frozen extended-history month-state and positive-month hourly-risk controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from extended_monthly_information import ExtendedMonthlyInformation
from extended_two_scale_zero import (ExtendedTwoScaleZero, METHODS, MONTH_METHODS, COMPONENTS,
                                     MONTH_HEADS, HURDLES, check_inputs)
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from monthly_history_features import MonthlyInformation
from monthly_history_zero_risk import METHODS as DIRECT_METHODS
from two_scale_zero import MONTH_METHODS as OLD_MONTH_METHODS
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import verify
from v171_two_scale_zero import METRICS, aggregate, score_query

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
OLD = ROOT / 'results/v171-two-scale-zero-r1'
DIRECT = ROOT / 'results/v186-monthly-history-zero-risk-r1'
CONTEXT = ROOT / 'results/v184-monthly-context-preflight-r1'
OUT = ROOT / 'results/v187-extended-two-scale-zero-r1'
PRE = ROOT / 'results/v187-extended-two-scale-zero-preflight-r1'
PROTOCOL = ROOT / 'docs/V187_EXTENDED_TWO_SCALE_ZERO_PROTOCOL.md'
COUNTS = {'hourly': 40488, 'monthly': 15424, 'components': 4544}


def checked_hashes() -> dict:
    hashes = {}
    for folder in (DIRECT, ROOT / 'results/v186-monthly-history-zero-risk-audit-r1',
                   OLD, ROOT / 'results/v171-zero-audit-r1', CONTEXT):
        summary = verify(folder)
        for name, value in summary['input_hashes'].items():
            if name in hashes and hashes[name] != value:
                raise ValueError('upstream hash conflict')
            hashes[name] = value
        for name in ('summary.json', *summary['output_hashes']):
            path = folder / name
            hashes[str(path.relative_to(ROOT))] = sha(path)
    for path in (Path(__file__), PROTOCOL, ROOT / 'scripts/extended_two_scale_zero.py'):
        hashes[str(path.relative_to(ROOT))] = sha(path)
    if any(sha(ROOT / name) != value for name, value in hashes.items()):
        raise ValueError('frozen input changed')
    return hashes


def make_predictor(data, short, long, inventory, fold: int):
    frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
    ids = inventory.loc[inventory.fold.eq(fold) & inventory.budget.eq('long_all'), 'profile_id'].to_numpy(int)
    return ExtendedTwoScaleZero(frozen, short, long, ids, DIRECT / f'fold_{fold}', OLD / f'fold_{fold}')


def training_arrays(predictor) -> dict:
    return {'raw_id': predictor.raw_ids, 'profile_id': predictor.profiles,
            'region': np.asarray(predictor.regions, str), 'zero': predictor.zero, 'positive_month': predictor.positive,
            **{'X_' + name: X for name, X in predictor.X.items()},
            'month_profile_id': predictor.month_ids, 'month_region': np.asarray(predictor.month_regions, str),
            'month_zero': predictor.month_zero, **{'month_X_' + name: X for name, X in predictor.month_X.items()}}


def monthly_prefix_rows(predictor) -> list[dict]:
    return [{'profile_id': int(q), 'region': predictor.long.index.region.iloc[q],
             'month': predictor.long.index.month.iloc[q], 'local': int(predictor.month_contexts[int(q)]['local']),
             'prefix': json.dumps([int(p) for p in predictor.month_contexts[int(q)]['prefix']])}
            for q in predictor.month_ids]


def reproduce(frames, fold: int) -> dict:
    result = {}
    specifications = [('hourly', DIRECT, DIRECT_METHODS, {}),
                      ('monthly', OLD, OLD_MONTH_METHODS, {}),
                      ('components', OLD, ('clock', 'short_cond43'), {'conditional43': 'short_cond43'})]
    for kind, archive, names, aliases in specifications:
        old = pd.read_csv(archive / f'fold_{fold}/{kind}_scores.csv')
        if aliases:
            old.method = old.method.replace(aliases)
        current = frames[kind].loc[frames[kind].method.isin(names)]
        pair = current.merge(old, on=['profile_id', 'method', 'fuel'], validate='one_to_one', suffixes=('_a', '_b'))
        error = max(float(np.max(abs(pair[m + '_a'] - pair[m + '_b']))) for m in METRICS)
        if len(pair) != len(old) or len(pair) != len(current) or error > 1e-12:
            raise ValueError(f'old {kind} scores not reproduced')
        result[kind + '_old_rows'] = len(pair)
        result[kind + '_old_max_error'] = error
    return result


def run_fold(data, short, long, inventory, fold: int, destination: Path, manifest_sha: str, preflight: bool = False):
    predictor = make_predictor(data, short, long, inventory, fold)
    for q in predictor.frozen.queries.profile_id.iloc[[0, -1]]:
        check_inputs(predictor, int(q))
    fingerprint = predictor.fingerprint()
    # Exercise exactly the formal prefix serializer before any outcome scoring.
    json.dumps(monthly_prefix_rows(predictor), allow_nan=False)
    if preflight:
        return fingerprint
    if fingerprint != json.loads((PRE / 'model_fingerprints.json').read_text())[str(fold)]:
        raise ValueError('preflight fit fingerprint differs')
    outputs = {k: [] for k in COUNTS}
    saved = {k: [] for k in COUNTS}
    contexts = []
    for rec in predictor.frozen.queries.itertuples(index=False):
        q = int(rec.profile_id)
        hourly, monthly, components, context = predictor.predict(q)
        for kind, values, names in [('hourly', hourly, METHODS), ('monthly', monthly, MONTH_METHODS),
                                    ('components', components, COMPONENTS)]:
            saved[kind].append(np.stack([values[m] for m in names]))
        contexts.append({'profile_id': q, 'local': context['local'], 'prefix': context['prefix'],
                         'base': context['base'].tolist(), 'history': context['history'].tolist()})
        # All unconditional and conditional forecasts precede outcome access.
        if rec.role == 'test':
            scored = score_query(data, rec, hourly, monthly, components)
            for kind, rows in scored.items():
                outputs[kind].extend({'fold': fold} | row for row in rows)
    frames = {k: pd.DataFrame(v) for k, v in outputs.items()}
    ntest = int(predictor.frozen.queries.role.eq('test').sum())
    if (len(frames['hourly']) != ntest * 168 or len(frames['monthly']) != ntest * 64
            or any(not np.isfinite(frame[list(METRICS)]).all().all() for frame in frames.values())):
        raise ValueError('fold score inventory/finiteness')
    reproduction = reproduce(frames, fold)
    staging = Path(tempfile.mkdtemp(prefix='.v187-zero-', dir=OUT.parent))
    for kind, frame in frames.items():
        frame.to_csv(staging / f'{kind}_scores.csv', index=False)
    np.savez_compressed(staging / 'probabilities.npz', profile_id=predictor.frozen.queries.profile_id.to_numpy(int),
                        **{k: np.asarray(v) for k, v in saved.items()})
    np.savez_compressed(staging / 'training_matrices.npz', **training_arrays(predictor))
    pd.DataFrame(monthly_prefix_rows(predictor)).to_csv(staging / 'monthly_training_prefix.csv', index=False)
    predictor.frozen.queries.to_csv(staging / 'query_inputs.csv', index=False)
    for name, value in [('model_fingerprints', fingerprint), ('training_provenance', predictor.provenance), ('contexts', contexts)]:
        (staging / f'{name}.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    result = {'assertion_failures': 0, 'fold': fold, 'proper_regions': predictor.frozen.fit,
              'manifest_sha256': manifest_sha, 'forests_fitted': 40, 'monthly_training_profiles': len(predictor.month_ids),
              'hourly_rows': len(frames['hourly']), 'monthly_rows': len(frames['monthly']),
              'conditional_task_rows': len(frames['components']), 'future_query_raw_guards': 2,
              **reproduction, 'output_hashes': {p.name: sha(p) for p in staging.iterdir()}}
    (staging / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    staging.rename(destination)
    print(f'completed V187 fold {fold}: {len(frames["hourly"])} hourly scores', flush=True)
    return fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    destination = PRE if args.preflight else OUT
    if destination.exists() and (args.preflight or not args.resume):
        raise FileExistsError(destination)
    hashes = checked_hashes()
    data, short = HourlyFuelData(ROOT), MonthlyInformation(ROOT)
    long = ExtendedMonthlyInformation(ROOT, ROOT / 'results/v183-coarse-history-export-r1/monthly_history.csv')
    inventory = pd.read_csv(CONTEXT / 'training_inventory.csv')
    if args.preflight:
        metadata = {}
        with threadpool_limits(limits=2):
            for fold in range(6):
                metadata[str(fold)] = run_fold(data, short, long, inventory, fold, destination, '', True)
                print(f'V187 preflight fold {fold} complete', flush=True)
        if checked_hashes() != hashes:
            raise ValueError('preflight input changed')
        destination.mkdir()
        (destination / 'model_fingerprints.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        result = {'assertion_failures': 0, 'forests_fitted': 240, 'future_query_raw_guards': 12,
                  'query_outcomes_scored': False, 'input_hashes': hashes,
                  'output_hashes': {'model_fingerprints.json': sha(destination / 'model_fingerprints.json')}}
    else:
        if verify(PRE)['input_hashes'] != hashes:
            raise ValueError('preflight code/data differs')
        for name in ('summary.json', 'model_fingerprints.json'):
            hashes[str((PRE / name).relative_to(ROOT))] = sha(PRE / name)
        manifest = json.dumps({'input_hashes': hashes, 'hourly_methods': METHODS, 'monthly_methods': MONTH_METHODS,
                               'conditional_methods': COMPONENTS, 'source_year': 2024, 'sample_budget': 'asof_30d'},
                              indent=2, sort_keys=True, allow_nan=False) + '\n'
        if destination.exists():
            if (destination / 'run_manifest.json').read_text() != manifest or (destination / 'summary.json').exists():
                raise ValueError('invalid resume')
        else:
            destination.mkdir()
            (destination / 'run_manifest.json').write_text(manifest)
        with threadpool_limits(limits=2):
            for fold in range(6):
                block = destination / f'fold_{fold}'
                if block.exists():
                    if verify(block)['manifest_sha256'] != sha(destination / 'run_manifest.json'):
                        raise ValueError('saved block manifest differs')
                else:
                    run_fold(data, short, long, inventory, fold, block, sha(destination / 'run_manifest.json'))
        for kind, expected in COUNTS.items():
            rows = pd.concat([pd.read_csv(destination / f'fold_{f}/{kind}_scores.csv') for f in range(6)], ignore_index=True)
            if len(rows) != expected or rows.duplicated(['profile_id', 'method', 'fuel']).any():
                raise ValueError(f'root {kind} score inventory')
            regional, macro = aggregate(rows)
            for name, frame in [('scores', rows), ('region_metrics', regional), ('macro_metrics', macro)]:
                frame.to_csv(destination / f'{kind}_{name}.csv', index=False)
        if any(sha(ROOT / p) != h for p, h in hashes.items()):
            raise ValueError('formal inputs changed')
        result = {'assertion_failures': 0, 'completed_folds': 6, 'forests_fitted': 240, **COUNTS,
                  'input_hashes': hashes, 'output_hashes': {str(p.relative_to(destination)): sha(p)
                                                           for p in destination.rglob('*') if p.is_file()},
                  'scope': 'Ordinary extended-history hurdle controls. Conditional current-positive task is diagnostic, '
                           'not deployment selection; no new CIF or original-method claim.'}
    (destination / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
