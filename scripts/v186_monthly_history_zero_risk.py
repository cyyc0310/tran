"""Frozen same-budget monthly-history/hourly-zero information-value experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from extended_monthly_information import ExtendedMonthlyInformation
from frozen_recomposition import FrozenRecomposition
from hourly_fuel_data import HourlyFuelData
from monthly_history_features import MonthlyInformation
from monthly_history_zero_risk import MonthlyHistoryZeroRisk, METHODS, OLD_ALIASES, check_inputs
from monthly_risk_features import NAMES
from v151_hourly_law import sha
from v170_fuel_marginal_diagnosis import FUELS, verify
from v171_two_scale_zero import METRICS, aggregate, score_probability

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/v159-fuel-activity-source-pilot-r1'
OLD = ROOT / 'results/v171-two-scale-zero-r1'
OUT = ROOT / 'results/v186-monthly-history-zero-risk-r1'
PRE = ROOT / 'results/v186-monthly-history-zero-risk-preflight-r1'
PROTOCOL = ROOT / 'docs/V186_MONTHLY_HISTORY_ZERO_RISK_PROTOCOL.md'


def checked_hashes() -> dict:
    hashes = {}
    for folder in (BASE, OLD, ROOT / 'results/v171-zero-audit-r1',
                   ROOT / 'results/v186-monthly-risk-feature-preflight-r2',
                   ROOT / 'results/v183-coarse-history-export-r1', ROOT / 'results/v183-coarse-history-export-audit-r1'):
        summary = verify(folder)
        for name, value in summary['input_hashes'].items():
            if name in hashes and hashes[name] != value:
                raise ValueError('upstream hash conflict')
            hashes[name] = value
        for name in ('summary.json', *summary['output_hashes']):
            path = folder / name
            hashes[str(path.relative_to(ROOT))] = sha(path)
    for path in (Path(__file__), PROTOCOL, ROOT / 'scripts/monthly_history_zero_risk.py',
                 ROOT / 'scripts/monthly_risk_features.py', ROOT / 'scripts/v171_two_scale_zero.py',
                 ROOT / 'scripts/two_scale_zero.py', ROOT / 'scripts/extended_monthly_information.py'):
        name = str(path.relative_to(ROOT))
        value = sha(path)
        if name in hashes and hashes[name] != value:
            raise ValueError('frozen helper differs from upstream')
        hashes[name] = value
    if any(sha(ROOT / name) != value for name, value in hashes.items()):
        raise ValueError('input changed')
    return hashes


def make_predictor(data, short, long, fold: int):
    frozen = FrozenRecomposition(data, BASE, fold, 'asof_30d')
    return MonthlyHistoryZeroRisk(frozen, short, long, OLD / f'fold_{fold}')


def query_scores(data, rec, probabilities: dict) -> list[dict]:
    q = int(rec.profile_id)
    row = data.index.iloc[q]
    state = data.states[int(row.offset):int(row.stop)]
    observed = data.raw[int(row.offset):int(row.stop)] == 0
    local = data.local(q, 'asof_30d')
    rows = []
    for f, fuel in enumerate(FUELS):
        identity = {'profile_id': q, 'region': rec.region, 'month': rec.month, 'hours': len(state),
                    'fuel': fuel, 'input_fuel_positive': bool(data.M[local, f] > 0),
                    'current_fuel_positive': bool(data.M[q, f] > 0)}
        for method in METHODS:
            rows.append(identity | {'method': method, **score_probability(probabilities[method][state, f], observed[:, f])})
    return rows


def run_fold(data, short, long, fold: int, destination: Path, manifest_sha: str, preflight: bool = False):
    predictor = make_predictor(data, short, long, fold)
    for q in predictor.frozen.queries.profile_id.iloc[[0, -1]]:
        check_inputs(predictor, int(q))
    fingerprints = predictor.fingerprint()
    if preflight:
        return fingerprints
    expected = json.loads((PRE / 'model_fingerprints.json').read_text())[str(fold)]
    if fingerprints != expected:
        raise ValueError('preflight fit fingerprint differs')
    saved, rows = [], []
    for rec in predictor.frozen.queries.itertuples(index=False):
        probabilities = predictor.predict(int(rec.profile_id))
        saved.append(np.stack([probabilities[name] for name in METHODS]))
        # Every method's complete forecast exists before query zero labels are read.
        if rec.role == 'test':
            rows.extend({'fold': fold} | row for row in query_scores(data, rec, probabilities))
    rows = pd.DataFrame(rows)
    ntest = int(predictor.frozen.queries.role.eq('test').sum())
    if len(rows) != ntest * 80 or not np.isfinite(rows[list(METRICS)]).all().all():
        raise ValueError('score inventory/finiteness')
    old = pd.read_csv(OLD / f'fold_{fold}/hourly_scores.csv')
    old.method = old.method.map(OLD_ALIASES)
    pair = rows.loc[rows.method.isin(OLD_ALIASES.values())].merge(old, on=['profile_id', 'method', 'fuel'],
                                                               validate='one_to_one', suffixes=('_a', '_b'))
    reproduction = max(float(max(abs(pair[m + '_a'] - pair[m + '_b']))) for m in METRICS)
    if len(pair) != ntest * 64 or reproduction > 1e-12:
        raise ValueError('V171 old score reproduction failed')
    staging = Path(tempfile.mkdtemp(prefix='.v186-zero-', dir=OUT.parent))
    rows.to_csv(staging / 'hourly_scores.csv', index=False)
    predictor.frozen.queries.to_csv(staging / 'query_inputs.csv', index=False)
    np.savez_compressed(staging / 'probabilities.npz', profile_id=predictor.frozen.queries.profile_id.to_numpy(int),
                        hourly=np.asarray(saved))
    (staging / 'training_provenance.json').write_text(json.dumps(predictor.provenance, indent=2, allow_nan=False) + '\n')
    (staging / 'model_fingerprints.json').write_text(json.dumps(fingerprints, indent=2, allow_nan=False) + '\n')
    for name in NAMES:
        np.savez_compressed(staging / f'{name}_training.npz', X=predictor.X[name], target=predictor.zero,
                            raw_id=predictor.raw_ids, profile_id=predictor.profiles, region=np.asarray(predictor.regions, str))
    summary = {'assertion_failures': 0, 'fold': fold, 'proper_regions': predictor.frozen.fit,
               'manifest_sha256': manifest_sha, 'score_rows': len(rows), 'forests_fitted': 32,
               'query_predictions': len(saved), 'old_score_rows': len(pair), 'old_score_max_error': reproduction,
               'direct_probabilities_bit_exact': True, 'future_month_raw_guards': 2,
               'output_hashes': {p.name: sha(p) for p in staging.iterdir()}}
    (staging / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    staging.rename(destination)
    print(f'completed V186 fold {fold}: {len(rows)} scores', flush=True)
    return fingerprints


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
    if args.preflight:
        metadata = {}
        with threadpool_limits(limits=2):
            for fold in range(6):
                metadata[str(fold)] = run_fold(data, short, long, fold, destination, '', True)
                print(f'V186 model preflight fold {fold} complete', flush=True)
        if checked_hashes() != hashes:
            raise ValueError('preflight input changed')
        destination.mkdir()
        (destination / 'model_fingerprints.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        result = {'assertion_failures': 0, 'forests_fitted': 192, 'query_future_raw_guards': 12,
                  'direct_probability_checks_scope': '12 first/last fold queries only; formal checks every query',
                  'query_outcomes_scored': False, 'input_hashes': hashes,
                  'output_hashes': {'model_fingerprints.json': sha(destination / 'model_fingerprints.json')}}
    else:
        if verify(PRE)['input_hashes'] != hashes:
            raise ValueError('preflight code/data differs')
        for name in ('summary.json', 'model_fingerprints.json'):
            hashes[str((PRE / name).relative_to(ROOT))] = sha(PRE / name)
        manifest = json.dumps({'input_hashes': hashes, 'methods': METHODS, 'source_year': 2024,
                               'sample_budget': 'asof_30d'}, indent=2, sort_keys=True, allow_nan=False) + '\n'
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
                        raise ValueError('saved fold manifest differs')
                else:
                    run_fold(data, short, long, fold, block, sha(destination / 'run_manifest.json'))
        rows = pd.concat([pd.read_csv(destination / f'fold_{f}/hourly_scores.csv') for f in range(6)], ignore_index=True)
        if len(rows) != 19280 or rows.duplicated(['profile_id', 'method', 'fuel']).any():
            raise ValueError('root score inventory')
        regional, macro = aggregate(rows)
        for name, table in [('hourly_scores', rows), ('hourly_region_metrics', regional), ('hourly_macro_metrics', macro)]:
            table.to_csv(destination / f'{name}.csv', index=False)
        if any(sha(ROOT / p) != h for p, h in hashes.items()):
            raise ValueError('formal input changed')
        result = {'assertion_failures': 0, 'completed_folds': 6, 'score_rows': len(rows), 'forests_fitted': 192,
                  'input_hashes': hashes, 'output_hashes': {str(p.relative_to(destination)): sha(p)
                                                           for p in destination.rglob('*') if p.is_file()},
                  'scope': 'Same-hour-budget ordinary zero-event forests, source development; no CIF forecast or originality claim.'}
    (destination / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if 'hash' not in k}, indent=2), flush=True)


if __name__ == '__main__':
    main()
