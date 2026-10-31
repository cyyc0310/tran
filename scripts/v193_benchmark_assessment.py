"""Read-only adapter/assessor for complete, already-viewed gate3 sealed forecasts."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd

from transcif.research.assessment_v193 import (ArtifactKind, ForecastArtifact, InformationContract,
                                              summarize_selection, validate_cohorts, validate_scoring_masks)
from transcif.research.references import proxy_values
from v151_hourly_law import sha
from v193_contract_checks import run_checks

BASE = ROOT / 'results/research-prefix'
OUT = ROOT / 'results/v193-benchmark-assessment-r3'
REGIONS = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
METHODS = ('prior', 'gaussian', 'diverse', 'level_template', 'ridge', 'extra_trees')
LIMITS = (50., 100., 200., 400., 1000.)
EF = np.array([1000., 400., 800., 0., 0., 0., 0., 400.])
OLD_METRICS = ('mae', 'rmse', 'crps', 'coverage90', 'width90', 'shape_mae', 'level_mae')


def contract(row: dict, sources: tuple[str, ...]) -> InformationContract:
    reference = row['prediction_kind'] == 'reference'
    return InformationContract(row['region'], row['origin'], row['available_at'], row['period_end'], sources,
                               sources if reference else (), 'source_residual_transfer' if reference else 'none',
                               '2024-01-01T00:00:00Z', '2023-01-01T00:00:00Z' if reference else None,
                               '2024-01-01T00:00:00Z' if reference else None)


def artifact(packed, row: dict, sources: tuple[str, ...]) -> ForecastArtifact:
    key = row['source_key']
    if row['prediction_kind'] == 'inference':
        predictions, weights = packed[key + '_pred'], packed[key + '_weights']
    else:
        predictions = np.clip(packed[key][None] + packed['residual_' + row['method']], 0, 1000)
        weights = np.ones(len(predictions))
    return ForecastArtifact(ArtifactKind.DISTRIBUTION, contract(row, sources), .1,
                            predictions=predictions, weights=weights)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = [Path(__file__), ROOT / 'scripts/v193_contract_checks.py',
             ROOT / 'src/transcif/research/assessment_v193.py',
             ROOT / 'src/transcif/research/weighted_metrics.py', ROOT / 'src/transcif/research/references.py',
             ROOT / 'scripts/v151_hourly_law.py', ROOT / 'docs/V193_BENCHMARK_ASSESSMENT_PROTOCOL.md']
    paths += [BASE / 'gate3' / f'{r}_other400' / name for r in REGIONS
              for name in ('prediction_seal.json', 'prediction_index.csv', 'predictions.npz')]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    checks = run_checks()
    rows, means, lowers, uppers = [], [], [], []
    cohorts = {method: [] for method in METHODS}
    for region in REGIONS:
        directory = BASE / 'gate3' / f'{region}_other400'
        seal = json.loads((directory / 'prediction_seal.json').read_text())
        if (seal['target'] != region or sha(directory / 'predictions.npz') != seal['prediction_sha256']
                or sha(directory / 'prediction_index.csv') != seal['index_sha256']
                or region in seal['model']['provided_source_ids']
                or set(seal['sources']) != set(REGIONS) - {region}):
            raise ValueError('old prediction seal/role/hash')
        source_regions = tuple(seal['sources'])
        index = pd.read_csv(directory / 'prediction_index.csv')
        index = index.loc[index.method.isin(METHODS) & (((index.kind == 'inference') & (index.seed == 0))
                                                      | ((index.kind == 'reference') & (index.seed == -1)))]
        if len(index) != 366 * 6 or index.duplicated(['method', 'origin']).any():
            raise ValueError('full-year method grid')
        expected_days = pd.date_range('2024-01-01', '2024-12-31', freq='D', tz='UTC')
        with np.load(directory / 'predictions.npz', allow_pickle=False) as packed:
            for method in METHODS:
                group = index.loc[index.method.eq(method)].sort_values('origin')
                if not pd.DatetimeIndex(pd.to_datetime(group.origin, utc=True)).equals(expected_days):
                    raise ValueError('full origin calendar')
                for rec in group.to_dict('records'):
                    row = {'region': region, 'origin': rec['origin'], 'method': method, 'seed': int(rec['seed']),
                           'source_key': rec['key'], 'prediction_kind': rec['kind'],
                           'period_end': rec['period_end'], 'available_at': rec['available_at']}
                    a = artifact(packed, row, source_regions)
                    row.update(artifact_id=len(rows), predictive_width=a.width,
                               calibration_kind=a.contract.calibration_kind)
                    cohorts[method].append(a.contract)
                    rows.append(row);means.append(a.prepared.mean);lowers.append(a.lower);uppers.append(a.upper)
        print(f'adapted/sealed-input {region} full year; no labels read', flush=True)
    calibration_policies = {method: ('none', 'empty', None, None) if method in METHODS[:3]
                            else ('source_residual_transfer', 'sources', '2023-01-01T00:00:00Z', '2024-01-01T00:00:00Z')
                            for method in METHODS}
    validate_cohorts(cohorts, calibration_policies)
    if len(rows) != 15372 or any(sha(ROOT / key) != value for key, value in hashes.items()):
        raise ValueError('forecast count/changed input')
    OUT.mkdir()
    pd.DataFrame(rows).to_csv(OUT / 'artifact_index.csv', index=False)
    np.savez_compressed(OUT / 'forecast_summaries.npz', mean=np.asarray(means), lower=np.asarray(lowers), upper=np.asarray(uppers))
    seal = {'prediction_index_sha256': sha(OUT / 'artifact_index.csv'),
            'prediction_summaries_sha256': sha(OUT / 'forecast_summaries.npz'),
            'all_forecasts_adapted_before_target_label_read': True, 'already_viewed_development_labels': True,
            'probability_artifacts': len(rows), 'origin_region_windows': 2562,
            'pre_score_input_hashes': hashes, 'calibration_difference_policies': calibration_policies}
    (OUT / 'prediction_seal.json').write_text(json.dumps(seal, indent=2) + '\n')
    # First label or old-score file read occurs after the complete all-region seal.
    results, max_old_error = [], 0.
    for region in REGIONS:
        path = BASE / 'data' / f'{region}_hourly.csv'
        old_path = BASE / 'gate3' / f'{region}_other400' / 'window_scores.csv'
        for p in (path, old_path):
            hashes[str(p.relative_to(ROOT))] = sha(p)
        labels = pd.read_csv(path)
        labels['time'] = pd.to_datetime(labels.time, utc=True)
        labels = labels.loc[labels.time.dt.year.eq(2024)]
        labels['day'] = labels.time.dt.floor('D')
        truth_by_day = {}
        for day, frame in labels.groupby('day'):
            frame = frame.sort_values('time')
            if (len(frame) == 24 and frame.valid.all()
                    and pd.DatetimeIndex(frame.time).equals(pd.date_range(day, periods=24, freq='h'))):
                truth = proxy_values(frame, EF)
                if np.isfinite(truth).all():
                    truth_by_day[day.isoformat()] = truth
        old = pd.read_csv(old_path)
        old = old.loc[old.method.isin(METHODS) & old.seed.isin([0, -1])]
        if old.duplicated(['method', 'origin']).any():
            raise ValueError('old score duplicate')
        source_regions = tuple(json.loads((BASE / 'gate3' / f'{region}_other400/prediction_seal.json').read_text())['sources'])
        with np.load(BASE / 'gate3' / f'{region}_other400/predictions.npz', allow_pickle=False) as packed:
            for row in (r for r in rows if r['region'] == region):
                rec = row.copy();y = truth_by_day.get(row['origin'])
                if y is None:
                    rec.update(status='missing_label', hours=0)
                else:
                    a = artifact(packed, row, source_regions)
                    values = a.score(y)
                    old_row = old.loc[old.method.eq(row['method']) & old.origin.eq(row['origin'])]
                    if len(old_row) != 1:
                        raise ValueError('old scored cohort differs')
                    error = max(abs(values[m] - float(old_row[m].iloc[0])) for m in OLD_METRICS)
                    if not np.isfinite(error) or error > 1e-8:
                        raise ValueError(f'old metric reproduction {region}: {error}')
                    max_old_error = max(max_old_error, error)
                    rec.update(status='scored', hours=24, **values)
                results.append(rec)
        if set(old.origin) != set(truth_by_day):
            raise ValueError('old/new label day cohorts differ')
        print(f'scored {region}, valid days={len(truth_by_day)}', flush=True)
    scores = pd.DataFrame(results)
    validate_scoring_masks(scores)
    scores.to_csv(OUT / 'window_scores.csv', index=False)
    regional = []
    metric_columns = [*OLD_METRICS, 'winkler90']
    for (region, method), group in scores.loc[scores.status.eq('scored')].groupby(['region', 'method']):
        value = {m: float(np.average(group[m], weights=group.hours)) for m in metric_columns}
        value['rmse'] = float(np.sqrt(np.average(group.rmse ** 2, weights=group.hours)))
        regional.append({'region': region, 'method': method, 'windows': len(group),
                         'hours': int(group.hours.sum()), **value})
    regional = pd.DataFrame(regional)
    macro = regional.groupby('method', as_index=False)[metric_columns].mean()
    regional.to_csv(OUT / 'region_metrics.csv', index=False);macro.to_csv(OUT / 'macro.csv', index=False)
    month = scores.loc[scores.status.eq('scored')].copy();month['month'] = pd.to_datetime(month.origin, utc=True).dt.month
    month['rmse'] = month['rmse'] ** 2
    month_metrics = month.groupby(['region', 'month', 'method'], as_index=False)[metric_columns].mean()
    month_metrics['rmse'] = np.sqrt(month_metrics['rmse'])
    month_metrics.to_csv(OUT / 'region_month_metrics.csv', index=False)
    selective = [{'method': method, **summarize_selection(scores.loc[scores.method.eq(method)], limit)}
                 for method in METHODS for limit in LIMITS]
    (OUT / 'selection.json').write_text(json.dumps(selective, indent=2, allow_nan=False) + '\n')
    if (sha(OUT / 'artifact_index.csv') != seal['prediction_index_sha256']
            or sha(OUT / 'forecast_summaries.npz') != seal['prediction_summaries_sha256']
            or any(sha(ROOT / key) != value for key, value in hashes.items())):
        raise ValueError('final frozen inputs/seal changed')
    summary = {'assertion_failures': 0, 'regions': list(REGIONS), 'methods': list(METHODS),
               'probability_artifacts': len(rows), 'scored_method_windows': int(scores.status.eq('scored').sum()),
               'missing_method_windows': int(scores.status.eq('missing_label').sum()),
               'scored_origin_region_windows': int(scores.status.eq('scored').sum()) // 6,
               'scored_days_by_region': scores.loc[scores.method.eq(METHODS[0]) & scores.status.eq('scored')].groupby('region').size().to_dict(),
               'old_metric_reproduction_max_error': max_old_error, 'selection_rows': len(selective),
               'contract_checks': checks, 'new_models_fitted': False, 'new_regions_opened': False,
               'identified_set_empirical_ranking': False, 'new_coverage_guarantee': False,
               'effective_version': 'r3', 'r2_retired_reason': 'Monthly RMSE was mean daily RMSE, corrected to sqrt(mean daily MSE); main region/macro/selection unaffected.',
               'input_hashes': hashes, 'output_hashes': {p.name: sha(p) for p in OUT.iterdir()}}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print({k: v for k, v in summary.items() if k not in ('input_hashes', 'output_hashes', 'contract_checks')}, flush=True)


if __name__ == '__main__':
    main()
