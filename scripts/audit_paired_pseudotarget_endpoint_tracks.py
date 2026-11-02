#!/usr/bin/env python3
"""Re-score sealed pseudo-target branches with their exact paired endpoint baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_eia923_pseudotarget_transfer as monthly_inputs
import run_official_intensity_transfer as official
from run_cross_country import ROOT
import run_pseudotarget_recomposition_residual_generation_proxy as generation_runner
from run_pseudotarget_recomposition_residual_generation_proxy import proxy_label


ARMS = ('hourly_reconstruction', 'eia923_monthly')
TRACKS = ('generation_proxy', 'consumption_official')
BASE_METHODS = tuple(official.METHODS)
PSEUDO_METHODS = ('pseudo_residual_residual_ridge', 'pseudo_residual_residual_hgb')
METRICS = ('mae', 'shape_mae', 'crps_gaussian', 'coverage90', 'width90', 'interval_score90')
PAIRED_ROOT = ROOT/'docs/evidence/paired-endpoint-track-transfer-r2'
PSEUDO_ROOTS = {
    'generation_proxy': ROOT/'docs/evidence/pseudotarget-recomposition-residual-generation-proxy-r2',
    'consumption_official': ROOT/'docs/evidence/official-pseudotarget-recomposition-residual-r1',
}
PROTOCOL = ROOT/'docs/PAIRED_PSEUDOTARGET_ENDPOINT_TRACK_PROTOCOL_20261101.md'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_proxy(raw: pd.DataFrame) -> pd.Series:
    times = pd.DatetimeIndex(pd.to_datetime(raw.time, utc=True, errors='raise'))
    values = proxy_label(raw)
    keep = times.year == 2024
    return pd.Series(values[keep], index=times[keep], name='label').sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)

    code_paths = [Path(__file__), PROTOCOL, Path(official.__file__), Path(monthly_inputs.__file__),
                  Path(generation_runner.__file__), ROOT/'scripts/run_cross_country.py',
                  ROOT/'src/transcif/research/cross_country.py',
                  ROOT/'src/transcif/research/capacity_transfer.py']
    input_paths = [official.EIA_ROOT/'acquisition_manifest.json',
                   official.EIA_ROOT/'coverage_audit_2022_2024.json',
                   official.EIA_ROOT/'feature_inputs/GB_generation_2022_2024.csv.gz',
                   official.SOURCE_LABEL_ROOT/'GB_consumption_2022_2023_halfhour.csv.gz',
                   official.TARGET_LABEL_ROOT/'GB_consumption_2024_halfhour.csv.gz',
                   ROOT/'data/neso-gb/df_fuel_ckan.csv', official.CAPACITY_TABLE_PATH]
    input_paths += [ROOT/f'results/research/data/{r}_hourly.csv' for r in official.US_REGIONS]
    input_paths += [monthly_inputs.eia_archive(y) for y in range(2020, 2025)]
    input_paths += [official.SOURCE_LABEL_ROOT/f'{r}_consumption_2022_2023.csv.gz'
                    for r in official.US_REGIONS]
    input_paths += [official.TARGET_LABEL_ROOT/f'{r}_consumption_2024.csv.gz'
                    for r in official.US_REGIONS]
    capacity = official.capacity_transfer.load_capacity(official.CAPACITY_TABLE_PATH)
    input_paths += [ROOT/rel for rel in capacity['input_hashes']]
    for track, pseudo_root in PSEUDO_ROOTS.items():
        for arm in ARMS:
            input_paths += [pseudo_root/f'{arm}_US_to_GB_predictions.csv.gz',
                            pseudo_root/f'{arm}_source_calibration.json',
                            pseudo_root/'monthly_metrics.csv',
                            PAIRED_ROOT/f'{arm}_{track}_US_to_GB_predictions.csv.gz']
    input_paths += [PAIRED_ROOT/'startup_hashes.json']
    hash_manifest = {
        'code': {str(p): sha(p) for p in dict.fromkeys(code_paths)},
        'inputs': {str(p): sha(p) for p in dict.fromkeys(input_paths)},
    }
    (out/'startup_hashes.json').write_text(json.dumps(hash_manifest, indent=2))

    raw, base_prepared, _, audit = official.prepare(ROOT, capacity)
    monthly = monthly_inputs.parse_monthly(tuple(range(2020, 2025)), official.US_REGIONS)
    prepared_by_arm, support = monthly_inputs.replace_source_features(base_prepared, monthly)
    (out/'input_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    (out/'common_support.json').write_text(json.dumps(support, indent=2))

    scored, comparison_rows, provenance = [], [], {}
    for track in TRACKS:
        pseudo_root = PSEUDO_ROOTS[track]
        for arm in ARMS:
            pseudo_path = pseudo_root/f'{arm}_US_to_GB_predictions.csv.gz'
            pseudo_pred = pd.read_csv(pseudo_path)
            paired_path = PAIRED_ROOT/f'{arm}_{track}_US_to_GB_predictions.csv.gz'
            paired_pred = pd.read_csv(paired_path)
            if len(pseudo_pred) != len(paired_pred):
                raise ValueError(f'row-count mismatch: {track}/{arm}')
            if not pseudo_pred.time_utc.equals(paired_pred.time_utc):
                raise ValueError(f'timestamp mismatch: {track}/{arm}')
            if not pseudo_pred.observed_month.equals(paired_pred.observed_month):
                raise ValueError(f'observed-month mismatch: {track}/{arm}')
            for method in BASE_METHODS:
                if not np.array_equal(pseudo_pred[method].to_numpy(), paired_pred[method].to_numpy()):
                    raise ValueError(f'paired baseline differs: {track}/{arm}/{method}')

            query = prepared_by_arm[arm]['GB'].loc[
                prepared_by_arm[arm]['GB'].year.eq(2024)].copy().reset_index(drop=True)
            query_times = pd.to_datetime(query.time, utc=True).astype(str).to_numpy()
            if not np.array_equal(query_times, pseudo_pred.time_utc.to_numpy()):
                raise ValueError(f'prepared query differs from sealed timestamps: {track}/{arm}')

            if track == 'generation_proxy':
                label = target_proxy(raw['GB'])
                target_hash = sha(ROOT/'data/neso-gb/df_fuel_ckan.csv')
            else:
                label = official.source_labels('GB', years=(2024,))
                target_hash = sha(official.TARGET_LABEL_ROOT/'GB_consumption_2024_halfhour.csv.gz')
            truth = official.aligned_labels(query, label)
            if len(truth) != len(pseudo_pred) or not np.isfinite(truth).all():
                raise ValueError(f'incomplete target labels: {track}/{arm}')

            calibration_path = pseudo_root/f'{arm}_source_calibration.json'
            calibration = json.loads(calibration_path.read_text())
            methods = [*BASE_METHODS, *PSEUDO_METHODS]
            for month in sorted(query.clock.dt.month.unique()):
                take = query.clock.dt.month.eq(month).to_numpy()
                for method in methods:
                    if method not in calibration:
                        raise ValueError(f'missing H2 calibration for {track}/{arm}/{method}')
                    result = official.score(
                        truth[take], pseudo_pred.loc[take, method].to_numpy(float),
                        calibration[method]['bias'], calibration[method]['sigma'],
                        query.loc[take, 'clock'].reset_index(drop=True))
                    scored.append({'track': track, 'arm': arm, 'method': method,
                                   'month': int(month), **result})

            old_scores = pd.read_csv(pseudo_root/'monthly_metrics.csv')
            old_scores = old_scores.loc[old_scores.arm.eq(arm)].set_index('method')
            new_rows = [r for r in scored if r['track'] == track and r['arm'] == arm]
            new_scores = pd.DataFrame(new_rows).set_index('method')
            max_diff = {}
            for metric in METRICS:
                old_by_method = old_scores.groupby(level=0)[metric].mean()
                new_by_method = new_scores.groupby(level=0)[metric].mean()
                common = old_by_method.index.intersection(new_by_method.index)
                max_diff[metric] = float((old_by_method.loc[common]-new_by_method.loc[common]).abs().max())
            if max(max_diff.values()) >= 1e-10:
                raise ValueError(f'rescoring does not reproduce sealed metrics: {track}/{arm}/{max_diff}')
            comparison_rows.append({'track': track, 'arm': arm,
                                    'target_hours': len(query),
                                    'timestamps_match_paired_replay': True,
                                    'baseline_predictions_match_paired_replay': True,
                                    'max_metric_difference_vs_original': max(max_diff.values()),
                                    'metric_differences': max_diff})
            provenance[f'{track}:{arm}'] = {
                'pseudo_prediction': str(pseudo_path.relative_to(ROOT)),
                'pseudo_prediction_sha256': sha(pseudo_path),
                'pseudo_prediction_seal_sha256': sha(pseudo_root/f'{arm}_prediction_seal.json'),
                'paired_prediction': str(paired_path.relative_to(ROOT)),
                'paired_prediction_sha256': sha(paired_path),
                'calibration': str(calibration_path.relative_to(ROOT)),
                'calibration_sha256': sha(calibration_path),
                'target_label_sha256': target_hash,
                'hours': len(query),
                'H2_calibration_hours': int(calibration['fixed_ef_recomposition']['calibration_hours']),
            }

    monthly_metrics = pd.DataFrame(scored)
    monthly_metrics.to_csv(out/'monthly_metrics.csv', index=False)
    summary = monthly_metrics.groupby(['track','arm','method'], as_index=False)[list(METRICS)].mean()
    summary.to_csv(out/'summary_metrics.csv', index=False)
    base = summary.loc[summary.method.eq('fixed_ef_recomposition'),
                       ['track','arm',*METRICS]].rename(columns={m:f'baseline_{m}' for m in METRICS})
    pseudo_hgb = summary.loc[summary.method.eq('pseudo_residual_residual_hgb'),
                            ['track','arm',*METRICS]].rename(columns={m:f'pseudo_hgb_{m}' for m in METRICS})
    delta = base.merge(pseudo_hgb, on=['track','arm'])
    for metric in METRICS:
        delta[f'delta_{metric}'] = delta[f'pseudo_hgb_{metric}']-delta[f'baseline_{metric}']
    delta.to_csv(out/'pseudo_hgb_vs_recomposition.csv', index=False)
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_json(out/'integrity_comparison.json', orient='records', indent=2)
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2))
    (out/'status.json').write_text(json.dumps({
        'status': 'SEALED_PSEUDOTARGET_ENDPOINT_RESCORING',
        'source_train_year': 2022, 'pseudo_selection': '2023H1 leave-BA-out',
        'source_calibration': '2023H2 leave-BA-out', 'target_year': 2024,
        'target_labels_previously_viewed': True,
        'directions': ['US_to_GB'],
        'tracks': list(TRACKS),
        'target_inputs': 'hourly reconstruction or EIA-923 five-category mix at simulated t-2',
        'labels': {'generation_proxy': 'fixed-factor hourly domestic-generation proxy',
                   'consumption_official': 'official EIA/NESO consumption intensity'},
        'gb_to_us_pseudo_target': 'not identified: GB has one source system, no geographic pseudo-region',
        'reused_prediction_seals': True,
    }, indent=2))
    for group in ('code', 'inputs'):
        for path, digest in hash_manifest[group].items():
            if sha(Path(path)) != digest:
                raise RuntimeError(f'{group} changed during audit: {path}')
    print(summary[['track','arm','method','mae','shape_mae','coverage90','width90','interval_score90']]
          .to_string(index=False))
    print('\nIntegrity comparison:')
    print(comparisons[['track','arm','target_hours','max_metric_difference_vs_original']]
          .to_string(index=False))


if __name__ == '__main__':
    main()
