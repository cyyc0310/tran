#!/usr/bin/env python3
"""Paired EIA-923 monthly-input experiment on official consumption labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_eia923_pseudotarget_transfer as pseudo
import run_official_intensity_transfer as official
from run_cross_country import ROOT

PROTOCOL = ROOT / 'docs/EIA923_OFFICIAL_CONSUMPTION_INPUT_BUDGET_PROTOCOL_20261101.md'
CAPACITY_TABLE_PATH = official.CAPACITY_TABLE_PATH
ARMS = ('hourly_reconstruction', 'eia923_monthly')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hashes() -> tuple[list[Path], list[Path]]:
    code = [Path(__file__), Path(pseudo.__file__), Path(official.__file__),
            ROOT / 'scripts/run_cross_country.py',
            ROOT / 'src/transcif/research/cross_country.py',
            ROOT / 'src/transcif/research/capacity_transfer.py', PROTOCOL,
            ROOT / 'scripts/prepare_official_intensity_panel.py']
    data = [official.EIA_ROOT / 'acquisition_manifest.json',
            official.EIA_ROOT / 'coverage_audit_2022_2024.json', CAPACITY_TABLE_PATH]
    data += [official.SOURCE_LABEL_ROOT / f'{r}_consumption_2022_2023.csv.gz'
             for r in official.US_REGIONS]
    data += [official.SOURCE_LABEL_ROOT / 'GB_consumption_2022_2023_halfhour.csv.gz',
             official.EIA_ROOT / 'feature_inputs/GB_generation_2022_2024.csv.gz']
    data += [ROOT / f'results/research/data/{r}_hourly.csv' for r in official.US_REGIONS]
    data += [pseudo.eia_archive(y) for y in range(2020, 2025)]
    capacity = official.capacity_transfer.load_capacity(CAPACITY_TABLE_PATH)
    data += [ROOT / name for name in capacity['input_hashes']]
    return code, list(dict.fromkeys(data))


def official_fuel_for(train: pd.DataFrame, raw: dict[str, pd.DataFrame]) -> np.ndarray:
    parts = []
    for region in train.region.unique():
        q = train.loc[train.region.eq(region) & train.year.eq(2022)]
        parts.append(raw[region].iloc[q.raw_row.to_numpy()][list(official.FUELS)].to_numpy(float))
    return np.concatenate(parts)


def run_direction(arm: str, source_country: str, target_country: str,
                  prepared: dict[str, pd.DataFrame], raw: dict[str, pd.DataFrame],
                  labels: dict[str, pd.Series], out: Path) -> tuple[list[dict], list[tuple]]:
    sources = [r for r, frame in prepared.items() if frame.country.iloc[0] == source_country]
    targets = [r for r, frame in prepared.items() if frame.country.iloc[0] == target_country]
    train_parts, response_parts = [], []
    for region in sources:
        frame = prepared[region].loc[prepared[region].year.eq(2022)].copy()
        response = official.aligned_labels(frame, labels[region])
        valid = np.isfinite(response)
        frame, response = frame.loc[valid].copy(), response[valid]
        train_parts.append(frame)
        response_parts.append(response)
    train = pd.concat(train_parts, ignore_index=True)
    y_train = np.concatenate(response_parts)
    if not len(train):
        raise ValueError(f'no 2022 source responses for {source_country}/{arm}')
    response_models = official.train_official(train, y_train)
    proxy_models = official.fit_models(train.reset_index(drop=True), official_fuel_for(train, raw))

    residuals = {method: [] for method in official.METHODS}
    for region in sources:
        cal = prepared[region].loc[prepared[region].year.eq(2023)].copy()
        response = official.aligned_labels(cal, labels[region])
        valid = np.isfinite(response)
        cal, response = cal.loc[valid].copy(), response[valid]
        predictions = official.point_predictions(response_models, proxy_models, cal)
        for method in official.METHODS:
            residuals[method].append(response - predictions[method])
    calibration = official.calibrate(residuals)
    cal_path = out / f'{arm}_{source_country}_to_{target_country}_calibration.json'
    cal_path.write_text(json.dumps(calibration, indent=2))

    records, pending = [], []
    for region in targets:
        query = prepared[region].loc[prepared[region].year.eq(2024)].copy()
        prediction = official.point_predictions(response_models, proxy_models, query)
        path = out / f'{arm}_{source_country}_to_{region}_predictions.csv.gz'
        pd.DataFrame({
            'time_utc': pd.to_datetime(query.time, utc=True).astype(str).to_numpy(),
            'observed_month': query.observed_month.to_numpy(),
            **prediction,
        }).to_csv(path, index=False, compression='gzip')
        record = {'arm': arm, 'source': source_country, 'target': target_country,
                  'region': region, 'path': path.name, 'sha256': sha(path),
                  'hours_predicted': len(query)}
        records.append(record)
        pending.append((arm, source_country, target_country, region, query, path, calibration))
        print(arm, source_country, 'to', region, 'sealed', len(query), flush=True)
    return records, pending


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)

    capacity = official.capacity_transfer.load_capacity(CAPACITY_TABLE_PATH)
    for name, digest in capacity['input_hashes'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'annual capacity input changed: {name}')
    raw, base_prepared, labels, audit = official.prepare(ROOT, capacity)
    monthly = pseudo.parse_monthly(tuple(range(2020, 2025)), official.US_REGIONS)
    prepared_by_arm, support = pseudo.replace_source_features(base_prepared, monthly)
    code_paths, data_paths = input_hashes()
    hashes = {'code': {str(p): sha(p) for p in code_paths},
              'data': {str(p): sha(p) for p in data_paths}}
    (out / 'startup_hashes.json').write_text(json.dumps(hashes, indent=2))
    (out / 'input_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    monthly.to_csv(out / 'eia923_monthly_ba_mix.csv', index=False)
    (out / 'common_support.json').write_text(json.dumps(support, indent=2))
    (out / 'preseal_target_label_audit.json').write_text(json.dumps({
        'target_year': 2024,
        'target_labels_previously_viewed_in_development': True,
        'target_labels_read_by_this_run_before_prediction_seal': False,
        'assessment': 'development replay, not a blind confirmation',
    }, indent=2))

    seals, pending = [], []
    for arm in ARMS:
        prepared = prepared_by_arm[arm]
        for source, target in (('US', 'GB'), ('GB', 'US')):
            arm_seals, arm_pending = run_direction(
                arm, source, target, prepared, raw, labels, out)
            seals.extend(arm_seals)
            pending.extend(arm_pending)
    (out / 'prediction_seal.json').write_text(json.dumps(seals, indent=2))
    if any(sha(Path(path)) != digest for path, digest in hashes['code'].items()):
        raise RuntimeError('code or protocol changed during execution')
    if any(sha(Path(path)) != digest for path, digest in hashes['data'].items()):
        raise RuntimeError('input data changed during execution')

    # Target responses are joined only after all arm/direction forecasts are sealed.
    target_hashes = {}
    score_records = []
    for arm, source_country, target_country, region, query, path, calibration in pending:
        label = official.source_labels(region, years=(2024,))
        target_hashes[region] = sha(official.TARGET_LABEL_ROOT / (
            'GB_consumption_2024_halfhour.csv.gz' if region == 'GB'
            else f'{region}_consumption_2024.csv.gz'))
        response = official.aligned_labels(query, label)
        prediction = pd.read_csv(path)
        for month in sorted(query.clock.dt.month.unique()):
            mask = query.clock.dt.month.eq(month).to_numpy()
            for method in official.METHODS:
                result = official.score(
                    response[mask], prediction.loc[mask, method].to_numpy(float),
                    calibration[method]['bias'], calibration[method]['sigma'],
                    query.loc[mask, 'clock'].reset_index(drop=True))
                score_records.append({'arm': arm, 'source': source_country,
                                      'target': target_country, 'region': region,
                                      'method': method, 'month': int(month), **result})
    (out / 'target_scoring_hashes.json').write_text(json.dumps(target_hashes, indent=2))
    monthly_metrics = pd.DataFrame(score_records)
    monthly_metrics.to_csv(out / 'monthly_metrics.csv', index=False)
    score_cols = ['mae', 'shape_mae', 'crps_gaussian', 'coverage90', 'width90', 'interval_score90']
    region = monthly_metrics.groupby(
        ['arm', 'source', 'target', 'region', 'method'], as_index=False)[score_cols].mean()
    counts = monthly_metrics.groupby(
        ['arm', 'source', 'target', 'region', 'method'], as_index=False)[
            ['scored_hours', 'missing_label_hours']].sum()
    region = region.merge(counts, on=['arm', 'source', 'target', 'region', 'method'])
    region.to_csv(out / 'region_metrics.csv', index=False)
    country = region.groupby(['arm', 'source', 'target', 'method'], as_index=False)[score_cols].mean()
    country.to_csv(out / 'country_metrics.csv', index=False)
    (out / 'status.json').write_text(json.dumps({
        'status': 'DEVELOPMENT_EIA923_OFFICIAL_CONSUMPTION_INPUT_BUDGET',
        'source_train_year': 2022, 'source_calibration_year': 2023,
        'target_year': 2024, 'input_delay_months': 2,
        'target_label_previously_viewed': True,
        'target_labels_read_by_this_run_before_prediction_seal': False,
        'eia923_vintage': 'current final archives, lag simulated, not as-issued',
        'label_track': 'EIA/NESO estimated consumption intensity',
        'arms': list(ARMS), 'annual_capacity_budget': '2020 coal/gas, separate reference',
    }, indent=2))
    print(country.groupby(['arm', 'target', 'method'])[score_cols].mean().to_string())


if __name__ == '__main__':
    main()
