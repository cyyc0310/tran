#!/usr/bin/env python3
"""Paired US↔GB endpoint replay for generation and consumption intensity tracks."""
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
from transcif.research.cross_country import EF


ARMS = ('hourly_reconstruction', 'eia923_monthly')
DIRECTIONS = (('US', 'GB'), ('GB', 'US'))
TRACKS = ('generation_proxy', 'consumption_official')
PROTOCOL = ROOT / 'docs/PAIRED_ENDPOINT_TRACK_TRANSFER_PROTOCOL_20261101.md'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fuel_for(frame: pd.DataFrame, raw: dict[str, pd.DataFrame]) -> np.ndarray:
    return np.concatenate([
        raw[region].iloc[part.raw_row.to_numpy()][list(official.FUELS)].to_numpy(float)
        for region in frame.region.unique()
        for part in [frame.loc[frame.region.eq(region)]]
    ])


def generation_labels(raw: dict[str, pd.DataFrame], years: tuple[int, ...]) -> dict[str, pd.Series]:
    labels = {}
    for region in (*official.US_REGIONS, 'GB'):
        frame = raw[region]
        times = pd.DatetimeIndex(pd.to_datetime(frame.time, utc=True, errors='raise'))
        fuels = frame[list(official.FUELS)].to_numpy(float)
        total = fuels.sum(axis=1)
        keep = times.year.isin(years) & np.isfinite(fuels).all(axis=1) & (total > 0)
        y = fuels[keep] @ EF / total[keep]
        labels[region] = pd.Series(y, index=times[keep], name='label').sort_index()
    return labels


def align(frame: pd.DataFrame, labels: dict[str, pd.Series]) -> tuple[pd.DataFrame, np.ndarray]:
    values = official.aligned_labels(frame, labels[frame.region.iloc[0]])
    keep = np.isfinite(values)
    return frame.loc[keep].copy().reset_index(drop=True), values[keep]


def response_labels(track: str, raw: dict[str, pd.DataFrame], years: tuple[int, ...]) -> dict[str, pd.Series]:
    if track == 'generation_proxy':
        return generation_labels(raw, years)
    return {region: official.source_labels(region, years=years)
            for region in (*official.US_REGIONS, 'GB')}


def train_frame(regions: list[str], prepared: dict[str, pd.DataFrame], labels: dict[str, pd.Series],
                year: int) -> tuple[pd.DataFrame, np.ndarray]:
    frames, values = [], []
    for region in regions:
        query = prepared[region].loc[prepared[region].year.eq(year)].copy()
        query, y = align(query, labels)
        if len(query):
            frames.append(query)
            values.append(y)
    if not frames:
        raise ValueError(f'no labeled source rows for {regions}, {year}')
    return pd.concat(frames, ignore_index=True), np.concatenate(values)


def fit(track: str, regions: list[str], prepared: dict[str, pd.DataFrame], raw: dict[str, pd.DataFrame]):
    labels = response_labels(track, raw, (2022, 2023))
    train, y = train_frame(regions, prepared, labels, 2022)
    response_models = official.train_official(train, y)
    fuel_models = official.fit_models(train.reset_index(drop=True), fuel_for(train, raw))
    return response_models, fuel_models, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)

    capacity = official.capacity_transfer.load_capacity(official.CAPACITY_TABLE_PATH)
    for rel, digest in capacity['input_hashes'].items():
        if sha(ROOT / rel) != digest:
            raise ValueError(f'capacity input changed: {rel}')
    raw, base_prepared, _, audit = official.prepare(ROOT, capacity)
    mix = monthly_inputs.parse_monthly(tuple(range(2020, 2025)), official.US_REGIONS)
    prepared_by_arm, support = monthly_inputs.replace_source_features(base_prepared, mix)
    code_files = [Path(__file__), PROTOCOL, Path(official.__file__), Path(monthly_inputs.__file__),
                  ROOT/'scripts/run_cross_country.py', ROOT/'src/transcif/research/cross_country.py',
                  ROOT/'src/transcif/research/capacity_transfer.py']
    input_files = [official.EIA_ROOT/'acquisition_manifest.json',
                   official.EIA_ROOT/'coverage_audit_2022_2024.json',
                   official.EIA_ROOT/'feature_inputs/GB_generation_2022_2024.csv.gz',
                   official.CAPACITY_TABLE_PATH, ROOT/'data/neso-gb/df_fuel_ckan.csv']
    input_files += [ROOT/f'results/research/data/{r}_hourly.csv' for r in official.US_REGIONS]
    input_files += [monthly_inputs.eia_archive(y) for y in range(2020, 2025)]
    input_files += [official.SOURCE_LABEL_ROOT/f'{r}_consumption_2022_2023.csv.gz'
                    for r in official.US_REGIONS]
    input_files += [official.SOURCE_LABEL_ROOT/'GB_consumption_2022_2023_halfhour.csv.gz']
    hashes = {'code': {str(p): sha(p) for p in dict.fromkeys(code_files)},
              'inputs': {str(p): sha(p) for p in dict.fromkeys(input_files)}}
    (out/'startup_hashes.json').write_text(json.dumps(hashes, indent=2))
    (out/'input_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    (out/'common_support.json').write_text(json.dumps(support, indent=2))
    mix.to_csv(out/'eia923_monthly_ba_mix.csv', index=False)

    # All model predictions are sealed before target responses are opened.
    pending, seals = [], []
    for arm in ARMS:
        prepared = prepared_by_arm[arm]
        for track in TRACKS:
            for source_country, target_country in DIRECTIONS:
                sources = [r for r, frame in prepared.items()
                           if frame.country.iloc[0] == source_country]
                targets = [r for r, frame in prepared.items()
                           if frame.country.iloc[0] == target_country]
                models, fuel_models, source_response = fit(track, sources, prepared, raw)
                residuals = {method: [] for method in official.METHODS}
                for region in sources:
                    cal = prepared[region].loc[prepared[region].year.eq(2023)].copy()
                    cal, truth = align(cal, source_response)
                    pred = official.point_predictions(models, fuel_models, cal)
                    for method in official.METHODS:
                        residuals[method].append(truth - pred[method])
                calibration = official.calibrate(residuals)
                cal_path = out/f'{arm}_{track}_{source_country}_to_{target_country}_calibration.json'
                cal_path.write_text(json.dumps(calibration, indent=2))
                for region in targets:
                    query = prepared[region].loc[prepared[region].year.eq(2024)].copy().reset_index(drop=True)
                    pred = official.point_predictions(models, fuel_models, query)
                    path = out/f'{arm}_{track}_{source_country}_to_{region}_predictions.csv.gz'
                    pd.DataFrame({'time_utc': pd.to_datetime(query.time, utc=True).astype(str).to_numpy(),
                                  'observed_month': query.observed_month.to_numpy(), **pred}).to_csv(
                                      path, index=False, compression='gzip')
                    seals.append({'arm': arm, 'track': track, 'source': source_country,
                                  'target': region, 'sha256': sha(path), 'path': path.name,
                                  'hours_predicted': len(query)})
                    pending.append((arm, track, source_country, target_country, region,
                                    query, path, calibration))
    (out/'prediction_seal.json').write_text(json.dumps(seals, indent=2))
    for group in ('code', 'inputs'):
        for path, digest in hashes[group].items():
            if sha(Path(path)) != digest:
                raise RuntimeError(f'{group} changed during run: {path}')

    records, target_hashes = [], {}
    for arm, track, source_country, target_country, region, query, path, calibration in pending:
        if track == 'generation_proxy':
            labels = generation_labels(raw, (2024,))
        else:
            labels = {region: official.source_labels(region, years=(2024,))}
        target_label = labels[region]
        target_hashes[f'{track}:{region}'] = (
            sha(official.TARGET_LABEL_ROOT/('GB_consumption_2024_halfhour.csv.gz' if region == 'GB'
                else f'{region}_consumption_2024.csv.gz')) if track == 'consumption_official'
            else sha(ROOT/('data/neso-gb/df_fuel_ckan.csv' if region == 'GB'
                else f'results/research/data/{region}_hourly.csv')))
        truth = official.aligned_labels(query, target_label)
        pred = pd.read_csv(path)
        for month in sorted(query.clock.dt.month.unique()):
            take = query.clock.dt.month.eq(month).to_numpy()
            for method in official.METHODS:
                scored = official.score(truth[take], pred.loc[take, method].to_numpy(float),
                                        calibration[method]['bias'], calibration[method]['sigma'],
                                        query.loc[take, 'clock'].reset_index(drop=True))
                records.append({'arm': arm, 'track': track, 'source': source_country,
                                'target': target_country, 'region': region,
                                'method': method, 'month': int(month), **scored})
    frame = pd.DataFrame(records)
    frame.to_csv(out/'monthly_metrics.csv', index=False)
    metrics = ['mae', 'shape_mae', 'crps_gaussian', 'coverage90', 'width90', 'interval_score90']
    by_region = frame.groupby(['arm','track','source','target','region','method'], as_index=False)[metrics].mean()
    by_region.to_csv(out/'region_metrics.csv', index=False)
    by_endpoint = by_region.groupby(['arm','track','source','target','method'], as_index=False)[metrics].mean()
    by_endpoint.to_csv(out/'endpoint_metrics.csv', index=False)
    (out/'target_scoring_hashes.json').write_text(json.dumps(target_hashes, indent=2))
    (out/'status.json').write_text(json.dumps({
        'status': 'DEVELOPMENT_PAIRED_ENDPOINT_TRACK_REPLAY',
        'target_year': 2024, 'source_train_year': 2022, 'source_calibration_year': 2023,
        'target_labels_previously_viewed': True,
        'tracks_scored_separately': list(TRACKS),
        'input_delay_months': 2,
        'eia923_vintage': 'current final archives; lag simulated, not as-issued',
        'M_plus_C': '2020 US coal/gas annual capacity; extra-budget reference, not China-verified',
        'generation_track': 'fixed-factor hourly domestic-generation proxy',
        'consumption_track': 'official EIA/NESO estimated consumption intensity',
    }, indent=2))
    print(by_endpoint.to_string(index=False))


if __name__ == '__main__':
    main()
