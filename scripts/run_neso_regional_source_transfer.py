"""Test whether multi-region NESO source labels improve GB-to-US transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_official_intensity_transfer as official  # noqa: E402

PROTOCOL = ROOT / 'docs/NESO_REGIONAL_SOURCE_TRANSFER_PROTOCOL_20261101.md'
REGIONAL_DATA = ROOT / 'data/neso-gb/regional_carbon_intensity_history.csv'
REGIONAL_MANIFEST = ROOT / 'data/neso-gb/regional_carbon_intensity_history.manifest.json'
CAPACITY_TABLE = ROOT / 'data/capacity-public/annual-2021/annual-coal-gas-budget-2020.json'
TARGET_MONTHS = (3, 4, 5, 7, 8, 9, 10, 11, 12)
M_METHODS = ('source_mean', 'calendar_ridge', 'coarse_ridge', 'coarse_boosting')
ALL_METHODS = (*M_METHODS, 'capacity_ridge', 'capacity_boosting', 'fixed_ef_recomposition')
REGION_COLUMNS = (
    'North Scotland', 'South Scotland', 'North West England', 'North East England',
    'Yorkshire', 'North Wales and Merseyside', 'South Wales', 'West Midlands',
    'East Midlands', 'East England', 'South West England', 'South England',
    'London', 'South East England',
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_regional_labels() -> dict[str, pd.Series]:
    source = pd.read_csv(REGIONAL_DATA, parse_dates=['datetime'])
    if tuple(source.columns[1:]) != REGION_COLUMNS:
        raise ValueError('NESO regional label schema changed')
    source['hour_utc'] = pd.to_datetime(source.datetime, utc=True).dt.floor('h')
    labels = {}
    for region in REGION_COLUMNS:
        grouped = source.groupby('hour_utc')[region].agg(['mean', 'count'])
        hourly = grouped.loc[grouped['count'].eq(2), 'mean'].astype(float)
        labels[region] = hourly.rename('label')
    return labels


def align(frame: pd.DataFrame, label: pd.Series) -> np.ndarray:
    time = pd.DatetimeIndex(pd.to_datetime(frame.time, utc=True, errors='raise'))
    return label.reindex(time).to_numpy(float)


def region_frame(base: pd.DataFrame, name: str) -> pd.DataFrame:
    result = base.copy()
    result['region'] = name
    result['country'] = 'GB'
    return result


def prepare_regional_source(base: pd.DataFrame, labels: dict[str, pd.Series],
                            year: int, months: tuple[int, ...] | None = None
                            ) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    pieces, responses, used = [], [], []
    query = base.loc[base.year.eq(year)].copy()
    if months is not None:
        query = query.loc[query.clock.dt.month.isin(months)]
    for region in REGION_COLUMNS:
        frame = region_frame(query, region)
        y = align(frame, labels[region])
        good = np.isfinite(y)
        if good.any():
            pieces.append(frame.loc[good].copy())
            responses.append(y[good])
            used.append(region)
    if not pieces:
        raise ValueError(f'no regional source rows for {year=} {months=}')
    return pd.concat(pieces, ignore_index=True), np.concatenate(responses), used


def national_source(base: pd.DataFrame, labels: dict[str, pd.Series],
                    year: int, months: tuple[int, ...] | None = None
                    ) -> tuple[pd.DataFrame, np.ndarray]:
    frame = base.loc[base.year.eq(year)].copy()
    if months is not None:
        frame = frame.loc[frame.clock.dt.month.isin(months)].copy()
    y = official.aligned_labels(frame, labels['GB'])
    good = np.isfinite(y)
    return frame.loc[good].copy(), y[good]


def fit_proxy(base: pd.DataFrame, raw: pd.DataFrame, year: int) -> dict:
    query = base.loc[base.year.eq(year)].copy()
    fuels = raw.iloc[query.raw_row.to_numpy()][list(official.FUELS)].to_numpy(float)
    return official.fit_models(query.reset_index(drop=True), fuels)


def evaluate_source_variant(base: pd.DataFrame, labels: dict[str, pd.Series],
                            raw: pd.DataFrame, targets: dict[str, pd.DataFrame],
                            regional: bool) -> tuple[list[dict], dict]:
    if regional:
        train, y_train, source_regions = prepare_regional_source(base, labels, 2022)
        calibration_source = lambda: prepare_regional_source(
            base, labels, 2023, tuple(range(7, 13)))
        source_name = 'GB_14_regions'
    else:
        train, y_train = national_source(base, labels, 2022)
        source_regions = ['GB']
        def calibration_source():
            frame, response = national_source(
                base, labels, 2023, tuple(range(7, 13)))
            return frame, response, ['GB']
        source_name = 'GB_national'
    models = official.train_official(train.reset_index(drop=True), y_train)
    proxy = fit_proxy(base, raw, 2022)

    residuals = {method: [] for method in ALL_METHODS}
    cal_frame, cal_y, cal_regions = calibration_source()
    for region in cal_regions:
        part = cal_frame.loc[cal_frame.region.eq(region)].copy()
        if part.empty:
            continue
        truth = cal_y[cal_frame.region.eq(region).to_numpy()]
        point = official.point_predictions(models, proxy, part)
        for method in ALL_METHODS:
            residuals[method].append(truth - point[method])
    calibration = official.calibrate(residuals)

    records, predictions = [], {}
    for region, query in targets.items():
        query = query.loc[query.clock.dt.month.isin(TARGET_MONTHS)].copy()
        point = official.point_predictions(models, proxy, query)
        path = query[['time', 'clock', 'observed_month']].copy()
        for method in ALL_METHODS:
            path[method] = point[method]
        predictions[region] = (query, path)
        for method in ALL_METHODS:
            records.append({'source_variant': source_name, 'source_regions': len(source_regions),
                            'target_region': region, 'method': method,
                            'prediction': path[method].to_numpy(float),
                            'query': query, 'calibration': calibration[method]})
    return records, {'source_variant': source_name, 'source_regions': source_regions,
                     'train_rows': len(train), 'calibration': calibration,
                     'predictions': predictions}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)

    manifest = json.loads((official.EIA_ROOT / 'acquisition_manifest.json').read_text())
    capacity = official.capacity_transfer.load_capacity(CAPACITY_TABLE)
    for path, expected in capacity['input_hashes'].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f'capacity source hash mismatch: {path}')
    raw_all, prepared_all, source_label_series, audit = official.prepare(ROOT, capacity)
    regional_labels = read_regional_labels()
    gb = prepared_all['GB'].dropna(subset=list(official.COARSE)).copy()

    source_labels = {'GB': source_label_series['GB'], **regional_labels}
    target_regions = {}
    for region in official.US_REGIONS:
        frame = prepared_all[region].dropna(subset=list(official.COARSE)).copy()
        target_regions[region] = frame.loc[frame.year.eq(2024)].copy()

    code_files = [Path(__file__), Path(official.__file__), official.PREPARE_LABELS,
                  ROOT / 'scripts/run_cross_country.py',
                  ROOT / 'src/transcif/research/cross_country.py',
                  ROOT / 'src/transcif/research/capacity_transfer.py',
                  official.PROTOCOL, PROTOCOL, official.CAPACITY_PROTOCOL,
                  ROOT / 'scripts/prepare_official_annual_capacity_budget.py']
    data_files = [REGIONAL_DATA, REGIONAL_MANIFEST,
                  official.EIA_ROOT / 'acquisition_manifest.json',
                  official.EIA_ROOT / 'coverage_audit_2022_2024.json',
                  ROOT / 'data/neso-gb/df_fuel_ckan.csv', CAPACITY_TABLE]
    data_files += [ROOT / name for name in capacity['input_hashes']]
    data_files += [ROOT / audit[region]['path'] for region in official.US_REGIONS]
    data_files += [official.SOURCE_LABEL_ROOT / f'{region}_consumption_2022_2023.csv.gz'
                   for region in official.US_REGIONS]
    data_files += [official.SOURCE_LABEL_ROOT / 'GB_consumption_2022_2023_halfhour.csv.gz']
    data_files += [official.EIA_ROOT / 'feature_inputs' / f'{region}_generation_2022_2024.csv.gz'
                   for region in ('GB',)]
    hashes = {'code': {str(path): sha(path) for path in dict.fromkeys(code_files)},
              'data': {str(path): sha(path) for path in dict.fromkeys(data_files)}}
    (output / 'startup_hashes.json').write_text(json.dumps(hashes, indent=2))
    (output / 'regional_source_audit.json').write_text(json.dumps({
        'data': str(REGIONAL_DATA.relative_to(ROOT)),
        'sha256': sha(REGIONAL_DATA),
        'regions': list(REGION_COLUMNS),
        'label_semantics': 'NESO official regional carbon intensity forecast/estimate',
        'generation_mix_history_by_region_available': False,
        'source_feature_contract': 'shared national GB five-category mix, exact t-2 or older',
        'target_months': list(TARGET_MONTHS),
        'target_regions': list(official.US_REGIONS),
        'target_labels_previously_opened': True,
    }, indent=2, ensure_ascii=False))

    targets = {region: frame for region, frame in target_regions.items()}
    all_records, summaries = [], []
    for regional in (False, True):
        records, summary = evaluate_source_variant(
            gb, source_labels, raw_all['GB'], targets, regional)
        summaries.append(summary)
        for record in records:
            # Store each point forecast before loading any target response.
            query = record['query']
            row = {'time_utc': pd.to_datetime(query.time, utc=True).astype(str).to_numpy(),
                   'observed_month': query.observed_month.to_numpy(),
                   'source_variant': record['source_variant'],
                   'target_region': record['target_region'], 'method': record['method'],
                   'prediction': record['prediction'],
                   'calibration_bias': record['calibration']['bias'],
                   'calibration_sigma': record['calibration']['sigma']}
            all_records.append(pd.DataFrame(row))
    sealed = pd.concat(all_records, ignore_index=True)
    seal_path = output / 'sealed_predictions.csv.gz'
    sealed.to_csv(seal_path, index=False, compression='gzip')
    (output / 'prediction_seal.json').write_text(json.dumps({
        'path': seal_path.name, 'sha256': sha(seal_path),
        'rows': len(sealed), 'unique_target_hour_region_pairs': int(
            sealed[['target_region', 'time_utc']].drop_duplicates().shape[0]),
        'target_labels_used_before_seal': False,
        'target_label_dataset_previously_opened_in_project': True,
    }, indent=2))
    if any(sha(Path(path)) != digest for path, digest in hashes['code'].items()):
        raise RuntimeError('code/protocol changed during prediction')
    if any(sha(Path(path)) != digest for path, digest in hashes['data'].items()):
        raise RuntimeError('source input changed during prediction')

    # Score only after the full point-forecast file is sealed and rehashed.
    target_label = {region: official.source_labels(region, years=(2024,))
                    for region in official.US_REGIONS}
    score_rows = []
    for variant in ('GB_national', 'GB_14_regions'):
        for region, query in targets.items():
            y = official.aligned_labels(query, target_label[region])
            records = sealed.loc[(sealed.source_variant.eq(variant)) &
                                 (sealed.target_region.eq(region))]
            for method in ALL_METHODS:
                pred = records.loc[records.method.eq(method)].copy()
                times = pd.DatetimeIndex(pd.to_datetime(pred.time_utc, utc=True))
                query_times = pd.DatetimeIndex(pd.to_datetime(query.time, utc=True))
                location = query_times.get_indexer(times)
                if (location < 0).any():
                    raise ValueError('sealed prediction timestamps do not align to target query')
                point = pred.prediction.to_numpy(float)
                bias = float(pred.calibration_bias.iloc[0])
                sigma = float(pred.calibration_sigma.iloc[0])
                for month in TARGET_MONTHS:
                    mask = query.clock.dt.month.eq(month).to_numpy()[location]
                    if not mask.any():
                        continue
                    metric = official.score(y[location][mask], point[mask], bias, sigma,
                                            query.clock.iloc[location[mask]].reset_index(drop=True))
                    score_rows.append({'source_variant': variant, 'target_region': region,
                                       'method': method, 'month': month, **metric})
    metrics = pd.DataFrame(score_rows)
    metrics.to_csv(output / 'monthly_metrics.csv', index=False)
    columns = ['mae', 'shape_mae', 'crps_gaussian', 'coverage90', 'width90', 'interval_score90']
    region = metrics.groupby(['source_variant', 'target_region', 'method'])[columns].mean().reset_index()
    region.to_csv(output / 'region_metrics.csv', index=False)
    country = region.groupby(['source_variant', 'method'])[columns].mean().reset_index()
    country.to_csv(output / 'country_metrics.csv', index=False)
    (output / 'post_run_verification.json').write_text(json.dumps({
        'source_variants': [summary['source_variant'] for summary in summaries],
        'source_region_counts': {summary['source_variant']: len(summary['source_regions'])
                                 for summary in summaries},
        'target_regions': list(official.US_REGIONS),
        'target_months': list(TARGET_MONTHS),
        'scored_region_month_rows': len(metrics),
        'prediction_seal_match': sha(seal_path) == json.loads(
            (output / 'prediction_seal.json').read_text())['sha256'],
        'input_hash_mismatches': [path for path, digest in hashes['data'].items()
                                  if sha(Path(path)) != digest],
        'code_hash_mismatches': [path for path, digest in hashes['code'].items()
                                 if sha(Path(path)) != digest],
    }, indent=2))
    print(country.to_string(index=False))


if __name__ == '__main__':
    main()
