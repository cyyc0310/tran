#!/usr/bin/env python3
"""US→GB 2025 official consumption-intensity holdout under lagged CN5 inputs."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
import run_official_intensity_transfer as official
from transcif.research.cross_country import (
    CALENDAR, FEATURES, FUELS, fit_models, monthly_inputs, predict_models,
)

PROTOCOL = ROOT / 'docs/CROSS_COUNTRY_2025_US_TO_GB_OFFICIAL_CONSUMPTION_PROTOCOL_20261102.md'
NESO_PATH = ROOT / 'data/neso-gb/df_fuel_ckan.csv'
METHODS = ('source_mean', 'calendar_ridge', 'coarse_ridge',
           'coarse_boosting', 'fixed_ef_recomposition')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def neso_hourly_generation() -> pd.DataFrame:
    cols = ['DATETIME', 'GAS', 'COAL', 'NUCLEAR', 'WIND', 'WIND_EMB',
            'HYDRO', 'BIOMASS', 'OTHER', 'SOLAR']
    raw = pd.read_csv(NESO_PATH, usecols=cols)
    stamp = pd.to_datetime(raw.DATETIME, errors='raise').dt.tz_localize('UTC')
    if stamp.duplicated().any() or not stamp.dt.minute.isin([0, 30]).all():
        raise ValueError('NESO half-hour timestamps are duplicated or malformed')
    raw['stamp'] = stamp
    raw = raw.loc[raw.stamp.dt.year.between(2022, 2025)].copy()
    raw['coal'], raw['gas'], raw['oil'] = raw.COAL, raw.GAS, 0.
    raw['other'] = raw.BIOMASS + raw.OTHER
    raw['nuclear'], raw['hydro'] = raw.NUCLEAR, raw.HYDRO
    raw['wind'], raw['solar'] = raw.WIND + raw.WIND_EMB, raw.SOLAR
    values = raw[list(FUELS)].to_numpy(float)
    raw = raw.loc[np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)].copy()
    raw['time'] = raw.stamp.dt.floor('h')
    halfhour_count = raw.groupby('time').size()
    hourly = raw.groupby('time')[list(FUELS)].mean()
    hourly = hourly.loc[halfhour_count.eq(2)].reset_index()
    hourly['clock'] = hourly.time.dt.tz_convert('Europe/London').dt.tz_localize(None)
    period = hourly.clock.dt.to_period('M')
    expected = {}
    for month in period.unique():
        start = month.start_time.tz_localize('Europe/London')
        end = (month + 1).start_time.tz_localize('Europe/London')
        expected[month] = (end - start).total_seconds() / 3600
    hourly['expected_hours'] = period.map(expected).to_numpy(float)
    hourly['country'], hourly['region'] = 'GB', 'GB'
    return hourly.reset_index(drop=True)


def neso_consumption_label() -> pd.Series:
    raw = pd.read_csv(NESO_PATH, usecols=['DATETIME', 'CARBON_INTENSITY'])
    time = pd.to_datetime(raw.DATETIME, errors='raise').dt.tz_localize('UTC')
    y = pd.to_numeric(raw.CARBON_INTENSITY, errors='coerce')
    labels = pd.DataFrame({'time': time, 'y': y}).dropna(subset=['y'])
    labels = labels.loc[labels.time.dt.year.eq(2025)]
    grouped = labels.groupby(labels.time.dt.floor('h')).y.agg(['mean', 'count'])
    grouped = grouped.loc[grouped['count'].eq(2), 'mean']
    grouped.index.name = 'time'
    grouped.name = 'label'
    return grouped.sort_index()


def predict_selected(models: dict, proxy_models: dict, query: pd.DataFrame) -> dict:
    """Predict only the frozen M-budget methods; no capacity features enter."""
    out = {'source_mean': np.full(len(query), models['source_mean'])}
    out['calendar_ridge'] = models['calendar_ridge'].predict(query[list(CALENDAR)])
    out['coarse_ridge'] = models['coarse_ridge'].predict(query[list(FEATURES)])
    out['coarse_boosting'] = models['coarse_boosting'].predict(query[list(FEATURES)])
    out['fixed_ef_recomposition'] = predict_models(proxy_models, query)['coarse_recomposition']
    return {name: np.clip(value, 0., 1000.) for name, value in out.items()}


def train_selected(frame: pd.DataFrame, y: np.ndarray) -> dict:
    """Fit only the frozen M-budget consumption-response methods."""
    weights = official.region_weights(frame)
    models = {'source_mean': float(np.sum(weights * y) / np.sum(weights))}
    for name, columns in (('calendar_ridge', CALENDAR), ('coarse_ridge', FEATURES)):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.))
        model.fit(frame[list(columns)], y,
                  standardscaler__sample_weight=weights,
                  ridge__sample_weight=weights * len(weights))
        models[name] = model
    booster = HistGradientBoostingRegressor(
        max_iter=100, max_leaf_nodes=15, min_samples_leaf=100,
        l2_regularization=10., random_state=412, early_stopping=False)
    booster.fit(frame[list(FEATURES)], y, sample_weight=weights * len(weights))
    models['coarse_boosting'] = booster
    return models


def calibrate_selected(residual_parts: dict[str, list[np.ndarray]]) -> dict:
    result = {}
    for method in METHODS:
        parts = residual_parts[method]
        bias = float(np.mean([values.mean() for values in parts]))
        sigma = float(np.sqrt(np.mean([
            np.mean((values - bias) ** 2) for values in parts])))
        result[method] = {'bias': bias, 'sigma': max(sigma, 1.),
                          'regions': len(parts),
                          'calibration_hours': int(sum(map(len, parts)))}
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output if args.output.is_absolute() else ROOT / args.output
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)

    capacity = official.capacity_transfer.load_capacity(official.CAPACITY_TABLE_PATH)
    raw_us, prepared_us, labels_us, source_audit = official.prepare(ROOT, capacity)
    gb_hourly = neso_hourly_generation()
    gb_features = monthly_inputs(gb_hourly, lag=2)
    query = gb_features.loc[
        gb_features.year.eq(2025) & gb_features.clock.dt.month.le(10)
    ].copy().reset_index(drop=True)
    if query.empty or query[list(official.COARSE)].isna().any().any():
        raise ValueError('GB 2025 query lacks legal t-2 coarse-month features')

    code_paths = [Path(__file__), PROTOCOL, Path(official.__file__),
                  ROOT / 'scripts/prepare_official_intensity_panel.py',
                  ROOT / 'scripts/run_cross_country.py',
                  ROOT / 'src/transcif/research/cross_country.py',
                  ROOT / 'src/transcif/research/capacity_transfer.py']
    data_paths = [NESO_PATH, official.EIA_ROOT / 'acquisition_manifest.json',
                  official.EIA_ROOT / 'coverage_audit_2022_2024.json',
                  official.CAPACITY_TABLE_PATH]
    data_paths += [ROOT / f'results/research/data/{r}_hourly.csv' for r in official.US_REGIONS]
    data_paths += [official.SOURCE_LABEL_ROOT / f'{r}_consumption_2022_2023.csv.gz'
                   for r in official.US_REGIONS]
    data_paths += [official.TARGET_LABEL_ROOT / f'{r}_consumption_2024.csv.gz'
                   for r in official.US_REGIONS]
    data_paths += [ROOT / name for name in capacity['input_hashes']]
    hashes = {'code': {str(p.resolve()): sha(p) for p in code_paths},
              'data': {str(p.resolve()): sha(p) for p in dict.fromkeys(data_paths)}}
    (out / 'startup_hashes.json').write_text(json.dumps(hashes, indent=2))
    (out / 'target_feature_audit.json').write_text(json.dumps({
        'target': 'GB', 'target_year': 2025, 'hours_predicted': len(query),
        'score_window_utc': ['2025-01-01', '2025-10-31'],
        'target_features': 'NESO historic generation mix, exact t-2 monthly five-group shares',
        'hourly_target_label_read_before_prediction': False,
        'current_historical_revision_not_as_issued': True,
        'target_features_hash': hashes['data'][str(NESO_PATH.resolve())],
    }, indent=2))

    train_parts, y_parts = [], []
    for region in official.US_REGIONS:
        frame = prepared_us[region]
        train = frame.loc[frame.year.isin([2022, 2023])].copy()
        y = official.aligned_labels(train, labels_us[region])
        good = np.isfinite(y)
        train = train.loc[good].copy()
        y = y[good]
        train['region'] = region
        train_parts.append(train)
        y_parts.append(y)
    train = pd.concat(train_parts, ignore_index=True)
    y_train = np.concatenate(y_parts)
    response_models = train_selected(train, y_train)

    proxy_fuel = np.concatenate([
        raw_us[r].iloc[q.raw_row.to_numpy()][list(FUELS)].to_numpy(float)
        for r, q in zip(official.US_REGIONS, train_parts)
    ])
    proxy_models = fit_models(train.reset_index(drop=True), proxy_fuel)

    residual_parts = {method: [] for method in METHODS}
    calibration_records = []
    for region in official.US_REGIONS:
        cal = prepared_us[region].loc[prepared_us[region].year.eq(2024)].copy()
        y = official.aligned_labels(cal, official.source_labels(region, years=(2024,)))
        good = np.isfinite(y)
        cal, y = cal.loc[good].copy(), y[good]
        prediction = predict_selected(response_models, proxy_models, cal)
        for method in METHODS:
            residual_parts[method].append(y - prediction[method])
        calibration_records.append({'region': region, 'hours': len(y)})
    calibration = calibrate_selected(residual_parts)
    (out / 'calibration.json').write_text(json.dumps(calibration, indent=2))
    (out / 'calibration_support.json').write_text(json.dumps(calibration_records, indent=2))

    point = predict_selected(response_models, proxy_models, query)
    prediction_path = out / 'sealed_predictions.csv.gz'
    prediction_frame = pd.DataFrame({
        'time_utc': pd.to_datetime(query.time, utc=True).astype(str).to_numpy(),
        'observed_month': query.observed_month.to_numpy(),
        **point,
    })
    prediction_frame.to_csv(prediction_path, index=False, compression='gzip')
    prediction_hash = sha(prediction_path)
    (out / 'prediction_seal.json').write_text(json.dumps({
        'sha256': prediction_hash, 'rows': len(prediction_frame),
        'target_response_read_before_seal': False,
    }, indent=2))
    if any(sha(Path(p)) != digest for p, digest in hashes['code'].items()):
        raise RuntimeError('code or protocol changed during run')
    if any(sha(Path(p)) != digest for p, digest in hashes['data'].items()):
        raise RuntimeError('source inputs changed during run')

    # Target intensity is opened only after the forecast artifact is sealed.
    target_labels = neso_consumption_label()
    aligned = target_labels.reindex(pd.DatetimeIndex(pd.to_datetime(query.time, utc=True))).to_numpy(float)
    prediction_read = pd.read_csv(prediction_path)
    monthly_rows = []
    for month in range(1, 11):
        mask = query.clock.dt.month.eq(month).to_numpy()
        for method in METHODS:
            scored = official.score(aligned[mask], prediction_read.loc[mask, method].to_numpy(float),
                                   calibration[method]['bias'], calibration[method]['sigma'],
                                   query.loc[mask, 'clock'].reset_index(drop=True))
            monthly_rows.append({'method': method, 'month': month, **scored})
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(out / 'monthly_metrics.csv', index=False)
    metric_cols = ['mae', 'shape_mae', 'crps_gaussian', 'coverage90', 'width90', 'interval_score90']
    overall = monthly.groupby('method')[metric_cols].mean().reset_index()
    counts = monthly.groupby('method')[['scored_hours', 'missing_label_hours']].sum().reset_index()
    overall = overall.merge(counts, on='method')
    overall.to_csv(out / 'metrics.csv', index=False)
    (out / 'target_scoring_audit.json').write_text(json.dumps({
        'target_response': 'NESO CARBON_INTENSITY consumption-side estimate',
        'target_year': 2025, 'target_hours_requested': len(query),
        'target_hours_scored': int(np.isfinite(aligned).sum()),
        'target_missing_hours': int((~np.isfinite(aligned)).sum()),
        'target_label_file_sha256': hashes['data'][str(NESO_PATH.resolve())],
        'target_label_accessed_after_prediction_seal': True,
        'target_score_window': '2025-01 through 2025-10 UTC',
    }, indent=2))
    (out / 'status.json').write_text(json.dumps({
        'status': 'DEVELOPMENT_TIME_HOLDOUT_OFFICIAL_CONSUMPTION_ESTIMATES',
        'direction': 'US_to_GB', 'source_train_years': [2022, 2023],
        'source_calibration_year': 2024, 'target_year': 2025,
        'target_region': 'GB', 'source_regions': list(official.US_REGIONS),
        'input_lag_months': 2, 'target_information': 'calendar + five coarse monthly generation shares',
        'labels': 'EIA CO2.CER source; NESO CARBON_INTENSITY target; both current estimates, not as-issued',
        'target_label_read_before_prediction': False,
        'target_new_country_confirmation': False,
        'prediction_sha256': prediction_hash,
    }, indent=2))
    print(overall.to_string(index=False))


if __name__ == '__main__':
    main()
