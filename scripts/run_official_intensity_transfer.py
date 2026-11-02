"""US↔GB transfer with official hourly consumption-intensity responses."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from transcif.research.cross_country import (
    CALENDAR, COARSE, FEATURES, FUELS, fit_models, load_inputs, monthly_inputs,
    predict_models,
)
from transcif.research import capacity_transfer
from run_cross_country import ROOT

EIA_ROOT = ROOT / 'data/eia930-hourly-co2-r1'
SOURCE_LABEL_ROOT = EIA_ROOT/'source_labels_2022_2023'
TARGET_LABEL_ROOT = EIA_ROOT/'sealed_target_labels_2024'
PREPARE_LABELS = ROOT/'scripts/prepare_official_intensity_panel.py'
PROTOCOL = ROOT / 'docs/OFFICIAL_CONSUMPTION_INTENSITY_TRANSFER_PROTOCOL_20261101.md'
CAPACITY_PROTOCOL = ROOT/'docs/OFFICIAL_ANNUAL_COAL_GAS_CAPACITY_PROTOCOL_20261101.md'
CAPACITY_TABLE_PATH = ROOT/'data/capacity-public/annual-2021/annual-coal-gas-budget-2020.json'
US_REGIONS = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
EIA_TIME_ZONES = {'Pacific': 'America/Los_Angeles', 'Central': 'America/Chicago',
                  'Eastern': 'America/New_York', 'Eastern Standard': 'Etc/GMT+5'}
METHODS = ('source_mean', 'calendar_ridge', 'coarse_ridge', 'coarse_boosting',
           'fixed_ef_recomposition', 'capacity_ridge', 'capacity_boosting')
CAPACITY_FEATURES = (*FEATURES, 'capacity_coal_share', 'log_capacity_coal_gas_mw')
def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_labels(region: str, years: tuple[int, ...] | None = None) -> pd.Series:
    target = years == (2024,)
    base = TARGET_LABEL_ROOT if target else SOURCE_LABEL_ROOT
    if region == 'GB':
        path=base/('GB_consumption_2024_halfhour.csv.gz' if target else
                   'GB_consumption_2022_2023_halfhour.csv.gz')
        raw = pd.read_csv(path)
        time = pd.to_datetime(raw.time_utc, utc=True, errors='raise')
        values = pd.to_numeric(raw.label_g_kwh, errors='coerce')
        frame = pd.DataFrame({'time': time, 'y': values})
        frame = frame.dropna(subset=['y']).groupby('time', as_index=True).y.mean()
        hourly_groups = frame.groupby(frame.index.floor('h'))
        if not hourly_groups.size().eq(2).all():
            raise ValueError('NESO hourly target requires both half-hour observations')
        hourly = hourly_groups.mean()
        result = hourly.rename('label').sort_index()
        return result.loc[result.index.year.isin(years)] if years else result
    path = base/(f'{region}_consumption_2024.csv.gz' if target else
                 f'{region}_consumption_2022_2023.csv.gz')
    raw = pd.read_csv(path)
    time = pd.to_datetime(raw.time_utc, utc=True, errors='raise')
    intensity = pd.to_numeric(raw.label_g_kwh, errors='coerce')
    result = pd.Series(intensity.to_numpy(float), index=pd.DatetimeIndex(time), name='label')
    if result.index.has_duplicates:
        raise ValueError(f'duplicate EIA intensity timestamps for {region}')
    result = result.sort_index()
    return result.loc[result.index.year.isin(years)] if years else result


def prepare(root: Path, capacity_table: dict) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], dict, dict]:
    raw, audit = load_inputs(root)
    coverage = json.loads((EIA_ROOT/'coverage_audit_2022_2024.json').read_text())
    for region in US_REGIONS:
        source_tz_values = coverage[region]['timezone_values']
        if len(source_tz_values) != 1 or source_tz_values[0] not in EIA_TIME_ZONES:
            raise ValueError(f'unrecognized EIA reporting timezone for {region}: {source_tz_values}')
        zone = EIA_TIME_ZONES[source_tz_values[0]]
        frame = raw[region].copy()
        utc = pd.to_datetime(frame.time, utc=True, errors='raise')
        frame['time'] = utc
        frame['clock'] = utc.dt.tz_convert(zone).dt.tz_localize(None)
        period = frame.clock.dt.to_period('M')
        expected = {}
        for month in period.unique():
            start = month.start_time.tz_localize(zone)
            end = (month + 1).start_time.tz_localize(zone)
            expected[month] = (end - start).total_seconds() / 3600
        frame['expected_hours'] = period.map(expected).to_numpy(float)
        raw[region] = frame.reset_index(drop=True)
        audit[region]['calendar'] = zone
        audit[region]['official_eia_timezone'] = source_tz_values[0]
    gb = raw['GB'].copy()
    # NESO publishes DATETIME in UTC. Convert to a local civil calendar for
    # month shares and calendar predictors, retaining UTC in `time` for joins.
    utc = pd.to_datetime(gb.time, errors='raise').dt.tz_localize('UTC')
    gb['time'] = utc
    gb['clock'] = utc.dt.tz_convert('Europe/London').dt.tz_localize(None)
    period = gb.clock.dt.to_period('M')
    expected = {}
    for month in period.unique():
        start = month.start_time.tz_localize('Europe/London')
        end = (month + 1).start_time.tz_localize('Europe/London')
        expected[month] = (end - start).total_seconds() / 3600
    gb['expected_hours'] = period.map(expected).to_numpy(float)
    raw['GB'] = gb.reset_index(drop=True)

    prepared = {}
    labels = {}
    for region, frame in raw.items():
        features = monthly_inputs(frame, lag=2)
        features['raw_row'] = np.arange(len(frame))
        features = features.dropna(subset=list(COARSE)).copy()
        features = capacity_transfer.add_capacity(features,capacity_table)
        features['log_capacity_coal_gas_mw'] = np.log1p(
            features.capacity_coal_mw + features.capacity_gas_mw)
        prepared[region] = features.reset_index(drop=True)
        # Keep target-year official responses sealed outside this phase.
        labels[region] = source_labels(region, years=(2022, 2023))
    audit['GB']['official_label_timezone'] = 'UTC labels; local calendar Europe/London'
    audit['US_official_consumption_labels'] = 'EIA BA workbook fields; lb/kWh converted to g/kWh'
    return raw, prepared, labels, audit


def aligned_labels(frame: pd.DataFrame, labels: pd.Series) -> np.ndarray:
    times = pd.DatetimeIndex(pd.to_datetime(frame.time, utc=True, errors='raise'))
    return labels.reindex(times).to_numpy(float)


def region_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby('region').region.transform('size').to_numpy(float)
    n_regions = frame.region.nunique()
    return 1. / (counts * n_regions)


def train_official(frame: pd.DataFrame, y: np.ndarray) -> dict:
    good = np.isfinite(y)
    train = frame.loc[good].reset_index(drop=True)
    response = y[good]
    w = region_weights(train)
    models = {'source_mean': float(np.sum(w * response) / np.sum(w))}
    for name, columns in (('calendar_ridge', CALENDAR), ('coarse_ridge', FEATURES)):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.))
        model.fit(train[list(columns)], response, standardscaler__sample_weight=w,
                  ridge__sample_weight=w * len(w))
        models[name] = model
    booster = HistGradientBoostingRegressor(
        max_iter=100, max_leaf_nodes=15, min_samples_leaf=100,
        l2_regularization=10., random_state=412, early_stopping=False,
    )
    booster.fit(train[list(FEATURES)], response, sample_weight=w * len(w))
    models['coarse_boosting'] = booster
    capacity_ridge=make_pipeline(StandardScaler(),Ridge(alpha=10.))
    capacity_ridge.fit(train[list(CAPACITY_FEATURES)],response,
                       standardscaler__sample_weight=w,ridge__sample_weight=w*len(w))
    models['capacity_ridge']=capacity_ridge
    capacity_booster=HistGradientBoostingRegressor(
        max_iter=100,max_leaf_nodes=15,min_samples_leaf=100,
        l2_regularization=10.,random_state=412,early_stopping=False)
    capacity_booster.fit(train[list(CAPACITY_FEATURES)],response,sample_weight=w*len(w))
    models['capacity_boosting']=capacity_booster
    models['train_hours'] = int(len(train))
    models['train_regions'] = int(train.region.nunique())
    return models


def point_predictions(models: dict, proxy: dict, query: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {'source_mean': np.full(len(query), models['source_mean'])}
    for name, cols in (('calendar_ridge', CALENDAR), ('coarse_ridge', FEATURES)):
        out[name] = models[name].predict(query[list(cols)])
    out['coarse_boosting'] = models['coarse_boosting'].predict(query[list(FEATURES)])
    out['capacity_ridge'] = models['capacity_ridge'].predict(query[list(CAPACITY_FEATURES)])
    out['capacity_boosting'] = models['capacity_boosting'].predict(query[list(CAPACITY_FEATURES)])
    out['fixed_ef_recomposition'] = predict_models(proxy, query)['coarse_recomposition']
    return {name: np.clip(value, 0., 1000.) for name, value in out.items()}


def calibrate(residual_parts: dict[str, list[np.ndarray]]) -> dict:
    result = {}
    for method in METHODS:
        parts = residual_parts[method]
        bias = float(np.mean([v.mean() for v in parts]))
        sigma = float(np.sqrt(np.mean([np.mean((v - bias) ** 2) for v in parts])))
        result[method] = {'bias': bias, 'sigma': max(sigma, 1.),
                          'regions': len(parts),
                          'calibration_hours': int(sum(len(v) for v in parts))}
    return result


def score(y: np.ndarray, point_raw: np.ndarray, bias: float, sigma: float,
          clock: pd.Series) -> dict:
    keep = np.isfinite(y)
    y, point = y[keep], point_raw[keep] + bias
    times = pd.to_datetime(clock[keep]).reset_index(drop=True)
    error = y - point
    z = error / sigma
    crps = sigma * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-z*z/2) / np.sqrt(2*np.pi)
                    - 1 / np.sqrt(np.pi))
    radius = 1.6448536269514722 * sigma
    tmp = pd.DataFrame({'day': times.dt.strftime('%Y-%m-%d'), 'y': y, 'p': point})
    centered = tmp[['y', 'p']] - tmp.groupby('day')[['y', 'p']].transform('mean')
    return {'scored_hours': int(len(y)), 'missing_label_hours': int((~keep).sum()),
            'mae': float(np.abs(error).mean()),
            'shape_mae': float(np.abs(centered.y-centered.p).mean()),
            'crps_gaussian': float(crps.mean()),
            'coverage90': float((np.abs(error) <= radius).mean()),
            'width90': float(2*radius),
            'interval_score90': float(np.mean(2*radius + 20*np.maximum(np.abs(error)-radius,0)))}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    preexisting_targets=list(TARGET_LABEL_ROOT.glob('*')) if TARGET_LABEL_ROOT.exists() else []
    if preexisting_targets:
        raise RuntimeError('held-out target-label files must be absent before prediction')
    (out/'preseal_target_label_audit.json').write_text(json.dumps({
        'target_label_directory':str(TARGET_LABEL_ROOT.relative_to(ROOT)),
        'target_label_files_present_before_prediction':0,
        'target_year':2024,
        'source_label_years':[2022,2023],
    },indent=2))
    source_assets=[SOURCE_LABEL_ROOT/f'{r}_consumption_2022_2023.csv.gz' for r in US_REGIONS]
    source_assets += [SOURCE_LABEL_ROOT/'GB_consumption_2022_2023_halfhour.csv.gz',
                      EIA_ROOT/'feature_inputs/GB_generation_2022_2024.csv.gz']
    if not all(p.is_file() for p in source_assets):
        subprocess.run([sys.executable,str(PREPARE_LABELS)],check=True)

    capacity_table=capacity_transfer.load_capacity(CAPACITY_TABLE_PATH)
    for name,expected in capacity_table['input_hashes'].items():
        if sha(ROOT/name)!=expected:raise ValueError(f'annual capacity input changed: {name}')
    code_files = [Path(__file__), PREPARE_LABELS, ROOT/'scripts/run_cross_country.py',
                  ROOT/'src/transcif/research/cross_country.py', PROTOCOL]
    code_files += [ROOT/'src/transcif/research/capacity_transfer.py',CAPACITY_PROTOCOL,
                   ROOT/'scripts/prepare_official_annual_capacity_budget.py']
    data_files = [EIA_ROOT/'acquisition_manifest.json',
                  EIA_ROOT/'coverage_audit_2022_2024.json']
    data_files += [SOURCE_LABEL_ROOT/f'{r}_consumption_2022_2023.csv.gz' for r in US_REGIONS]
    data_files += [SOURCE_LABEL_ROOT/'GB_consumption_2022_2023_halfhour.csv.gz']
    data_files += [ROOT/f'results/research/data/{r}_hourly.csv' for r in US_REGIONS]
    data_files += [EIA_ROOT/'feature_inputs/GB_generation_2022_2024.csv.gz']
    data_files += [CAPACITY_TABLE_PATH]
    data_files += [ROOT/name for name in capacity_table['input_hashes']]
    prepared_raw, prepared, official, audit = prepare(ROOT,capacity_table)
    data_files += [ROOT/a['path'] for k,a in audit.items() if k in US_REGIONS]
    hashes = {'code': {str(p): sha(p) for p in code_files},
              'data': {str(p): sha(p) for p in dict.fromkeys(data_files)}}
    (out/'startup_hashes.json').write_text(json.dumps(hashes, indent=2))
    (out/'input_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    all_seals, calibration_records = [], {}
    source_directions = [('US', 'GB'), ('GB', 'US')]
    directions_data = {}

    # Fit, calibrate, and seal every 2024 target forecast before opening any
    # 2024 official response labels for scoring.
    for source_country, target_country in source_directions:
        sources = [r for r, f in prepared.items() if f.country.iloc[0] == source_country]
        targets = [r for r, f in prepared.items() if f.country.iloc[0] == target_country]
        train_parts, y_parts = [], []
        for region in sources:
            q = prepared[region].loc[prepared[region].year.eq(2022)].copy()
            y = aligned_labels(q, official[region])
            good = np.isfinite(y)
            q = q.loc[good].copy()
            q['official_y'] = y[good]
            train_parts.append(q)
            y_parts.append(y[good])
        train = pd.concat(train_parts, ignore_index=True)
        y_train = np.concatenate(y_parts)
        if not len(train):
            raise ValueError(f'no official 2022 source labels for {source_country}')
        official_models = train_official(train, y_train)

        fuel_parts = []
        for region, q in zip(sources, train_parts):
            raw = prepared_raw[region]
            fuel_parts.append(raw.iloc[q.raw_row.to_numpy()][list(FUELS)].to_numpy(float))
        proxy_fuel = np.concatenate(fuel_parts)
        proxy_models = fit_models(train.reset_index(drop=True), proxy_fuel)

        residual_parts = {m: [] for m in METHODS}
        for region in sources:
            cal = prepared[region].loc[prepared[region].year.eq(2023)].copy()
            y = aligned_labels(cal, official[region])
            good = np.isfinite(y)
            cal = cal.loc[good].copy()
            y = y[good]
            predictions = point_predictions(official_models, proxy_models, cal)
            for method in METHODS:
                residual_parts[method].append(y - predictions[method])
        calibration = calibrate(residual_parts)
        calibration_records[f'{source_country}_to_{target_country}'] = calibration
        (out/f'{source_country}_source_calibration.json').write_text(
            json.dumps(calibration, indent=2))

        directions_data[(source_country, target_country)] = []
        for region in targets:
            query = prepared[region].loc[prepared[region].year.eq(2024)].copy()
            prediction = point_predictions(official_models, proxy_models, query)
            path = out/f'{source_country}_to_{region}_official_predictions.csv.gz'
            pd.DataFrame({'time_utc': pd.to_datetime(query.time, utc=True).astype(str).to_numpy(),
                          'observed_month': query.observed_month.to_numpy(),
                          **prediction}).to_csv(path, index=False, compression='gzip')
            seal = {'source': source_country, 'target': target_country, 'region': region,
                    'path': path.name, 'sha256': sha(path), 'hours_predicted': len(query)}
            all_seals.append(seal)
            directions_data[(source_country, target_country)].append((region, query, path))
            print(source_country, 'to', region, 'sealed', len(query), flush=True)

    (out/'prediction_seal.json').write_text(json.dumps(all_seals, indent=2))
    (out/'source_calibration_all.json').write_text(json.dumps(calibration_records, indent=2))
    if any(sha(Path(path)) != digest for path, digest in hashes['code'].items()):
        raise RuntimeError('code or protocol changed during execution')
    if any(sha(Path(path)) != digest for path, digest in hashes['data'].items()):
        raise RuntimeError('input data changed during execution')

    # Target response labels enter only after all prediction files are sealed.
    subprocess.run([sys.executable,str(PREPARE_LABELS),'--target-labels'],check=True)
    target_files=[TARGET_LABEL_ROOT/f'{r}_consumption_2024.csv.gz' for r in US_REGIONS]
    target_files.append(TARGET_LABEL_ROOT/'GB_consumption_2024_halfhour.csv.gz')
    (out/'target_scoring_hashes.json').write_text(json.dumps(
        {str(p.relative_to(ROOT)):sha(p) for p in target_files},indent=2))
    metric_rows = []
    for direction, records in directions_data.items():
        source_country, target_country = direction
        calibration = calibration_records[f'{source_country}_to_{target_country}']
        for region, query, path in records:
            target_labels = source_labels(region, years=(2024,))
            y = aligned_labels(query, target_labels)
            prediction = pd.read_csv(path)
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                for method in METHODS:
                    scored = score(y[mask], prediction.loc[mask, method].to_numpy(float),
                                   calibration[method]['bias'], calibration[method]['sigma'],
                                   query.loc[mask, 'clock'].reset_index(drop=True))
                    metric_rows.append({'source': source_country, 'target': target_country,
                                        'region': region, 'method': method,
                                        'month': int(month), **scored})
    monthly = pd.DataFrame(metric_rows)
    monthly.to_csv(out/'monthly_metrics.csv', index=False)
    performance = ['mae','shape_mae','crps_gaussian','coverage90','width90','interval_score90']
    group_cols = ['source','target','region','method']
    region_month = monthly.groupby(group_cols)[performance].mean().reset_index()
    counts = monthly.groupby(group_cols)[['scored_hours','missing_label_hours']].sum().reset_index()
    region_month = region_month.merge(counts,on=group_cols)
    region_month.to_csv(out/'region_metrics.csv', index=False)
    country_cols = ['source','target','method']
    country = region_month.groupby(country_cols)[performance].mean().reset_index()
    counts = region_month.groupby(country_cols)[['scored_hours','missing_label_hours']].sum().reset_index()
    country = country.merge(counts,on=country_cols)
    country.to_csv(out/'country_metrics.csv', index=False)
    (out/'status.json').write_text(json.dumps({
        'status':'DEVELOPMENT_OFFICIAL_CONSUMPTION_INTENSITY_TRANSFER',
        'source_train_year':2022,'source_calibration_year':2023,'target_test_year':2024,
        'target_labels_used_before_seal':False,'input_delay_months':2,
        'inputs_from_revised_hourly_derived_months':True,
        'labels':'EIA BA official estimated consumption intensity / NESO revised estimated consumption intensity',
        'target_labels_are_direct_measurements':False,
        'annual_capacity_budget_stat_year':2020,
        'annual_capacity_information_type':'coal_gas_nameplate_capacity_soft_features',
    },indent=2))
    print(country.groupby(['target','method'])[['mae','shape_mae','coverage90','width90',
          'interval_score90']].mean().to_string())


if __name__ == '__main__':
    main()
