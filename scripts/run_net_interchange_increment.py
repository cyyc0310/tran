#!/usr/bin/env python3
"""Development replay of lagged net-interchange value for consumption CIF."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_official_intensity_transfer as official
from run_cross_country import ROOT
from transcif.research.cross_country import FEATURES

FLOW_ROOT = ROOT / 'data/neso-interconnector-flows-r1'
FLOW_COLS = tuple(f'{name}_FLOW' for name in (
    'IFA', 'IFA2', 'BRITNED', 'MOYLE', 'EAST_WEST', 'NEMO', 'NSL',
    'ELECLINK', 'VIKING', 'GREENLINK'))
FLOW_FEATURE = 'lag2_net_import_share'
FLOW_MODELS = ('coarse_ridge', 'coarse_boosting')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_hours(period: pd.Period, zone: str) -> int:
    start = period.start_time.tz_localize(zone)
    end = (period + 1).start_time.tz_localize(zone)
    return int((end - start).total_seconds() / 3600)


def us_flow_months(region: str) -> pd.DataFrame:
    path = official.EIA_ROOT / f'{region}.xlsx'
    zone = official.EIA_TIME_ZONES[json.loads(
        (official.EIA_ROOT / 'coverage_audit_2022_2024.json').read_text()
    )[region]['timezone_values'][0]]
    raw = pd.read_excel(path, sheet_name='Published Hourly Data',
                        usecols=['UTC time', 'Demand', 'Total interchange'])
    raw['clock'] = pd.to_datetime(raw['UTC time'], utc=True).dt.tz_convert(zone).dt.tz_localize(None)
    raw['period'] = raw.clock.dt.to_period('M')
    raw = raw.loc[raw.period.dt.year.isin(range(2020, 2025))].copy()
    raw['net_import_mwh'] = -pd.to_numeric(raw['Total interchange'], errors='coerce')
    raw['demand_mwh'] = pd.to_numeric(raw.Demand, errors='coerce')
    raw['complete'] = np.isfinite(raw[['net_import_mwh', 'demand_mwh']]).all(axis=1)
    raw.loc[~raw.complete, ['net_import_mwh', 'demand_mwh']] = np.nan
    grouped = raw.groupby('period').agg(
        net_import_mwh=('net_import_mwh', 'sum'), demand_mwh=('demand_mwh', 'sum'),
        records=('complete', 'sum'),
    )
    grouped['expected_records'] = [expected_hours(p, zone) for p in grouped.index]
    grouped['coverage'] = grouped.records / grouped.expected_records
    grouped[FLOW_FEATURE] = grouped.net_import_mwh / grouped.demand_mwh
    grouped.loc[grouped.coverage < .99, FLOW_FEATURE] = np.nan
    return grouped.reset_index()


def gb_flow_months() -> pd.DataFrame:
    pieces = []
    for year in range(2020, 2025):
        raw = pd.read_csv(FLOW_ROOT / f'demanddata_{year}.csv')
        date = pd.to_datetime(raw.SETTLEMENT_DATE, format='mixed', dayfirst=True)
        raw['clock'] = date + pd.to_timedelta((raw.SETTLEMENT_PERIOD.astype(int) - 1) * 30, unit='min')
        raw['period'] = raw.clock.dt.to_period('M')
        flow_columns = [name for name in FLOW_COLS if name in raw]
        raw['net_import_mw'] = raw[flow_columns].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=len(flow_columns))
        raw['demand_mw'] = pd.to_numeric(raw.TSD, errors='coerce')
        raw['complete'] = np.isfinite(raw[['net_import_mw', 'demand_mw']]).all(axis=1)
        raw.loc[~raw.complete, ['net_import_mw', 'demand_mw']] = np.nan
        pieces.append(raw[['period', 'net_import_mw', 'demand_mw']])
    raw = pd.concat(pieces, ignore_index=True)
    grouped = raw.groupby('period').agg(
        net_import_mwh=('net_import_mw', 'sum'), demand_mwh=('demand_mw', 'sum'),
        records=('net_import_mw', 'count'),
    )
    grouped['expected_records'] = [expected_hours(p, 'Europe/London') * 2 for p in grouped.index]
    grouped['coverage'] = grouped.records / grouped.expected_records
    grouped[FLOW_FEATURE] = grouped.net_import_mwh / grouped.demand_mwh
    grouped.loc[grouped.coverage < .99, FLOW_FEATURE] = np.nan
    return grouped.reset_index()


def attach_lagged_flow(prepared: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    result, audit = {}, {}
    flow_tables = {region: us_flow_months(region) for region in official.US_REGIONS}
    flow_tables['GB'] = gb_flow_months()
    for region, frame in prepared.items():
        flow = flow_tables[region].copy()
        flow['period'] = flow.period + 2
        flow = flow.rename(columns={'period': 'feature_period'})
        local = frame.copy()
        local['feature_period'] = pd.to_datetime(local.clock).dt.to_period('M')
        local = local.merge(flow[['feature_period', FLOW_FEATURE]], on='feature_period', how='left', validate='many_to_one')
        result[region] = local
        audit[region] = flow_tables[region].assign(
            period=lambda x: x.period.astype(str)
        ).to_dict(orient='records')
    return result, audit


def features_for(frame: pd.DataFrame, include_flow: bool) -> list[str]:
    return list(FEATURES) + ([FLOW_FEATURE] if include_flow else [])


def fit_model(frame: pd.DataFrame, y: np.ndarray, name: str, include_flow: bool):
    cols = features_for(frame, include_flow)
    good = np.isfinite(y) & np.isfinite(frame[cols]).all(axis=1).to_numpy()
    train, response = frame.loc[good], y[good]
    weights = official.region_weights(train)
    if name == 'coarse_ridge':
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.))
        model.fit(train[cols], response, standardscaler__sample_weight=weights,
                  ridge__sample_weight=weights * len(weights))
    else:
        model = HistGradientBoostingRegressor(
            max_iter=100, max_leaf_nodes=15, min_samples_leaf=100,
            l2_regularization=10., random_state=412, early_stopping=False,
        )
        model.fit(train[cols], response, sample_weight=weights * len(weights))
    return model, int(good.sum())


def predict(model, frame: pd.DataFrame, name: str, include_flow: bool) -> np.ndarray:
    return np.clip(model.predict(frame[features_for(frame, include_flow)]), 0., 1000.)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)

    capacity = official.capacity_transfer.load_capacity(official.CAPACITY_TABLE_PATH)
    raw, base_prepared, _, input_audit = official.prepare(ROOT, capacity)
    mix = __import__('run_eia923_pseudotarget_transfer').parse_monthly(tuple(range(2020, 2025)), official.US_REGIONS)
    prepared_by_arm, support = __import__('run_eia923_pseudotarget_transfer').replace_source_features(base_prepared, mix)
    flow_prepared, flow_audit = attach_lagged_flow(prepared_by_arm['eia923_monthly'])
    targets = {r: official.source_labels(r, years=(2024,)) for r in (*official.US_REGIONS, 'GB')}
    source = {r: official.source_labels(r, years=(2022, 2023)) for r in (*official.US_REGIONS, 'GB')}

    evidence_inputs = [FLOW_ROOT / f'demanddata_{y}.csv' for y in range(2020, 2025)]
    evidence_inputs += [official.EIA_ROOT / f'{r}.xlsx' for r in official.US_REGIONS]
    evidence_inputs += [ROOT/'data/neso-gb/df_fuel_ckan.csv']
    code_paths = [Path(__file__), Path(official.__file__), ROOT/'src/transcif/research/cross_country.py']
    (out/'hashes.json').write_text(json.dumps({
        'code': {str(p): sha(p) for p in code_paths},
        'inputs': {str(p): sha(p) for p in evidence_inputs},
    }, indent=2))
    (out/'flow_monthly_audit.json').write_text(json.dumps(flow_audit, indent=2, default=str))
    (out/'generation_input_audit.json').write_text(json.dumps(input_audit, indent=2, default=str))

    prediction_rows, calibrations, seals = [], [], []
    for source_country, target_country in (('US', 'GB'), ('GB', 'US')):
        sources = [r for r in flow_prepared if raw[r].country.iloc[0] == source_country]
        targets_regions = [r for r in flow_prepared if raw[r].country.iloc[0] == target_country]
        for method in FLOW_MODELS:
            for arm, include_flow in (('M', False), ('M_plus_X', True)):
                train_frames, train_y = [], []
                for region in sources:
                    f = flow_prepared[region].loc[flow_prepared[region].year.eq(2022)].copy()
                    y = official.aligned_labels(f, source[region])
                    good = np.isfinite(y) & np.isfinite(f[list(FEATURES)].to_numpy(float)).all(axis=1)
                    # Keep identical source rows for M and M+X.
                    good &= np.isfinite(f[FLOW_FEATURE].to_numpy(float))
                    if good.any():
                        train_frames.append(f.loc[good]); train_y.append(y[good])
                train = pd.concat(train_frames, ignore_index=True)
                response = np.concatenate(train_y)
                model, n_train = fit_model(train, response, method, include_flow)

                cal_errors = []
                for region in sources:
                    cal = flow_prepared[region].loc[flow_prepared[region].year.eq(2023)].copy()
                    truth = official.aligned_labels(cal, source[region])
                    ok = np.isfinite(truth) & np.isfinite(cal[list(FEATURES)].to_numpy(float)).all(axis=1)
                    ok &= np.isfinite(cal[FLOW_FEATURE].to_numpy(float))
                    if ok.any():
                        pred = predict(model, cal.loc[ok], method, include_flow)
                        cal_errors.append(truth[ok] - pred)
                bias = float(np.mean([err.mean() for err in cal_errors]))
                sigma = float(np.sqrt(np.mean([np.mean((err - bias) ** 2) for err in cal_errors])))
                sigma = max(sigma, 1.)
                calibrations.append({'source': source_country, 'target': target_country,
                                     'method': method, 'arm': arm, 'bias': bias, 'sigma': sigma,
                                     'calibration_regions': len(cal_errors),
                                     'calibration_hours': int(sum(map(len, cal_errors)))})
                for region in targets_regions:
                    q = flow_prepared[region].loc[flow_prepared[region].year.eq(2024)].copy().reset_index(drop=True)
                    ok = np.isfinite(q[list(FEATURES)].to_numpy(float)).all(axis=1)
                    ok &= np.isfinite(q[FLOW_FEATURE].to_numpy(float))
                    q = q.loc[ok].copy().reset_index(drop=True)
                    pred = predict(model, q, method, include_flow)
                    path = out/f'{source_country}_to_{region}_{method}_{arm}.csv.gz'
                    pd.DataFrame({'time_utc': pd.to_datetime(q.time, utc=True).astype(str),
                                  'prediction': pred, 'bias': bias, 'sigma': sigma}).to_csv(path, index=False, compression='gzip')
                    seals.append({'arm': arm, 'method': method, 'source': source_country,
                                  'target': region, 'rows': len(q), 'sha256': sha(path), 'file': path.name})
                    y = official.aligned_labels(q, targets[region])
                    keep = np.isfinite(y)
                    error = y[keep] - (pred[keep] + bias)
                    radius = 1.6448536269514722 * sigma
                    crps_z = error / sigma
                    crps = sigma * (crps_z * (2 * __import__('scipy').special.ndtr(crps_z) - 1)
                                    + 2 * np.exp(-crps_z**2 / 2) / np.sqrt(2*np.pi) - 1/np.sqrt(np.pi))
                    clock = pd.to_datetime(q.loc[keep, 'clock']).reset_index(drop=True)
                    daily = pd.DataFrame({'day': clock.dt.strftime('%Y-%m-%d'), 'truth': y[keep], 'pred': pred[keep] + bias})
                    centered = daily[['truth','pred']] - daily.groupby('day')[['truth','pred']].transform('mean')
                    prediction_rows.append({'arm': arm, 'method': method, 'source': source_country,
                        'target': target_country, 'region': region, 'scored_hours': int(keep.sum()),
                        'feature_rows': int(len(q)), 'mae': float(np.abs(error).mean()),
                        'shape_mae': float(np.abs(centered.truth-centered.pred).mean()),
                        'crps_gaussian': float(crps.mean()), 'coverage90': float((np.abs(error) <= radius).mean()),
                        'width90': float(2*radius),
                        'interval_score90': float(np.mean(2*radius + 20*np.maximum(np.abs(error)-radius,0))),
                        'flow_missing_feature_hours': int((~np.isfinite(flow_prepared[region].loc[
                            flow_prepared[region].year.eq(2024), FLOW_FEATURE].to_numpy(float))).sum())})

    results = pd.DataFrame(prediction_rows)
    results.to_csv(out/'region_metrics.csv', index=False)
    metrics = ['mae','shape_mae','crps_gaussian','coverage90','width90','interval_score90']
    summary = results.groupby(['source','target','method','arm'], as_index=False)[metrics].mean()
    summary.to_csv(out/'endpoint_metrics.csv', index=False)
    (out/'calibration.json').write_text(json.dumps(calibrations, indent=2))
    (out/'prediction_seal.json').write_text(json.dumps(seals, indent=2))
    (out/'status.json').write_text(json.dumps({
        'status':'DEVELOPMENT_ONLY_TARGET_LABELS_PREVIOUSLY_VIEWED',
        'target_year':2024,'source_train_year':2022,'source_calibration_year':2023,
        'feature':'t-2 monthly net interchange energy / demand energy',
        'M':'lagged CN5 monthly shares and calendar features',
        'M_plus_X':'M plus lagged monthly net import share',
        'fairness':'M and M+X use identical source training rows, source calibration rows, and target scoring hours within each direction and model.',
        'limitations':'US Demand and GB TSD differ in publication scope; final revised data, not true as-issued vintage.'
    }, indent=2))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
