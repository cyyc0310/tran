#!/usr/bin/env python3
"""GB→US 2025 EIA consumption-intensity holdout under lagged CN5 inputs."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
import run_eia_2025_gb_to_us_official_generation as generation
import run_official_intensity_transfer as official

PROTOCOL = ROOT / 'docs/CROSS_COUNTRY_2025_GB_TO_US_OFFICIAL_CONSUMPTION_PROTOCOL_20261102.md'
NESO_PATH = ROOT / 'data/neso-gb/df_fuel_ckan.csv'
TARGET_LABEL_PATH = ROOT / 'docs/evidence/eia930-official-co2-2025-r1/eia930_official_co2_2025.csv'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_consumption_data() -> tuple[pd.DataFrame, dict]:
    """Build GB M-budget source features from fuels; load only 2022–24 CIF labels."""
    selected = ['DATETIME', 'GAS', 'COAL', 'NUCLEAR', 'WIND', 'WIND_EMB',
                'HYDRO', 'BIOMASS', 'OTHER', 'SOLAR']
    raw = pd.read_csv(NESO_PATH, usecols=selected)
    stamp = pd.to_datetime(raw.DATETIME, errors='raise')
    if stamp.duplicated().any() or not stamp.dt.minute.isin([0, 30]).all():
        raise ValueError('NESO source fuel timestamps are duplicated or malformed')
    raw['time'] = stamp.dt.floor('h')
    raw['coal'], raw['gas'], raw['oil'] = raw.COAL, raw.GAS, 0.
    raw['other'] = raw.BIOMASS + raw.OTHER
    raw['nuclear'], raw['hydro'] = raw.NUCLEAR, raw.HYDRO
    raw['wind'], raw['solar'] = raw.WIND + raw.WIND_EMB, raw.SOLAR
    raw = raw.loc[stamp.dt.year.between(2021, 2024)].copy()
    values = raw[list(generation.FUELS)].to_numpy(float)
    valid = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
    raw = raw.loc[valid].copy()
    by_hour = raw.groupby('time')[list(generation.FUELS)].agg(['mean', 'count'])
    hourly = pd.DataFrame({f: by_hour[(f, 'mean')] for f in generation.FUELS})
    complete = by_hour[(generation.FUELS[0], 'count')].eq(2)
    hourly = hourly.loc[complete].copy()
    hourly = hourly.loc[hourly[list(generation.FUELS)].sum(axis=1).gt(0)].copy()

    # Source features use only lagged fuel composition. Compute monthly shares
    # from the same fuel-only data; hourly carbon labels never enter features.
    month = hourly.index.to_period('M')
    totals = hourly.groupby(month)[list(generation.FUELS)].sum()
    coarse = pd.DataFrame(index=totals.index)
    coarse['thermal'] = totals[['coal', 'gas', 'oil', 'other']].sum(axis=1)
    for f in ('hydro', 'nuclear', 'wind', 'solar'):
        coarse[f] = totals[f]
    shares = coarse.div(coarse.sum(axis=1), axis=0)
    shares.index = shares.index + 2
    shares.index.name = 'period'

    source = hourly.reset_index()
    source['time'] = pd.to_datetime(source.time).dt.tz_localize('UTC')
    source['period'] = source.time.dt.tz_localize(None).dt.to_period('M')
    source['period_year'] = source.time.dt.year
    source = source.loc[source.period_year.between(2022, 2024)].copy()
    source = source.join(shares, on='period', rsuffix='_lagged')
    for f in generation.COARSE:
        source[f'm_{f}'] = source[f'{f}_lagged'] if f'{f}_lagged' in source else source[f]
    source = source.dropna(subset=[f'm_{f}' for f in generation.COARSE]).copy()
    source['country'] = 'GB'
    source['region'] = 'GB'

    source_label = official.source_labels('GB', years=(2022, 2023))
    cal_label_path = official.TARGET_LABEL_ROOT / 'GB_consumption_2024_halfhour.csv.gz'
    raw_cal = pd.read_csv(cal_label_path)
    cal_times = pd.to_datetime(raw_cal.time_utc, utc=True, errors='raise')
    cal_values = pd.to_numeric(raw_cal.label_g_kwh, errors='coerce')
    cal_frame = pd.DataFrame({'time': cal_times, 'y': cal_values}).dropna(subset=['y'])
    cal_labels = cal_frame.groupby(cal_frame.time.dt.floor('h')).y.mean()
    labels = pd.concat([source_label, cal_labels.rename('label')]).sort_index()
    aligned = labels.reindex(pd.DatetimeIndex(source.time))
    source['y'] = aligned.to_numpy(float)
    source = source.dropna(subset=['y']).copy()
    audit = {'source_data': str(NESO_PATH.relative_to(ROOT)),
             'source_train_label': str((official.SOURCE_LABEL_ROOT / 'GB_consumption_2022_2023_halfhour.csv.gz').relative_to(ROOT)),
             'source_calibration_label': str(cal_label_path.relative_to(ROOT)),
             'source_rows_after_alignment': len(source),
             'source_years': sorted(source.period_year.unique().tolist()),
             'training_years': [2022, 2023], 'calibration_year': 2024,
             'label': 'NESO CARBON_INTENSITY consumption-side estimate'}
    return source.reset_index(drop=True), audit


def score_consumption(target: pd.DataFrame, predictions: pd.DataFrame,
                      calibration: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(TARGET_LABEL_PATH)
    labels = labels.loc[labels.year.eq(2025)].copy()
    labels['time'] = pd.to_datetime(labels.local_hour_ending, utc=True) - pd.Timedelta(hours=1)
    labels['y'] = pd.to_numeric(labels.co2_consumed_intensity_lb_per_kwh, errors='coerce') * generation.LB_TO_G
    labels = labels.loc[labels.time.dt.month.le(10), ['ba', 'time', 'y']]
    merged = predictions.merge(labels, on=['ba', 'time'], how='inner', validate='one_to_one')
    rows = []
    for ba, group in merged.groupby('ba'):
        for method, cal in calibration.items():
            point = group[method].to_numpy(float) + cal['bias_g_per_kwh']
            y = group.y.to_numpy(float)
            error = y - point
            radius = cal['split_conformal_90_radius_g_per_kwh']
            day = group.time.dt.strftime('%Y-%m-%d').to_numpy()
            paired = pd.DataFrame({'y': y, 'p': point, 'day': day})
            centered = paired[['y', 'p']] - paired.groupby('day')[['y', 'p']].transform('mean')
            rows.append({'ba': ba, 'method': method, 'hours': len(y),
                         'mae_g_per_kwh': float(np.abs(error).mean()),
                         'bias_g_per_kwh': float(error.mean()),
                         'shape_mae_g_per_kwh': float(np.abs(centered.y - centered.p).mean()),
                         'coverage90': float((np.abs(error) <= radius).mean()),
                         'width90_g_per_kwh': 2 * radius,
                         'interval_score90': float(np.mean(2 * radius + 20 * np.maximum(np.abs(error) - radius, 0))),
                         'target_mean_g_per_kwh': float(y.mean())})
    detail = pd.DataFrame(rows)
    macro = detail.groupby('method').agg(
        ba_count=('ba', 'nunique'), hours=('hours', 'sum'),
        mae_ba_macro=('mae_g_per_kwh', 'mean'), shape_mae_ba_macro=('shape_mae_g_per_kwh', 'mean'),
        bias_ba_macro=('bias_g_per_kwh', 'mean'), coverage90_ba_macro=('coverage90', 'mean'),
        width90_ba_macro=('width90_g_per_kwh', 'mean'),
        interval_score90_ba_macro=('interval_score90', 'mean')).reset_index()
    return detail, macro


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output if args.output.is_absolute() else ROOT / args.output
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)

    source, source_audit = source_consumption_data()
    monthly, target_audit = generation.target_month_features()
    target = generation.make_target_hours(monthly)
    target = target.loc[target.time.dt.month.le(10)].copy().reset_index(drop=True)

    data_paths = [NESO_PATH, official.SOURCE_LABEL_ROOT / 'GB_consumption_2022_2023_halfhour.csv.gz',
                  official.TARGET_LABEL_ROOT / 'GB_consumption_2024_halfhour.csv.gz',
                  ROOT / 'docs/evidence/eia923-ba-mapping-r1/f923_2024.zip',
                  ROOT / 'docs/evidence/eia923-ba-mapping-r1/f923_2025.zip',
                  ROOT / 'docs/evidence/eia930-official-co2-2025-r1/acquisition_manifest.json']
    code_paths = [Path(__file__), PROTOCOL, Path(generation.__file__), Path(official.__file__),
                  ROOT / 'scripts/audit_eia923_ba_mix_alignment.py',
                  ROOT / 'scripts/run_cross_country.py',
                  ROOT / 'src/transcif/research/cross_country.py']
    hashes = {'code': {str(p.resolve()): sha(p) for p in code_paths},
              'data': {str(p.resolve()): sha(p) for p in dict.fromkeys(data_paths)}}
    (out / 'startup_hashes.json').write_text(json.dumps(hashes, indent=2))
    (out / 'source_audit.json').write_text(json.dumps(source_audit, indent=2))
    (out / 'target_feature_audit.json').write_text(json.dumps({
        **target_audit, 'target_region_count': 7, 'target_hours_predicted': len(target),
        'target_feature_window': '2025-01 through 2025-10 UTC',
        'target_information': 'EIA-923 current final monthly five-group shares at t-2 plus calendar',
        'target_response_read_before_seal': False,
    }, indent=2))

    prediction_values, calibration, model_audit = generation.fit_predict(source, target)
    predictions = pd.DataFrame({'ba': target.ba.to_numpy(), 'time': pd.to_datetime(target.time, utc=True),
                                'period': target.period.to_numpy(), 'observed_month': target.observed_month.to_numpy()})
    for method, values in prediction_values.items():
        predictions[method] = values
    prediction_path = out / 'sealed_predictions.csv.gz'
    predictions.to_csv(prediction_path, index=False, compression='gzip')
    prediction_hash = sha(prediction_path)
    (out / 'calibration.json').write_text(json.dumps(calibration, indent=2))
    (out / 'model_audit.json').write_text(json.dumps(model_audit, indent=2))
    (out / 'prediction_seal.json').write_text(json.dumps({
        'sha256': prediction_hash, 'rows': len(predictions),
        'target_response_read_before_seal': False,
    }, indent=2))
    if any(sha(Path(p)) != digest for p, digest in hashes['code'].items()):
        raise RuntimeError('code or protocol changed during run')
    if any(sha(Path(p)) != digest for p, digest in hashes['data'].items()):
        raise RuntimeError('inputs changed during run')

    # EIA-930 consumption response is opened only after the forecasts are sealed.
    detail, macro = score_consumption(target, predictions, calibration)
    detail.to_csv(out / 'ba_metrics.csv', index=False)
    macro.to_csv(out / 'macro_metrics.csv', index=False)
    labels = pd.read_csv(TARGET_LABEL_PATH)
    labels = labels.loc[labels.year.eq(2025)].copy()
    labels['time'] = pd.to_datetime(labels.local_hour_ending, utc=True) - pd.Timedelta(hours=1)
    labels['month'] = labels.time.dt.month
    # Keep only a post-seal diagnostic detail, joined with the same target labels.
    pred_long = predictions.melt(id_vars=['ba', 'time', 'period', 'observed_month'],
                                 var_name='method', value_name='point')
    lab = labels.loc[labels.month.le(10), ['ba', 'time', 'month', 'co2_consumed_intensity_lb_per_kwh']].copy()
    lab['y'] = pd.to_numeric(lab.co2_consumed_intensity_lb_per_kwh, errors='coerce') * generation.LB_TO_G
    joined = pred_long.merge(lab[['ba', 'time', 'month', 'y']], on=['ba', 'time'], how='inner', validate='many_to_one')
    month_rows = []
    for (ba, method, month), group in joined.groupby(['ba', 'method', 'month']):
        cal = calibration[method]
        err = group.y.to_numpy(float) - (group.point.to_numpy(float) + cal['bias_g_per_kwh'])
        radius = cal['split_conformal_90_radius_g_per_kwh']
        month_rows.append({'ba': ba, 'method': method, 'month': int(month),
                           'hours': len(group), 'mae_g_per_kwh': float(np.abs(err).mean()),
                           'coverage90': float((np.abs(err) <= radius).mean()),
                           'width90_g_per_kwh': 2 * radius,
                           'interval_score90': float(np.mean(2 * radius + 20 * np.maximum(np.abs(err) - radius, 0)))})
    ba_month = pd.DataFrame(month_rows)
    ba_month.to_csv(out / 'ba_month_metrics.csv', index=False)
    ba_month.groupby(['method', 'month'])[['mae_g_per_kwh', 'coverage90', 'width90_g_per_kwh', 'interval_score90']].mean().reset_index().to_csv(out / 'ba_month_macro_metrics.csv', index=False)
    (out / 'target_scoring_audit.json').write_text(json.dumps({
        'response': 'EIA-930 CO2.CER current historical estimate',
        'target_response_sha256': sha(TARGET_LABEL_PATH),
        'target_labels_read_after_prediction_seal': True,
        'hours_scored': int(detail.hours.sum()), 'ba_count': int(detail.ba.nunique()),
        'window': '2025-01 through 2025-10 UTC',
    }, indent=2))
    (out / 'status.json').write_text(json.dumps({
        'status': 'DEVELOPMENT_TIME_HOLDOUT_OFFICIAL_CONSUMPTION_ESTIMATES',
        'direction': 'GB_to_US', 'source_train_years': [2022, 2023],
        'source_calibration_year': 2024, 'target_year': 2025,
        'target_bas': list(generation.BAS), 'target_response': 'EIA-930 CO2.CER',
        'target_features': 'EIA-923 current final t-2 monthly five-group shares',
        'source_label': 'NESO CARBON_INTENSITY hourly mean; one source country',
        'target_information_budget': 'CN5 lagged monthly mix + calendar',
        'target_labels_read_before_prediction': False,
        'target_country_confirmation': False,
        'prediction_sha256': prediction_hash,
    }, indent=2))
    print(macro.to_string(index=False))


if __name__ == '__main__':
    main()
