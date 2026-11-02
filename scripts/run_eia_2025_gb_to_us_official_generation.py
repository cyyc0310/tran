#!/usr/bin/env python3
"""GB-source to US-target 2025 development holdout on official generation CO2 rates."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BAS = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
FUELS = ('coal', 'gas', 'oil', 'other', 'nuclear', 'hydro', 'wind', 'solar')
COARSE = ('thermal', 'hydro', 'nuclear', 'wind', 'solar')
LB_TO_G = 453.59237


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_parser():
    spec = importlib.util.spec_from_file_location('eia923_audit', ROOT / 'scripts/audit_eia923_ba_mix_alignment.py')
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.parse_eia923


def calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    time = pd.to_datetime(frame.time)
    out = pd.DataFrame(index=frame.index)
    for name, value, period in [('hour', time.dt.hour, 24), ('dow', time.dt.dayofweek, 7), ('month', time.dt.month, 12)]:
        out[f'{name}_sin'] = np.sin(2 * np.pi * value / period)
        out[f'{name}_cos'] = np.cos(2 * np.pi * value / period)
    return out


def add_coarse(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    out['thermal'] = out[['coal', 'gas', 'oil', 'other']].sum(axis=1)
    total = out[list(FUELS)].sum(axis=1)
    for fuel in COARSE:
        out[fuel] = out[fuel].div(total.replace(0, np.nan)) if fuel != 'thermal' else out[fuel].div(total.replace(0, np.nan))
    return out


def source_data() -> tuple[pd.DataFrame, dict]:
    data_dir = ROOT / 'data/neso-gb'
    fuels_path = data_dir / 'df_fuel_ckan.csv'
    selected = ['DATETIME', 'GAS', 'COAL', 'NUCLEAR', 'WIND', 'WIND_EMB', 'HYDRO', 'BIOMASS', 'OTHER', 'SOLAR']
    fuels = pd.read_csv(fuels_path, usecols=selected)
    fuels['time'] = pd.to_datetime(fuels.DATETIME, errors='raise').dt.floor('h')
    fuels['coal'], fuels['gas'], fuels['oil'] = fuels.COAL, fuels.GAS, 0.
    fuels['other'] = fuels.BIOMASS + fuels.OTHER
    fuels['nuclear'], fuels['hydro'] = fuels.NUCLEAR, fuels.HYDRO
    fuels['wind'], fuels['solar'] = fuels.WIND + fuels.WIND_EMB, fuels.SOLAR
    fuels = fuels.loc[pd.to_datetime(fuels.DATETIME).dt.year.between(2021, 2024)]
    hourly_fuels = fuels.groupby('time')[list(FUELS)].agg(['mean', 'count'])
    values = pd.DataFrame({f: hourly_fuels[(f, 'mean')] for f in FUELS})
    values['fuel_half_hours'] = hourly_fuels[(FUELS[0], 'count')]

    ci_path = ROOT / 'docs/evidence/neso-national-intensity-2022-2024-r1/neso_national_intensity_history.csv'
    ci = pd.read_csv(ci_path, parse_dates=['from'])
    ci['time'] = ci['from'].dt.floor('h').dt.tz_localize(None)
    actual = ci.groupby('time').actual_gco2_per_kwh.agg(['mean', 'count']).rename(columns={'mean': 'y', 'count': 'ci_half_hours'})
    # Build monthly source features from the fuel archive (including 2021 months
    # required to form the Jan/Feb 2022 source-side t-2 training inputs).
    fuel_valid = values.fuel_half_hours.eq(2) & values[list(FUELS)].notna().all(axis=1) & (values[list(FUELS)] >= 0).all(axis=1)
    monthly = values.loc[fuel_valid, list(FUELS)].groupby(values.loc[fuel_valid].index.to_period('M')).sum()
    merged = values.join(actual, how='inner').reset_index()
    merged = merged.loc[(merged.time.dt.year >= 2022) & (merged.time.dt.year <= 2024)
                        & merged.fuel_half_hours.eq(2) & merged.ci_half_hours.eq(2)
                        & merged[list(FUELS)].notna().all(axis=1) & (merged[list(FUELS)] >= 0).all(axis=1)
                        & merged.y.notna() & (merged.y >= 0)].copy()
    source = add_coarse(merged)
    # Build only lagged monthly composition features from source fuel observations.
    grouped = pd.DataFrame(index=monthly.index)
    grouped['thermal'] = monthly[['coal', 'gas', 'oil', 'other']].sum(axis=1)
    for f in ('hydro', 'nuclear', 'wind', 'solar'):
        grouped[f] = monthly[f]
    shares = grouped.div(grouped.sum(axis=1), axis=0)
    shares.index = shares.index + 2
    shares.index.name = 'target_period'
    source['period'] = source.time.dt.to_period('M')
    source = source.join(shares, on='period', rsuffix='_lagged')
    # Keep target features under standard column names; hour-level source mix remains separately stored.
    for f in COARSE:
        source[f'm_{f}'] = source[f'{f}_lagged']
    source = source.dropna(subset=[f'm_{f}' for f in COARSE]).copy()
    source['period_year'] = source.time.dt.year
    source['country'] = 'GB'
    return source, {'fuel_source': str(fuels_path.relative_to(ROOT)), 'intensity_source': str(ci_path.relative_to(ROOT)),
                    'valid_hour_rows': len(source), 'years': sorted(source.period_year.unique().tolist()),
                    'hours_with_two_halfhour_fuel_and_intensity_rows': True}


def target_month_features() -> tuple[pd.DataFrame, dict]:
    parse = load_parser()
    archive_paths = [ROOT / f'docs/evidence/eia923-ba-mapping-r1/f923_{year}.zip' for year in (2024, 2025)]
    parts = [parse(path, BAS, year) for year, path in zip((2024, 2025), archive_paths)]
    monthly = pd.concat(parts, ignore_index=True)
    records = []
    for row in monthly.to_dict('records'):
        current = pd.Period(row['month'], freq='M')
        target_period = current + 2
        total = sum(row[f'eia923_{f}_mwh'] for f in FUELS)
        if not np.isfinite(total) or total <= 0:
            continue
        record = {'ba': row['ba'], 'target_period': str(target_period), 'observed_period': row['month']}
        record['thermal'] = sum(row[f'eia923_{f}_mwh'] for f in ('coal', 'gas', 'oil', 'other')) / total
        for f in ('hydro', 'nuclear', 'wind', 'solar'):
            record[f] = row[f'eia923_{f}_mwh'] / total
        records.append(record)
    out = pd.DataFrame(records)
    return out, {'archives': {str(path.relative_to(ROOT)): sha(path) for path in archive_paths},
                 'target_ba_month_rows': int(len(out)), 'statistic_vintage': 'current EIA-923 annual final, including post-target revisions'}


def make_target_hours(month_features: pd.DataFrame) -> pd.DataFrame:
    times = pd.date_range('2025-01-01T00:00:00', '2026-01-01T00:00:00', freq='h', inclusive='left', tz='UTC')
    rows = []
    for ba in BAS:
        local = pd.DataFrame({'time': times})
        local['ba'] = ba
        local['period'] = local.time.dt.tz_localize(None).dt.to_period('M').astype(str)
        feats = month_features.loc[month_features.ba.eq(ba)].drop(columns='ba').rename(columns={'target_period': 'period', 'observed_period': 'observed_month'})
        local = local.merge(feats, on='period', how='left', validate='many_to_one')
        rows.append(local)
    target = pd.concat(rows, ignore_index=True)
    if target[[*COARSE, 'observed_month']].isna().any().any():
        raise ValueError('missing EIA-923 t-2 feature for a target hour')
    for f in COARSE:
        target[f'm_{f}'] = target[f]
    return target


def feature_matrix(frame: pd.DataFrame, *, m_prefix: str = 'm_') -> tuple[pd.DataFrame, tuple[str, ...]]:
    base = calendar_features(frame)
    names = list(base.columns)
    mcols = [f'{m_prefix}{f}' for f in COARSE]
    for f, col in zip(COARSE, mcols):
        base[col] = frame[col].to_numpy()
        names.append(col)
    for f in COARSE:
        for suffix in ('sin', 'cos'):
            col = f'{f}_hour_{suffix}'
            base[col] = frame[f'm_{f}'].to_numpy() * base[f'hour_{suffix}'].to_numpy()
            names.append(col)
    return base, tuple(names)


def fit_predict(source: pd.DataFrame, target: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict, dict]:
    xtrain, mix_columns = feature_matrix(source)
    xcal, _ = feature_matrix(source.loc[source.period_year.eq(2024)])
    xall, _ = feature_matrix(source)
    target_x, _ = feature_matrix(target)
    train_mask = source.period_year.le(2023).to_numpy()
    ytrain = source.loc[train_mask, 'y'].to_numpy(float)
    xfit = xtrain.loc[train_mask]
    ycal = source.loc[source.period_year.eq(2024), 'y'].to_numpy(float)
    models = {}
    cal_predictions = {}
    test_predictions = {}

    models['calendar_ridge'] = make_pipeline(StandardScaler(), Ridge(alpha=10.))
    models['calendar_ridge'].fit(xfit[[c for c in xfit if c in ('hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos')]], ytrain)
    cal_predictions['calendar_ridge'] = models['calendar_ridge'].predict(xcal[['hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos']])
    test_predictions['calendar_ridge'] = models['calendar_ridge'].predict(target_x[['hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos']])

    models['mix_ridge'] = make_pipeline(StandardScaler(), Ridge(alpha=10.))
    models['mix_ridge'].fit(xfit[list(mix_columns)], ytrain)
    cal_predictions['mix_ridge'] = models['mix_ridge'].predict(xcal[list(mix_columns)])
    test_predictions['mix_ridge'] = models['mix_ridge'].predict(target_x[list(mix_columns)])

    models['mix_hgb'] = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, min_samples_leaf=100,
                                                       l2_regularization=10., random_state=412)
    models['mix_hgb'].fit(xfit[list(mix_columns)], ytrain)
    cal_predictions['mix_hgb'] = models['mix_hgb'].predict(xcal[list(mix_columns)])
    test_predictions['mix_hgb'] = models['mix_hgb'].predict(target_x[list(mix_columns)])

    # Simple target-level anchors derived only from target monthly composition.
    target_mix = target[[f'm_{f}' for f in COARSE]]
    test_predictions['anchor700'] = 700. * target_mix.m_thermal.to_numpy()
    train_total = source.loc[train_mask, list(FUELS)].sum(axis=1).to_numpy(float)
    train_thermal = source.loc[train_mask, ['coal','gas','oil','other']].sum(axis=1).to_numpy(float)
    train_y = ytrain
    source_thermal_ef = float(np.sum(train_y * train_total) / np.sum(train_thermal))
    test_predictions['source_thermal_anchor'] = source_thermal_ef * target_mix.m_thermal.to_numpy()

    # Transfer source-learned coarse fuel responses and daily shapes into target monthly budgets.
    train_coarse = source.loc[train_mask, list(COARSE)].to_numpy(float)
    fuel_response = LinearRegression(positive=True, fit_intercept=False).fit(train_coarse, ytrain)
    hour = source.loc[train_mask, 'time'].dt.hour.to_numpy()
    profiles = []
    for index, f in enumerate(COARSE):
        by_hour = pd.Series(train_coarse[:, index]).groupby(hour).mean().reindex(range(24))
        overall = float(np.mean(train_coarse[:, index]))
        profiles.append(by_hour.to_numpy() / overall if overall > 0 else np.ones(24))
    profile = np.asarray(profiles).T
    target_hour = target.time.dt.hour.to_numpy()
    q = target_mix.to_numpy(float) * profile[target_hour]
    q = q / q.sum(axis=1, keepdims=True)
    test_predictions['source_fuel_recomposition'] = fuel_response.predict(q)

    cal_features = xcal.index
    cal_y = ycal
    calibration = {}
    for method, pred in cal_predictions.items():
        residual = cal_y - pred
        bias = float(np.mean(residual))
        abs_centered = np.abs(residual - bias)
        rank = min(len(abs_centered), int(np.ceil((len(abs_centered) + 1) * .90)))
        radius = float(np.partition(abs_centered, rank - 1)[rank - 1])
        calibration[method] = {'bias_g_per_kwh': bias, 'split_conformal_90_radius_g_per_kwh': radius,
                               'calibration_hours': int(len(residual)), 'calibration_year': 2024}
    for method in ('anchor700', 'source_thermal_anchor', 'source_fuel_recomposition'):
        # Calibration uses exact source-year t-2 mixes and the corresponding frozen baseline rule.
        cal = source.loc[source.period_year.eq(2024)].copy()
        mix = cal[[f'm_{f}' for f in COARSE]]
        if method == 'anchor700':
            pred = 700. * mix.m_thermal.to_numpy()
        elif method == 'source_thermal_anchor':
            pred = source_thermal_ef * mix.m_thermal.to_numpy()
        else:
            cmix = cal[[f'm_{f}' for f in COARSE]].to_numpy(float)
            chour = cal.time.dt.hour.to_numpy()
            cq = cmix * profile[chour]
            cq = cq / cq.sum(axis=1, keepdims=True)
            pred = fuel_response.predict(cq)
        residual = cal_y - pred
        bias = float(np.mean(residual))
        abs_centered = np.abs(residual - bias)
        rank = min(len(abs_centered), int(np.ceil((len(abs_centered) + 1) * .90)))
        radius = float(np.partition(abs_centered, rank - 1)[rank - 1])
        calibration[method] = {'bias_g_per_kwh': bias, 'split_conformal_90_radius_g_per_kwh': radius,
                               'calibration_hours': int(len(residual)), 'calibration_year': 2024}
    return test_predictions, calibration, {'source_thermal_ef_g_per_kwh': source_thermal_ef,
                                           'fuel_response_g_per_kwh': dict(zip(COARSE, fuel_response.coef_.tolist())),
                                           'source_hours_train': int(train_mask.sum()),
                                           'source_hours_calibration': int((~train_mask).sum()),
                                           'features': list(mix_columns)}


def score_predictions(target: pd.DataFrame, predictions: pd.DataFrame, labels: pd.DataFrame,
                       calibration: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = labels.copy()
    labels['valid_hour_start_utc'] = pd.to_datetime(labels.local_hour_ending, utc=True) - pd.Timedelta(hours=1)
    labels['y_g_per_kwh'] = labels.co2_generated_intensity_lb_per_kwh * LB_TO_G
    labels = labels.loc[labels.valid_hour_start_utc.dt.year.eq(2025), ['ba','valid_hour_start_utc','y_g_per_kwh']].dropna(subset=['y_g_per_kwh'])
    merged = predictions.merge(labels, left_on=['ba','time'], right_on=['ba','valid_hour_start_utc'], how='inner', validate='one_to_one')
    rows = []
    methods = list(calibration)
    for ba, group in merged.groupby('ba'):
        for method in methods:
            raw = group[method].to_numpy(float)
            cal = calibration[method]
            point = raw + cal['bias_g_per_kwh']
            y = group.y_g_per_kwh.to_numpy(float)
            error = y - point
            radius = cal['split_conformal_90_radius_g_per_kwh']
            width = 2*radius
            score = width + 20*np.maximum(np.abs(error)-radius, 0.)
            day = pd.to_datetime(group.time, utc=True).dt.strftime('%Y-%m-%d').to_numpy()
            tmp = pd.DataFrame({'y':y,'p':point,'day':day})
            centered = tmp[['y','p']] - tmp.groupby('day')[['y','p']].transform('mean')
            rows.append({'ba':ba,'method':method,'hours':len(y),'mae_g_per_kwh':float(np.abs(error).mean()),
                         'bias_g_per_kwh':float(error.mean()),'shape_mae_g_per_kwh':float(np.abs(centered.y-centered.p).mean()),
                         'coverage90':float((np.abs(error)<=radius).mean()),'width90_g_per_kwh':width,
                         'interval_score90':float(score.mean()),'median_target_g_per_kwh':float(np.median(y)),
                         'target_mean_g_per_kwh':float(np.mean(y))})
    detail=pd.DataFrame(rows)
    macro=detail.groupby('method').agg(ba_count=('ba','nunique'),hours=('hours','sum'),mae_ba_macro=('mae_g_per_kwh','mean'),
        shape_mae_ba_macro=('shape_mae_g_per_kwh','mean'),bias_ba_macro=('bias_g_per_kwh','mean'),
        coverage90_ba_macro=('coverage90','mean'),width90_ba_macro=('width90_g_per_kwh','mean'),
        interval_score90_ba_macro=('interval_score90','mean')).reset_index()
    return detail,macro


def main() -> None:
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    src,source_audit=source_data()
    monthly,target_audit=target_month_features()
    target=make_target_hours(monthly)
    predictions,calibration,model_audit=fit_predict(src,target)
    pred_frame=target[['ba','time','period','observed_month']].copy()
    for name,value in predictions.items(): pred_frame[name]=value
    pred_path=out/'sealed_predictions.csv.gz'
    pred_frame.to_csv(pred_path,index=False,compression='gzip')
    # Freeze predictions before opening the target response column for scoring.
    seal={'prediction_sha256':sha(pred_path),'rows':int(len(pred_frame)),'created_before_target_score':True,
          'target_response_file_sha256':sha(ROOT/'docs/evidence/eia930-official-co2-2025-r1/eia930_official_co2_2025.csv')}
    (out/'prediction_seal.json').write_text(json.dumps(seal,indent=2)+'\n')
    label_file=pd.read_csv(ROOT/'docs/evidence/eia930-official-co2-2025-r1/eia930_official_co2_2025.csv')
    detail,macro=score_predictions(target,pred_frame,label_file,calibration)
    detail.to_csv(out/'ba_metrics.csv',index=False)
    macro.to_csv(out/'macro_metrics.csv',index=False)
    (out/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    (out/'model_audit.json').write_text(json.dumps(model_audit,indent=2)+'\n')
    input_files=[ROOT/'docs/CROSS_COUNTRY_2025_OFFICIAL_GENERATION_TRANSFER_PROTOCOL_20261102.md',
                 Path(__file__),ROOT/'data/neso-gb/df_fuel_ckan.csv',
                 ROOT/'docs/evidence/neso-national-intensity-2022-2024-r1/neso_national_intensity_history.csv',
                 ROOT/'docs/evidence/eia930-official-co2-2025-r1/eia930_official_co2_2025.csv']
    provenance={str(p.relative_to(ROOT)):sha(p) for p in input_files}
    provenance.update(target_audit['archives'])
    status={'status':'DEVELOPMENT_TIME_HOLDOUT_OFFICIAL_GENERATION_ESTIMATES',
            'source':'GB NESO national official estimated actual generation carbon intensity',
            'source_train_years':[2022,2023],'source_calibration_year':2024,'target_year':2025,
            'target_bas':list(BAS),'target_response':'EIA-930 CO2.GER current historical snapshot',
            'target_feature_release':'EIA-923 annual final current revisions; t-2 lag simulated retrospectively',
            'label_vintage':'not as-issued; retrieved 2026-11-01',
            'consumption_track_included':False,'new_country_confirmation':False,
            'source_audit':source_audit,'target_feature_audit':target_audit,'model_audit':model_audit,
            'prediction_seal':seal,'provenance_sha256':provenance}
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(macro.to_string(index=False))
    print('\nBA outcomes:')
    print(detail.pivot(index='ba',columns='method',values='mae_g_per_kwh').round(1).to_string())

if __name__=='__main__': main()
