"""Country-held-out test of lagged monthly capacity-utilization features."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import run_energy_charts_loco as base
from transcif.research.multicountry import (
    FACTORS, FINE, GROUPS, HISTORY, coarse_values, fit, predict, public_features,
    public_months, weights,
)
from transcif.research.monthly_adaptation import weighted_quantile

ROOT = base.ROOT
CAPACITY_CSV = ROOT/'data/eurostat-capacity-energy-charts-r1/capacity.csv'
CAPACITY_MANIFEST = ROOT/'data/eurostat-capacity-energy-charts-r1/capacity.manifest.json'
PROTOCOL = ROOT/'docs/MONTHLY_CAPACITY_UTILIZATION_TRANSFER_PROTOCOL_20261101.md'
CAPACITY_SHARE_FIELDS = ('thermal', 'hydro', 'nuclear', 'wind', 'solar')
CAPACITY_COLUMNS = {'thermal': 'thermal_mw', 'hydro': 'hydro_mw',
                    'nuclear': 'nuclear_other_mw', 'wind': 'wind_mw', 'solar': 'solar_pv_mw'}
UTILIZATION_FIELDS = (*CAPACITY_SHARE_FIELDS, 'total')
UTILIZATION_COLUMNS = {**CAPACITY_COLUMNS, 'total': 'total_mw'}
UTIL_FEATURES = tuple(f'util_{g}' for g in UTILIZATION_FIELDS)
DELTA_FEATURES = tuple(f'util_delta_{g}' for g in UTILIZATION_FIELDS)
CAP_SHARE_FEATURES = tuple(f'capacity_share_{g}' for g in CAPACITY_SHARE_FIELDS)
MODEL_GROUPS = {
    'capacity_mix': (*HISTORY, *CAP_SHARE_FEATURES),
    'utilization': (*HISTORY, *CAP_SHARE_FEATURES, *UTIL_FEATURES),
    'utilization_delta': (*HISTORY, *CAP_SHARE_FEATURES, *UTIL_FEATURES, *DELTA_FEATURES),
}
MODEL_KINDS = ('direct', 'ridge', 'factor')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_capacity_utilization(features: pd.DataFrame, months: pd.DataFrame,
                             capacity: pd.DataFrame, timezone: str) -> tuple[pd.DataFrame, list[dict]]:
    """Join only t-2/t-3 monthly MWh and forecast-year-minus-two MW values."""
    out = features.copy()
    capacity = capacity.set_index(['country', 'stat_year'])
    audit = []
    country = str(out.country.iloc[0]).upper()
    by_period = {}
    for period in out.period.drop_duplicates():
        cap_year = int(period.year)-2
        key = (country, cap_year)
        if key not in capacity.index:
            raise ValueError(f'missing capacity row for {country} {cap_year}')
        cap = capacity.loc[key]
        latest_month, previous_month = period-2, period-3
        period_mask = out.period.eq(period)
        if latest_month not in months.index or previous_month not in months.index:
            for field in [*CAP_SHARE_FEATURES, *UTIL_FEATURES, *DELTA_FEATURES]:
                out.loc[period_mask, field] = np.nan
            audit.append({'country': country, 'prediction_period': str(period),
                          'lagged_month': str(latest_month), 'previous_month': str(previous_month),
                          'capacity_stat_year': cap_year, 'monthly_input_available': False})
            continue
        latest_hours = base.expected_local_hours(latest_month, timezone)
        previous_hours = base.expected_local_hours(previous_month, timezone)
        current, previous = {}, {}
        for group in CAPACITY_SHARE_FIELDS:
            capacity_mw = float(cap[CAPACITY_COLUMNS[group]])
            if not np.isfinite(capacity_mw) or capacity_mw < 0:
                raise ValueError(f'invalid {group} capacity for {country} {cap_year}')
            current_mwh = float(months.loc[latest_month, group])
            previous_mwh = float(months.loc[previous_month, group])
            if capacity_mw == 0 and (current_mwh > 1e-8 or previous_mwh > 1e-8):
                raise ValueError(f'positive {group} output but zero lagged capacity: {country} {period}')
            current[group] = current_mwh/(capacity_mw*latest_hours) if capacity_mw else 0.
            previous[group] = previous_mwh/(capacity_mw*previous_hours) if capacity_mw else 0.
            out.loc[period_mask, f'capacity_share_{group}'] = (
                capacity_mw/float(cap.total_mw) if float(cap.total_mw)>0 else np.nan)
            out.loc[period_mask, f'util_{group}'] = current[group]
            out.loc[period_mask, f'util_delta_{group}'] = current[group]-previous[group]
        total_capacity = float(cap.total_mw)
        if not np.isfinite(total_capacity) or total_capacity <= 0:
            raise ValueError(f'invalid total capacity for {country} {cap_year}')
        current_total = float(months.loc[latest_month, list(GROUPS)].sum())
        previous_total = float(months.loc[previous_month, list(GROUPS)].sum())
        total_current_cf = current_total/(total_capacity*latest_hours)
        total_previous_cf = previous_total/(total_capacity*previous_hours)
        out.loc[period_mask, 'util_total'] = total_current_cf
        out.loc[period_mask, 'util_delta_total'] = total_current_cf-total_previous_cf
        by_period[period] = current | {'total': total_current_cf}
        audit.append({'country': country, 'prediction_period': str(period),
                      'lagged_month': str(latest_month), 'previous_month': str(previous_month),
                      'capacity_stat_year': cap_year, 'lagged_local_hours': latest_hours,
                      'monthly_input_available': True,
                      **{f'util_{k}': v for k, v in by_period[period].items()}})
    required = [*CAP_SHARE_FEATURES, *UTIL_FEATURES, *DELTA_FEATURES]
    complete = out[required].notna().all(axis=1)
    if not np.isfinite(out.loc[complete, required].to_numpy(float)).all():
        raise ValueError(f'nonfinite monthly utilization features for {country}')
    return out, audit


def fit_utilization_heads(frame: pd.DataFrame, fuel: np.ndarray) -> dict:
    if len(frame) != len(fuel) or not np.isfinite(fuel).all() or (fuel < 0).any():
        raise ValueError('invalid source hourly response rows')
    w = weights(frame)
    thermal = fuel[:, :4].sum(axis=1)
    y = fuel@FACTORS/fuel.sum(axis=1)
    thermal_y = np.divide(fuel@FACTORS, thermal, out=np.full(len(fuel), 400.), where=thermal>0)
    options = dict(max_iter=80, max_leaf_nodes=15, min_samples_leaf=100,
                   l2_regularization=10., random_state=416, early_stopping=False)
    output = {}
    for group, columns in MODEL_GROUPS.items():
        direct = HistGradientBoostingRegressor(**options).fit(
            frame[list(columns)], y, sample_weight=w*len(w))
        scaler = StandardScaler().fit(frame[list(columns)], sample_weight=w)
        ridge = Ridge(alpha=10.).fit(scaler.transform(frame[list(columns)]), y,
                                     sample_weight=w*len(w))
        active = thermal > 0
        factor = HistGradientBoostingRegressor(**options).fit(
            frame.loc[active, list(columns)], thermal_y[active], sample_weight=w[active]*len(w))
        output[group] = {'columns': columns, 'direct': direct,
                         'ridge': (scaler, ridge), 'factor': factor}
    return output


def predict_utilization_heads(heads: dict, public: pd.DataFrame, reference: dict) -> dict:
    q = public[list(GROUPS)].to_numpy()*reference['profile'][public.clock.dt.hour.to_numpy()]
    hourly_thermal = q[:, 0]/q.sum(axis=1)
    out = {}
    for group, models in heads.items():
        columns = list(models['columns'])
        out[f'{group}_direct'] = np.clip(models['direct'].predict(public[columns]), 0., 1000.)
        scaler, ridge = models['ridge']
        out[f'{group}_ridge'] = np.clip(ridge.predict(scaler.transform(public[columns])), 0., 1000.)
        factor = np.clip(models['factor'].predict(public[columns]), 400., 1000.)
        out[f'{group}_factor'] = factor*hourly_thermal
    return out


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((base.DATA/'acquisition_manifest.json').read_text())
    capacity = pd.read_csv(CAPACITY_CSV)
    capacity_manifest = json.loads(CAPACITY_MANIFEST.read_text())
    if capacity_manifest['output_sha256'] != sha(CAPACITY_CSV):
        raise ValueError('Eurostat capacity table does not match its manifest')
    inputs = {str(base.DATA/f'{c}_2022_2024.json'): sha(base.DATA/f'{c}_2022_2024.json')
              for c in base.COUNTRIES}
    inputs[str(base.DATA/'acquisition_manifest.json')] = sha(base.DATA/'acquisition_manifest.json')
    inputs[str(base.DATA/'audit.json')] = sha(base.DATA/'audit.json')
    inputs[str(CAPACITY_CSV)] = sha(CAPACITY_CSV)
    inputs[str(CAPACITY_MANIFEST)] = sha(CAPACITY_MANIFEST)
    code_files = [Path(__file__), Path(base.__file__), ROOT/'src/transcif/research/multicountry.py',
                  ROOT/'src/transcif/research/cross_country.py', ROOT/'src/transcif/research/monthly_adaptation.py',
                  ROOT/'scripts/parse_eurostat_energy_capacity.py', PROTOCOL,
                  ROOT/'docs/ENERGY_CHARTS_LOCO_PROTOCOL_20261101.md',
                  ROOT/'docs/ENERGY_CHARTS_CAPACITY_PROTOCOL_20261101.md']
    code_hashes = {str(p): sha(p) for p in code_files}
    prepared, data_audit, utilization_audit = {}, {}, []
    for code in base.COUNTRIES:
        raw, data_audit[code] = base.load_country(code, manifest)
        months = public_months(raw)
        features = public_features(months, raw.clock)
        features['country'], features['region'] = code.upper(), code.upper()
        features['year'] = raw.clock.dt.year.to_numpy()
        features['raw_row'] = np.arange(len(raw))
        features = base.add_capacity_features(features, capacity)
        features, audit_rows = add_capacity_utilization(features, months, capacity, data_audit[code]['timezone'])
        utilization_audit.extend(audit_rows)
        # Score every method on the same rows; early periods without both
        # lagged monthly reports are outside this experiment's common support.
        common_features = [*HISTORY, *CAP_SHARE_FEATURES, *UTIL_FEATURES, *DELTA_FEATURES]
        prepared[code] = (raw, features.dropna(subset=common_features).copy())
    (out/'input_audit.json').write_text(json.dumps(data_audit, indent=2, ensure_ascii=False))
    pd.DataFrame(utilization_audit).to_csv(out/'utilization_input_audit.csv', index=False)
    (out/'startup_hashes.json').write_text(json.dumps({'code': code_hashes, 'inputs': inputs}, indent=2))

    seals, predictions_by_target = [], {}
    for target in base.COUNTRIES:
        sources = [code for code in base.COUNTRIES if code != target]
        train_parts = [prepared[code][1].loc[prepared[code][1].year.eq(2022)] for code in sources]
        train = pd.concat(train_parts, ignore_index=True)
        fuel = np.concatenate([prepared[code][0].iloc[part.raw_row.to_numpy()][list(FINE)].to_numpy(float)
                               for code, part in zip(sources, train_parts)])
        coarse = np.concatenate([coarse_values(prepared[code][0].iloc[part.raw_row.to_numpy()])
                                 for code, part in zip(sources, train_parts)])
        if train.empty or len(train) != len(fuel) or len(fuel) != len(coarse):
            raise ValueError(f'empty/misaligned training cohort for {target}')
        models = fit(train, fuel, coarse)
        capacity_heads = base.fit_capacity_heads(train, fuel)
        utilization_heads = fit_utilization_heads(train, fuel)

        cal_parts = [prepared[code][1].loc[prepared[code][1].year.eq(2023)
                                           & prepared[code][1].clock.dt.month.le(6)] for code in sources]
        calibration = pd.concat(cal_parts, ignore_index=True)
        cal_fuel = np.concatenate([prepared[code][0].iloc[part.raw_row.to_numpy()][list(FINE)].to_numpy(float)
                                   for code, part in zip(sources, cal_parts)])
        cal_y = cal_fuel@FACTORS/cal_fuel.sum(axis=1)
        cal_weights = weights(calibration)
        cal_predictions = predict(models, calibration)
        cal_predictions.update(base.predict_capacity_heads(capacity_heads, calibration, models))
        cal_predictions.update(predict_utilization_heads(utilization_heads, calibration, models))
        radii = {name: weighted_quantile(np.abs(cal_y-value), cal_weights, .9)
                 for name, value in cal_predictions.items()}

        target_raw, target_all = prepared[target]
        query = target_all.loc[target_all.year.eq(2024)].copy()
        point = predict(models, query)
        point.update(base.predict_capacity_heads(capacity_heads, query, models))
        point.update(predict_utilization_heads(utilization_heads, query, models))
        path = out/f'{target.upper()}_heldout_predictions.csv.gz'
        pd.DataFrame({'time': target_raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                      **point}).to_csv(path, index=False, compression='gzip')
        seal = {'target': target.upper(), 'path': path.name, 'sha256': sha(path), 'hours': len(query)}
        seals.append(seal)
        predictions_by_target[target] = (query, path, radii)
        print(target.upper(), 'prediction sealed', len(query), 'hours', flush=True)
    (out/'prediction_seal.json').write_text(json.dumps(seals, indent=2))
    if any(sha(Path(path)) != digest for path, digest in inputs.items()):
        raise RuntimeError('research input changed during fitting')
    if any(sha(Path(path)) != digest for path, digest in code_hashes.items()):
        raise RuntimeError('code/protocol changed during fitting')

    rows = []
    for target, (query, path, radii) in predictions_by_target.items():
        target_raw = prepared[target][0]
        fuel = target_raw.iloc[query.raw_row.to_numpy()][list(FINE)].to_numpy(float)
        truth = fuel@FACTORS/fuel.sum(axis=1)
        scored = pd.read_csv(path)
        for method, radius in radii.items():
            prediction = scored[method].to_numpy(float)
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                error = truth[mask]-prediction[mask]
                daily = pd.DataFrame({'day': query.loc[mask, 'clock'].dt.strftime('%Y-%m-%d').to_numpy(),
                                      'truth': truth[mask], 'prediction': prediction[mask]})
                centered = daily[['truth', 'prediction']]-daily.groupby('day')[['truth', 'prediction']].transform('mean')
                rows.append({'target_country': target.upper(), 'method': method, 'month': int(month),
                             'hours': int(mask.sum()), 'mae': float(np.abs(error).mean()),
                             'shape_mae': float(np.abs(centered.truth-centered.prediction).mean()),
                             'coverage90': float((np.abs(error)<=radius).mean()), 'width90': float(2*radius),
                             'interval_score90': float(np.mean(2*radius+20*np.maximum(np.abs(error)-radius,0)))})
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out/'monthly_metrics.csv', index=False)
    fields = ['mae', 'shape_mae', 'coverage90', 'width90', 'interval_score90']
    country = monthly.groupby(['target_country', 'method'])[fields].mean().reset_index()
    country.to_csv(out/'country_metrics.csv', index=False)
    macro = country.groupby('method')[fields].mean().reset_index()
    macro.to_csv(out/'macro_metrics.csv', index=False)
    status = {'status': 'DEVELOPMENT_COUNTRY_HELDOUT_MONTHLY_CAPACITY_UTILIZATION',
              'outer_targets': [c.upper() for c in base.COUNTRIES], 'training_year': 2022,
              'source_calibration_period': '2023H1', 'target_test_year': 2024,
              'target_inputs': 't-2 coarse monthly generation; annual category capacities at target-year minus two',
              'monthly_vintage': 'current revised Energy-Charts monthly aggregates; lag simulated, not as-issued',
              'capacity_vintage': 'current revised Eurostat series; lag simulated, not as-issued',
              'new_method_frozen': False, 'unseen_country_confirmation': False}
    (out/'status.json').write_text(json.dumps(status, indent=2))
    print(macro.to_string(index=False))


if __name__ == '__main__':
    main()
