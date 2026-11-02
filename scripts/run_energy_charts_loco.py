"""Run a country-held-out, delayed-coarse-month transfer benchmark on public-power data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from transcif.research.multicountry import (
    FACTORS, FINE, GROUPS, HISTORY, coarse_values, fit, predict, public_features,
    public_months, weights,
)
from transcif.research.monthly_adaptation import weighted_quantile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/energy-charts-v2-panel-r1'
PROTOCOL = ROOT / 'docs/ENERGY_CHARTS_LOCO_PROTOCOL_20261101.md'
CAPACITY_PROTOCOL = ROOT / 'docs/ENERGY_CHARTS_CAPACITY_PROTOCOL_20261101.md'
COUNTRIES = ('nl', 'es', 'be', 'at', 'se')

# A missing API series means zero for this declared schema mapping. Nulls inside
# a returned series are missing observations and are never converted to zero.
TYPE_MAP = {
    'nl': {
        'coal': ('fossil_hard_coal',), 'gas': ('fossil_gas',), 'oil': (),
        'other': ('biomass', 'waste', 'others'), 'nuclear': ('nuclear',),
        'hydro': (), 'wind': ('wind_onshore', 'wind_offshore'), 'solar': ('solar',),
        'other_renewable': (),
    },
    'es': {
        'coal': ('fossil_hard_coal',), 'gas': ('fossil_gas',), 'oil': ('fossil_oil',),
        'other': ('biomass', 'waste', 'others'), 'nuclear': ('nuclear',),
        'hydro': ('hydro_run_of_river', 'hydro_water_reservoir'),
        'wind': ('wind_onshore',), 'solar': ('solar',),
        'other_renewable': ('other_renewables',),
    },
    'be': {
        'coal': (), 'gas': ('fossil_gas',), 'oil': ('fossil_oil',),
        'other': ('biomass', 'waste', 'others'), 'nuclear': ('nuclear',),
        'hydro': ('hydro_run_of_river',),
        'wind': ('wind_onshore', 'wind_offshore'), 'solar': ('solar',),
        'other_renewable': (),
    },
    'at': {
        'coal': (), 'gas': ('fossil_gas',), 'oil': (),
        'other': ('biomass', 'waste', 'others'), 'nuclear': (),
        'hydro': ('hydro_run_of_river', 'hydro_water_reservoir'),
        'wind': ('wind_onshore',), 'solar': ('solar',),
        'other_renewable': ('geothermal',),
    },
    'se': {
        'coal': (), 'gas': ('fossil_gas',), 'oil': (), 'other': ('others',),
        'nuclear': ('nuclear',), 'hydro': ('hydro_water_reservoir',),
        'wind': ('wind_onshore',), 'solar': ('solar',),
        'other_renewable': (),
    },
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_local_hours(period: pd.Period, timezone: str) -> int:
    zone = ZoneInfo(timezone)
    start = pd.Timestamp(period.start_time).tz_localize(zone)
    end = pd.Timestamp((period + 1).start_time).tz_localize(zone)
    return len(pd.date_range(start, end, freq='h', inclusive='left'))


def load_country(code: str, manifest: dict) -> tuple[pd.DataFrame, dict]:
    path = DATA / f'{code}_2022_2024.json'
    expected_sha = next(x['sha256'] for x in manifest['countries'] if x['country'] == code)
    if sha(path) != expected_sha:
        raise ValueError(f'raw Energy-Charts SHA mismatch for {code}')
    payload = json.loads(path.read_text())
    timezone = payload['timezone']
    rows = [x for x in payload['data'] if x['timestamp'][:4] in {'2022', '2023', '2024'}]
    if not rows:
        raise ValueError(f'no 2022–2024 response data for {code}')
    stamps = pd.DatetimeIndex([pd.Timestamp(x['timestamp']).tz_convert('UTC') for x in rows])
    if stamps.has_duplicates or not stamps.is_monotonic_increasing:
        raise ValueError(f'duplicate or unordered source timestamps for {code}')
    values = pd.DataFrame([x['values'] for x in rows])
    mapping = TYPE_MAP[code]
    present = set(values.columns)
    missing_schema = sorted({k for group in mapping.values() for k in group} - present)
    if missing_schema:
        raise ValueError(f'expected mapped production types absent for {code}: {missing_schema}')
    fine = pd.DataFrame(index=values.index)
    null_audit = {}
    for fuel in FINE:
        keys = mapping[fuel]
        if not keys:
            fine[fuel] = 0.0
            continue
        block = values[list(keys)].apply(pd.to_numeric, errors='coerce')
        null_audit[fuel] = int(block.isna().sum().sum())
        fine[fuel] = block.sum(axis=1, min_count=len(keys))
    # Keep missing category observations missing through resampling. In
    # particular, do not let a 15-minute hourly mean silently use only 1–3
    # valid quarter-hours.
    fine['__complete'] = fine[list(FINE)].notna().all(axis=1)
    # API resolution can change within a country history (Spain). Infer and
    # validate the interval separately by local calendar year.
    table = pd.DataFrame({'timestamp': stamps, 'fine_row': np.arange(len(rows))})
    local_year = pd.Series([pd.Timestamp(x['timestamp']).year for x in rows], index=table.index)
    table['local_year'] = local_year
    hourly_parts = []
    resolution_audit = {}
    for year, positions in table.groupby('local_year', sort=True).groups.items():
        idx = np.asarray(list(positions), dtype=int)
        utc = stamps[idx].as_unit('ns')
        diffs = np.diff(utc.asi8) / 1e9
        if not len(diffs):
            raise ValueError(f'cannot infer interval for {code} {year}')
        unique, counts = np.unique(diffs, return_counts=True)
        interval_s = float(unique[np.argmax(counts)])
        if interval_s not in (900., 3600.):
            raise ValueError(f'unsupported {interval_s}s interval in {code} {year}')
        expected_steps = int(3600 / interval_s)
        resolution_audit[str(year)] = {
            'interval_seconds': int(interval_s),
            'dominant_step_fraction': float(counts.max() / len(diffs)),
            'source_rows': int(len(idx)),
        }
        hour_utc = utc.floor('h')
        values_year = fine.iloc[idx].copy()
        values_year['hour_utc'] = hour_utc
        interval_counts = values_year.groupby('hour_utc')['__complete'].sum()
        valid_hours = interval_counts.index[interval_counts.eq(expected_steps)]
        complete_rows = values_year['__complete'] & values_year.hour_utc.isin(valid_hours)
        grouped = values_year.loc[complete_rows].groupby('hour_utc')[list(FINE)].mean()
        if ((grouped[list(FINE)] < 0).any().any()
                or not np.isfinite(grouped[list(FINE)].to_numpy(float)).all()):
            raise ValueError(f'negative or nonfinite mapped generation for {code} {year}')
        grouped = grouped.loc[grouped[list(FINE)].sum(axis=1).gt(0)].copy()
        if grouped.empty:
            raise ValueError(f'no positive-generation hours for {code} {year}')
        grouped['time'] = grouped.index
        grouped['clock'] = grouped.index.tz_convert(timezone).tz_localize(None)
        grouped['country'], grouped['region'] = code.upper(), code.upper()
        period = grouped.clock.dt.to_period('M')
        grouped['expected_hours'] = [expected_local_hours(p, timezone) for p in period]
        hourly_parts.append(grouped.reset_index(drop=True))
    frame = pd.concat(hourly_parts, ignore_index=True)
    audit = {
        'raw_sha256': expected_sha, 'timezone': timezone, 'unit': payload.get('unit'),
        'reported_resolution': payload.get('resolution'),
        'observed_resolution_by_year': resolution_audit,
        'null_cells_by_mapped_fuel': null_audit,
        'schema_absent_categories_assumed_zero': [k for k, v in mapping.items() if not v],
        'excluded_source_types': sorted(present - {k for v in mapping.values() for k in v}),
        'hourly_rows_after_complete_mapping': int(len(frame)),
    }
    return frame, audit


def add_capacity_features(features: pd.DataFrame, capacities: pd.DataFrame) -> pd.DataFrame:
    """Join broad annual thermal capacity with a conservative two-year lag."""
    out = features.copy()
    lookup = capacities.set_index(['country', 'stat_year'])['thermal_capacity_share']
    key = pd.MultiIndex.from_arrays(
        [out.country, out.year.astype(int) - 2], names=['country', 'stat_year'])
    out['capacity_thermal_share'] = lookup.reindex(key).to_numpy()
    if out.capacity_thermal_share.isna().any():
        missing = out.loc[out.capacity_thermal_share.isna(), ['country', 'year']].drop_duplicates()
        raise ValueError(f'missing two-year-lagged capacity rows: {missing.to_dict("records")}')
    return out


def fit_capacity_heads(frame: pd.DataFrame, fuel: np.ndarray) -> dict:
    """Same HGB capacity, with the annual capacity budget added to HISTORY."""
    columns = [*HISTORY, 'capacity_thermal_share']
    w = weights(frame)
    thermal = fuel[:, :4].sum(axis=1)
    y = fuel @ FACTORS / fuel.sum(axis=1)
    thermal_y = np.divide(fuel @ FACTORS, thermal, out=np.full(len(fuel), 400.), where=thermal > 0)
    options = dict(max_iter=80, max_leaf_nodes=15, min_samples_leaf=100,
                   l2_regularization=10., random_state=416, early_stopping=False)
    direct = HistGradientBoostingRegressor(**options).fit(
        frame[columns], y, sample_weight=w * len(w))
    scaler = StandardScaler().fit(frame[columns], sample_weight=w)
    ridge = Ridge(alpha=10.).fit(
        scaler.transform(frame[columns]), y, sample_weight=w * len(w))
    active = thermal > 0
    factor = HistGradientBoostingRegressor(**options).fit(
        frame.loc[active, columns], thermal_y[active], sample_weight=w[active] * len(w))
    return {'columns': columns, 'direct': direct, 'ridge': (scaler, ridge), 'factor': factor}


def predict_capacity_heads(models: dict, public: pd.DataFrame, reference: dict) -> dict:
    columns = models['columns']
    factor = np.clip(models['factor'].predict(public[columns]), 400., 1000.)
    scaler, ridge = models['ridge']
    q = public[list(GROUPS)].to_numpy() * reference['profile'][public.clock.dt.hour.to_numpy()]
    hourly_thermal = q[:, 0] / q.sum(axis=1)
    return {
        'capacity_direct': np.clip(models['direct'].predict(public[columns]), 0., 1000.),
        'capacity_ridge': np.clip(ridge.predict(scaler.transform(public[columns])), 0., 1000.),
        'capacity_factor': factor * hourly_thermal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capacity-csv', type=Path,
                        help='Eurostat nrg_inf_epc extract; enables the separate annual-capacity budget')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((DATA / 'acquisition_manifest.json').read_text())
    audit_file = DATA / 'audit.json'
    if not audit_file.is_file():
        raise FileNotFoundError(audit_file)
    inputs = {str(DATA / f'{c}_2022_2024.json'): sha(DATA / f'{c}_2022_2024.json') for c in COUNTRIES}
    inputs[str(DATA / 'acquisition_manifest.json')] = sha(DATA / 'acquisition_manifest.json')
    inputs[str(audit_file)] = sha(audit_file)
    capacities = None
    if args.capacity_csv:
        capacities = pd.read_csv(args.capacity_csv)
        required = {'country', 'stat_year', 'thermal_capacity_share'}
        if not required.issubset(capacities.columns):
            raise ValueError(f'capacity file needs columns {sorted(required)}')
        inputs[str(args.capacity_csv.resolve())] = sha(args.capacity_csv)
    code_files = [Path(__file__), ROOT / 'src/transcif/research/multicountry.py',
                  ROOT / 'src/transcif/research/cross_country.py',
                  ROOT / 'src/transcif/research/monthly_adaptation.py', PROTOCOL]
    if capacities is not None:
        code_files.extend([CAPACITY_PROTOCOL, ROOT / 'scripts/parse_eurostat_energy_capacity.py'])
    code_hashes = {str(p): sha(p) for p in code_files}
    if any(p.read_bytes() == b'' for p in code_files):
        raise ValueError('empty frozen code/protocol input')
    prepared, data_audit = {}, {}
    for country in COUNTRIES:
        raw, data_audit[country] = load_country(country, manifest)
        months = public_months(raw)
        features = public_features(months, raw.clock)
        features['country'], features['region'] = country.upper(), country.upper()
        features['year'] = raw.clock.dt.year.to_numpy()
        features['raw_row'] = np.arange(len(raw))
        features['year'] = raw.clock.dt.year.to_numpy()
        if capacities is not None:
            features = add_capacity_features(features, capacities)
        prepared[country] = (raw, features.dropna(subset=list(HISTORY)).copy())
    (out / 'input_audit.json').write_text(json.dumps(data_audit, indent=2, ensure_ascii=False))
    (out / 'startup_hashes.json').write_text(json.dumps({'code': code_hashes, 'inputs': inputs}, indent=2))
    prediction_seals, predictions_by_target = [], {}
    for target in COUNTRIES:
        sources = [c for c in COUNTRIES if c != target]
        train_parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in sources]
        train = pd.concat(train_parts, ignore_index=True)
        fuel = np.concatenate([
            prepared[c][0].iloc[part.raw_row.to_numpy()][list(FINE)].to_numpy(float)
            for c, part in zip(sources, train_parts)
        ])
        coarse = np.concatenate([
            coarse_values(prepared[c][0].iloc[part.raw_row.to_numpy()])
            for c, part in zip(sources, train_parts)
        ])
        if train.empty or len(train) != len(fuel) or len(fuel) != len(coarse):
            raise ValueError(f'empty or misaligned source training cohort for {target}')
        models = fit(train, fuel, coarse)
        capacity_models = fit_capacity_heads(train, fuel) if capacities is not None else None
        cal_parts = [prepared[c][1].loc[
            prepared[c][1].year.eq(2023) & prepared[c][1].clock.dt.month.le(6)
        ] for c in sources]
        calibration = pd.concat(cal_parts, ignore_index=True)
        cal_fuel = np.concatenate([
            prepared[c][0].iloc[part.raw_row.to_numpy()][list(FINE)].to_numpy(float)
            for c, part in zip(sources, cal_parts)
        ])
        cal_y = cal_fuel @ FACTORS / cal_fuel.sum(axis=1)
        cal_weights = weights(calibration)
        cal_predictions = predict(models, calibration)
        if capacity_models is not None:
            cal_predictions.update(predict_capacity_heads(capacity_models, calibration, models))
        radii = {name: weighted_quantile(np.abs(cal_y - value), cal_weights, .9)
                 for name, value in cal_predictions.items()}
        target_raw, target_all = prepared[target]
        query = target_all.loc[target_all.year.eq(2024)].copy()
        point = predict(models, query)
        if capacity_models is not None:
            point.update(predict_capacity_heads(capacity_models, query, models))
        path = out / f'{target.upper()}_heldout_predictions.csv.gz'
        pd.DataFrame({'time': target_raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                      **point}).to_csv(path, index=False, compression='gzip')
        prediction_seals.append({'target': target.upper(), 'path': path.name,
                                 'sha256': sha(path), 'hours': len(query)})
        predictions_by_target[target] = (query, path, radii)
        print(target.upper(), 'outer target sealed', len(query), 'hours from', '+'.join(x.upper() for x in sources), flush=True)
    (out / 'prediction_seal.json').write_text(json.dumps(prediction_seals, indent=2))
    if any(sha(Path(p)) != v for p, v in inputs.items()):
        raise RuntimeError('frozen research input changed during fit')
    if any(sha(Path(p)) != v for p, v in code_hashes.items()):
        raise RuntimeError('frozen code/protocol changed during fit')

    rows = []
    for target, (query, path, radii) in predictions_by_target.items():
        target_raw = prepared[target][0]
        yfuel = target_raw.iloc[query.raw_row.to_numpy()][list(FINE)].to_numpy(float)
        y = yfuel @ FACTORS / yfuel.sum(axis=1)
        scored = pd.read_csv(path)
        for method in radii:
            p = scored[method].to_numpy(float)
            radius = float(radii[method])
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                error = y[mask] - p[mask]
                daily = pd.DataFrame({
                    'day': query.loc[mask, 'clock'].dt.strftime('%Y-%m-%d').to_numpy(),
                    'truth': y[mask], 'prediction': p[mask],
                })
                center = daily[['truth', 'prediction']] - daily.groupby('day')[['truth', 'prediction']].transform('mean')
                rows.append({
                    'target_country': target.upper(), 'method': method, 'month': int(month),
                    'hours': int(mask.sum()), 'mae': float(np.abs(error).mean()),
                    'shape_mae': float(np.abs(center.truth - center.prediction).mean()),
                    'coverage90': float((np.abs(error) <= radius).mean()),
                    'width90': float(2 * radius),
                    'interval_score90': float(np.mean(2 * radius + 20 * np.maximum(np.abs(error) - radius, 0))),
                })
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / 'monthly_metrics.csv', index=False)
    metric_cols = ['mae', 'shape_mae', 'coverage90', 'width90', 'interval_score90']
    country = monthly.groupby(['target_country', 'method'])[metric_cols].mean().reset_index()
    country.to_csv(out / 'country_metrics.csv', index=False)
    macro = country.groupby('method')[metric_cols].mean().reset_index()
    macro.to_csv(out / 'macro_metrics.csv', index=False)
    status = {
        'status': 'DEVELOPMENT_COUNTRY_HELDOUT_ENERGY_CHARTS_PROXY',
        'outer_targets': [c.upper() for c in COUNTRIES],
        'training_year': 2022, 'source_calibration_period': '2023-01 through 2023-06',
        'target_test_year': 2024, 'target_hourly_inputs_used': False,
        'target_monthly_inputs': 'two-month-delayed generation shares reconstructed from current revised hourly source data',
        'historical_as_issued_vintages': False, 'new_final_confirmation_countries': False,
        'source_attribution': 'Energy-Charts.info, CC BY 4.0',
        'capacity_budget': ('Eurostat nrg_inf_epc net maximum electrical capacity; two-year statistical lag; current revised extract, not historical vintage'
                            if capacities is not None else None),
    }
    (out / 'status.json').write_text(json.dumps(status, indent=2))
    print(macro.to_string(index=False))


if __name__ == '__main__':
    main()
