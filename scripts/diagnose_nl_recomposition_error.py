"""Post hoc decomposition of the Netherlands ratio-recomposition error."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_energy_charts_loco import COUNTRIES, DATA, load_country, sha
from transcif.research.multicountry import (
    FACTORS, FINE, GROUPS, HISTORY, coarse_values, fit, predict, public_features, public_months,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'docs/NL_RATIO_RECOMPOSITION_ERROR_DECOMPOSITION_PROTOCOL_20261101.md'


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = DATA/'acquisition_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    raw_by_country, feature_by_country, audits = {}, {}, {}
    inputs = {str(DATA/f'{c}_2022_2024.json'): sha(DATA/f'{c}_2022_2024.json') for c in COUNTRIES}
    inputs[str(manifest_path)] = sha(manifest_path)
    inputs[str(DATA/'audit.json')] = sha(DATA/'audit.json')
    code_files = [Path(__file__), ROOT/'scripts/run_energy_charts_loco.py',
                  ROOT/'src/transcif/research/multicountry.py',
                  ROOT/'src/transcif/research/cross_country.py', PROTOCOL]
    code = {str(p): sha(p) for p in code_files}
    for country in COUNTRIES:
        raw, audits[country] = load_country(country, manifest)
        features = public_features(public_months(raw), raw.clock)
        features['country'], features['region'] = country.upper(), country.upper()
        features['year'] = raw.clock.dt.year.to_numpy()
        features['raw_row'] = np.arange(len(raw))
        raw_by_country[country] = raw
        feature_by_country[country] = features.dropna(subset=list(HISTORY)).copy()
    sources = [c for c in COUNTRIES if c != 'nl']
    train_parts = [feature_by_country[c].loc[feature_by_country[c].year.eq(2022)] for c in sources]
    train = pd.concat(train_parts, ignore_index=True)
    fuel = np.concatenate([raw_by_country[c].iloc[p.raw_row.to_numpy()][list(FINE)].to_numpy(float)
                           for c, p in zip(sources, train_parts)])
    coarse = np.concatenate([coarse_values(raw_by_country[c].iloc[p.raw_row.to_numpy()])
                             for c, p in zip(sources, train_parts)])
    models = fit(train, fuel, coarse)
    target_raw = raw_by_country['nl']
    query = feature_by_country['nl'].loc[feature_by_country['nl'].year.eq(2024)].copy()
    source_point = predict(models, query)['ratio_recomposition']
    q0 = query[list(GROUPS)].to_numpy(float)*models['profile'][query.clock.dt.hour.to_numpy()]
    q_hat = q0[:, 0]/q0.sum(axis=1)
    q_month = query['thermal'].to_numpy(float)
    target = target_raw.iloc[query.raw_row.to_numpy()]
    fuel_target = target[list(FINE)].to_numpy(float)
    y_true = fuel_target@FACTORS/fuel_target.sum(axis=1)
    coarse_target = coarse_values(target)
    q_true = coarse_target[:, 0]/coarse_target.sum(axis=1)
    ef_true = np.divide(y_true, q_true, out=np.full(len(y_true), np.nan), where=q_true>1e-10)
    ef_source = float(models['ratio_ef'])
    predictions = {
        'ratio_recomposition': source_point,
        'lagged_month_share_only': q_month*float(models['ratio_ef']),
        'true_hourly_thermal_share_only': q_true*ef_source,
        'true_hourly_factor_only': q_hat*ef_true,
    }
    hourly = pd.DataFrame({'time': target.time.astype(str).to_numpy(), 'y_true': y_true,
                           'q_true': q_true, 'q_hat': q_hat, 'q_lagged_month': q_month,
                           'ef_true': ef_true, 'ef_source': ef_source,
                           **{f'{name}_point': value for name, value in predictions.items()}})
    hourly.to_csv(out/'hourly_decomposition.csv.gz', index=False, compression='gzip')
    rows = []
    for month in sorted(query.clock.dt.month.unique()):
        mask = query.clock.dt.month.eq(month).to_numpy()
        row = {'month': int(month), 'hours': int(mask.sum()), 'source_thermal_ef': ef_source,
               'hours_with_thermal': int(np.sum(q_true[mask]>1e-10)),
               'actual_thermal_share_mean': float(q_true[mask].mean()),
               'predicted_thermal_share_mean': float(q_hat[mask].mean()),
               'lagged_month_thermal_share_mean': float(q_month[mask].mean()),
               'lagged_month_share_mae': float(np.abs(q_true[mask]-q_month[mask]).mean()),
               'thermal_share_mae': float(np.abs(q_true[mask]-q_hat[mask]).mean()),
               'actual_hour_weighted_thermal_ef': float(np.sum(fuel_target[mask, :4]@FACTORS[:4])/
                                                           np.sum(fuel_target[mask, :4])),
               'actual_thermal_ef_hourly_sd': float(np.nanstd(ef_true[mask]))}
        for name, point in predictions.items():
            valid = np.isfinite(point[mask])
            err = y_true[mask][valid]-point[mask][valid]
            row[f'{name}_mae'] = float(np.abs(err).mean()) if len(err) else float('nan')
            row[f'{name}_bias'] = float(err.mean()) if len(err) else float('nan')
        rows.append(row)
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out/'monthly_decomposition.csv', index=False)
    macro = {col: float(monthly[col].mean()) for col in monthly.columns
             if col.endswith('_mae') or col == 'thermal_share_mae'}
    macro.update({'source_thermal_ef': ef_source,
                  'actual_hour_weighted_thermal_ef': float(np.sum(fuel_target[:, :4]@FACTORS[:4])/
                                                               np.sum(fuel_target[:, :4])),
                  'n_target_hours': int(len(query)),
                  'mae_ratio_recomposition': float(np.mean(np.abs(y_true-source_point))),
                  'mae_lagged_month_share_only': float(np.mean(np.abs(y_true-predictions['lagged_month_share_only']))),
                  'mae_true_hourly_thermal_share_only': float(np.mean(np.abs(y_true-predictions['true_hourly_thermal_share_only']))),
                  'mae_true_hourly_factor_only': float(np.nanmean(np.abs(y_true-predictions['true_hourly_factor_only'])))})
    (out/'summary.json').write_text(json.dumps(macro, indent=2))
    (out/'input_hashes.json').write_text(json.dumps(inputs, indent=2))
    (out/'startup_code_hashes.json').write_text(json.dumps(code, indent=2))
    (out/'input_audit.json').write_text(json.dumps(audits, indent=2, ensure_ascii=False))
    if any(sha(Path(path)) != value for path, value in inputs.items()):
        raise RuntimeError('input changed during diagnostic')
    if any(sha(Path(path)) != value for path, value in code.items()):
        raise RuntimeError('code/protocol changed during diagnostic')
    outputs = ['monthly_decomposition.csv', 'hourly_decomposition.csv.gz', 'summary.json',
               'input_hashes.json', 'startup_code_hashes.json', 'input_audit.json']
    verification = {'target': 'NL', 'scored_hours': int(len(query)),
                    'input_files': len(inputs), 'code_files': len(code),
                    'missing_outputs': [name for name in outputs if not (out/name).is_file()],
                    'input_hash_mismatches': [path for path, value in inputs.items()
                                              if sha(Path(path)) != value],
                    'code_hash_mismatches': [path for path, value in code.items()
                                             if sha(Path(path)) != value]}
    (out/'post_run_verification.json').write_text(json.dumps(verification, indent=2))
    print(json.dumps(macro, indent=2))


if __name__ == '__main__':
    main()
