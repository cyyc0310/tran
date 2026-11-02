"""Post hoc five-country decomposition of factor-head transfer errors."""
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
SOURCE_RESULT = ROOT/'results/pseudotarget-factor-response-shrinkage-r1'
PROTOCOL = ROOT/'docs/FACTOR_HEAD_TRANSFER_DECOMPOSITION_PROTOCOL_20261101.md'


def load_prepared(manifest: dict) -> tuple[dict, dict]:
    prepared, audits = {}, {}
    for country in COUNTRIES:
        raw, audits[country] = load_country(country, manifest)
        features = public_features(public_months(raw), raw.clock)
        features['country'], features['region'] = country.upper(), country.upper()
        features['year'] = raw.clock.dt.year.to_numpy()
        features['raw_row'] = np.arange(len(raw))
        prepared[country] = (raw, features.dropna(subset=list(HISTORY)).copy())
    return prepared, audits


def fit_sources(prepared: dict, sources: list[str]) -> dict:
    parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in sources]
    frame = pd.concat(parts, ignore_index=True)
    fuel = np.concatenate([prepared[c][0].iloc[p.raw_row.to_numpy()][list(FINE)].to_numpy(float)
                           for c, p in zip(sources, parts)])
    coarse = np.concatenate([coarse_values(prepared[c][0].iloc[p.raw_row.to_numpy()])
                             for c, p in zip(sources, parts)])
    if frame.empty or len(frame) != len(fuel) or len(frame) != len(coarse):
        raise ValueError(f'misaligned source training panel for {sources}')
    return fit(frame, fuel, coarse)


def daily_shape(frame: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> float:
    table = pd.DataFrame({'y': y, 'p': p, 'day': frame.clock.dt.strftime('%Y-%m-%d').to_numpy()})
    centered = table[['y', 'p']]-table.groupby('day')[['y', 'p']].transform('mean')
    return float(np.abs(centered.y-centered.p).mean())


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = DATA/'acquisition_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    inputs = {str(DATA/f'{c}_2022_2024.json'): sha(DATA/f'{c}_2022_2024.json') for c in COUNTRIES}
    inputs[str(manifest_path)] = sha(manifest_path)
    inputs[str(DATA/'audit.json')] = sha(DATA/'audit.json')
    source_seal = json.loads((SOURCE_RESULT/'prediction_seal.json').read_text())
    for row in source_seal:
        pred_path = SOURCE_RESULT/row['path']
        if sha(pred_path) != row['sha256']:
            raise ValueError(f'source prediction seal mismatch: {row["path"]}')
        inputs[str(pred_path)] = sha(pred_path)
    code_files = [Path(__file__), ROOT/'scripts/run_pseudotarget_factor_response_shrinkage.py',
                  ROOT/'scripts/run_energy_charts_loco.py',
                  ROOT/'src/transcif/research/multicountry.py',
                  ROOT/'src/transcif/research/cross_country.py', PROTOCOL]
    code = {str(path): sha(path) for path in code_files}
    (out/'input_hashes.json').write_text(json.dumps(inputs, indent=2))
    (out/'startup_code_hashes.json').write_text(json.dumps(code, indent=2))
    prepared, audits = load_prepared(manifest)
    (out/'input_audit.json').write_text(json.dumps(audits, indent=2, ensure_ascii=False))

    hourly_parts, monthly_rows, country_rows, seals = [], [], [], []
    for target in COUNTRIES:
        sources = [c for c in COUNTRIES if c != target]
        models = fit_sources(prepared, sources)
        raw, all_features = prepared[target]
        query = all_features.loc[all_features.year.eq(2024)].copy()
        raw_target = raw.iloc[query.raw_row.to_numpy()]
        fuel = raw_target[list(FINE)].to_numpy(float)
        y = fuel@FACTORS/fuel.sum(axis=1)
        q_true = coarse_values(raw_target)[:, 0]/coarse_values(raw_target).sum(axis=1)
        thermal = fuel[:, :4].sum(axis=1)
        ef_true = np.divide(fuel[:, :4]@FACTORS[:4], thermal,
                            out=np.full(len(thermal), np.nan), where=thermal>1e-10)
        point = predict(models, query)
        q0 = query[list(GROUPS)].to_numpy(float)*models['profile'][query.clock.dt.hour.to_numpy()]
        q_hat = q0[:, 0]/q0.sum(axis=1)
        q_month = query['thermal'].to_numpy(float)
        ef_hat = np.clip(models['factor_history'].predict(query[list(HISTORY)]), 400., 1000.)
        recomputed = {'ratio_recomposition': point['ratio_recomposition'],
                      'factor_history': point['factor_history'],
                      'macro_recomposition': point['macro_recomposition'],
                      'direct_history': point['direct_history']}
        saved = pd.read_csv(SOURCE_RESULT/f'{target.upper()}_heldout_predictions.csv.gz')
        for method, values in recomputed.items():
            np.testing.assert_allclose(values, saved[method].to_numpy(float), rtol=0., atol=1e-9,
                                       err_msg=f'{target.upper()} {method} does not reproduce sealed point')
        q_oracle = q_true*float(models['ratio_ef'])
        ef_oracle = q_hat*ef_true
        q_and_ef_oracle = q_true*ef_true
        frame_hour = pd.DataFrame({
            'target_country': target.upper(),
            'time': raw_target.time.astype(str).to_numpy(),
            'y_true': y, 'q_true': q_true, 'q_hat': q_hat, 'q_month': q_month,
            'ef_true': ef_true, 'ef_hat': ef_hat,
            'source_ratio_ef': float(models['ratio_ef']), 'source_macro_ef': float(models['macro_ef']),
            'ratio_recomposition': point['ratio_recomposition'],
            'factor_history': point['factor_history'],
            'macro_recomposition': point['macro_recomposition'],
            'direct_history': point['direct_history'],
            'oracle_q_true_source_ef': q_oracle,
            'oracle_q_hat_ef_true': ef_oracle,
            'oracle_q_true_ef_true': q_and_ef_oracle,
        })
        hourly_parts.append(frame_hour)
        for month in sorted(query.clock.dt.month.unique()):
            mask = query.clock.dt.month.eq(month).to_numpy()
            active = mask & (thermal>1e-10)
            row = {'target_country': target.upper(), 'month': int(month), 'hours': int(mask.sum()),
                   'active_thermal_hours': int(active.sum()),
                   'q_hat_mae': float(np.abs(q_hat[mask]-q_true[mask]).mean()),
                   'q_month_mae': float(np.abs(q_month[mask]-q_true[mask]).mean()),
                   'q_true_mean': float(q_true[mask].mean()), 'q_hat_mean': float(q_hat[mask].mean()),
                   'q_month_mean': float(q_month[mask].mean()),
                   'source_ratio_ef': float(models['ratio_ef']),
                   'target_energy_weighted_ef': float(np.sum(fuel[mask, :4]@FACTORS[:4])/
                                                       max(np.sum(thermal[mask]), 1e-12)),
                   'target_ef_hat_active_mean': float(np.mean(ef_hat[active])) if active.any() else np.nan,
                   'ef_active_mae': float(np.abs(ef_hat[active]-ef_true[active]).mean()) if active.any() else np.nan,
                   'ef_active_bias': float(np.mean(ef_hat[active]-ef_true[active])) if active.any() else np.nan}
            for method, values in recomputed.items():
                row[f'{method}_mae'] = float(np.abs(y[mask]-values[mask]).mean())
                row[f'{method}_shape_mae'] = daily_shape(query.loc[mask], y[mask], values[mask])
                lo = saved[f'{method}_lo'].to_numpy(float)[mask]
                hi = saved[f'{method}_hi'].to_numpy(float)[mask]
                row[f'{method}_coverage90'] = float(((y[mask]>=lo)&(y[mask]<=hi)).mean())
            for name, values in [('oracle_q_true_source_ef', q_oracle),
                                 ('oracle_q_hat_ef_true', ef_oracle),
                                 ('oracle_q_true_ef_true', q_and_ef_oracle)]:
                valid = np.isfinite(values[mask])
                row[f'{name}_mae'] = float(np.abs(y[mask][valid]-values[mask][valid]).mean()) if valid.any() else np.nan
            monthly_rows.append(row)
        country_row = {'target_country': target.upper(), 'source_countries': '+'.join(x.upper() for x in sources),
                       'source_ratio_ef': float(models['ratio_ef']),
                       'source_macro_ef': float(models['macro_ef']),
                       'target_energy_weighted_ef': float(np.sum(fuel[:, :4]@FACTORS[:4])/np.sum(thermal)),
                       'target_thermal_share_mae': float(np.mean(np.abs(q_hat-q_true))),
                       'target_month_share_mae': float(np.mean(np.abs(q_month-q_true))),
                       'ef_active_mae': float(np.nanmean(np.abs(ef_hat-ef_true))),
                       'ef_active_bias': float(np.nanmean(ef_hat-ef_true))}
        monthly_target = [r for r in monthly_rows if r['target_country']==target.upper()]
        for method in recomputed:
            country_row[f'{method}_mae_equal_month'] = float(np.mean([r[f'{method}_mae'] for r in monthly_target]))
            country_row[f'{method}_shape_mae_equal_month'] = float(np.mean([r[f'{method}_shape_mae'] for r in monthly_target]))
            country_row[f'{method}_coverage_equal_month'] = float(np.mean([r[f'{method}_coverage90'] for r in monthly_target]))
        for name in ('oracle_q_true_source_ef','oracle_q_hat_ef_true','oracle_q_true_ef_true'):
            country_row[f'{name}_mae_equal_month'] = float(np.nanmean([r[f'{name}_mae'] for r in monthly_target]))
        country_rows.append(country_row)
        seals.append({'target': target.upper(), 'rows': len(query),
                      'sealed_prediction': f'{target.upper()}_heldout_predictions.csv.gz',
                      'sealed_prediction_sha256': sha(SOURCE_RESULT/f'{target.upper()}_heldout_predictions.csv.gz')})

    hourly = pd.concat(hourly_parts, ignore_index=True)
    monthly = pd.DataFrame(monthly_rows)
    country = pd.DataFrame(country_rows)
    hourly.to_csv(out/'hourly_factor_decomposition.csv.gz', index=False, compression='gzip')
    monthly.to_csv(out/'monthly_factor_decomposition.csv', index=False)
    country.to_csv(out/'country_factor_decomposition.csv', index=False)
    fields = [col for col in country.columns if col.endswith('_equal_month')]
    macro = {col: float(country[col].mean()) for col in fields}
    macro.update({'outer_countries': int(len(country)), 'hourly_rows': int(len(hourly)),
                  'source_inputs_hash_files': len(inputs), 'startup_code_files': len(code),
                  'recomputed_base_predictions_match_sealed': True})
    (out/'summary.json').write_text(json.dumps(macro, indent=2))
    (out/'reconstruction_seal.json').write_text(json.dumps(seals, indent=2))
    if any(sha(Path(path)) != value for path, value in inputs.items()):
        raise RuntimeError('input changed during diagnostic')
    if any(sha(Path(path)) != value for path, value in code.items()):
        raise RuntimeError('code/protocol changed during diagnostic')
    output_names = ['hourly_factor_decomposition.csv.gz','monthly_factor_decomposition.csv',
                    'country_factor_decomposition.csv','summary.json','reconstruction_seal.json']
    verification = {'outer_countries': len(country), 'hourly_rows': len(hourly),
                    'input_files': len(inputs), 'code_files': len(code),
                    'missing_outputs': [name for name in output_names if not (out/name).is_file()],
                    'input_hash_mismatches': [path for path,value in inputs.items() if sha(Path(path))!=value],
                    'code_hash_mismatches': [path for path,value in code.items() if sha(Path(path))!=value],
                    'sealed_prediction_reconstruction_match': True}
    (out/'post_run_verification.json').write_text(json.dumps(verification, indent=2))
    print(country.to_string(index=False))


if __name__ == '__main__':
    main()
