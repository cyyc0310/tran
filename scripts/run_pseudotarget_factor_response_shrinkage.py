"""Source-only pseudo-target shrinkage of the thermal-factor response head."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_energy_charts_loco import COUNTRIES, DATA, PROTOCOL, load_country, sha
from transcif.research.multicountry import FACTORS, FINE, HISTORY, coarse_values, fit, predict, public_features, public_months, weights
from transcif.research.monthly_adaptation import weighted_quantile

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = (0., .25, .5, .75, 1.)
METHODS = ('macro_recomposition', 'ratio_recomposition', 'factor_history',
           'direct_history', 'source_calibrated_factor')


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


def source_fit(prepared: dict, sources: list[str]) -> dict:
    parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in sources]
    frame = pd.concat(parts, ignore_index=True)
    fuel = np.concatenate([prepared[c][0].iloc[p.raw_row.to_numpy()][list(FINE)].to_numpy(float)
                           for c, p in zip(sources, parts)])
    coarse = np.concatenate([coarse_values(prepared[c][0].iloc[p.raw_row.to_numpy()])
                             for c, p in zip(sources, parts)])
    if frame.empty or len(frame) != len(fuel) or len(frame) != len(coarse):
        raise ValueError(f'misaligned training frame for {sources}')
    return fit(frame, fuel, coarse)


def query_for(prepared: dict, country: str, half: int) -> tuple[pd.DataFrame, np.ndarray]:
    raw, features = prepared[country]
    query = features.loc[features.year.eq(2023)].copy()
    query = query.loc[query.clock.dt.month.le(6) if half == 1 else query.clock.dt.month.gt(6)].copy()
    if query.empty:
        raise ValueError(f'no 2023H{half} rows for {country}')
    fuel = raw.iloc[query.raw_row.to_numpy()][list(FINE)].to_numpy(float)
    y = fuel@FACTORS/fuel.sum(axis=1)
    return query, y


def oof_predictions(prepared: dict, sources: list[str]) -> dict[str, dict[int, dict]]:
    result = {}
    for pseudo in sources:
        model = source_fit(prepared, [c for c in sources if c != pseudo])
        halves = {}
        for half in (1, 2):
            query, y = query_for(prepared, pseudo, half)
            halves[half] = {'query': query, 'y': y, 'point': predict(model, query)}
        result[pseudo] = halves
        print('pseudo factor task complete', pseudo.upper(), flush=True)
    return result


def choose_alpha(tasks: dict[str, dict], countries: list[str]) -> tuple[float, pd.DataFrame]:
    rows = []
    for alpha in ALPHAS:
        country_mae = []
        for country in countries:
            p = tasks[country]['point']
            pred = p['ratio_recomposition']+alpha*(p['factor_history']-p['ratio_recomposition'])
            country_mae.append(float(np.abs(pred-tasks[country]['y']).mean()))
        rows.append({'alpha': alpha, 'country_equal_mae': float(np.mean(country_mae)),
                     'country_mae': json.dumps(country_mae)})
    table = pd.DataFrame(rows).sort_values(['country_equal_mae', 'alpha']).reset_index(drop=True)
    return float(table.iloc[0].alpha), table


def interval_radius(errors: np.ndarray, groups: np.ndarray) -> float:
    frame = pd.DataFrame({'country': groups, 'region': groups})
    return weighted_quantile(np.abs(errors), weights(frame), .9)


def daily_shape(query: pd.DataFrame, y: np.ndarray, pred: np.ndarray) -> float:
    frame = pd.DataFrame({'y': y, 'p': pred, 'day': query.clock.dt.strftime('%Y-%m-%d').to_numpy()})
    centered = frame[['y', 'p']]-frame.groupby('day')[['y', 'p']].transform('mean')
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
    code_files = [Path(__file__), ROOT/'scripts/run_energy_charts_loco.py',
                  ROOT/'src/transcif/research/multicountry.py',
                  ROOT/'src/transcif/research/cross_country.py',
                  ROOT/'src/transcif/research/monthly_adaptation.py', PROTOCOL]
    code_hashes = {str(path): sha(path) for path in code_files}
    (out/'startup_hashes.json').write_text(json.dumps({'code': code_hashes, 'inputs': inputs}, indent=2))
    prepared, audits = load_prepared(manifest)
    (out/'input_audit.json').write_text(json.dumps(audits, indent=2, ensure_ascii=False))
    seals, predictions, choices = [], {}, []

    for target in COUNTRIES:
        sources = [c for c in COUNTRIES if c != target]
        oof = oof_predictions(prepared, sources)
        h1 = {c: oof[c][1] for c in sources}
        alpha, score_table = choose_alpha(h1, sources)
        score_table.to_csv(out/f'{target.upper()}_alpha_scores.csv', index=False)
        nested = []
        for heldout in sources:
            subtrain = [c for c in sources if c != heldout]
            fold_alpha, fold_scores = choose_alpha(h1, subtrain)
            test = h1[heldout]
            p = test['point']
            fold_pred = p['ratio_recomposition']+fold_alpha*(p['factor_history']-p['ratio_recomposition'])
            nested.append({'heldout_pseudo': heldout.upper(), 'selected_alpha': fold_alpha,
                           'mae': float(np.abs(fold_pred-test['y']).mean())})
        (out/f'{target.upper()}_nested_alpha_cv.json').write_text(json.dumps(nested, indent=2))
        choices.append({'target': target.upper(), 'alpha': alpha,
                        'selected_inner_mae': float(score_table.iloc[0].country_equal_mae),
                        'nested_cv_mean_mae': float(np.mean([row['mae'] for row in nested]))})

        h2_parts = [oof[c][2] for c in sources]
        cal_y = np.concatenate([p['y'] for p in h2_parts])
        cal_groups = np.concatenate([np.repeat(c, len(p['y'])) for c, p in zip(sources, h2_parts)])
        cal_predictions = {method: np.concatenate([p['point'][method] for p in h2_parts])
                           for method in ('macro_recomposition', 'ratio_recomposition', 'factor_history', 'direct_history')}
        cal_predictions['source_calibrated_factor'] = (
            cal_predictions['ratio_recomposition']+alpha*(cal_predictions['factor_history']-
                                                            cal_predictions['ratio_recomposition']))
        radii = {method: interval_radius(cal_y-pred, cal_groups)
                 for method, pred in cal_predictions.items()}

        model = source_fit(prepared, sources)
        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        point = predict(model, query)
        point['source_calibrated_factor'] = (
            point['ratio_recomposition']+alpha*(point['factor_history']-point['ratio_recomposition']))
        payload = {'time': raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(), **point}
        for method in METHODS:
            payload[f'{method}_lo'] = point[method]-radii[method]
            payload[f'{method}_hi'] = point[method]+radii[method]
        path = out/f'{target.upper()}_heldout_predictions.csv.gz'
        pd.DataFrame(payload).to_csv(path, index=False, compression='gzip')
        seals.append({'target': target.upper(), 'path': path.name, 'sha256': sha(path), 'hours': len(query)})
        predictions[target] = (query, path)
        print(target.upper(), 'sealed', len(query), 'hours; alpha', alpha, flush=True)

    (out/'prediction_seal.json').write_text(json.dumps(seals, indent=2))
    (out/'alpha_selection.json').write_text(json.dumps(choices, indent=2))
    if any(sha(Path(path)) != value for path, value in inputs.items()):
        raise RuntimeError('research input changed during run')
    if any(sha(Path(path)) != value for path, value in code_hashes.items()):
        raise RuntimeError('code or protocol changed during run')
    prediction_hash_mismatches = [row['path'] for row in seals
                                  if sha(out/row['path']) != row['sha256']]
    if prediction_hash_mismatches:
        raise RuntimeError(f'prediction seal mismatch: {prediction_hash_mismatches}')

    rows = []
    for target, (query, path) in predictions.items():
        raw = prepared[target][0]
        fuel = raw.iloc[query.raw_row.to_numpy()][list(FINE)].to_numpy(float)
        y = fuel@FACTORS/fuel.sum(axis=1)
        scored = pd.read_csv(path)
        for method in METHODS:
            pred = scored[method].to_numpy(float)
            lo, hi = scored[f'{method}_lo'].to_numpy(float), scored[f'{method}_hi'].to_numpy(float)
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                error = y[mask]-pred[mask]
                rows.append({'target_country': target.upper(), 'method': method, 'month': int(month),
                             'hours': int(mask.sum()), 'mae': float(np.abs(error).mean()),
                             'shape_mae': daily_shape(query.loc[mask], y[mask], pred[mask]),
                             'coverage90': float(((y[mask] >= lo[mask])&(y[mask] <= hi[mask])).mean()),
                             'width90': float(np.mean(hi[mask]-lo[mask])),
                             'interval_score90': float(np.mean(hi[mask]-lo[mask]
                                 +20*np.maximum(lo[mask]-y[mask], 0)+20*np.maximum(y[mask]-hi[mask], 0)))})
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out/'monthly_metrics.csv', index=False)
    fields = ['mae', 'shape_mae', 'coverage90', 'width90', 'interval_score90']
    country = monthly.groupby(['target_country', 'method'])[fields].mean().reset_index()
    country.to_csv(out/'country_metrics.csv', index=False)
    macro = country.groupby('method')[fields].mean().reset_index()
    macro.to_csv(out/'macro_metrics.csv', index=False)
    outputs = ['prediction_seal.json', 'alpha_selection.json', 'monthly_metrics.csv',
               'country_metrics.csv', 'macro_metrics.csv']
    verification = {'outer_targets': len(seals), 'sealed_predictions': len(seals),
                    'input_files': len(inputs), 'code_files': len(code_hashes),
                    'prediction_hash_mismatches': prediction_hash_mismatches,
                    'missing_outputs': [name for name in outputs if not (out/name).is_file()],
                    'input_hash_mismatches': [path for path, value in inputs.items()
                                              if sha(Path(path)) != value],
                    'code_hash_mismatches': [path for path, value in code_hashes.items()
                                             if sha(Path(path)) != value],
                    'target_predictions_written_before_scoring': True}
    (out/'post_run_verification.json').write_text(json.dumps(verification, indent=2))
    print(macro.to_string(index=False))


if __name__ == '__main__':
    main()
