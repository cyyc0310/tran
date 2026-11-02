"""Cross-fitted pseudo-target training of a coarse-composition predictor gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from run_energy_charts_loco import (
    COUNTRIES, DATA, FINE, PROTOCOL, load_country, sha,
)
from transcif.research.multicountry import (
    FACTORS, GROUPS, HISTORY, METHODS, coarse_values, fit, predict,
    public_features, public_months, weights,
)
from transcif.research.monthly_adaptation import weighted_quantile

ROOT = Path(__file__).resolve().parents[1]
GATE_FEATURES = (*GROUPS, 'thermal_mean', 'thermal_std', 'thermal_change', 'total_cv', 'observations')
PENALTIES = (0.01, 0.1, 1.0, 10.0, 100.0)


def gate_weights(x: np.ndarray, beta: np.ndarray, mean: np.ndarray,
                 scale: np.ndarray, conditional: bool) -> np.ndarray:
    if conditional:
        z = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    else:
        z = np.ones((len(x), 1))
    logits = z@beta
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp/exp.sum(axis=1, keepdims=True)


def fit_gate(x: np.ndarray, expert: np.ndarray, y: np.ndarray, groups: np.ndarray,
             penalty: float, conditional: bool) -> dict:
    x, expert, y = np.asarray(x, float), np.asarray(expert, float), np.asarray(y, float)/1000.
    if len(x) != len(expert) or len(y) != len(x) or not np.isfinite(expert).all():
        raise ValueError('invalid pseudo-target gate training arrays')
    mean = np.zeros(x.shape[1]); scale = np.ones(x.shape[1])
    if conditional:
        row_weights = weights(pd.DataFrame({'country': groups, 'region': groups}))
        mean = np.sum(x*row_weights[:, None], axis=0)
        scale = np.sqrt(np.sum((x-mean)**2*row_weights[:, None], axis=0))
        scale[scale < 1e-8] = 1.
        design = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    else:
        design = np.ones((len(x), 1))
    # Country-balanced objective: each pseudo-target contributes equal mass.
    row_weights = weights(pd.DataFrame({'country': groups, 'region': groups}))
    n, m = expert.shape
    d = design.shape[1]
    def objective(flat):
        beta = flat.reshape(d, m)
        logits = design@beta
        logits -= logits.max(axis=1, keepdims=True)
        gate = np.exp(logits)
        gate /= gate.sum(axis=1, keepdims=True)
        point = np.sum(gate*expert/1000., axis=1)
        residual = point-y
        value = float(np.sum(row_weights*residual**2))
        if conditional and d > 1:
            value += .5*penalty*float(np.sum(beta[1:]**2))
        dlogit = (2*row_weights*residual)[:, None]*gate*(expert/1000.-point[:, None])
        grad = design.T@dlogit
        if conditional and d > 1:
            grad[1:] += penalty*beta[1:]
        return value, grad.ravel()
    result = minimize(objective, np.zeros(d*m), method='L-BFGS-B', jac=True,
                      options={'maxiter': 250, 'ftol': 1e-10, 'maxls': 30})
    if not result.success and result.status not in (1,):
        raise RuntimeError(f'gate optimizer failed: {result.message}')
    return {'beta': result.x.reshape(d, m), 'mean': mean, 'scale': scale,
            'conditional': conditional, 'penalty': float(penalty),
            'optimizer_message': str(result.message), 'objective': float(result.fun)}


def predict_gate(gate: dict, x: np.ndarray, expert: np.ndarray) -> np.ndarray:
    alpha = gate_weights(x, gate['beta'], gate['mean'], gate['scale'], gate['conditional'])
    return np.sum(alpha*expert, axis=1)


def target_rows(prepared: dict, country: str, year: int, half: int | None) -> tuple[pd.DataFrame, np.ndarray]:
    raw, features = prepared[country]
    query = features.loc[features.year.eq(year)].copy()
    if half == 1:
        query = query.loc[query.clock.dt.month.le(6)].copy()
    elif half == 2:
        query = query.loc[query.clock.dt.month.gt(6)].copy()
    if query.empty:
        raise ValueError(f'empty query: {country} {year} half={half}')
    fine = raw.iloc[query.raw_row.to_numpy()][list(FINE)].to_numpy(float)
    labels = fine@FACTORS/fine.sum(axis=1)
    return query, labels


def source_fit(prepared: dict, sources: list[str]) -> dict:
    parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in sources]
    train = pd.concat(parts, ignore_index=True)
    fuel = np.concatenate([prepared[c][0].iloc[p.raw_row.to_numpy()][list(FINE)].to_numpy(float)
                           for c, p in zip(sources, parts)])
    coarse = np.concatenate([coarse_values(prepared[c][0].iloc[p.raw_row.to_numpy()])
                             for c, p in zip(sources, parts)])
    if train.empty or len(train) != len(fuel) or len(train) != len(coarse):
        raise ValueError(f'misaligned source fit for {sources}')
    return fit(train, fuel, coarse)


def oof_pseudo_predictions(prepared: dict, sources: list[str]) -> dict[int, dict]:
    result = {}
    for pseudo in sources:
        train_sources = [c for c in sources if c != pseudo]
        models = source_fit(prepared, train_sources)
        rows = {}
        for half in (1, 2):
            query, y = target_rows(prepared, pseudo, 2023, half)
            point = predict(models, query)
            x = query[list(GATE_FEATURES)].to_numpy(float)
            rows[half] = {'query': query, 'y': y, 'point': point, 'x': x}
        result[pseudo] = rows
        print('pseudo target crossfit complete', pseudo.upper(), flush=True)
    return result


def tune_penalty(oof: dict) -> tuple[float, pd.DataFrame]:
    rows = []
    for penalty in PENALTIES:
        fold_mae = []
        for heldout in oof:
            train_parts = [oof[c][1] for c in oof if c != heldout]
            x = np.concatenate([p['x'] for p in train_parts])
            y = np.concatenate([p['y'] for p in train_parts])
            experts = np.concatenate([np.column_stack([p['point'][m] for m in METHODS]) for p in train_parts])
            groups = np.concatenate([np.repeat(c, len(p['y'])) for c, p in zip([c for c in oof if c != heldout], train_parts)])
            gate = fit_gate(x, experts, y, groups, penalty, True)
            test = oof[heldout][1]
            pred = predict_gate(gate, test['x'], np.column_stack([test['point'][m] for m in METHODS]))
            fold_mae.append(float(np.abs(pred-test['y']).mean()))
        score = float(np.mean(fold_mae))
        rows.append({'penalty': penalty, 'inner_country_mae': score,
                     'fold_scores': json.dumps(fold_mae)})
    table = pd.DataFrame(rows).sort_values(['inner_country_mae', 'penalty'])
    return float(table.iloc[0].penalty), table


def macro_daily_shape(query: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> float:
    daily = pd.DataFrame({'truth': y, 'prediction': p,
                          'day': query.clock.dt.strftime('%Y-%m-%d').to_numpy()})
    centered = daily[['truth', 'prediction']]-daily.groupby('day')[['truth', 'prediction']].transform('mean')
    return float(np.abs(centered.truth-centered.prediction).mean())


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
    code_hashes = {str(p): sha(p) for p in code_files}
    (out/'startup_hashes.json').write_text(json.dumps({'code': code_hashes, 'inputs': inputs}, indent=2))
    prepared, audits = {}, {}
    for country in COUNTRIES:
        raw, audits[country] = load_country(country, manifest)
        months = public_months(raw)
        features = public_features(months, raw.clock)
        features['country'], features['region'] = country.upper(), country.upper()
        features['year'] = raw.clock.dt.year.to_numpy()
        features['raw_row'] = np.arange(len(raw))
        prepared[country] = (raw, features.dropna(subset=list(HISTORY)).copy())
    (out/'input_audit.json').write_text(json.dumps(audits, indent=2, ensure_ascii=False))

    seals, prediction_sets, selection_records = [], {}, []
    for target in COUNTRIES:
        sources = [c for c in COUNTRIES if c != target]
        oof = oof_pseudo_predictions(prepared, sources)
        penalty, penalty_table = tune_penalty(oof)
        penalty_table.to_csv(out/f'{target.upper()}_inner_penalty_cv.csv', index=False)
        h1_parts = [oof[c][1] for c in sources]
        h2_parts = [oof[c][2] for c in sources]
        x1 = np.concatenate([p['x'] for p in h1_parts])
        y1 = np.concatenate([p['y'] for p in h1_parts])
        e1 = np.concatenate([np.column_stack([p['point'][m] for m in METHODS]) for p in h1_parts])
        c1 = np.concatenate([np.repeat(c, len(p['y'])) for c, p in zip(sources, h1_parts)])
        gate = fit_gate(x1, e1, y1, c1, penalty, True)
        static_gate = fit_gate(x1, e1, y1, c1, 0., False)
        model_mae = {m: float(np.mean([np.abs(oof[c][1]['point'][m]-oof[c][1]['y']).mean()
                                       for c in sources])) for m in METHODS}
        selected = min(model_mae, key=model_mae.get)
        selection_records.append({'target': target.upper(), 'selected_method': selected,
                                  'selected_inner_mae': model_mae[selected], 'penalty': penalty,
                                  'gate_objective': gate['objective']})
        h2_query = pd.concat([p['query'] for p in h2_parts], ignore_index=True)
        h2_y = np.concatenate([p['y'] for p in h2_parts])
        h2_expert = np.concatenate([np.column_stack([p['point'][m] for m in METHODS]) for p in h2_parts])
        h2_x = np.concatenate([p['x'] for p in h2_parts])
        h2_groups = np.concatenate([np.repeat(c, len(p['y'])) for c, p in zip(sources, h2_parts)])
        alpha = gate_weights(h2_x, gate['beta'], gate['mean'], gate['scale'], True)
        static_alpha = gate_weights(h2_x, static_gate['beta'], static_gate['mean'], static_gate['scale'], False)
        h2_points = {m: h2_expert[:, i] for i, m in enumerate(METHODS)}
        h2_points['uniform_ensemble'] = h2_expert.mean(axis=1)
        h2_points['static_stack'] = np.sum(static_alpha*h2_expert, axis=1)
        h2_points['conditional_stack'] = np.sum(alpha*h2_expert, axis=1)
        h2_points['pseudo_selected'] = h2_expert[:, METHODS.index(selected)]
        cal_weights = weights(pd.DataFrame({'country': h2_groups, 'region': h2_groups}))
        radii = {m: weighted_quantile(np.abs(h2_y-p), cal_weights, .9) for m, p in h2_points.items()}

        base_models = source_fit(prepared, sources)
        _, query = prepared[target]
        query = query.loc[query.year.eq(2024)].copy()
        candidates = predict(base_models, query)
        expert = np.column_stack([candidates[m] for m in METHODS])
        conditional = predict_gate(gate, query[list(GATE_FEATURES)].to_numpy(float), expert)
        static = predict_gate(static_gate, query[list(GATE_FEATURES)].to_numpy(float), expert)
        point = dict(candidates)
        point['uniform_ensemble'] = expert.mean(axis=1)
        point['static_stack'] = static
        point['conditional_stack'] = conditional
        point['pseudo_selected'] = candidates[selected]
        payload = {'time': prepared[target][0].iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(), **point}
        for method, pred in point.items():
            payload[f'{method}_lo'] = pred-radii[method]
            payload[f'{method}_hi'] = pred+radii[method]
        path = out/f'{target.upper()}_heldout_predictions.csv.gz'
        pd.DataFrame(payload).to_csv(path, index=False, compression='gzip')
        seals.append({'target': target.upper(), 'path': path.name, 'sha256': sha(path), 'hours': len(query)})
        prediction_sets[target] = (query.copy(), path, radii)
        print(target.upper(), 'sealed', len(query), 'hours; pseudo-selected', selected,
              'penalty', penalty, flush=True)
    (out/'prediction_seal.json').write_text(json.dumps(seals, indent=2))
    (out/'pseudo_target_selection.json').write_text(json.dumps(selection_records, indent=2))
    if any(sha(Path(p)) != v for p, v in inputs.items()):
        raise RuntimeError('frozen research input changed during execution')
    if any(sha(Path(p)) != v for p, v in code_hashes.items()):
        raise RuntimeError('frozen code/protocol changed during execution')

    rows = []
    for target, (query, path, radii) in prediction_sets.items():
        raw = prepared[target][0]
        fuel = raw.iloc[query.raw_row.to_numpy()][list(FINE)].to_numpy(float)
        y = fuel@FACTORS/fuel.sum(axis=1)
        scored = pd.read_csv(path)
        methods = [*METHODS, 'uniform_ensemble', 'static_stack', 'conditional_stack', 'pseudo_selected']
        for method in methods:
            p = scored[method].to_numpy(float)
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                error = y[mask]-p[mask]
                lo, hi = scored[f'{method}_lo'].to_numpy(float)[mask], scored[f'{method}_hi'].to_numpy(float)[mask]
                rows.append({'target_country': target.upper(), 'method': method, 'month': int(month),
                             'hours': int(mask.sum()), 'mae': float(np.abs(error).mean()),
                             'shape_mae': macro_daily_shape(query.loc[mask], y[mask], p[mask]),
                             'coverage90': float(((y[mask] >= lo)&(y[mask] <= hi)).mean()),
                             'width90': float(np.mean(hi-lo)),
                             'interval_score90': float(np.mean(hi-lo+20*np.maximum(lo-y[mask], 0)
                                                                 +20*np.maximum(y[mask]-hi, 0)))})
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out/'monthly_metrics.csv', index=False)
    fields = ['mae', 'shape_mae', 'coverage90', 'width90', 'interval_score90']
    country = monthly.groupby(['target_country', 'method'])[fields].mean().reset_index()
    country.to_csv(out/'country_metrics.csv', index=False)
    macro = country.groupby('method')[fields].mean().reset_index()
    macro.to_csv(out/'macro_metrics.csv', index=False)
    print(macro.to_string(index=False))


if __name__ == '__main__':
    main()
