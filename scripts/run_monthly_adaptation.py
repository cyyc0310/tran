"""Source-only hourly models with local coarse-month adaptation and risk references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import sklearn

from transcif.research.budget_transfer import budget_views
from transcif.research.cross_country import COARSE, FUELS, fit_models, labels, load_inputs, monthly_inputs
from transcif.research.monthly_adaptation import (
    METHODS, fit_risks, fit_source, monthly_table, predict_hours, risk_bounds,
)
from run_cross_country import ROOT, digest


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), ROOT/'src/transcif/research/monthly_adaptation.py',
             ROOT/'src/transcif/research/cross_country.py', ROOT/'src/transcif/research/budget_transfer.py',
             ROOT/'docs/CROSS_COUNTRY_MONTHLY_ADAPTATION_PROTOCOL.md']
    provenance = {str(p.relative_to(ROOT)): digest(p) for p in files}
    (out/'startup_code_hashes.json').write_text(json.dumps(provenance, indent=2))
    (out/'environment.json').write_text(json.dumps({'numpy': np.__version__, 'pandas': pd.__version__,
                                                   'sklearn': sklearn.__version__}, indent=2))
    raw, audit = load_inputs(ROOT)
    views = {r: {lag: monthly_inputs(f, lag) for lag in (2, 3, 4)} for r, f in raw.items()}
    tables = {r: monthly_table(f) for r, f in raw.items()}
    all_frames = {r: v[2].dropna(subset=list(COARSE)) for r, v in views.items()}
    records, seals, coarse_forecasts = [], [], []
    for source, target_country in [('US', 'GB'), ('GB', 'US')]:
        sources = [r for r in raw if raw[r].country.iloc[0] == source and r != 'BPAT']
        targets = [r for r in raw if raw[r].country.iloc[0] == target_country]
        source_frames = [budget_views(views[r], 2022, False).drop_duplicates('raw_row') for r in sources]
        train = pd.concat(source_frames, ignore_index=True)
        fuel = np.concatenate([raw[r].loc[f.raw_row, list(FUELS)].to_numpy() for r, f in zip(sources, source_frames)])
        native = fit_models(train, fuel)
        source_model = fit_source({r: tables[r] for r in sources})
        source_state = {'regions': sources, 'ridge_coef': source_model.coef_.tolist(),
                        'ridge_intercept': source_model.intercept_.tolist(),
                        'profile': native['profile'].tolist(), 'thermal_ef': native['thermal_ef']}
        (out/f'{source}_model.json').write_text(json.dumps(source_state, indent=2))
        for method in METHODS:
            chunks = {half: [] for half in (1, 2)}
            for r in sources:
                f = all_frames[r].loc[all_frames[r].year.eq(2023)]
                point, x = predict_hours(f, tables[r], source_model, method, native)
                error = labels(raw[r].loc[f.index])-point
                for half in (1, 2):
                    mask = (f.clock.dt.month.le(6) if half == 1 else f.clock.dt.month.gt(6)).to_numpy()
                    chunks[half].append((x[mask], error[mask], np.repeat(r, mask.sum())))
            parts = []
            for half in (1, 2):
                parts.extend(np.concatenate([row[j] for row in chunks[half]]) for j in range(3))
            fitted = fit_risks(*parts)
            joblib.dump(fitted, out/f'{source}_{method}_risk.joblib')
            restored = joblib.load(out/f'{source}_{method}_risk.joblib')
            for kind in ('global', 'hetero', 'cqr'):
                expected = risk_bounds(fitted, parts[3][:32], np.zeros(min(32, len(parts[3]))))[kind]
                actual = risk_bounds(restored, parts[3][:32], np.zeros(min(32, len(parts[3]))))[kind]
                for a, b in zip(expected, actual):
                    np.testing.assert_array_equal(a, b)
            (out/f'{source}_{method}_risk.json').write_text(json.dumps(
                {**{k: fitted[k] for k in ('bias', 'global', 'hetero', 'cqr')},
                 'actual_iterations': {k: int(fitted[k].n_iter_) for k in ('scale', 'lo', 'hi')}}, indent=2))
            for r in targets:
                f = all_frames[r].loc[all_frames[r].year.eq(2024)]
                point, x = predict_hours(f, tables[r], source_model, method, native)
                bounds = risk_bounds(fitted, x, point)
                data = {'time': f.time.astype(str), 'point_raw': point, 'point': point+fitted['bias']}
                for kind, (lo, hi) in bounds.items():
                    data[f'{kind}_lo'], data[f'{kind}_hi'] = lo, hi
                path = out/f'{source}_{method}_to_{r}.csv'
                pd.DataFrame(data).to_csv(path, index=False)
                seals.append({'path': path.name, 'sha256': digest(path)})
                records.append((source, target_country, r, method, f, path))
                for p, block in f.groupby('period'):
                    coarse_forecasts.append({'source': source, 'region': r, 'method': method,
                                             'period': str(p), 'latest_allowed_month': str(p-2),
                                             **dict(zip(COARSE, x[f.period.eq(p).to_numpy()][0, -5:]))})
            print(source, method, 'predictions sealed', flush=True)
    (out/'prediction_seal.json').write_text(json.dumps(seals, indent=2))
    pd.DataFrame(coarse_forecasts).to_csv(out/'coarse_forecasts.csv', index=False)
    rows = []
    for source, target_country, r, method, f, path in records:
        predictions = pd.read_csv(path)
        y = labels(raw[r].loc[f.index])
        for month in sorted(f.clock.dt.month.unique()):
            mask = f.clock.dt.month.eq(month).to_numpy()
            selected = f.loc[mask]
            a, b = y[mask], predictions.point.to_numpy()[mask]
            tmp = pd.DataFrame({'day': selected.clock.dt.strftime('%Y-%m-%d').to_numpy(), 'y': a, 'p': b})
            centered = tmp[['y', 'p']]-tmp.groupby('day')[['y', 'p']].transform('mean')
            for kind in ('global', 'hetero', 'cqr'):
                lo, hi = predictions[f'{kind}_lo'].to_numpy()[mask], predictions[f'{kind}_hi'].to_numpy()[mask]
                rows.append({'source': source, 'target': target_country, 'region': r, 'method': method,
                             'risk': kind, 'month': int(month), 'common_month': month in (2, 4, 6),
                             'hours': int(mask.sum()), 'mae': float(np.abs(a-b).mean()),
                             'raw_mae': float(np.abs(a-predictions.point_raw.to_numpy()[mask]).mean()),
                             'shape_mae': float(np.abs(centered.y-centered.p).mean()),
                             'coverage90': float(((a>=lo)&(a<=hi)).mean()), 'width90': float(np.mean(hi-lo)),
                             'interval_score90': float(np.mean(hi-lo+20*np.maximum(lo-a, 0)+20*np.maximum(a-hi, 0)))})
    scores = pd.DataFrame(rows)
    scores.to_csv(out/'monthly_metrics.csv', index=False)
    fields = ['mae', 'raw_mae', 'shape_mae', 'coverage90', 'width90', 'interval_score90']
    groups = ['source', 'target', 'region', 'method', 'risk']
    regional = scores.loc[scores.common_month].groupby(groups)[fields].mean().reset_index()
    regional.to_csv(out/'common_region_metrics.csv', index=False)
    country = regional.groupby(['source', 'target', 'method', 'risk'])[fields].mean().reset_index()
    country.to_csv(out/'common_country_metrics.csv', index=False)
    if any(digest(ROOT/k) != v for k, v in provenance.items()):
        raise RuntimeError('code/protocol changed during execution')
    for a in audit.values():
        provenance[a['path']] = digest(ROOT/a['path'])
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(country.to_string(index=False))


if __name__ == '__main__':
    main()
