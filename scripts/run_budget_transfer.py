"""Fixed-budget delayed-month and auxiliary-fuel cross-country ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transcif.research.budget_transfer import HEADS, budget_views, predict_model, save_model, train_model
from transcif.research.cross_country import (
    COARSE, FUELS, fit_models, labels, load_inputs, metrics, monthly_inputs, predict_models,
)
from run_cross_country import ROOT, digest

COMMON = (2, 4, 6)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--family', choices=['budget', 'structured'], default='budget')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    protocol = 'CROSS_COUNTRY_BUDGET_TRAINING_PROTOCOL.md' if args.family == 'budget' else 'CROSS_COUNTRY_STRUCTURED_PROTOCOL.md'
    files = [Path(__file__), ROOT/'src/transcif/research/budget_transfer.py',
             ROOT/'src/transcif/research/cross_country.py', ROOT/'docs'/protocol]
    if args.family == 'structured':
        files.append(ROOT/'src/transcif/research/structured_transfer.py')
    code_provenance = {str(p.relative_to(ROOT)): digest(p) for p in files}
    (out/'startup_code_hashes.json').write_text(json.dumps(code_provenance, indent=2))
    raw, audit = load_inputs(ROOT)
    views = {r: {lag: monthly_inputs(f, lag) for lag in (2, 3, 4)} for r, f in raw.items()}
    records, seals, inventory, calibrations = [], [], [], {}
    for source, target_country in [('US', 'GB'), ('GB', 'US')]:
        sources = [r for r in raw if raw[r].country.iloc[0] == source]
        targets = [r for r in raw if raw[r].country.iloc[0] == target_country]
        excluded = []
        for r in list(sources):
            valid_sets = [set(f.index[f.year.eq(2022) & f[list(COARSE)].notna().all(axis=1)])
                          for f in views[r].values()]
            if not set.intersection(*valid_sets):
                sources.remove(r)
                excluded.append(r)
        (out/f'{source}_cohort.json').write_text(json.dumps({
            'source_regions': sources, 'excluded_no_common_source_hours': excluded,
            'target_regions': targets}, indent=2))
        baseline_frames = [budget_views(views[r], 2022, False) for r in sources]
        augmented_frames = [budget_views(views[r], 2022, True) for r in sources]
        train = pd.concat(baseline_frames, ignore_index=True)
        augmented = pd.concat(augmented_frames, ignore_index=True)
        fuel = np.concatenate([raw[r].loc[f.raw_row, list(FUELS)].to_numpy()
                               for r, f in zip(sources, baseline_frames)])
        cal = {}
        for r in sources:
            f = views[r][2]
            cal[r] = f.loc[f.year.eq(2023)].dropna(subset=list(COARSE)).copy()
            cal[r]['age_months'] = 2.
        target_frames = {}
        for r in targets:
            f = views[r][2]
            target_frames[r] = f.loc[f.year.eq(2024)].dropna(subset=list(COARSE)).copy()
            target_frames[r]['age_months'] = 2.
        # One unique copy for legacy baselines, same distinct supervision hours.
        unique = train.drop_duplicates(['region', 'raw_row'])
        unique_fuel = np.concatenate([raw[r].loc[unique.loc[unique.region.eq(r), 'raw_row'], list(FUELS)].to_numpy()
                                      for r in sources])
        legacy = fit_models(unique, unique_fuel)
        configurations = [('reference', -1, legacy)]
        if args.family == 'structured':
            from transcif.research.structured_transfer import (
                HEADS as selected_heads, predict_structured, save_structured, train_structured,
            )
            def fit_candidate(frame, fuel, head, seed):
                return train_structured(frame, fuel, head, seed, legacy)
            predict_candidate, save_candidate = predict_structured, save_structured
            training_views = [(False, train)]
        else:
            selected_heads = HEADS
            fit_candidate, predict_candidate, save_candidate = train_model, predict_model, save_model
            training_views = [(False, train), (True, augmented)]
        for augment, frame in training_views:
            for head in selected_heads:
                for seed in (412, 413, 414):
                    name = f'{"augmented" if augment else "fixed"}_{head}'
                    fitted = fit_candidate(frame, fuel, head, seed)
                    save_candidate(fitted, out/f'{source}_{name}_{seed}.pt')
                    inventory.append({'source': source, 'method': name, 'seed': seed,
                                      'unique_source_hours': len(unique),
                                      **{k: fitted[k] for k in ('parameters', 'last_loss', 'steps', 'batch')}})
                    configurations.append((name, seed, fitted))
                    print(source, name, seed, 'trained', flush=True)
        for name, seed, fitted in configurations:
            cal_predictions = {r: predict_models(fitted, f) if name == 'reference'
                               else {name: predict_candidate(fitted, f)} for r, f in cal.items()}
            methods = list(next(iter(cal_predictions.values())))
            for method in methods:
                residuals = [labels(raw[r].loc[f.index])-cal_predictions[r][method] for r, f in cal.items()]
                bias = float(np.mean([v.mean() for v in residuals]))
                sigma = max(1., float(np.sqrt(np.mean([np.mean((v-bias)**2) for v in residuals]))))
                key = f'{source}_{method}_{seed}'
                calibrations[key] = [bias, sigma]
                for r, f in target_frames.items():
                    p = predict_models(fitted, f)[method] if name == 'reference' else predict_candidate(fitted, f)
                    path = out/f'{key}_to_{r}.csv'
                    pd.DataFrame({'time': f.time.astype(str), 'point_raw': p}).to_csv(path, index=False)
                    seals.append({'path': path.name, 'sha256': digest(path)})
                    records.append((source, target_country, r, method, seed, f, path, bias, sigma))
    (out/'prediction_seal.json').write_text(json.dumps(seals, indent=2))
    (out/'source_calibration.json').write_text(json.dumps(calibrations, indent=2))
    (out/'training_inventory.json').write_text(json.dumps(inventory, indent=2))
    rows = []
    for source, target_country, r, method, seed, f, path, bias, sigma in records:
        pred = pd.read_csv(path).point_raw.to_numpy()
        y = labels(raw[r].loc[f.index])
        for month in sorted(f.clock.dt.month.unique()):
            mask = f.clock.dt.month.eq(month).to_numpy()
            rows.append({'source': source, 'target': target_country, 'region': r,
                         'method': method, 'seed': seed, 'month': int(month),
                         'common_month': month in COMMON,
                         'raw_mae': float(np.abs(y[mask]-pred[mask]).mean()),
                         **metrics(f.loc[mask], y[mask], pred[mask], bias, sigma)})
    scores = pd.DataFrame(rows)
    scores.to_csv(out/'monthly_metrics.csv', index=False)
    numeric = ['raw_mae', 'mae', 'shape_mae', 'crps_gaussian', 'coverage90', 'width90', 'interval_score90']
    regional = scores.loc[scores.common_month].groupby(['source', 'target', 'region', 'method', 'seed'])
    summary = regional[numeric].mean().reset_index()
    counts = regional.hours.sum().reset_index(name='hours_total')
    summary = summary.merge(counts)
    summary.to_csv(out/'common_region_metrics.csv', index=False)
    national = summary.groupby(['source', 'target', 'method', 'seed'])
    country = national[numeric].mean().reset_index()
    country = country.merge(national.agg(hours_total=('hours_total', 'sum'), regions=('region', 'nunique')).reset_index())
    country.to_csv(out/'common_country_metrics.csv', index=False)
    if any(digest(ROOT/key) != value for key, value in code_provenance.items()):
        raise RuntimeError('code or protocol changed during execution')
    provenance = dict(code_provenance)
    for a in audit.values():
        provenance[a['path']] = digest(ROOT/a['path'])
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(country.groupby(['target', 'method'])[numeric].mean().to_string())


if __name__ == '__main__':
    main()
