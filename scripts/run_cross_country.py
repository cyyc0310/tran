"""Two-country delayed-month development pilot; predictions sealed before scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transcif.research.cross_country import (
    COARSE, FUELS, METHODS, fit_models, labels, load_inputs,
    metrics, monthly_inputs, predict_models,
)

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    raw, audit = load_inputs(ROOT)
    prepared = {r: monthly_inputs(f) for r, f in raw.items()}
    # Features constructed independently of contemporaneous target outcomes.
    usable = {r: f.dropna(subset=list(COARSE)).copy() for r, f in prepared.items()}
    for r, f in usable.items():
        f['raw_row'] = f.index
    protocols = [ROOT/'docs/CROSS_COUNTRY_PILOT_PROTOCOL.md', Path(__file__),
                 ROOT/'src/transcif/research/cross_country.py']
    provenance = {str(p.relative_to(ROOT)): digest(p) for p in protocols}
    for a in audit.values():
        provenance[a['path']] = digest(ROOT/a['path'])
    (out/'provenance.json').write_text(json.dumps({'sha256': provenance, 'audit': audit}, indent=2))
    all_rows, seals = [], []
    for source_country, target_country in [('US', 'GB'), ('GB', 'US')]:
        source_regions = [r for r in raw if raw[r].country.iloc[0] == source_country]
        target_regions = [r for r in raw if raw[r].country.iloc[0] == target_country]
        train = pd.concat([usable[r].loc[usable[r].year.eq(2022)] for r in source_regions])
        fuel = np.concatenate([raw[r].loc[usable[r].loc[usable[r].year.eq(2022), 'raw_row'], list(FUELS)].to_numpy(float)
                               for r in source_regions])
        if train.empty:
            raise ValueError('empty source training set')
        models = fit_models(train, fuel)
        residuals = {m: [] for m in METHODS}
        for r in source_regions:
            cal = usable[r].loc[usable[r].year.eq(2023)]
            pred = predict_models(models, cal)
            y = labels(raw[r].loc[cal.raw_row])
            for method in METHODS:
                residuals[method].append(y - pred[method])
        calibration = {}
        for method in METHODS:
            # Equal-region mixture moments, not pooled hourly weighting.
            parts = residuals[method]
            bias = float(np.mean([v.mean() for v in parts]))
            sigma = float(np.sqrt(np.mean([np.mean((v-bias)**2) for v in parts])))
            calibration[method] = (bias, max(sigma, 1.))
        (out/f'{source_country}_calibration.json').write_text(json.dumps(calibration, indent=2))
        for r in target_regions:
            target = usable[r].loc[usable[r].year.eq(2024)]
            prediction = predict_models(models, target)
            path = out/f'{source_country}_to_{r}_predictions.csv'
            pd.DataFrame({'time': target.time.astype(str),
                          'observed_month': target.observed_month, **prediction}).to_csv(path, index=False)
            seals.append({'path': path.name, 'sha256': digest(path)})
    (out/'prediction_seal.json').write_text(json.dumps(seals, indent=2))
    # Only after every direction's prediction is on disk, load held-out responses.
    for source_country, target_country in [('US', 'GB'), ('GB', 'US')]:
        calibration = json.loads((out/f'{source_country}_calibration.json').read_text())
        for r in raw:
            if raw[r].country.iloc[0] != target_country:
                continue
            target = usable[r].loc[usable[r].year.eq(2024)]
            predictions = pd.read_csv(out/f'{source_country}_to_{r}_predictions.csv')
            y = labels(raw[r].loc[target.raw_row])
            for method in METHODS:
                all_rows.append({'source': source_country, 'target': target_country,
                                 'region': r, 'method': method,
                                 **metrics(target, y, predictions[method].to_numpy(), *calibration[method])})
    scores = pd.DataFrame(all_rows)
    scores.to_csv(out/'region_metrics.csv', index=False)
    scores.groupby(['source', 'target', 'method']).mean(numeric_only=True).to_csv(out/'country_metrics.csv')
    (out/'status.json').write_text(json.dumps({'status': 'DEVELOPMENT_PROXY_TWO_COUNTRIES',
        'train_year': 2022, 'calibration_year': 2023, 'test_year': 2024,
        'target_hourly_inputs': False, 'release_delay_months': 2,
        'historical_release_vintages': False, 'new_blind_countries': False}, indent=2))
    print(scores.groupby(['target', 'method'])[['mae', 'shape_mae', 'crps_gaussian', 'coverage90', 'width90']].mean().to_string())


if __name__ == '__main__':
    main()
