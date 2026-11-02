"""Temporal development check for five-group factor transfer (2023 H2)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_episodic_coldstart_transfer import (
    COUNTRIES, DATA, make_panel, panel_labels,
)
from run_energy_charts_loco import load_country
from run_five_group_capacity_increment import (
    CAPACITY, CAPACITY_MANIFEST, PROTOCOL as CAPACITY_PROTOCOL,
    capacity_join, fit_heads, predict_heads, recomposition_reference, sha,
    region_equal_weights,
)
from transcif.research.multicountry import FINE
from transcif.research.monthly_adaptation import weighted_quantile

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/FIVE_GROUP_FACTOR_TEMPORAL_CHECK_PROTOCOL_20261101.md'


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = DATA / 'acquisition_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    capacity = pd.read_csv(CAPACITY)
    raw, panels, audits = {}, {}, {}
    input_paths = [DATA / f'{c}_2022_2024.json' for c in COUNTRIES]
    input_paths += [manifest_path, DATA / 'audit.json', CAPACITY, CAPACITY_MANIFEST]
    code_paths = [Path(__file__), PROTOCOL, CAPACITY_PROTOCOL,
                  ROOT / 'scripts/run_episodic_coldstart_transfer.py',
                  ROOT / 'scripts/run_energy_charts_loco.py',
                  ROOT / 'scripts/run_five_group_capacity_increment.py',
                  ROOT / 'src/transcif/research/multicountry.py',
                  ROOT / 'src/transcif/research/monthly_adaptation.py']
    hashes = {'inputs': {str(p): sha(p) for p in input_paths},
              'code': {str(p): sha(p) for p in code_paths}}
    for c in COUNTRIES:
        raw[c.upper()], audits[c] = load_country(c, manifest)
        panel, _ = make_panel(raw[c.upper()])
        panel['country'] = c.upper()
        panels[c.upper()] = capacity_join(panel, capacity)
    (out / 'startup_hashes.json').write_text(json.dumps(hashes, indent=2))
    (out / 'input_audit.json').write_text(json.dumps(audits, indent=2, default=str))

    predictions, seals, rows = [], [], []
    for target in COUNTRIES:
        target = target.upper()
        sources = sorted(c for c in panels if c != target)
        train_parts = [panels[c].loc[panels[c].year.eq(2022)] for c in sources]
        train = pd.concat(train_parts, ignore_index=True)
        train_fuel = np.concatenate([
            raw[c].iloc[panels[c].loc[panels[c].year.eq(2022), 'raw_row'].to_numpy()][
                list(FINE)].to_numpy(float) for c in sources])
        train_y = np.concatenate([
            panel_labels(raw[c], panels[c].loc[panels[c].year.eq(2022)].reset_index(drop=True))
            for c in sources])
        cal_parts = [panels[c].loc[panels[c].year.eq(2023) &
                                   panels[c].clock.dt.month.le(6)] for c in sources]
        calibration = pd.concat(cal_parts, ignore_index=True)
        cal_y = np.concatenate([
            panel_labels(raw[c], panels[c].loc[panels[c].year.eq(2023) &
                panels[c].clock.dt.month.le(6)].reset_index(drop=True)) for c in sources])
        query = panels[target].loc[panels[target].year.eq(2023) &
                                   panels[target].clock.dt.month.ge(7)].reset_index(drop=True)
        if query.empty:
            raise ValueError(f'No 2023H2 target rows for {target}')
        cal_w = region_equal_weights(calibration)
        cal_predictions = {'five_group_recomposition':
                           recomposition_reference(train, train_fuel, calibration)}
        test_predictions = {'five_group_recomposition':
                            recomposition_reference(train, train_fuel, query)}
        for budget, has_capacity in (('M', False), ('MplusC', True)):
            models = fit_heads(train, train_y, train_fuel, has_capacity)
            cal = predict_heads(models, calibration, train, train_fuel)
            test = predict_heads(models, query, train, train_fuel)
            for method in ('direct', 'factor', 'ridge'):
                name = f'{budget}_{method}'
                cal_predictions[name] = cal[method]
                test_predictions[name] = test[method]
        if any(not np.isfinite(v).all() for v in test_predictions.values()):
            raise ValueError(f'Nonfinite target predictions for {target}')
        radii = {name: float(weighted_quantile(np.abs(cal_y - pred), cal_w, .9))
                 for name, pred in cal_predictions.items()}
        path = out / f'{target}_heldout_predictions.csv.gz'
        pd.DataFrame({'time': query.clock.astype(str).to_numpy(), **test_predictions}).to_csv(
            path, index=False, compression='gzip')
        radius_path = out / f'{target}_source_radii.json'
        radius_path.write_text(json.dumps(radii, indent=2))
        seals.append({'target_country': target, 'prediction_file': path.name,
                      'prediction_sha256': sha(path), 'radii_file': radius_path.name,
                      'radii_sha256': sha(radius_path), 'hours': len(query)})
        predictions.append((target, query, path, radii))
        if any(sha(Path(p)) != v for p, v in hashes['inputs'].items()):
            raise RuntimeError('Input changed during model fit')
        if any(sha(Path(p)) != v for p, v in hashes['code'].items()):
            raise RuntimeError('Code/protocol changed during model fit')
        print(f'{target}: sealed {len(query)} target hours', flush=True)
    (out / 'prediction_seal.json').write_text(json.dumps(seals, indent=2))

    for target, query, path, radii in predictions:
        y = panel_labels(raw[target], query)
        forecast = pd.read_csv(path)
        for method, radius in radii.items():
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                from run_five_group_capacity_increment import score
                rows.append({'target_country': target, 'method': method, 'month': int(month),
                    **score(y[mask], forecast.loc[mask, method].to_numpy(float), radius,
                            query.loc[mask, 'clock'].reset_index(drop=True))})
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / 'monthly_metrics.csv', index=False)
    metrics = ['mae', 'shape_mae', 'coverage90', 'width90', 'interval_score90']
    country = monthly.groupby(['target_country', 'method'])[metrics].mean().reset_index()
    country.to_csv(out / 'country_metrics.csv', index=False)
    country.groupby('method')[metrics].mean().reset_index().to_csv(out / 'macro_metrics.csv', index=False)
    (out / 'status.json').write_text(json.dumps({
        'status': 'DEVELOPMENT_FIVE_GROUP_TEMPORAL_HOLDOUT',
        'target_countries': list(COUNTRIES), 'training_year': 2022,
        'source_calibration': '2023H1', 'target_period': '2023H2',
        'target_budget_M': 'five groups t-2 through t-7, calendar and hourly interactions',
        'target_budget_MplusC': 'M plus two-year-lagged annual thermal capacity share',
        'capacity_vintage': 'current revised extract, lagged statistical simulation',
        'target_labels_used_before_seal': False,
        'all_countries_previously_viewed': True,
    }, indent=2))
    print(country.groupby('method')[metrics].mean().to_string())


if __name__ == '__main__':
    main()
