"""Select cross-country transfer references using nested pseudo-target countries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transcif.research.multicountry import (
    FACTORS, FINE, HISTORY, METHODS, coarse_values, fit, predict, public_features,
    public_months, weights,
)
from transcif.research.monthly_adaptation import weighted_quantile
from run_energy_charts_loco import COUNTRIES, DATA, PROTOCOL as DATA_PROTOCOL, load_country

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/PSEUDOTARGET_LOCO_SELECTION_PROTOCOL_20261101.md'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shape_mae(clock: pd.Series, y: np.ndarray, p: np.ndarray) -> float:
    sample = pd.DataFrame({'day': clock.dt.strftime('%Y-%m-%d').to_numpy(), 'y': y, 'p': p})
    centered = sample[['y', 'p']] - sample.groupby('day')[['y', 'p']].transform('mean')
    return float(np.abs(centered.y - centered.p).mean())


def aligned_arrays(prepared: dict, countries: list[str], predicate) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    parts = [prepared[c][1].loc[predicate(prepared[c][1])] for c in countries]
    frame = pd.concat(parts, ignore_index=True)
    fuel = np.concatenate([
        prepared[c][0].iloc[part.raw_row.to_numpy()][list(FINE)].to_numpy(float)
        for c, part in zip(countries, parts)
    ])
    coarse = np.concatenate([
        coarse_values(prepared[c][0].iloc[part.raw_row.to_numpy()])
        for c, part in zip(countries, parts)
    ])
    if len(frame) != len(fuel) or len(fuel) != len(coarse) or frame.empty:
        raise ValueError('empty or misaligned pseudo-target cohort')
    return frame, fuel, coarse


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = DATA / 'acquisition_manifest.json'
    audit_path = DATA / 'audit.json'
    code_files = [Path(__file__), ROOT / 'scripts/run_energy_charts_loco.py',
                  ROOT / 'src/transcif/research/multicountry.py',
                  ROOT / 'src/transcif/research/cross_country.py',
                  ROOT / 'src/transcif/research/monthly_adaptation.py', PROTOCOL, DATA_PROTOCOL]
    code_hashes = {str(p): sha(p) for p in code_files}
    inputs = {str(DATA / f'{c}_2022_2024.json'): sha(DATA / f'{c}_2022_2024.json')
              for c in COUNTRIES}
    inputs[str(manifest_path)] = sha(manifest_path)
    inputs[str(audit_path)] = sha(audit_path)
    (out / 'startup_hashes.json').write_text(json.dumps(
        {'code': code_hashes, 'inputs': inputs}, indent=2))

    manifest = json.loads(manifest_path.read_text())
    prepared, data_audit = {}, {}
    for country in COUNTRIES:
        raw, data_audit[country] = load_country(country, manifest)
        features = public_features(public_months(raw), raw.clock)
        features['country'], features['region'] = country.upper(), country.upper()
        features['year'] = raw.clock.dt.year.to_numpy()
        features['raw_row'] = np.arange(len(raw))
        prepared[country] = (raw, features.dropna(subset=list(HISTORY)).copy())
    (out / 'input_audit.json').write_text(json.dumps(data_audit, indent=2, ensure_ascii=False))

    selections, prediction_rows, seals = {}, {}, []
    for outer_target in COUNTRIES:
        source_countries = [c for c in COUNTRIES if c != outer_target]
        pseudo_metrics = []
        # Each source country is a whole pseudo-target. Its hourly labels are
        # excluded from that fold's fit and used only to score model selection.
        for pseudo_target in source_countries:
            pseudo_sources = [c for c in source_countries if c != pseudo_target]
            train, train_fuel, train_coarse = aligned_arrays(
                prepared, pseudo_sources,
                lambda f: f.year.eq(2022))
            model = fit(train, train_fuel, train_coarse)
            pseudo_raw, pseudo_all = prepared[pseudo_target]
            query = pseudo_all.loc[pseudo_all.year.eq(2023) & pseudo_all.clock.dt.month.le(6)].copy()
            if query.empty:
                raise ValueError(f'empty pseudo-target validation cohort: {pseudo_target}')
            p = predict(model, query)
            truth = pseudo_raw.iloc[query.raw_row.to_numpy()][list(FINE)].to_numpy(float)
            y = truth @ FACTORS / truth.sum(axis=1)
            for method in METHODS:
                pseudo_metrics.append({
                    'outer_target': outer_target.upper(),
                    'pseudo_target': pseudo_target.upper(), 'method': method,
                    'mae': float(np.abs(y - p[method]).mean()),
                    'shape_mae': shape_mae(query.clock, y, p[method]),
                    'hours': len(query),
                })
            print(outer_target.upper(), 'pseudo-heldout', pseudo_target.upper(),
                  'scored', len(query), flush=True)
        inner = pd.DataFrame([r for r in pseudo_metrics if r['outer_target'] == outer_target.upper()])
        inner_country = inner.groupby(['pseudo_target', 'method'])[['mae', 'shape_mae']].mean().reset_index()
        selection_score = inner_country.groupby('method').mae.mean().sort_values(kind='mergesort')
        selected = str(selection_score.index[0])
        selections[outer_target] = {
            'selected_method': selected,
            'inner_macro_mae': {k: float(v) for k, v in selection_score.items()},
            'inner_country_metrics': inner_country.to_dict('records'),
        }

        train, train_fuel, train_coarse = aligned_arrays(
            prepared, source_countries, lambda f: f.year.eq(2022))
        model = fit(train, train_fuel, train_coarse)
        cal, cal_fuel, _ = aligned_arrays(
            prepared, source_countries,
            lambda f: f.year.eq(2023) & f.clock.dt.month.le(6))
        cal_y = cal_fuel @ FACTORS / cal_fuel.sum(axis=1)
        cal_p = predict(model, cal)
        cal_weights = weights(cal)
        radii = {m: weighted_quantile(np.abs(cal_y - cal_p[m]), cal_weights, .9)
                 for m in METHODS}
        target_raw, target_all = prepared[outer_target]
        query = target_all.loc[target_all.year.eq(2024)].copy()
        p = predict(model, query)
        p['pseudo_selected'] = p[selected]
        path = out / f'{outer_target.upper()}_heldout_predictions.csv.gz'
        pd.DataFrame({
            'time': target_raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
            **p,
        }).to_csv(path, index=False, compression='gzip')
        seals.append({'target': outer_target.upper(), 'path': path.name,
                      'sha256': sha(path), 'hours': len(query),
                      'selected_method': selected})
        prediction_rows[outer_target] = (query, target_raw, path, radii, selected)
        print(outer_target.upper(), 'outer prediction sealed', len(query),
              'selected', selected, flush=True)

    (out / 'pseudo_target_validation.csv').write_text(
        pd.DataFrame(pseudo_metrics).to_csv(index=False))
    (out / 'selection.json').write_text(json.dumps(selections, indent=2))
    (out / 'prediction_seal.json').write_text(json.dumps(seals, indent=2))
    if any(sha(Path(p)) != value for p, value in inputs.items()):
        raise RuntimeError('frozen input changed during execution')
    if any(sha(Path(p)) != value for p, value in code_hashes.items()):
        raise RuntimeError('frozen code/protocol changed during execution')

    rows = []
    for target, (query, raw, path, radii, selected) in prediction_rows.items():
        truth = raw.iloc[query.raw_row.to_numpy()][list(FINE)].to_numpy(float)
        y = truth @ FACTORS / truth.sum(axis=1)
        scores = pd.read_csv(path)
        for method in (*METHODS, 'pseudo_selected'):
            radius = float(radii[selected] if method == 'pseudo_selected' else radii[method])
            point = scores[method].to_numpy(float)
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                error = y[mask] - point[mask]
                rows.append({
                    'target_country': target.upper(), 'method': method,
                    'selected_reference': selected, 'month': int(month),
                    'hours': int(mask.sum()), 'mae': float(np.abs(error).mean()),
                    'shape_mae': shape_mae(query.loc[mask, 'clock'], y[mask], point[mask]),
                    'coverage90': float((np.abs(error) <= radius).mean()),
                    'width90': float(2 * radius),
                    'interval_score90': float(np.mean(
                        2 * radius + 20 * np.maximum(np.abs(error) - radius, 0))),
                })
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / 'monthly_metrics.csv', index=False)
    columns = ['mae', 'shape_mae', 'coverage90', 'width90', 'interval_score90']
    country = monthly.groupby(['target_country', 'method'])[columns].mean().reset_index()
    country.to_csv(out / 'country_metrics.csv', index=False)
    country.groupby('method')[columns].mean().reset_index().to_csv(
        out / 'macro_metrics.csv', index=False)
    (out / 'status.json').write_text(json.dumps({
        'status': 'DEVELOPMENT_NESTED_PSEUDOTARGET_MODEL_SELECTION',
        'outer_targets': [c.upper() for c in COUNTRIES],
        'pseudo_target_inputs': 'local calendar plus t-2 and older coarse monthly generation shares',
        'pseudo_target_fit_labels': 'other source countries only, 2022',
        'pseudo_target_validation_labels': 'held-out source country 2023H1, selection only',
        'outer_target_labels_used_before_seal': False,
        'historical_as_issued_months': False,
        'actual_meta_learning_or_gradient_based_task_training': False,
    }, indent=2))
    print(pd.read_csv(out / 'macro_metrics.csv').to_string(index=False))


if __name__ == '__main__':
    main()
