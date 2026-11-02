#!/usr/bin/env python3
"""Evaluate Energy-Charts source training on a held-out SMARD Germany domain."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_energy_charts_loco as ec
from run_cross_country import ROOT
from transcif.data.smard import load_native
from transcif.research import multicountry as mc
from transcif.research.monthly_adaptation import weighted_quantile

PROTOCOL = ROOT / 'docs/ENERGY_CHARTS_TO_SMARD_GERMANY_HOLDOUT_PROTOCOL_20261101.md'
SOURCES = ('nl', 'es', 'be', 'at', 'se')
SMARD_MANIFESTS = [ROOT / 'results/smard-de-2022-full.json',
                   ROOT / 'results/smard-de-2023-full.json']


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smard_germany() -> tuple[pd.DataFrame, dict, list[Path]]:
    native, audit = load_native(SMARD_MANIFESTS, ROOT)
    for item in audit['years'].values():
        if item['hours'] != item['expected_hours'] or item['nonfinite'] or item['negative']:
            raise ValueError('SMARD DE raw data failed its year completeness/quality gate')
    frame = pd.DataFrame(index=native.index)
    frame['coal'] = native.lignite + native.hard_coal
    frame['gas'], frame['oil'] = native.gas, 0.
    frame['other'] = native.biomass + native.other_conventional
    frame['nuclear'], frame['hydro'] = native.nuclear, native.hydro
    frame['wind'] = native.wind_onshore + native.wind_offshore
    frame['solar'], frame['other_renewable'] = native.solar, native.other_renewable
    total_ex_storage = native.drop(columns='clock').sum(axis=1) - native.pumped_storage
    np.testing.assert_allclose(frame[list(mc.FINE)].sum(axis=1), total_ex_storage, atol=1e-6)
    frame['time'] = native.index
    frame['clock'] = native.clock
    frame['country'], frame['region'] = 'DE', 'DE'
    periods = frame.clock.dt.to_period('M')
    expected = {}
    for period in periods.unique():
        start = period.start_time.tz_localize('Europe/Berlin')
        end = (period + 1).start_time.tz_localize('Europe/Berlin')
        expected[period] = (end - start).total_seconds() / 3600
    frame['expected_hours'] = periods.map(expected).to_numpy(float)
    frame = frame.reset_index(drop=True)
    files = [*SMARD_MANIFESTS]
    for manifest in SMARD_MANIFESTS:
        body = json.loads(manifest.read_text())
        files.extend(ROOT / rec['path'] for rec in body['records'])
    return frame, audit, list(dict.fromkeys(files))


def features_for(raw: pd.DataFrame) -> pd.DataFrame:
    months = mc.public_months(raw)
    features = mc.public_features(months, raw.clock)
    features['country'], features['region'] = raw.country.to_numpy(), raw.region.to_numpy()
    features['year'] = raw.clock.dt.year.to_numpy()
    features['raw_row'] = np.arange(len(raw))
    return features.dropna(subset=list(mc.HISTORY)).copy().reset_index(drop=True)


def response_for(raw: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    fuel = raw.iloc[frame.raw_row.to_numpy()][list(mc.FINE)].to_numpy(float)
    total = fuel.sum(axis=1)
    if (total <= 0).any() or not np.isfinite(fuel).all():
        raise ValueError('invalid hourly proxy response rows')
    return fuel @ mc.FACTORS / total


def load_sources(manifest: dict) -> tuple[dict, dict]:
    prepared, audits = {}, {}
    for country in SOURCES:
        raw, audit = ec.load_country(country, manifest)
        prepared[country] = (raw, features_for(raw))
        audits[country] = audit
    return prepared, audits


def fit_source_models(prepared: dict) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    frames, fuels, coarses = [], [], []
    for country in SOURCES:
        raw, features = prepared[country]
        part = features.loc[features.year.eq(2022)].copy()
        frame_fuel = raw.iloc[part.raw_row.to_numpy()][list(mc.FINE)].to_numpy(float)
        frame_coarse = mc.coarse_values(raw.iloc[part.raw_row.to_numpy()])
        frames.append(part)
        fuels.append(frame_fuel)
        coarses.append(frame_coarse)
    train = pd.concat(frames, ignore_index=True)
    fuel, coarse = np.concatenate(fuels), np.concatenate(coarses)
    if len(train) != len(fuel) or len(train) != len(coarse) or train.empty:
        raise ValueError('source training arrays are empty or misaligned')
    return mc.fit(train, fuel, coarse), train, fuel, coarse


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)

    manifest_path = ec.DATA / 'acquisition_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    code = [Path(__file__), Path(ec.__file__), ROOT / 'src/transcif/data/smard.py',
            ROOT / 'src/transcif/research/multicountry.py',
            ROOT / 'src/transcif/research/cross_country.py',
            ROOT / 'src/transcif/research/monthly_adaptation.py', PROTOCOL]
    input_files = [manifest_path, *(ec.DATA / f'{c}_2022_2024.json' for c in SOURCES)]
    de, de_audit, smard_files = smard_germany()
    input_files += smard_files
    startup = {'code': {str(p): sha(p) for p in code},
               'inputs': {str(p): sha(p) for p in dict.fromkeys(input_files)}}
    (out / 'startup_hashes.json').write_text(json.dumps(startup, indent=2))

    source, source_audit = load_sources(manifest)
    de_features = features_for(de)
    prepared = {**source, 'de': (de, de_features)}
    (out / 'input_audit.json').write_text(json.dumps(
        {'sources': source_audit, 'DE': de_audit}, indent=2, ensure_ascii=False))
    (out / 'preseal_target_audit.json').write_text(json.dumps({
        'target': 'DE', 'target_year': '2023H2',
        'target_country_previously_viewed': True,
        'target_response_read_before_prediction_seal': False,
        'assessment': 'cross-platform development holdout, not blind confirmation',
    }, indent=2))

    models, train, _, _ = fit_source_models(source)
    cal_parts = []
    for country in SOURCES:
        _, features = source[country]
        cal_parts.append(features.loc[features.year.eq(2023)
                                     & features.clock.dt.month.le(6)].copy())
    calibration = pd.concat(cal_parts, ignore_index=True)
    cal_truth = np.concatenate([response_for(source[country][0], part)
                                for country, part in zip(SOURCES, cal_parts)])
    radii = {name: float(weighted_quantile(
        np.abs(cal_truth - value), mc.weights(calibration), .9))
        for name, value in mc.predict(models, calibration).items()}
    (out / 'source_calibration.json').write_text(json.dumps({
        name: {'radius90': value, 'hours': int(len(calibration)),
               'source_countries': list(SOURCES)} for name, value in radii.items()}, indent=2))

    target_query = de_features.loc[
        de_features.year.eq(2023) & de_features.clock.dt.month.ge(7)].copy().reset_index(drop=True)
    if target_query.empty or target_query.clock.dt.month.nunique() != 6:
        raise ValueError('Germany 2023H2 target window must contain all six months')
    target_predictions = mc.predict(models, target_query)
    pred_path = out / 'DE_2023H2_predictions.csv.gz'
    pd.DataFrame({'time_utc': de.iloc[target_query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                  'observed_month': target_query.period.astype(str).to_numpy(),
                  **target_predictions}).to_csv(pred_path, index=False, compression='gzip')
    seal = {'target': 'DE', 'period': '2023H2', 'path': pred_path.name,
            'sha256': sha(pred_path), 'hours': len(target_query),
            'source_countries': list(SOURCES)}
    (out / 'prediction_seal.json').write_text(json.dumps(seal, indent=2))
    if any(sha(Path(p)) != h for p, h in startup['code'].items()):
        raise RuntimeError('code/protocol changed during run')
    if any(sha(Path(p)) != h for p, h in startup['inputs'].items()):
        raise RuntimeError('input data changed during run')

    # Create the Germany response only after the target prediction has been sealed.
    truth = response_for(de, target_query)
    methods = pd.read_csv(pred_path)
    rows = []
    for name, radius in radii.items():
        point = methods[name].to_numpy(float)
        for month in sorted(target_query.clock.dt.month.unique()):
            mask = target_query.clock.dt.month.eq(month).to_numpy()
            error = truth[mask] - point[mask]
            daily = pd.DataFrame({'day': target_query.loc[mask, 'clock'].dt.strftime('%Y-%m-%d'),
                                  'truth': truth[mask], 'prediction': point[mask]})
            center = daily[['truth', 'prediction']] - daily.groupby('day')[
                ['truth', 'prediction']].transform('mean')
            rows.append({'target': 'DE', 'period': '2023H2', 'method': name,
                         'month': int(month), 'hours': int(mask.sum()),
                         'mae': float(np.abs(error).mean()),
                         'shape_mae': float(np.abs(center.truth - center.prediction).mean()),
                         'coverage90': float((np.abs(error) <= radius).mean()),
                         'width90': float(2 * radius),
                         'interval_score90': float(np.mean(
                             2 * radius + 20 * np.maximum(np.abs(error) - radius, 0)))})
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / 'monthly_metrics.csv', index=False)
    metric_cols = ['mae', 'shape_mae', 'coverage90', 'width90', 'interval_score90']
    monthly.groupby('method', as_index=False)[metric_cols].mean().to_csv(
        out / 'summary_metrics.csv', index=False)
    (out / 'target_scoring_hashes.json').write_text(json.dumps({
        str(p): startup['inputs'][str(p)] for p in smard_files if p.name.endswith('.json')
    }, indent=2))
    (out / 'status.json').write_text(json.dumps({
        'status': 'DEVELOPMENT_ENERGY_CHARTS_TO_SMARD_GERMANY_LOCO',
        'source_countries': list(SOURCES), 'training_year': 2022,
        'source_calibration_period': '2023H1', 'target_period': 'DE 2023H2',
        'target_country_previously_viewed': True, 'blind_confirmation': False,
        'monthly_input_delay_months': 2, 'historical_as_issued_vintages': False,
        'response': 'fixed-factor generation-intensity proxy',
    }, indent=2))
    print(monthly.groupby('method')[metric_cols].mean().to_string())


if __name__ == '__main__':
    main()
