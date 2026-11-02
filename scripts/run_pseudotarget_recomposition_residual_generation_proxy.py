#!/usr/bin/env python3
"""Paired generation-proxy response replication of the official CIF residual test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_official_intensity_transfer as official
import run_eia923_pseudotarget_transfer as pseudo
import run_official_pseudotarget_recomposition_residual as engine
from run_cross_country import ROOT
from transcif.research.cross_country import EF

PROTOCOL = ROOT / 'docs/PSEUDOTARGET_RECOMPOSITION_RESIDUAL_GENERATION_PROXY_PROTOCOL_20261101.md'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes() -> tuple[list[Path], list[Path]]:
    code = [Path(__file__), Path(engine.__file__), Path(official.__file__),
            Path(pseudo.__file__), ROOT / 'scripts/run_cross_country.py',
            ROOT / 'src/transcif/research/cross_country.py',
            ROOT / 'src/transcif/research/capacity_transfer.py',
            ROOT / 'scripts/prepare_official_intensity_panel.py', PROTOCOL,
            engine.PROTOCOL]
    capacity = official.capacity_transfer.load_capacity(official.CAPACITY_TABLE_PATH)
    data = [official.EIA_ROOT / 'acquisition_manifest.json',
            official.EIA_ROOT / 'coverage_audit_2022_2024.json',
            official.EIA_ROOT / 'feature_inputs/GB_generation_2022_2024.csv.gz',
            official.CAPACITY_TABLE_PATH, ROOT / 'data/neso-gb/df_fuel_ckan.csv']
    data += [ROOT / f'results/research/data/{r}_hourly.csv' for r in official.US_REGIONS]
    data += [pseudo.eia_archive(y) for y in range(2020, 2025)]
    data += [ROOT / name for name in capacity['input_hashes']]
    return code, list(dict.fromkeys(data))


def proxy_label(frame: pd.DataFrame) -> np.ndarray:
    fuel = frame[list(official.FUELS)].to_numpy(float)
    total = fuel.sum(axis=1)
    if (total <= 0).any() or not np.isfinite(fuel).all():
        raise ValueError('proxy response requires finite positive generation')
    return fuel @ EF / total


def source_proxy_labels(raw: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    labels = {}
    for region in (*official.US_REGIONS, 'GB'):
        frame = raw[region]
        times = pd.DatetimeIndex(pd.to_datetime(frame.time, utc=True, errors='raise'))
        y = proxy_label(frame)
        keep = times.year.isin([2022, 2023])
        labels[region] = pd.Series(y[keep], index=times[keep], name='label').sort_index()
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    capacity = official.capacity_transfer.load_capacity(official.CAPACITY_TABLE_PATH)
    for name, digest in capacity['input_hashes'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'annual capacity input changed: {name}')
    code, data = hashes()
    startup = {'code': {str(p): sha(p) for p in code},
               'inputs': {str(p): sha(p) for p in data}}
    (out / 'startup_hashes.json').write_text(json.dumps(startup, indent=2))

    # The target response is not computed here. prepare() supplies only lagged
    # features; source proxy labels are restricted to 2022/2023.
    original_source_labels = official.source_labels
    official.source_labels = lambda region, years=None: pd.Series(
        [], index=pd.DatetimeIndex([], tz='UTC'), dtype=float, name='label')
    try:
        raw, base_prepared, _, audit = official.prepare(ROOT, capacity)
    finally:
        official.source_labels = original_source_labels
    labels = source_proxy_labels(raw)
    monthly = pseudo.parse_monthly(tuple(range(2020, 2025)), official.US_REGIONS)
    prepared, support = pseudo.replace_source_features(base_prepared, monthly)
    monthly.to_csv(out / 'eia923_monthly_ba_mix.csv', index=False)
    (out / 'input_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    (out / 'common_support.json').write_text(json.dumps(support, indent=2))
    (out / 'preseal_target_audit.json').write_text(json.dumps({
        'target_year': 2024, 'target_proxy_label_computed_before_seal': False,
        'target_country_previously_viewed_in_development': True,
        'response': 'fixed-factor hourly domestic-generation proxy',
    }, indent=2))

    validation, selections, seals = [], [], []
    for arm in ('hourly_reconstruction', 'eia923_monthly'):
        rows, picked = engine.evaluate_arm(arm, prepared[arm], raw, labels, out)
        validation.extend(rows)
        selections.extend(picked)
        seals.append(json.loads((out / f'{arm}_prediction_seal.json').read_text()))
    pd.DataFrame(validation).to_csv(out / 'source_h1_validation.csv', index=False)
    (out / 'prediction_seal.json').write_text(json.dumps(seals, indent=2))
    if any(sha(Path(p)) != h for p, h in startup['code'].items()):
        raise RuntimeError('code/protocol changed during run')
    if any(sha(Path(p)) != h for p, h in startup['inputs'].items()):
        raise RuntimeError('input data changed during run')

    # Only now derive the held-out 2024 generation-proxy response.
    target = raw['GB'].copy()
    target_times = pd.DatetimeIndex(pd.to_datetime(target.time, utc=True, errors='raise'))
    keep = target_times.year == 2024
    target_proxy = pd.Series(proxy_label(target.loc[keep]), index=target_times[keep], name='label')
    target_raw_path = ROOT / 'data/neso-gb/df_fuel_ckan.csv'
    (out / 'target_scoring_hash.json').write_text(json.dumps(
        {str(target_raw_path): sha(target_raw_path)}, indent=2))

    metrics = []
    for arm in ('hourly_reconstruction', 'eia923_monthly'):
        seal = json.loads((out / f'{arm}_prediction_seal.json').read_text())
        frame = prepared[arm]['GB'].loc[prepared[arm]['GB'].year.eq(2024)].copy().reset_index(drop=True)
        truth = official.aligned_labels(frame, target_proxy)
        prediction = pd.read_csv(out / seal['path'])
        calibration_info = json.loads((out / f'{arm}_source_calibration.json').read_text())
        for month in sorted(frame.clock.dt.month.unique()):
            mask = frame.clock.dt.month.eq(month).to_numpy()
            for method in [*engine.OUTER_METHODS,
                           *(f'pseudo_residual_{c}' for c in engine.CORRECTORS)]:
                result = official.score(
                    truth[mask], prediction.loc[mask, method].to_numpy(float),
                    calibration_info[method]['bias'], calibration_info[method]['sigma'],
                    frame.loc[mask, 'clock'].reset_index(drop=True))
                metrics.append({'arm': arm, 'method': method, 'month': int(month), **result})
    monthly_metrics = pd.DataFrame(metrics)
    monthly_metrics.to_csv(out / 'monthly_metrics.csv', index=False)
    score_cols = ['mae', 'shape_mae', 'crps_gaussian', 'coverage90', 'width90', 'interval_score90']
    monthly_metrics.groupby(['arm', 'method'], as_index=False)[score_cols].mean().to_csv(
        out / 'summary_metrics.csv', index=False)
    (out / 'status.json').write_text(json.dumps({
        'status': 'DEVELOPMENT_PSEUDOTARGET_GENERATION_PROXY_RESIDUAL',
        'source_train_year': 2022, 'source_select_year': '2023H1',
        'source_calibration_year': '2023H2', 'target_year': 2024,
        'target_label_previously_viewed': True,
        'target_proxy_computed_after_seal': True,
        'response': 'fixed-factor hourly domestic-generation proxy',
        'target_input_delay_months': 2,
    }, indent=2))
    print(pd.read_csv(out / 'summary_metrics.csv').to_string(index=False))


if __name__ == '__main__':
    main()
