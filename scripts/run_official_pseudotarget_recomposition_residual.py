#!/usr/bin/env python3
"""Cross-fitted BA pseudo-target residual correction for US-to-GB CIF transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_eia923_pseudotarget_transfer as pseudo
import run_official_intensity_transfer as official
from run_cross_country import ROOT

PROTOCOL = ROOT / 'docs/OFFICIAL_PSEUDOTARGET_RECOMPOSITION_RESIDUAL_PROTOCOL_20261101.md'
OUTER_METHODS = tuple(official.METHODS)
CORRECTORS = ('residual_ridge', 'residual_hgb')
SHRINKS = (0., .1, .25, .5, .75, 1.)
RESIDUAL_FEATURES = (*official.FEATURES, 'base_prediction')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths() -> tuple[list[Path], list[Path]]:
    code = [Path(__file__), Path(official.__file__), Path(pseudo.__file__),
            ROOT / 'scripts/run_cross_country.py',
            ROOT / 'src/transcif/research/cross_country.py',
            ROOT / 'src/transcif/research/capacity_transfer.py',
            ROOT / 'scripts/prepare_official_intensity_panel.py', PROTOCOL]
    capacity = official.capacity_transfer.load_capacity(official.CAPACITY_TABLE_PATH)
    data = [official.EIA_ROOT / 'acquisition_manifest.json',
            official.EIA_ROOT / 'coverage_audit_2022_2024.json', official.CAPACITY_TABLE_PATH,
            official.SOURCE_LABEL_ROOT / 'GB_consumption_2022_2023_halfhour.csv.gz',
            official.EIA_ROOT / 'feature_inputs/GB_generation_2022_2024.csv.gz']
    data += [official.SOURCE_LABEL_ROOT / f'{r}_consumption_2022_2023.csv.gz'
             for r in official.US_REGIONS]
    data += [ROOT / f'results/research/data/{r}_hourly.csv' for r in official.US_REGIONS]
    data += [pseudo.eia_archive(y) for y in range(2020, 2025)]
    data += [ROOT / name for name in capacity['input_hashes']]
    return code, list(dict.fromkeys(data))


def align_response(frame: pd.DataFrame, labels: dict[str, pd.Series]) -> tuple[pd.DataFrame, np.ndarray]:
    value = official.aligned_labels(frame, labels[frame.region.iloc[0]])
    keep = np.isfinite(value)
    return frame.loc[keep].copy().reset_index(drop=True), value[keep]


def fuel_for(frame: pd.DataFrame, raw: dict[str, pd.DataFrame]) -> np.ndarray:
    parts = []
    for region in frame.region.unique():
        q = frame.loc[frame.region.eq(region)]
        parts.append(raw[region].iloc[q.raw_row.to_numpy()][list(official.FUELS)].to_numpy(float))
    return np.concatenate(parts)


def training_rows(regions: list[str], prepared: dict[str, pd.DataFrame],
                  labels: dict[str, pd.Series], year: int) -> tuple[pd.DataFrame, np.ndarray]:
    frames, values = [], []
    for region in regions:
        frame = prepared[region].loc[prepared[region].year.eq(year)].copy()
        frame, response = align_response(frame, labels)
        frames.append(frame)
        values.append(response)
    if not frames:
        raise ValueError('empty source training group')
    return pd.concat(frames, ignore_index=True), np.concatenate(values)


def fit_reference(regions: list[str], prepared: dict[str, pd.DataFrame],
                  raw: dict[str, pd.DataFrame], labels: dict[str, pd.Series]) -> tuple[dict, dict]:
    frame, response = training_rows(regions, prepared, labels, 2022)
    response_models = official.train_official(frame, response)
    fuel_models = official.fit_models(frame.reset_index(drop=True), fuel_for(frame, raw))
    return response_models, fuel_models


def recompose(fuel_models: dict, frame: pd.DataFrame) -> np.ndarray:
    return official.predict_models(fuel_models, frame)['coarse_recomposition']


def pseudo_residual_data(regions: list[str], prepared: dict[str, pd.DataFrame],
                         raw: dict[str, pd.DataFrame], labels: dict[str, pd.Series]
                         ) -> tuple[pd.DataFrame, np.ndarray]:
    """Create cross-fitted 2022 residual labels, each from other BAs only."""
    meta_frames, residuals = [], []
    for pseudo_ba in regions:
        train_ba = [r for r in regions if r != pseudo_ba]
        if len(train_ba) < 2:
            raise ValueError('residual pseudo targets require at least two training BAs')
        _, fuel_models = fit_reference(train_ba, prepared, raw, labels)
        query = prepared[pseudo_ba].loc[prepared[pseudo_ba].year.eq(2022)].copy()
        query, response = align_response(query, labels)
        base = recompose(fuel_models, query)
        query['base_prediction'] = base
        meta_frames.append(query)
        residuals.append(response - base)
    return pd.concat(meta_frames, ignore_index=True), np.concatenate(residuals)


def fit_corrector(frame: pd.DataFrame, residual: np.ndarray, family: str):
    x = frame[list(RESIDUAL_FEATURES)]
    weights = official.region_weights(frame)
    if family == 'residual_ridge':
        return make_pipeline(StandardScaler(), Ridge(alpha=10.)).fit(
            x, residual, standardscaler__sample_weight=weights,
            ridge__sample_weight=weights * len(weights))
    if family == 'residual_hgb':
        return HistGradientBoostingRegressor(
            max_iter=80, max_leaf_nodes=15, min_samples_leaf=100,
            l2_regularization=10., random_state=417, early_stopping=False,
        ).fit(x, residual, sample_weight=weights * len(weights))
    raise ValueError(f'unknown residual learner: {family}')


def correct(model, frame: pd.DataFrame, base: np.ndarray) -> np.ndarray:
    query = frame.copy()
    query['base_prediction'] = base
    return np.asarray(model.predict(query[list(RESIDUAL_FEATURES)]), dtype=float)


def calibration(errors: dict[str, list[np.ndarray]]) -> dict[str, dict]:
    output = {}
    for method, parts in errors.items():
        bias = float(np.mean([v.mean() for v in parts]))
        sigma = max(float(np.sqrt(np.mean([np.mean((v - bias) ** 2) for v in parts]))), 1.)
        output[method] = {'bias': bias, 'sigma': sigma,
                          'regions': len(parts), 'calibration_hours': int(sum(map(len, parts)))}
    return output


def macro_mae(rows: list[dict], method: str, family: str | None = None,
              shrink: float | None = None) -> float:
    selected = [r for r in rows if r['method'] == method
                and (family is None or r.get('family') == family)
                and (shrink is None or r.get('shrink') == shrink)]
    by_ba = {}
    for row in selected:
        by_ba.setdefault(row['pseudo_ba'], []).append(row['mae'])
    return float(np.mean([np.mean(values) for values in by_ba.values()]))


def evaluate_arm(arm: str, prepared: dict[str, pd.DataFrame], raw: dict[str, pd.DataFrame],
                 labels: dict[str, pd.Series], out: Path) -> tuple[list[dict], list[dict]]:
    bas = list(official.US_REGIONS)
    validation_rows, calibration_errors = [], {method: [] for method in OUTER_METHODS}
    calibration_parts: dict[str, list[np.ndarray]] = {m: [] for m in OUTER_METHODS}
    calibration_records = []

    # Geographic H1 validation: held-out BA is absent from its residual tasks and base.
    for held in bas:
        train_ba = [r for r in bas if r != held]
        meta, residual = pseudo_residual_data(train_ba, prepared, raw, labels)
        fitted = {family: fit_corrector(meta, residual, family) for family in CORRECTORS}
        response_models, fuel_models = fit_reference(train_ba, prepared, raw, labels)
        valid = prepared[held].loc[prepared[held].year.eq(2023)
                                   & prepared[held].clock.dt.month.le(6)].copy()
        valid, truth = align_response(valid, labels)
        predictions = official.point_predictions(response_models, fuel_models, valid)
        base = predictions['fixed_ef_recomposition']
        for method in OUTER_METHODS:
            point = predictions[method]
            for month in sorted(valid.clock.dt.month.unique()):
                mask = valid.clock.dt.month.eq(month).to_numpy()
                validation_rows.append({'arm': arm, 'pseudo_ba': held, 'method': method,
                                        'family': '', 'shrink': 0., 'month': int(month),
                                        'mae': float(np.abs(truth[mask] - point[mask]).mean()),
                                        'hours': int(mask.sum())})
        for family, model in fitted.items():
            correction = correct(model, valid, base)
            for shrink in SHRINKS:
                point = np.clip(base + shrink * correction, 0., 1000.)
                for month in sorted(valid.clock.dt.month.unique()):
                    mask = valid.clock.dt.month.eq(month).to_numpy()
                    validation_rows.append({'arm': arm, 'pseudo_ba': held,
                                            'method': 'pseudo_residual', 'family': family,
                                            'shrink': shrink, 'month': int(month),
                                            'mae': float(np.abs(truth[mask] - point[mask]).mean()),
                                            'hours': int(mask.sum())})
        print(arm, 'H1 held BA', held, 'validated', len(valid), flush=True)

    selection_rows = []
    for family in CORRECTORS:
        scores = [(macro_mae(validation_rows, 'pseudo_residual', family, shrink), shrink)
                  for shrink in SHRINKS]
        selected = min(scores, key=lambda pair: (pair[0], pair[1]))[1]
        selection_rows.append({'arm': arm, 'family': family, 'selected_shrink': selected,
                               'source_h1_macro_mae': min(score for score, _ in scores),
                               'candidate_scores': {str(shrink): score for score, shrink in scores}})

    # Independent H2 cross-fitted calibration by BA, using the H1-selected correction.
    for held in bas:
        train_ba = [r for r in bas if r != held]
        meta, residual = pseudo_residual_data(train_ba, prepared, raw, labels)
        corrections = {family: fit_corrector(meta, residual, family) for family in CORRECTORS}
        response_models, fuel_models = fit_reference(train_ba, prepared, raw, labels)
        valid = prepared[held].loc[prepared[held].year.eq(2023)
                                   & prepared[held].clock.dt.month.gt(6)].copy()
        valid, truth = align_response(valid, labels)
        predictions = official.point_predictions(response_models, fuel_models, valid)
        for method in OUTER_METHODS:
            calibration_parts[method].append(truth - predictions[method])
        base = predictions['fixed_ef_recomposition']
        for item in selection_rows:
            family, shrink = item['family'], item['selected_shrink']
            correction = correct(corrections[family], valid, base)
            point = np.clip(base + shrink * correction, 0., 1000.)
            name = f'pseudo_residual_{family}'
            calibration_parts.setdefault(name, []).append(truth - point)
        calibration_records.append({'arm': arm, 'held_ba': held, 'h2_hours': len(valid)})
        print(arm, 'H2 calibration BA', held, 'calibrated', len(valid), flush=True)

    calibrated = calibration(calibration_parts)
    selection_map = {row['family']: row['selected_shrink'] for row in selection_rows}

    # Final all-source base and residual model use only 2022 labels.
    meta, residual = pseudo_residual_data(bas, prepared, raw, labels)
    final_corrections = {family: fit_corrector(meta, residual, family) for family in CORRECTORS}
    train, y = training_rows(bas, prepared, labels, 2022)
    response_models = official.train_official(train, y)
    fuel_models = official.fit_models(train.reset_index(drop=True), fuel_for(train, raw))
    target = prepared['GB'].loc[prepared['GB'].year.eq(2024)].copy().reset_index(drop=True)
    predictions = official.point_predictions(response_models, fuel_models, target)
    base = predictions['fixed_ef_recomposition']
    for family, model in final_corrections.items():
        alpha = selection_map[family]
        predictions[f'pseudo_residual_{family}'] = np.clip(
            base + alpha * correct(model, target, base), 0., 1000.)
    path = out / f'{arm}_US_to_GB_predictions.csv.gz'
    pd.DataFrame({'time_utc': pd.to_datetime(target.time, utc=True).astype(str).to_numpy(),
                  'observed_month': target.observed_month.to_numpy(),
                  **predictions}).to_csv(path, index=False, compression='gzip')
    seal = {'arm': arm, 'source': 'US', 'target': 'GB', 'path': path.name,
            'sha256': sha(path), 'hours_predicted': len(target),
            'selected_shrink': selection_map}
    (out / f'{arm}_selection.json').write_text(json.dumps(selection_rows, indent=2))
    (out / f'{arm}_source_calibration.json').write_text(json.dumps(calibrated, indent=2))
    (out / f'{arm}_prediction_seal.json').write_text(json.dumps(seal, indent=2))
    return validation_rows, [{'arm': arm, **row} for row in selection_rows]


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
    code, data = source_paths()
    hashes = {'code': {str(p): sha(p) for p in code},
              'inputs': {str(p): sha(p) for p in data}}
    (out / 'startup_hashes.json').write_text(json.dumps(hashes, indent=2))
    raw, base_prepared, labels, audit = official.prepare(ROOT, capacity)
    monthly = pseudo.parse_monthly(tuple(range(2020, 2025)), official.US_REGIONS)
    prepared, support = pseudo.replace_source_features(base_prepared, monthly)
    monthly.to_csv(out / 'eia923_monthly_ba_mix.csv', index=False)
    (out / 'input_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    (out / 'common_support.json').write_text(json.dumps(support, indent=2))
    (out / 'preseal_target_audit.json').write_text(json.dumps({
        'target_year': 2024, 'target_label_previously_viewed': True,
        'target_label_read_by_this_run_before_seal': False,
        'assessment': 'development replay, not blind confirmation',
    }, indent=2))

    validation, selections, seals = [], [], []
    for arm in ('hourly_reconstruction', 'eia923_monthly'):
        rows, picked = evaluate_arm(arm, prepared[arm], raw, labels, out)
        validation.extend(rows)
        selections.extend(picked)
        seals.append(json.loads((out / f'{arm}_prediction_seal.json').read_text()))
    pd.DataFrame(validation).to_csv(out / 'source_h1_validation.csv', index=False)
    (out / 'prediction_seal.json').write_text(json.dumps(seals, indent=2))
    if any(sha(Path(path)) != digest for path, digest in hashes['code'].items()):
        raise RuntimeError('code or protocol changed during execution')
    if any(sha(Path(path)) != digest for path, digest in hashes['inputs'].items()):
        raise RuntimeError('input data changed during execution')

    target_labels = official.source_labels('GB', years=(2024,))
    target_path = official.TARGET_LABEL_ROOT / 'GB_consumption_2024_halfhour.csv.gz'
    (out / 'target_scoring_hash.json').write_text(json.dumps({str(target_path): sha(target_path)}, indent=2))
    metrics = []
    for arm in ('hourly_reconstruction', 'eia923_monthly'):
        seal = json.loads((out / f'{arm}_prediction_seal.json').read_text())
        frame = prepared[arm]['GB'].loc[prepared[arm]['GB'].year.eq(2024)].copy().reset_index(drop=True)
        truth = official.aligned_labels(frame, target_labels)
        prediction = pd.read_csv(out / seal['path'])
        calibration_info = json.loads((out / f'{arm}_source_calibration.json').read_text())
        for month in sorted(frame.clock.dt.month.unique()):
            mask = frame.clock.dt.month.eq(month).to_numpy()
            for method in [*OUTER_METHODS, *(f'pseudo_residual_{c}' for c in CORRECTORS)]:
                result = official.score(
                    truth[mask], prediction.loc[mask, method].to_numpy(float),
                    calibration_info[method]['bias'], calibration_info[method]['sigma'],
                    frame.loc[mask, 'clock'].reset_index(drop=True))
                metrics.append({'arm': arm, 'method': method, 'month': int(month), **result})
    monthly_metrics = pd.DataFrame(metrics)
    monthly_metrics.to_csv(out / 'monthly_metrics.csv', index=False)
    score_cols = ['mae', 'shape_mae', 'crps_gaussian', 'coverage90', 'width90', 'interval_score90']
    summary = monthly_metrics.groupby(['arm', 'method'], as_index=False)[score_cols].mean()
    summary.to_csv(out / 'summary_metrics.csv', index=False)
    (out / 'status.json').write_text(json.dumps({
        'status': 'DEVELOPMENT_OFFICIAL_PSEUDOTARGET_RECOMPOSITION_RESIDUAL',
        'source_train_year': 2022, 'source_select_year': '2023H1',
        'source_calibration_year': '2023H2', 'target_year': 2024,
        'target_label_previously_viewed': True,
        'target_label_read_after_prediction_seal': True,
        'arms': ['hourly_reconstruction', 'eia923_monthly'],
        'target_input_delay_months': 2,
    }, indent=2))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
