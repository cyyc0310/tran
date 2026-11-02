"""Train coarse-only residual correctors from cross-fitted country pseudo-target tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from transcif.research.multicountry import (
    FACTORS, FINE, GROUPS, HISTORY, fit, predict, public_features, public_months,
)
from transcif.research.monthly_adaptation import weighted_quantile
from run_coarse_conditional_experts import expert_prediction, fit_expert
from run_energy_charts_loco import COUNTRIES, DATA, PROTOCOL as DATA_PROTOCOL, load_country

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/CROSS_FITTED_PSEUDOTARGET_BACKOFF_PROTOCOL_20261101.md'
BASE_METHODS = ('ratio_recomposition', 'macro_recomposition', 'factor_history', 'direct_history')
CORRECTORS = ('residual_ridge', 'residual_hgb')
BACKOFF_METHODS = ('residual_ridge_backoff', 'residual_hgb_backoff')
METHODS = (*BASE_METHODS, *CORRECTORS, *BACKOFF_METHODS, 'coarse_expert_gate')
RESIDUAL_FEATURES = (*HISTORY, 'ratio_recomposition')
SHRINK_VALUES = (0., .1, .25, .5, .75, 1.)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def truth_for(raw: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    fuel = raw.iloc[frame.raw_row.to_numpy()][list(FINE)].to_numpy(float)
    return fuel @ FACTORS / fuel.sum(axis=1)


def cohort(prepared: dict, countries: list[str], predicate):
    frames, fuels, coarse = [], [], []
    for country in countries:
        raw, features = prepared[country]
        frame = features.loc[predicate(features)].copy()
        if frame.empty:
            continue
        rows = frame.raw_row.to_numpy()
        frames.append(frame)
        fuels.append(raw.iloc[rows][list(FINE)].to_numpy(float))
        coarse.append(np.column_stack([
            raw.iloc[rows][list(FINE[:4])].sum(axis=1).to_numpy(float),
            raw.iloc[rows][['hydro', 'nuclear', 'wind', 'solar', 'other_renewable']].to_numpy(float),
        ]))
    if not frames:
        raise ValueError(f'empty cohort for {countries}')
    return pd.concat(frames, ignore_index=True), np.concatenate(fuels), np.concatenate(coarse)


def predict_base(model: dict, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    result = predict(model, frame)
    return {name: np.asarray(result[name], dtype=float) for name in BASE_METHODS}


def equal_country_weights(frame: pd.DataFrame) -> np.ndarray:
    n = frame.groupby('country').country.transform('size').to_numpy(float)
    value = 1. / n
    return value / value.sum() * len(value)


def fit_corrector(frame: pd.DataFrame, residual: np.ndarray, family: str):
    x = frame[list(RESIDUAL_FEATURES)].to_numpy(float)
    w = equal_country_weights(frame)
    if family == 'residual_ridge':
        scaler = StandardScaler().fit(x, sample_weight=w)
        model = Ridge(alpha=10.).fit(scaler.transform(x), residual, sample_weight=w)
        return ('ridge', scaler, model)
    if family == 'residual_hgb':
        model = HistGradientBoostingRegressor(
            max_iter=80, max_leaf_nodes=15, min_samples_leaf=100,
            l2_regularization=10., random_state=417, early_stopping=False,
        ).fit(x, residual, sample_weight=w)
        return ('hgb', None, model)
    raise ValueError(f'unknown correction family: {family}')


def apply_corrector(model, frame: pd.DataFrame) -> np.ndarray:
    kind, scaler, fitted = model
    x = frame[list(RESIDUAL_FEATURES)].to_numpy(float)
    if kind == 'ridge':
        x = scaler.transform(x)
    return fitted.predict(x)


def shape_mae(clock: pd.Series, y: np.ndarray, p: np.ndarray) -> float:
    sample = pd.DataFrame({'day': clock.dt.strftime('%Y-%m-%d').to_numpy(), 'y': y, 'p': p})
    center = sample[['y', 'p']] - sample.groupby('day')[['y', 'p']].transform('mean')
    return float(np.abs(center.y - center.p).mean())


def calibration_sigma(errors: dict[str, np.ndarray]) -> float:
    return max(float(np.sqrt(np.mean([np.mean(v ** 2) for v in errors.values()]))), 1e-6)


def crps_gaussian(y: np.ndarray, p: np.ndarray, sigma: float) -> float:
    z = (y - p) / sigma
    return float(np.mean(sigma * (z * (2. * norm.cdf(z) - 1.) +
                                  2. * norm.pdf(z) - 1. / np.sqrt(np.pi))))


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)

    manifest_path, audit_path = DATA / 'acquisition_manifest.json', DATA / 'audit.json'
    expert_root = ROOT / 'results/coarse-conditional-experts-r5'
    selection_path = expert_root / 'selection_and_calibration.json'
    input_paths = [manifest_path, audit_path, selection_path]
    input_paths += [DATA / f'{c}_2022_2024.json' for c in COUNTRIES]
    input_paths += [expert_root / f'{c.upper()}_heldout_predictions.csv.gz' for c in COUNTRIES]
    code_paths = [Path(__file__), ROOT / 'scripts/run_energy_charts_loco.py',
                  ROOT / 'scripts/run_coarse_conditional_experts.py',
                  ROOT / 'src/transcif/research/multicountry.py',
                  ROOT / 'src/transcif/research/cross_country.py',
                  ROOT / 'src/transcif/research/monthly_adaptation.py',
                  PROTOCOL, DATA_PROTOCOL]
    hashes = {'code': {str(p): sha(p) for p in code_paths},
              'inputs': {str(p): sha(p) for p in input_paths}}
    (out / 'startup_hashes.json').write_text(json.dumps(hashes, indent=2))

    manifest = json.loads(manifest_path.read_text())
    prior_gate = json.loads(selection_path.read_text())
    prepared, audits = {}, {}
    for country in COUNTRIES:
        raw, audits[country] = load_country(country, manifest)
        features = public_features(public_months(raw), raw.clock)
        features['country'], features['region'] = country.upper(), country.upper()
        features['year'] = raw.clock.dt.year.to_numpy()
        features['raw_row'] = np.arange(len(raw))
        prepared[country] = (raw, features.dropna(subset=list(HISTORY)).copy())
    (out / 'input_audit.json').write_text(json.dumps(audits, indent=2, ensure_ascii=False))

    seals, target_cache, summaries, all_shrink_selection = [], {}, {}, []
    for target in COUNTRIES:
        sources = [c for c in COUNTRIES if c != target]
        task_h1, task_h2 = {}, {}

        # Each source country is a pseudo target: its labels never train its base model.
        for pseudo in sources:
            fit_countries = [c for c in sources if c != pseudo]
            train, train_y, train_coarse = cohort(
                prepared, fit_countries, lambda f: f.year.eq(2022))
            base = fit(train, train_y, train_coarse)
            raw, features = prepared[pseudo]
            for half, months in (('h1', lambda m: m.le(6)), ('h2', lambda m: m.gt(6))):
                frame = features.loc[features.year.eq(2023) & months(features.clock.dt.month)].copy()
                if frame.empty:
                    raise ValueError(f'empty {half} pseudo target for {pseudo}')
                base_predictions = predict_base(base, frame)
                frame['ratio_recomposition'] = base_predictions['ratio_recomposition']
                y = truth_for(raw, frame)
                residual = y - base_predictions['ratio_recomposition']
                (task_h1 if half == 'h1' else task_h2)[pseudo] = {
                    'frame': frame, 'truth': y, 'base': base_predictions,
                    'residual': residual,
                }

        # Independent source calibration: each H2 pseudo target is corrected
        # only by residual tasks from the other source countries' H1.
        cal_errors = {name: {} for name in (*BASE_METHODS, *CORRECTORS)}
        cal_corrections = {name: {} for name in CORRECTORS}
        for held in sources:
            train_frames = [task_h1[c]['frame'] for c in sources if c != held]
            meta_frame = pd.concat(train_frames, ignore_index=True)
            meta_residual = np.concatenate([task_h1[c]['residual'] for c in sources if c != held])
            fitted = {family: fit_corrector(meta_frame, meta_residual, family)
                      for family in ('residual_ridge', 'residual_hgb')}
            valid = task_h2[held]
            frame = valid['frame']
            y = valid['truth']
            for method in BASE_METHODS:
                cal_errors[method][held] = y - valid['base'][method]
            for family, model in fitted.items():
                correction = apply_corrector(model, frame)
                cal_corrections[family][held] = correction
                point = np.clip(
                    valid['base']['ratio_recomposition'] + correction,
                    0., 1000.,
                )
                cal_errors[family][held] = y - point

        # Reproduce the frozen expert gate's source H2 cross-fit calibration.
        expert_models, centroids = {}, {}
        for source in sources:
            expert_models[source], centroids[source] = fit_expert(prepared, source)
        train_2022, _, _ = cohort(prepared, sources, lambda f: f.year.eq(2022))
        scales = np.maximum(train_2022[list(GROUPS)].std(ddof=0).to_numpy(float), .02)
        tau_name = str(prior_gate[target]['selected_tau'])
        tau = np.inf if tau_name == 'inf' else float(tau_name)
        cal_errors['coarse_expert_gate'] = {}
        for held in sources:
            frame = task_h2[held]['frame']
            available = [c for c in sources if c != held]
            point = expert_prediction(frame, available, expert_models, centroids, scales, tau)
            cal_errors['coarse_expert_gate'][held] = task_h2[held]['truth'] - point

        # Select residual shrinkage on source-only Jul-Sep pseudo-target folds.
        shrink_selection, selected_shrink = [], {}
        for family in CORRECTORS:
            candidate_scores = []
            for shrink in SHRINK_VALUES:
                country_scores = []
                for held in sources:
                    frame = task_h2[held]['frame']
                    keep = frame.clock.dt.month.between(7, 9).to_numpy()
                    point = np.clip(
                        task_h2[held]['base']['ratio_recomposition'] +
                        shrink * cal_corrections[family][held], 0., 1000.,
                    )
                    month_scores = []
                    for month in sorted(frame.loc[keep, 'clock'].dt.month.unique()):
                        mask = keep & frame.clock.dt.month.eq(month).to_numpy()
                        score = float(np.abs(task_h2[held]['truth'][mask] - point[mask]).mean())
                        month_scores.append(score)
                        shrink_selection.append({
                            'outer_target': target.upper(), 'family': family,
                            'shrink': shrink, 'pseudo_target': held.upper(),
                            'month': int(month), 'hours': int(mask.sum()), 'mae': score,
                        })
                    country_scores.append(float(np.mean(month_scores)))
                candidate_scores.append((float(np.mean(country_scores)), shrink))
            # Stable tie behavior favors the smaller correction.
            selected_shrink[family] = min(candidate_scores, key=lambda item: (item[0], item[1]))[1]
            for score, shrink in candidate_scores:
                shrink_selection.append({
                    'outer_target': target.upper(), 'family': family,
                    'shrink': shrink, 'pseudo_target': 'MACRO', 'month': 0,
                    'hours': 0, 'mae': score,
                })
            for held in sources:
                point = np.clip(
                    task_h2[held]['base']['ratio_recomposition'] +
                    selected_shrink[family] * cal_corrections[family][held], 0., 1000.,
                )
                cal_errors[f'{family}_backoff'] = cal_errors.get(f'{family}_backoff', {})
                cal_errors[f'{family}_backoff'][held] = task_h2[held]['truth'] - point
        all_shrink_selection.extend(shrink_selection)

        radii, sigmas = {}, {}
        for method, by_country in cal_errors.items():
            # Oct-Dec is held aside from the Jul-Sep shrinkage selection.
            late_errors = {}
            for country in sources:
                frame = task_h2[country]['frame']
                late = frame.clock.dt.month.between(10, 12).to_numpy()
                late_errors[country] = by_country[country][late]
            sigma = calibration_sigma(late_errors)
            pooled_errors = np.concatenate([np.abs(late_errors[c]) for c in sources])
            country_weight = np.concatenate([
                np.full(len(late_errors[c]), 1. / len(sources) / len(late_errors[c]))
                for c in sources
            ])
            radii[method] = float(weighted_quantile(pooled_errors, country_weight, .9))
            sigmas[method] = sigma

        # Final source fit and correction trained only on source pseudo tasks.
        train, train_y, train_coarse = cohort(prepared, sources, lambda f: f.year.eq(2022))
        final_base = fit(train, train_y, train_coarse)
        meta_frame = pd.concat([task_h1[c]['frame'] for c in sources], ignore_index=True)
        meta_residual = np.concatenate([task_h1[c]['residual'] for c in sources])
        final_correctors = {family: fit_corrector(meta_frame, meta_residual, family)
                            for family in ('residual_ridge', 'residual_hgb')}

        target_raw, target_features = prepared[target]
        query = target_features.loc[target_features.year.eq(2024)].copy().reset_index(drop=True)
        if query.empty:
            raise ValueError(f'empty external target: {target}')
        target_base = predict_base(final_base, query)
        forecast = dict(target_base)
        correction_query = query.copy()
        correction_query['ratio_recomposition'] = target_base['ratio_recomposition']
        for family, model in final_correctors.items():
            raw_correction = apply_corrector(model, correction_query)
            forecast[family] = np.clip(target_base['ratio_recomposition'] + raw_correction, 0., 1000.)
            forecast[f'{family}_backoff'] = np.clip(
                target_base['ratio_recomposition'] + selected_shrink[family] * raw_correction,
                0., 1000.,
            )
            if selected_shrink[family] == 0. and not np.array_equal(
                    forecast[f'{family}_backoff'], target_base['ratio_recomposition']):
                raise RuntimeError(f'{family} zero-shrink did not exactly recover the strong anchor')
        # Reuse the already sealed gate point forecast as a comparator.
        gate_path = expert_root / f'{target.upper()}_heldout_predictions.csv.gz'
        gate_frame = pd.read_csv(gate_path)
        target_times = target_raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy()
        aligned_gate = pd.DataFrame({'time': target_times}).merge(
            gate_frame[['time', 'coarse_expert_gate']], on='time', how='left', validate='one_to_one')
        if aligned_gate.coarse_expert_gate.isna().any():
            raise ValueError(f'gate forecast timestamps do not align for {target}')
        forecast['coarse_expert_gate'] = aligned_gate.coarse_expert_gate.to_numpy(float)
        prediction_path = out / f'{target.upper()}_heldout_predictions.csv.gz'
        pd.DataFrame({'time': target_times, **forecast}).to_csv(
            prediction_path, index=False, compression='gzip')
        seal = {'target': target.upper(), 'path': prediction_path.name,
                'sha256': sha(prediction_path), 'hours': len(query),
                'outer_target_label_used_before_seal': False,
                'source_pseudo_tasks': [c.upper() for c in sources]}
        seals.append(seal)
        target_cache[target] = (target_raw, query, prediction_path, radii, sigmas)
        summaries[target] = {
            'selected_expert_tau_from_frozen_gate_run': tau_name,
            'source_selected_shrink': selected_shrink,
            'source_h2_oof_radius90': radii,
            'source_h2_oof_sigma': sigmas,
            'h1_pseudo_task_hours': {c: len(task_h1[c]['frame']) for c in sources},
            'h2_pseudo_calibration_hours': {c: len(task_h2[c]['frame']) for c in sources},
        }
        print(target.upper(), 'source tasks fitted; forecast sealed', len(query), flush=True)

        if any(sha(Path(p)) != v for p, v in hashes['code'].items()):
            raise RuntimeError('frozen code changed during model fitting')
        if any(sha(Path(p)) != v for p, v in hashes['inputs'].items()):
            raise RuntimeError('frozen input changed during model fitting')

    (out / 'prediction_seal.json').write_text(json.dumps(seals, indent=2))
    (out / 'source_calibration.json').write_text(json.dumps(summaries, indent=2))

    # Target fixed-factor responses are read only after all target forecasts are sealed.
    rows = []
    for target, (raw, query, path, radii, sigmas) in target_cache.items():
        truth = truth_for(raw, query)
        predictions = pd.read_csv(path)
        score_methods = (*BASE_METHODS, *CORRECTORS, *BACKOFF_METHODS, 'coarse_expert_gate')
        for month in sorted(query.clock.dt.month.unique()):
            mask = query.clock.dt.month.eq(month).to_numpy()
            clock = query.loc[mask, 'clock'].reset_index(drop=True)
            y = truth[mask]
            for method in score_methods:
                point = predictions.loc[mask, method].to_numpy(float)
                error = y - point
                radius, sigma = radii[method], sigmas[method]
                rows.append({
                    'target_country': target.upper(), 'method': method, 'month': int(month),
                    'hours': int(mask.sum()), 'mae': float(np.abs(error).mean()),
                    'shape_mae': shape_mae(clock, y, point),
                    'crps_gaussian': crps_gaussian(y, point, sigma),
                    'coverage90': float((np.abs(error) <= radius).mean()),
                    'width90': float(2. * radius),
                    'interval_score90': float(np.mean(
                        2. * radius + 20. * np.maximum(np.abs(error) - radius, 0.))),
                })
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / 'monthly_metrics.csv', index=False)
    metric_columns = ['mae', 'shape_mae', 'crps_gaussian', 'coverage90', 'width90', 'interval_score90']
    country = monthly.groupby(['target_country', 'method'])[metric_columns].mean().reset_index()
    country.to_csv(out / 'country_metrics.csv', index=False)
    country.groupby('method')[metric_columns].mean().reset_index().to_csv(
        out / 'macro_metrics.csv', index=False)
    (out / 'status.json').write_text(json.dumps({
        'status': 'DEVELOPMENT_CROSSFIT_PSEUDOTARGET_RESIDUAL_TRANSFER',
        'outer_targets': [c.upper() for c in COUNTRIES],
        'source_labels_for_residual_tasks': 'held-out source countries 2023H1 only',
        'source_h2_calibration': 'nested source-country cross-fit; each held source excluded from corrector task training',
        'shrinkage_selection_period': 'source-only pseudo-target 2023 Jul-Sep, country-equal macro MAE',
        'final_interval_calibration_period': 'source-only pseudo-target 2023 Oct-Dec absolute residuals',
        'target_inputs': 'local calendar and t-2-or-older coarse monthly generation composition',
        'target_responses_used_before_forecast_seal': False,
        'target_countries_previously_viewed': True,
        'target_labels': 'fixed-EF generation-intensity proxy; not measured CIF',
        'monthly_inputs_are_historical_as_issued': False,
        'actual_meta_learning': False,
    }, indent=2))
    print(pd.read_csv(out / 'macro_metrics.csv').to_string(index=False))
    selection_rows = []
    for target, summary in summaries.items():
        for family, shrink in summary['source_selected_shrink'].items():
            selection_rows.append({'outer_target': target.upper(), 'family': family,
                                   'selected_shrink': shrink})
    pd.DataFrame(all_shrink_selection).to_csv(out / 'source_shrinkage_cv.csv', index=False)
    pd.DataFrame(selection_rows).to_csv(out / 'selected_shrinkage.csv', index=False)


if __name__ == '__main__':
    main()
