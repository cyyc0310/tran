"""Fit country experts and gate them using only target coarse composition."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transcif.research.multicountry import (
    FACTORS, FINE, GROUPS, HISTORY, METHODS, coarse_values, fit, predict, public_features,
    public_months, weights,
)
from transcif.research.monthly_adaptation import weighted_quantile
from run_energy_charts_loco import COUNTRIES, DATA, PROTOCOL as DATA_PROTOCOL, load_country

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/COARSE_CONDITIONAL_EXPERT_TRANSFER_PROTOCOL_20261101.md'
TAUS = (0.05, 0.10, 0.20, 0.40, 0.80, np.inf)
BASELINES = ('anchor700', 'macro_recomposition', 'ratio_recomposition', 'factor_history')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aligned(prepared: dict, countries: list[str], mask) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    parts = [prepared[c][1].loc[mask(prepared[c][1])] for c in countries]
    frame = pd.concat(parts, ignore_index=True)
    fuel = np.concatenate([prepared[c][0].iloc[p.raw_row.to_numpy()][list(FINE)].to_numpy(float)
                           for c, p in zip(countries, parts)])
    coarse = np.concatenate([coarse_values(prepared[c][0].iloc[p.raw_row.to_numpy()])
                             for c, p in zip(countries, parts)])
    if not len(frame) or len(frame) != len(fuel) or len(fuel) != len(coarse):
        raise ValueError('empty or misaligned cohort')
    return frame, fuel, coarse


def fit_expert(prepared: dict, country: str) -> tuple[dict, np.ndarray]:
    frame, fuel, coarse = aligned(prepared, [country], lambda f: f.year.eq(2022))
    model = fit(frame, fuel, coarse)
    centroid = frame[list(GROUPS)].mean().to_numpy(float)
    return model, centroid


def gate(public: pd.DataFrame, source_codes: list[str], centroids: dict,
         scales: np.ndarray, tau: float) -> np.ndarray:
    target = public[list(GROUPS)].to_numpy(float)
    means = np.stack([centroids[c] for c in source_codes])
    distance = ((target[:, None, :] - means[None, :, :]) / scales[None, None, :]) ** 2
    distance = distance.mean(axis=2)
    if np.isinf(tau):
        return np.full(distance.shape, 1. / len(source_codes))
    logits = -distance / (2. * tau * tau)
    logits -= logits.max(axis=1, keepdims=True)
    value = np.exp(logits)
    return value / value.sum(axis=1, keepdims=True)


def expert_prediction(public: pd.DataFrame, source_codes: list[str], experts: dict,
                      centroids: dict, scales: np.ndarray, tau: float) -> np.ndarray:
    g = gate(public, source_codes, centroids, scales, tau)
    p = np.column_stack([predict(experts[c], public)['factor_history'] for c in source_codes])
    return np.sum(g * p, axis=1)


def shape_mae(clock: pd.Series, y: np.ndarray, p: np.ndarray) -> float:
    values = pd.DataFrame({'day': clock.dt.strftime('%Y-%m-%d').to_numpy(), 'y': y, 'p': p})
    center = values[['y', 'p']] - values.groupby('day')[['y', 'p']].transform('mean')
    return float(np.abs(center.y - center.p).mean())


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    manifest_path, audit_path = DATA / 'acquisition_manifest.json', DATA / 'audit.json'
    code_files = [Path(__file__), ROOT / 'scripts/run_energy_charts_loco.py',
                  ROOT / 'src/transcif/research/multicountry.py',
                  ROOT / 'src/transcif/research/cross_country.py',
                  ROOT / 'src/transcif/research/monthly_adaptation.py', PROTOCOL, DATA_PROTOCOL]
    code_hashes = {str(p): sha(p) for p in code_files}
    inputs = {str(DATA / f'{c}_2022_2024.json'): sha(DATA / f'{c}_2022_2024.json')
              for c in COUNTRIES}
    inputs[str(manifest_path)], inputs[str(audit_path)] = sha(manifest_path), sha(audit_path)
    (out / 'startup_hashes.json').write_text(json.dumps({'code': code_hashes, 'inputs': inputs}, indent=2))

    source_manifest = json.loads(manifest_path.read_text())
    prepared, audit = {}, {}
    for country in COUNTRIES:
        raw, audit[country] = load_country(country, source_manifest)
        features = public_features(public_months(raw), raw.clock)
        features['country'], features['region'] = country.upper(), country.upper()
        features['year'] = raw.clock.dt.year.to_numpy()
        features['raw_row'] = np.arange(len(raw))
        prepared[country] = (raw, features.dropna(subset=list(HISTORY)).copy())
    (out / 'input_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    sealed, details = [], {}
    output_data = {}
    for target in COUNTRIES:
        sources = [c for c in COUNTRIES if c != target]
        # Country-specific experts and support prototypes use source 2022 only.
        experts, centroids = {}, {}
        source_train = []
        for source in sources:
            experts[source], centroids[source] = fit_expert(prepared, source)
            source_train.append(prepared[source][1].loc[prepared[source][1].year.eq(2022)])
        all_train = pd.concat(source_train, ignore_index=True)
        scale = all_train[list(GROUPS)].std(ddof=0).to_numpy(float)
        scale = np.maximum(scale, 0.02)

        # Pseudo-target tuning uses 2023H1 only, with each pseudo-target's own
        # expert removed from its mixture.
        tune_rows = []
        for pseudo in sources:
            pseudo_frame = prepared[pseudo][1].loc[
                prepared[pseudo][1].year.eq(2023) & prepared[pseudo][1].clock.dt.month.le(6)
            ].copy()
            raw = prepared[pseudo][0]
            fuel = raw.iloc[pseudo_frame.raw_row.to_numpy()][list(FINE)].to_numpy(float)
            y = fuel @ FACTORS / fuel.sum(axis=1)
            available = [c for c in sources if c != pseudo]
            for tau in TAUS:
                p = expert_prediction(pseudo_frame, available, experts, centroids, scale, tau)
                tune_rows.append({'pseudo_target': pseudo.upper(), 'tau': 'inf' if np.isinf(tau) else tau,
                                  'mae': float(np.abs(y - p).mean()),
                                  'shape_mae': shape_mae(pseudo_frame.clock, y, p),
                                  'hours': len(y)})
        tune = pd.DataFrame(tune_rows)
        macro_tune = tune.groupby('tau').mae.mean().sort_values(kind='mergesort')
        selected_tau_name = str(macro_tune.index[0])
        selected_tau = np.inf if selected_tau_name == 'inf' else float(selected_tau_name)

        # H2 calibration is disjoint from H1 bandwidth selection and uses
        # cross-country predictions that exclude each pseudo-target expert.
        cal_frames, cal_residuals, uniform_residuals = [], [], []
        calibration_residual_by_country = {}
        calibration_frame_by_country = {}
        for pseudo in sources:
            frame = prepared[pseudo][1].loc[
                prepared[pseudo][1].year.eq(2023) & prepared[pseudo][1].clock.dt.month.gt(6)
            ].copy()
            available = [c for c in sources if c != pseudo]
            p = expert_prediction(frame, available, experts, centroids, scale, selected_tau)
            p_uniform = expert_prediction(frame, available, experts, centroids, scale, np.inf)
            fuel = prepared[pseudo][0].iloc[frame.raw_row.to_numpy()][list(FINE)].to_numpy(float)
            y = fuel @ FACTORS / fuel.sum(axis=1)
            cal_frames.append(frame)
            cal_residuals.append(np.abs(y - p))
            uniform_residuals.append(np.abs(y - p_uniform))
            calibration_residual_by_country[pseudo] = np.abs(y - p)
            calibration_frame_by_country[pseudo] = frame
        cal = pd.concat(cal_frames, ignore_index=True)
        cal_w = weights(cal)
        residual = np.concatenate(cal_residuals)
        gated_radius = weighted_quantile(residual, cal_w, .9)
        uniform_radius = weighted_quantile(np.concatenate(uniform_residuals), cal_w, .9)

        # Choose a global expansion from source-only leave-one-country-out
        # validation on Jul–Sep; use Oct–Dec residuals for final target radii.
        lambda_candidates = (1.0, 1.25, 1.5, 1.75, 2.0)
        risk_cv_rows = []
        for pseudo in sources:
            validation = calibration_frame_by_country[pseudo]
            validation = validation.loc[validation.clock.dt.month.between(7, 9)]
            raw = prepared[pseudo][0]
            fuel = raw.iloc[validation.raw_row.to_numpy()][list(FINE)].to_numpy(float)
            y_val = fuel @ FACTORS / fuel.sum(axis=1)
            for period in validation.clock.dt.to_period('M').unique():
                mask = validation.clock.dt.to_period('M').eq(period).to_numpy()
                center = validation.loc[mask, list(GROUPS)].iloc[[0]].copy()
                calibration_countries = [c for c in sources if c != pseudo]
                domain_w = gate(center, calibration_countries, centroids, scale, selected_tau)[0]
                calibration_errors, calibration_weights = [], []
                for index, country in enumerate(calibration_countries):
                    frame_c = calibration_frame_by_country[country]
                    keep = frame_c.clock.dt.month.between(7, 12).to_numpy()
                    errors_c = calibration_residual_by_country[country][keep]
                    calibration_errors.append(errors_c)
                    calibration_weights.append(np.full(
                        len(errors_c), domain_w[index] / len(errors_c)))
                base_radius = weighted_quantile(
                    np.concatenate(calibration_errors), np.concatenate(calibration_weights), .9)
                for multiplier in lambda_candidates:
                    radius = base_radius * multiplier
                    error = y_val[mask] - expert_prediction(
                        validation.loc[mask], calibration_countries, experts,
                        centroids, scale, selected_tau)
                    risk_cv_rows.append({
                        'pseudo_target': pseudo.upper(), 'period': str(period),
                        'lambda': multiplier, 'hours': int(mask.sum()),
                        'coverage90': float((np.abs(error) <= radius).mean()),
                        'width90': float(2 * radius),
                        'interval_score90': float(np.mean(
                            2 * radius + 20 * np.maximum(np.abs(error) - radius, 0))),
                    })
        risk_cv = pd.DataFrame(risk_cv_rows)
        risk_macro = risk_cv.groupby('lambda').coverage90.mean()
        risk_worst = risk_cv.groupby(['pseudo_target', 'lambda']).coverage90.mean().groupby('lambda').min()
        acceptable = [lam for lam in lambda_candidates
                      if risk_macro.loc[lam] >= .90 and risk_worst.loc[lam] >= .85]
        macro_acceptable = [lam for lam in lambda_candidates if risk_macro.loc[lam] >= .90]
        selected_lambda = (min(acceptable) if acceptable else
                           min(macro_acceptable) if macro_acceptable else max(lambda_candidates))
        (out / f'{target.upper()}_risk_lambda_cv.csv').write_text(risk_cv.to_csv(index=False))

        # Fixed baselines share the same fit years and H2 calibration interval.
        train, train_fuel, train_coarse = aligned(prepared, sources, lambda f: f.year.eq(2022))
        pooled = fit(train, train_fuel, train_coarse)
        cal_h2, cal_h2_fuel, _ = aligned(
            prepared, sources, lambda f: f.year.eq(2023) & f.clock.dt.month.gt(6))
        cal_y = cal_h2_fuel @ FACTORS / cal_h2_fuel.sum(axis=1)
        cal_p = predict(pooled, cal_h2)
        pooled_w = weights(cal_h2)
        radii = {m: weighted_quantile(np.abs(cal_y - cal_p[m]), pooled_w, .9)
                 for m in BASELINES}
        radii['coarse_expert_gate'] = float(gated_radius)
        radii['coarse_expert_uniform'] = float(uniform_radius)

        target_raw, target_features = prepared[target]
        query = target_features.loc[target_features.year.eq(2024)].copy()
        forecast = {'coarse_expert_gate': expert_prediction(
            query, sources, experts, centroids, scale, selected_tau)}
        forecast['coarse_expert_uniform'] = expert_prediction(
            query, sources, experts, centroids, scale, np.inf)
        forecast['coarse_expert_gate_conditional'] = forecast['coarse_expert_gate'].copy()
        forecast['coarse_expert_gate_conditional_scaled'] = forecast['coarse_expert_gate'].copy()
        conditional_radii = {}
        scaled_conditional_radii = {}
        for period in query.clock.dt.to_period('M').unique():
            month_mask = query.clock.dt.to_period('M').eq(period).to_numpy()
            representative = query.loc[month_mask, list(GROUPS)].iloc[[0]].copy()
            domain_weights = gate(representative, sources, centroids, scale, selected_tau)[0]
            errors = np.concatenate([calibration_residual_by_country[c] for c in sources])
            w_parts = [np.full(len(calibration_residual_by_country[c]), domain_weights[i] /
                                len(calibration_residual_by_country[c]))
                       for i, c in enumerate(sources)]
            conditional_radii[str(period)] = weighted_quantile(
                errors, np.concatenate(w_parts), .9)
            late_errors, late_weights = [], []
            for i, c in enumerate(sources):
                frame_c = calibration_frame_by_country[c]
                keep = frame_c.clock.dt.month.between(10, 12).to_numpy()
                errors_c = calibration_residual_by_country[c][keep]
                late_errors.append(errors_c)
                late_weights.append(np.full(len(errors_c), domain_weights[i] / len(errors_c)))
            scaled_conditional_radii[str(period)] = selected_lambda * weighted_quantile(
                np.concatenate(late_errors), np.concatenate(late_weights), .9)
        radii['coarse_expert_gate_conditional'] = conditional_radii
        radii['coarse_expert_gate_conditional_scaled'] = scaled_conditional_radii
        forecast.update({k: v for k, v in predict(pooled, query).items() if k in BASELINES})
        path = out / f'{target.upper()}_heldout_predictions.csv.gz'
        pd.DataFrame({'time': target_raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                      **forecast}).to_csv(path, index=False, compression='gzip')
        sealed.append({'target': target.upper(), 'path': path.name, 'sha256': sha(path),
                       'hours': len(query), 'selected_tau': selected_tau_name})
        output_data[target] = (query, target_raw, path, radii, selected_tau_name)
        details[target] = {
            'selected_tau': selected_tau_name,
            'tuning_macro_mae': {str(k): float(v) for k, v in macro_tune.items()},
            'tuning_by_pseudo_country': tune.to_dict('records'),
            'crossfit_h2_radius90': float(gated_radius),
            'conditional_h2_radius90_by_target_month': conditional_radii,
            'scaled_conditional_h2_radius90_by_target_month': scaled_conditional_radii,
            'risk_lambda_selected_by_source_cv': float(selected_lambda),
            'risk_lambda_cv_macro_coverage': {str(k): float(v) for k, v in risk_macro.items()},
            'risk_lambda_cv_worst_country_coverage': {str(k): float(v) for k, v in risk_worst.items()},
            'baseline_h2_radius90': radii,
            'source_group_means': {c: centroids[c].tolist() for c in sources},
            'group_scale': scale.tolist(),
        }
        print(target.upper(), 'sealed', len(query), 'tau', selected_tau_name, flush=True)

    (out / 'pseudo_target_tuning.csv').write_text(
        pd.concat([pd.DataFrame(v['tuning_by_pseudo_country']).assign(outer_target=k.upper())
                   for k, v in details.items()], ignore_index=True).to_csv(index=False))
    (out / 'selection_and_calibration.json').write_text(json.dumps(details, indent=2))
    (out / 'prediction_seal.json').write_text(json.dumps(sealed, indent=2))
    if any(sha(Path(p)) != h for p, h in inputs.items()):
        raise RuntimeError('input changed during run')
    if any(sha(Path(p)) != h for p, h in code_hashes.items()):
        raise RuntimeError('code/protocol changed during run')

    rows = []
    for target, (query, raw, path, radii, tau_name) in output_data.items():
        values = pd.read_csv(path)
        fuel = raw.iloc[query.raw_row.to_numpy()][list(FINE)].to_numpy(float)
        y = fuel @ FACTORS / fuel.sum(axis=1)
        for method in (*BASELINES, 'coarse_expert_uniform', 'coarse_expert_gate',
                       'coarse_expert_gate_conditional', 'coarse_expert_gate_conditional_scaled'):
            point = values[method].to_numpy(float)
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                if method == 'coarse_expert_gate_conditional':
                    radius = float(radii[method][str(query.loc[mask, 'clock'].dt.to_period('M').iloc[0])])
                elif method == 'coarse_expert_gate_conditional_scaled':
                    radius = float(radii[method][str(query.loc[mask, 'clock'].dt.to_period('M').iloc[0])])
                else:
                    radius = float(radii[method])
                err = y[mask] - point[mask]
                rows.append({'target_country': target.upper(), 'method': method,
                             'selected_tau': tau_name, 'month': int(month), 'hours': int(mask.sum()),
                             'mae': float(np.abs(err).mean()),
                             'shape_mae': shape_mae(query.loc[mask, 'clock'], y[mask], point[mask]),
                             'coverage90': float((np.abs(err) <= radius).mean()),
                             'width90': float(2 * radius),
                             'interval_score90': float(np.mean(
                                 2 * radius + 20 * np.maximum(np.abs(err) - radius, 0)))})
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / 'monthly_metrics.csv', index=False)
    metric_cols = ['mae', 'shape_mae', 'coverage90', 'width90', 'interval_score90']
    country = monthly.groupby(['target_country', 'method'])[metric_cols].mean().reset_index()
    country.to_csv(out / 'country_metrics.csv', index=False)
    country.groupby('method')[metric_cols].mean().reset_index().to_csv(out / 'macro_metrics.csv', index=False)
    (out / 'status.json').write_text(json.dumps({
        'status': 'DEVELOPMENT_PSEUDOTARGET_COARSE_CONDITIONAL_SOURCE_EXPERTS',
        'target_hourly_labels_used_before_seal': False,
        'pseudo_target_selection_period': '2023H1',
        'cross_fitted_interval_calibration_period': '2023H2',
        'target_test_period': '2024',
        'coarse_month_vintage_is_as_issued': False,
        'target_labels_are_measured_cif': False,
    }, indent=2))
    print(pd.read_csv(out / 'macro_metrics.csv').to_string(index=False))


if __name__ == '__main__':
    main()
