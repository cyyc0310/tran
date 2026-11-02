"""Train coarse-conditioned fuel-factor heads with pseudo-target instance weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from transcif.research.multicountry import (
    FACTORS, FINE, GROUPS, HISTORY, METHODS, coarse_values, fit, predict,
    public_features, public_months, weights,
)
from transcif.research.monthly_adaptation import weighted_quantile
from run_energy_charts_loco import COUNTRIES, DATA, PROTOCOL as DATA_PROTOCOL, load_country

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/PSEUDOTARGET_INSTANCE_REWEIGHTING_PROTOCOL_20261101.md'
GATE_RUN = ROOT / 'results/coarse-conditional-experts-r5'
TAUS = (0.25, 0.5, 1.0, 2.0, np.inf)
BASELINES = ('anchor700', 'macro_recomposition', 'ratio_recomposition', 'factor_history')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aligned(prepared: dict, countries: list[str], predicate) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    parts = [prepared[c][1].loc[predicate(prepared[c][1])] for c in countries]
    frame = pd.concat(parts, ignore_index=True)
    fuel = np.concatenate([prepared[c][0].iloc[p.raw_row.to_numpy()][list(FINE)].to_numpy(float)
                           for c, p in zip(countries, parts)])
    coarse = np.concatenate([coarse_values(prepared[c][0].iloc[p.raw_row.to_numpy()])
                             for c, p in zip(countries, parts)])
    if not len(frame) or len(frame) != len(fuel) or len(fuel) != len(coarse):
        raise ValueError('empty or misaligned source cohort')
    return frame, fuel, coarse


def training_scale(frame: pd.DataFrame) -> np.ndarray:
    return np.maximum(frame[list(GROUPS)].std(ddof=0).to_numpy(float), 0.02)


def conditioned_weights(frame: pd.DataFrame, center: np.ndarray,
                        scale: np.ndarray, tau: float) -> tuple[np.ndarray, dict]:
    base = weights(frame)
    if np.isinf(tau):
        value = base
    else:
        delta = (frame[list(GROUPS)].to_numpy(float) - center[None, :]) / scale[None, :]
        distance = np.mean(delta * delta, axis=1)
        kernel = np.exp(-0.5 * distance / (tau * tau))
        value = base * kernel
        if not np.isfinite(value).all() or value.sum() <= 0:
            raise ValueError('invalid pseudo-target importance weights')
        value = value / value.sum()
    ess = float(1. / np.sum(value * value))
    country_mass = pd.Series(value, index=frame.index).groupby(frame.country).sum().to_dict()
    return value * len(frame), {'effective_sample_size': ess,
                                 'country_weight_mass': {str(k): float(v) for k, v in country_mass.items()}}


def profile_from_training(frame: pd.DataFrame, coarse: np.ndarray) -> np.ndarray:
    per_country = []
    for country in frame.country.drop_duplicates():
        mask = frame.country.to_numpy() == country
        values = pd.DataFrame(coarse[mask], columns=GROUPS)
        values['hour'] = frame.loc[mask, 'clock'].dt.hour.to_numpy()
        means = values.groupby('hour')[list(GROUPS)].mean().reindex(range(24)).fillna(0.)
        per_country.append(means.div(means.mean().replace(0., np.nan)).fillna(1.).to_numpy())
    return np.mean(per_country, axis=0)


def fit_factor(frame: pd.DataFrame, fuel: np.ndarray, center: np.ndarray,
               scale: np.ndarray, tau: float, coarse: np.ndarray) -> tuple[dict, dict]:
    if len(frame) != len(fuel) or len(frame) != len(coarse):
        raise ValueError('factor training arrays are misaligned')
    all_weights, audit = conditioned_weights(frame, center, scale, tau)
    thermal = fuel[:, :4].sum(axis=1)
    active = thermal > 0
    target = np.divide(fuel @ FACTORS, thermal, out=np.full(len(fuel), 400.), where=active)
    model = HistGradientBoostingRegressor(
        max_iter=80, max_leaf_nodes=15, min_samples_leaf=100,
        l2_regularization=10., random_state=416, early_stopping=False,
    )
    model.fit(frame.loc[active, list(HISTORY)], target[active], sample_weight=all_weights[active])
    return {'factor': model, 'profile': profile_from_training(frame, coarse)}, audit


def predict_factor(model: dict, public: pd.DataFrame) -> np.ndarray:
    q = public[list(GROUPS)].to_numpy(float) * model['profile'][public.clock.dt.hour.to_numpy()]
    thermal_share = q[:, 0] / q.sum(axis=1)
    ef = np.clip(model['factor'].predict(public[list(HISTORY)]), 400., 1000.)
    return ef * thermal_share


def shape_mae(clock: pd.Series, y: np.ndarray, p: np.ndarray) -> float:
    sample = pd.DataFrame({'day': clock.dt.strftime('%Y-%m-%d').to_numpy(), 'y': y, 'p': p})
    centered = sample[['y', 'p']] - sample.groupby('day')[['y', 'p']].transform('mean')
    return float(np.abs(centered.y - centered.p).mean())


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    manifest_path, audit_path = DATA / 'acquisition_manifest.json', DATA / 'audit.json'
    gate_predictions = {c: GATE_RUN / f'{c.upper()}_heldout_predictions.csv.gz' for c in COUNTRIES}
    gate_calibration = GATE_RUN / 'selection_and_calibration.json'
    code_files = [Path(__file__), ROOT / 'scripts/run_energy_charts_loco.py',
                  ROOT / 'src/transcif/research/multicountry.py',
                  ROOT / 'src/transcif/research/cross_country.py',
                  ROOT / 'src/transcif/research/monthly_adaptation.py', PROTOCOL, DATA_PROTOCOL]
    code_hashes = {str(p): sha(p) for p in code_files}
    inputs = {str(DATA / f'{c}_2022_2024.json'): sha(DATA / f'{c}_2022_2024.json') for c in COUNTRIES}
    inputs[str(manifest_path)], inputs[str(audit_path)] = sha(manifest_path), sha(audit_path)
    inputs[str(gate_calibration)] = sha(gate_calibration)
    inputs.update({str(path): sha(path) for path in gate_predictions.values()})
    (out / 'startup_hashes.json').write_text(json.dumps({'code': code_hashes, 'inputs': inputs}, indent=2))

    source_manifest = json.loads(manifest_path.read_text())
    prepared, data_audit = {}, {}
    for country in COUNTRIES:
        raw, data_audit[country] = load_country(country, source_manifest)
        public = public_features(public_months(raw), raw.clock)
        public['country'], public['region'] = country.upper(), country.upper()
        public['year'] = raw.clock.dt.year.to_numpy()
        public['raw_row'] = np.arange(len(raw))
        prepared[country] = (raw, public.dropna(subset=list(HISTORY)).copy())
    (out / 'input_audit.json').write_text(json.dumps(data_audit, indent=2, ensure_ascii=False))
    gate_radii = json.loads(gate_calibration.read_text())

    sealed, outputs, selections = [], {}, {}
    for target in COUNTRIES:
        sources = [c for c in COUNTRIES if c != target]
        tune_rows = []
        # Inner pseudo-target folds: q labels never enter the fitted model.
        for pseudo in sources:
            pseudo_sources = [c for c in sources if c != pseudo]
            train, train_fuel, train_coarse = aligned(
                prepared, pseudo_sources, lambda f: f.year.eq(2022))
            scale = training_scale(train)
            validation_all = prepared[pseudo][1].loc[
                prepared[pseudo][1].year.eq(2023) & prepared[pseudo][1].clock.dt.month.le(6)
            ]
            raw = prepared[pseudo][0]
            for month in range(1, 7):
                validation = validation_all.loc[validation_all.clock.dt.month.eq(month)].copy()
                fuel = raw.iloc[validation.raw_row.to_numpy()][list(FINE)].to_numpy(float)
                y = fuel @ FACTORS / fuel.sum(axis=1)
                if not len(validation):
                    raise ValueError(f'missing pseudo-target month {pseudo} 2023-{month:02d}')
                center = validation[list(GROUPS)].iloc[0].to_numpy(float)
                for tau in TAUS:
                    model, audit = fit_factor(train, train_fuel, center, scale, tau, train_coarse)
                    p = predict_factor(model, validation)
                    tune_rows.append({
                        'outer_target': target.upper(), 'pseudo_target': pseudo.upper(),
                        'month': month, 'tau': 'inf' if np.isinf(tau) else tau,
                        'mae': float(np.abs(y - p).mean()),
                        'shape_mae': shape_mae(validation.clock, y, p),
                        'hours': len(validation), **audit,
                    })
            print(target.upper(), 'pseudo-target trained/validated', pseudo.upper(), flush=True)
        tune = pd.DataFrame(tune_rows)
        inner = tune.groupby(['pseudo_target', 'tau']).mae.mean().reset_index()
        macro = inner.groupby('tau').mae.mean().sort_values(kind='mergesort')
        selected_tau_name = str(macro.index[0])
        selected_tau = np.inf if selected_tau_name == 'inf' else float(selected_tau_name)

        # Out-of-country source calibration on H2; H1 was reserved for tau choice.
        cal_frames, cal_abs_errors = [], []
        for pseudo in sources:
            pseudo_sources = [c for c in sources if c != pseudo]
            train, train_fuel, train_coarse = aligned(
                prepared, pseudo_sources, lambda f: f.year.eq(2022))
            scale = training_scale(train)
            query_all = prepared[pseudo][1].loc[
                prepared[pseudo][1].year.eq(2023) & prepared[pseudo][1].clock.dt.month.gt(6)
            ]
            raw = prepared[pseudo][0]
            for month in range(7, 13):
                query = query_all.loc[query_all.clock.dt.month.eq(month)].copy()
                fuel = raw.iloc[query.raw_row.to_numpy()][list(FINE)].to_numpy(float)
                y = fuel @ FACTORS / fuel.sum(axis=1)
                center = query[list(GROUPS)].iloc[0].to_numpy(float)
                model, _ = fit_factor(train, train_fuel, center, scale, selected_tau, train_coarse)
                cal_abs_errors.append(np.abs(y - predict_factor(model, query)))
                cal_frames.append(query)
        cal_frame = pd.concat(cal_frames, ignore_index=True)
        candidate_radius = weighted_quantile(np.concatenate(cal_abs_errors), weights(cal_frame), .9)

        # Existing pooled baselines use the same train year and H2 source period.
        train, train_fuel, train_coarse = aligned(prepared, sources, lambda f: f.year.eq(2022))
        pooled = fit(train, train_fuel, train_coarse)
        cal, cal_fuel, _ = aligned(
            prepared, sources, lambda f: f.year.eq(2023) & f.clock.dt.month.gt(6))
        cal_y = cal_fuel @ FACTORS / cal_fuel.sum(axis=1)
        cal_p = predict(pooled, cal)
        cal_w = weights(cal)
        radii = {m: weighted_quantile(np.abs(cal_y - cal_p[m]), cal_w, .9) for m in BASELINES}
        radii['pseudo_target_weighted_factor'] = float(candidate_radius)
        radii['coarse_expert_gate'] = float(gate_radii[target]['crossfit_h2_radius90'])

        target_raw, target_public = prepared[target]
        query_all = target_public.loc[target_public.year.eq(2024)].copy()
        scale = training_scale(train)
        weighted_point = np.full(len(query_all), np.nan)
        audit_monthly = []
        for month in sorted(query_all.clock.dt.month.unique()):
            query = query_all.loc[query_all.clock.dt.month.eq(month)].copy()
            center = query[list(GROUPS)].iloc[0].to_numpy(float)
            model, audit = fit_factor(train, train_fuel, center, scale, selected_tau, train_coarse)
            weighted_point[query_all.clock.dt.month.eq(month).to_numpy()] = predict_factor(model, query)
            audit_monthly.append({'month': int(month), 'tau': selected_tau_name, **audit})
        if not np.isfinite(weighted_point).all():
            raise ValueError(f'incomplete weighted prediction for {target}')
        pooled_point = predict(pooled, query_all)
        gate_frame = pd.read_csv(gate_predictions[target])
        if len(gate_frame) != len(query_all):
            raise ValueError(f'gating reference sample mismatch for {target}')
        forecast = {k: pooled_point[k] for k in BASELINES}
        forecast['pseudo_target_weighted_factor'] = weighted_point
        forecast['coarse_expert_gate'] = gate_frame['coarse_expert_gate'].to_numpy(float)
        path = out / f'{target.upper()}_heldout_predictions.csv.gz'
        pd.DataFrame({'time': target_raw.iloc[query_all.raw_row.to_numpy()].time.astype(str).to_numpy(),
                      **forecast}).to_csv(path, index=False, compression='gzip')
        sealed.append({'target': target.upper(), 'path': path.name, 'sha256': sha(path),
                       'hours': len(query_all), 'selected_tau': selected_tau_name})
        outputs[target] = (query_all, target_raw, path, radii)
        selections[target] = {
            'selected_tau': selected_tau_name,
            'inner_macro_mae': {str(k): float(v) for k, v in macro.items()},
            'pseudo_target_country_mae': inner.to_dict('records'),
            'target_month_weight_audit': audit_monthly,
            'candidate_h2_cross_country_radius90': float(candidate_radius),
            'baseline_h2_radius90': radii,
        }
        (out / f'{target.upper()}_inner_tuning.csv').write_text(tune.to_csv(index=False))
        print(target.upper(), 'outer forecast sealed', len(query_all), 'tau', selected_tau_name, flush=True)

    (out / 'selection_and_calibration.json').write_text(json.dumps(selections, indent=2))
    (out / 'prediction_seal.json').write_text(json.dumps(sealed, indent=2))
    if any(sha(Path(p)) != h for p, h in inputs.items()):
        raise RuntimeError('input changed during run')
    if any(sha(Path(p)) != h for p, h in code_hashes.items()):
        raise RuntimeError('code/protocol changed during run')

    rows = []
    for target, (query, raw, path, radius) in outputs.items():
        prediction = pd.read_csv(path)
        fuel = raw.iloc[query.raw_row.to_numpy()][list(FINE)].to_numpy(float)
        y = fuel @ FACTORS / fuel.sum(axis=1)
        for method in (*BASELINES, 'pseudo_target_weighted_factor', 'coarse_expert_gate'):
            point = prediction[method].to_numpy(float)
            r = float(radius[method])
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                error = y[mask] - point[mask]
                rows.append({
                    'target_country': target.upper(), 'method': method, 'month': int(month),
                    'hours': int(mask.sum()), 'mae': float(np.abs(error).mean()),
                    'shape_mae': shape_mae(query.loc[mask, 'clock'], y[mask], point[mask]),
                    'coverage90': float((np.abs(error) <= r).mean()), 'width90': 2 * r,
                    'interval_score90': float(np.mean(2 * r + 20 * np.maximum(np.abs(error) - r, 0))),
                })
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / 'monthly_metrics.csv', index=False)
    metric_cols = ['mae', 'shape_mae', 'coverage90', 'width90', 'interval_score90']
    country = monthly.groupby(['target_country', 'method'])[metric_cols].mean().reset_index()
    country.to_csv(out / 'country_metrics.csv', index=False)
    country.groupby('method')[metric_cols].mean().reset_index().to_csv(out / 'macro_metrics.csv', index=False)
    (out / 'status.json').write_text(json.dumps({
        'status': 'DEVELOPMENT_PSEUDOTARGET_COARSE_CONDITIONED_INSTANCE_REWEIGHTING',
        'training_labels': 'source country 2022 only; pseudo target labels used only for source-side bandwidth selection',
        'target_hourly_labels_used_before_seal': False,
        'tuning_period': 'source pseudo target 2023H1',
        'cross_country_calibration_period': 'source pseudo targets 2023H2',
        'outer_target_test_period': '2024',
        'monthly_inputs_historical_as_issued': False,
        'labels_measured_cif': False,
    }, indent=2))
    print(pd.read_csv(out / 'macro_metrics.csv').to_string(index=False))


if __name__ == '__main__':
    main()
