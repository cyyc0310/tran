"""Paired five-group M vs M+C country-held-out capacity ablation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from transcif.research.multicountry import FACTORS, FINE
from transcif.research.monthly_adaptation import weighted_quantile
from run_episodic_coldstart_transfer import (
    COUNTRIES, FEATURES, GROUPS5, DATA, make_panel, panel_labels, score,
)
from run_energy_charts_loco import load_country

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/FIVE_GROUP_CAPACITY_INCREMENT_PROTOCOL_20261101.md'
CAPACITY = ROOT / 'data/eurostat-capacity-energy-charts-r1/capacity.csv'
CAPACITY_MANIFEST = ROOT / 'data/eurostat-capacity-energy-charts-r1/capacity.manifest.json'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hgb() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=15,
        min_samples_leaf=100, l2_regularization=10., random_state=416,
        early_stopping=False)


def capacity_join(panel: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    lookup = capacity.set_index(['country', 'stat_year']).thermal_capacity_share
    keys = pd.MultiIndex.from_arrays(
        [out.country.str.upper(), out.year.astype(int) - 2],
        names=['country', 'stat_year'])
    out['capacity_thermal_share'] = lookup.reindex(keys).to_numpy()
    if out.capacity_thermal_share.isna().any():
        raise ValueError('Missing lagged annual capacity for one or more rows')
    return out


def region_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    # One region per country in this dataset; give each source country equal mass.
    counts = frame.groupby('country').country.transform('size').to_numpy(float)
    return 1. / (frame.country.nunique() * counts)


def fit_heads(train: pd.DataFrame, y: np.ndarray, fuel: np.ndarray,
              with_capacity: bool) -> dict:
    cols = [*FEATURES, *(('capacity_thermal_share',) if with_capacity else ())]
    w = region_equal_weights(train)
    models: dict = {'columns': cols}
    models['direct'] = hgb().fit(train[cols], y, sample_weight=w * len(w))
    # Match the existing five-group Ridge preprocessing exactly.
    scale = StandardScaler().fit(train[cols])
    ridge = Ridge(alpha=10.).fit(scale.transform(train[cols]), y,
                                 sample_weight=w * len(w))
    models['ridge'] = (scale, ridge)
    thermal = fuel[:, :4].sum(axis=1) + fuel[:, -1]
    numerator = fuel @ FACTORS
    ef = np.divide(numerator, thermal, out=np.full(len(fuel), 400.), where=thermal > 0)
    active = thermal > 0
    models['factor'] = hgb().fit(train.loc[active, cols], ef[active],
                                 sample_weight=w[active] * len(w))
    return models


def recomposition_reference(train: pd.DataFrame, fuel: np.ndarray,
                            query: pd.DataFrame) -> np.ndarray:
    weights = region_equal_weights(train)
    thermal = fuel[:, :4].sum(axis=1) + fuel[:, -1]
    emissions = fuel @ FACTORS
    macro_ef = float(np.sum(weights * emissions) / np.sum(weights * thermal))
    c5 = np.column_stack([thermal, fuel[:, 5], fuel[:, 4], fuel[:, 6], fuel[:, 7]])
    profiles = []
    for country in sorted(train.country.unique()):
        mask = train.country.to_numpy() == country
        block = pd.DataFrame(c5[mask], columns=GROUPS5)
        block['hour'] = train.loc[mask, 'clock'].dt.hour.to_numpy()
        profile = block.groupby('hour')[list(GROUPS5)].mean().reindex(range(24)).fillna(0.)
        profiles.append(profile.div(profile.mean().replace(0., np.nan)).fillna(1.).to_numpy())
    profile = np.mean(profiles, axis=0)
    mix = query[[f'{g}_lag2' for g in GROUPS5]].to_numpy(float)
    hourly = profile[query.clock.dt.hour.to_numpy()] * mix
    return np.clip(macro_ef * hourly[:, 0] / hourly.sum(axis=1), 0., 1000.)


def predict_heads(models: dict, frame: pd.DataFrame,
                  train: pd.DataFrame, fuel: np.ndarray) -> dict[str, np.ndarray]:
    cols = models['columns']
    direct = np.clip(models['direct'].predict(frame[cols]), 0., 1000.)
    scaler, ridge = models['ridge']
    ridge_pred = np.clip(ridge.predict(scaler.transform(frame[cols])), 0., 1000.)
    factor = np.clip(models['factor'].predict(frame[cols]), 400., 1000.)
    ref = recomposition_reference(train, fuel, frame)
    base_mix = frame[[f'{g}_lag2' for g in GROUPS5]].to_numpy(float)
    # Recover the same hourly thermal share used by the fixed-factor reference.
    w = region_equal_weights(train)
    thermal = fuel[:, :4].sum(axis=1) + fuel[:, -1]
    macro_ef = np.sum(w * (fuel @ FACTORS)) / np.sum(w * thermal)
    c5 = np.column_stack([thermal, fuel[:, 5], fuel[:, 4], fuel[:, 6], fuel[:, 7]])
    profiles = []
    for country in sorted(train.country.unique()):
        mask = train.country.to_numpy() == country
        block = pd.DataFrame(c5[mask], columns=GROUPS5)
        block['hour'] = train.loc[mask, 'clock'].dt.hour.to_numpy()
        prof = block.groupby('hour')[list(GROUPS5)].mean().reindex(range(24)).fillna(0.)
        profiles.append(prof.div(prof.mean().replace(0., np.nan)).fillna(1.).to_numpy())
    profile = np.mean(profiles, axis=0)
    hourly = profile[frame.clock.dt.hour.to_numpy()] * base_mix
    thermal_share = np.divide(hourly[:, 0], hourly.sum(axis=1),
                              out=np.zeros(len(frame)), where=hourly.sum(axis=1) > 0)
    factor_pred = np.clip(factor * thermal_share, 0., 1000.)
    # Numerical identity check protects the recomposition implementation.
    if not np.allclose(ref, np.clip(macro_ef * thermal_share, 0., 1000.), atol=1e-10):
        raise ValueError('Factor-head shape differs from frozen five-group recomposition')
    return {'direct': direct, 'ridge': ridge_pred, 'factor': factor_pred}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    data_manifest = json.loads((DATA / 'acquisition_manifest.json').read_text())
    capacity = pd.read_csv(CAPACITY)
    raw, panels, audits = {}, {}, {}
    input_paths = [DATA / f'{c}_2022_2024.json' for c in COUNTRIES]
    input_paths += [DATA / 'acquisition_manifest.json', DATA / 'audit.json', CAPACITY,
                    CAPACITY_MANIFEST]
    code_paths = [Path(__file__), PROTOCOL, ROOT / 'scripts/run_episodic_coldstart_transfer.py',
                  ROOT / 'scripts/run_energy_charts_loco.py',
                  ROOT / 'src/transcif/research/multicountry.py',
                  ROOT / 'src/transcif/research/monthly_adaptation.py']
    hashes = {'inputs': {str(p): sha(p) for p in input_paths},
              'code': {str(p): sha(p) for p in code_paths}}
    for c in COUNTRIES:
        raw[c.upper()], audits[c] = load_country(c, data_manifest)
        panel, _ = make_panel(raw[c.upper()])
        panel['country'] = c.upper()
        panels[c.upper()] = capacity_join(panel, capacity)
    (out / 'startup_hashes.json').write_text(json.dumps(hashes, indent=2))
    (out / 'input_audit.json').write_text(json.dumps(audits, indent=2, default=str))

    predictions, seals, metric_rows = [], [], []
    for target in COUNTRIES:
        target = target.upper()
        source_names = sorted(c for c in panels if c != target)
        train = pd.concat([panels[c].loc[panels[c].year.eq(2022)] for c in source_names],
                          ignore_index=True)
        train_fuel = np.concatenate([
            raw[c].iloc[panels[c].loc[panels[c].year.eq(2022), 'raw_row'].to_numpy()][
                list(FINE)].to_numpy(float) for c in source_names])
        train_y = np.concatenate([
            panel_labels(raw[c], panels[c].loc[panels[c].year.eq(2022)].reset_index(drop=True))
            for c in source_names])
        calibration = pd.concat([panels[c].loc[
            panels[c].year.eq(2023) & panels[c].clock.dt.month.le(6)]
            for c in source_names], ignore_index=True)
        cal_y = np.concatenate([
            panel_labels(raw[c], panels[c].loc[
                panels[c].year.eq(2023) & panels[c].clock.dt.month.le(6)].reset_index(drop=True))
            for c in source_names])
        query = panels[target].loc[panels[target].year.eq(2024)].reset_index(drop=True)
        cal_w = region_equal_weights(calibration)
        cal_predictions: dict[str, np.ndarray] = {
            'M_five_group_recomposition': recomposition_reference(train, train_fuel, calibration)}
        query_forecast: dict[str, np.ndarray] = {
            'M_five_group_recomposition': recomposition_reference(train, train_fuel, query)}
        for budget_name, has_capacity in (('M', False), ('MplusC', True)):
            models = fit_heads(train, train_y, train_fuel, has_capacity)
            cal_pred = predict_heads(models, calibration, train, train_fuel)
            test_pred = predict_heads(models, query, train, train_fuel)
            for method in ('direct', 'factor', 'ridge'):
                tag = f'{budget_name}_{method}'
                cal_predictions[tag] = cal_pred[method]
                query_forecast[tag] = test_pred[method]
        sealed = {'time': query.clock.astype(str).to_numpy(), **query_forecast}
        for method, values in query_forecast.items():
            if len(values) != len(query) or not np.isfinite(values).all():
                raise ValueError(f'Invalid or incomplete held-out prediction: {method}')
        radii = {name: float(weighted_quantile(np.abs(cal_y - pred), cal_w, .9))
                 for name, pred in cal_predictions.items()}
        path = out / f'{target}_heldout_predictions.csv.gz'
        pd.DataFrame(sealed).to_csv(path, index=False, compression='gzip')
        radius_path = out / f'{target}_source_radii.json'
        radius_path.write_text(json.dumps(radii, indent=2))
        seals.append({'target_country': target, 'prediction_file': path.name,
                      'prediction_sha256': sha(path), 'radii_file': radius_path.name,
                      'radii_sha256': sha(radius_path), 'hours': len(query)})
        predictions.append((target, query, path, radii))
        if any(sha(Path(p)) != value for p, value in hashes['inputs'].items()):
            raise RuntimeError('Input changed during fitting')
        if any(sha(Path(p)) != value for p, value in hashes['code'].items()):
            raise RuntimeError('Code or protocol changed during fitting')
        print(f'{target}: sealed {len(query)} target hours', flush=True)
    (out / 'prediction_seal.json').write_text(json.dumps(seals, indent=2))

    # Read target labels only after all held-out predictions and radii are sealed.
    for target, query, path, radii in predictions:
        y = panel_labels(raw[target], query)
        sealed = pd.read_csv(path)
        for method, radius in radii.items():
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                metric_rows.append({'target_country': target, 'method': method, 'month': int(month),
                    **score(y[mask], sealed.loc[mask, method].to_numpy(float), radius,
                            query.loc[mask, 'clock'].reset_index(drop=True))})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / 'monthly_metrics.csv', index=False)
    cols = ['mae', 'shape_mae', 'coverage90', 'width90', 'interval_score90']
    country = metrics.groupby(['target_country', 'method'])[cols].mean().reset_index()
    country.to_csv(out / 'country_metrics.csv', index=False)
    macro = country.groupby('method')[cols].mean().reset_index()
    macro.to_csv(out / 'macro_metrics.csv', index=False)
    (out / 'status.json').write_text(json.dumps({
        'status': 'DEVELOPMENT_FIVE_GROUP_M_VS_MC_COUNTRY_HELDOUT',
        'target_countries': list(COUNTRIES), 'training_year': 2022,
        'source_calibration': '2023H1', 'target_year': 2024,
        'target_budget_M': 'five groups t-2 through t-7, calendar and five t-2 hourly interactions',
        'target_budget_MplusC': 'M plus annual Eurostat thermal capacity share with two-year statistical lag',
        'capacity_vintage': 'current revised 2026-10-22 extract; lagged statistical simulation',
        'target_hourly_labels_used_before_seal': False,
        'target_labels': 'fixed-factor generation-side proxy; all countries previously viewed',
        'new_confirmation': False,
    }, indent=2))
    print(macro.to_string(index=False))


if __name__ == '__main__':
    main()
