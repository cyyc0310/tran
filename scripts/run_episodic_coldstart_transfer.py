"""Compare country-episodic training with ERM under a China-style monthly budget."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.func import functional_call

from transcif.research.multicountry import FACTORS, FINE
from transcif.research.monthly_adaptation import weighted_quantile
from run_energy_charts_loco import load_country

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/energy-charts-v2-panel-r1'
PROTOCOL = ROOT / 'docs/EPISODIC_COLDSTART_TRANSFER_PROTOCOL_20261101.md'
COUNTRIES = ('nl', 'es', 'be', 'at', 'se')
GROUPS5 = ('thermal', 'hydro', 'nuclear', 'wind', 'solar')
FEATURES = tuple([f'{g}_lag{i}' for i in range(2, 8) for g in GROUPS5] +
                 ['thermal_mean', 'thermal_std', 'thermal_change', 'total_cv', 'observations'] +
                 [f'{x}_{s}' for x in ('hour', 'dow', 'month') for s in ('sin', 'cos')] +
                 [f'{g}_hour_{s}' for g in GROUPS5 for s in ('sin', 'cos')])
SEEDS = (412, 413, 414)
STEPS = 500
BATCH = 64
INNER_LR = 0.01
OUTER_LR = 0.001
META_WEIGHT = 0.5


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_panel(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build five public classes; unreported generation is absorbed into thermal."""
    # Current-year monthly totals are usable only as lagged inputs: the exact
    # joins below map period p to statistics no newer than p-2.
    past = raw
    thermal = past[list(FINE[:4])].sum(axis=1) + past.other_renewable
    values = pd.DataFrame({'thermal': thermal, 'hydro': past.hydro, 'nuclear': past.nuclear,
                           'wind': past.wind, 'solar': past.solar}, index=past.index)
    period = past.clock.dt.to_period('M')
    values['period'] = period
    months = values.groupby('period')[list(GROUPS5)].sum()
    expected = past.groupby(period).expected_hours.first()
    coverage = values.groupby('period').size() / expected
    months = months.loc[coverage.reindex(months.index).ge(.99)]
    period = raw.clock.dt.to_period('M')
    out = pd.DataFrame({'clock': raw.clock, 'period': period})
    out['raw_row'] = np.arange(len(raw))
    for name, v, denom in (('hour', raw.clock.dt.hour, 24),
                           ('dow', raw.clock.dt.dayofweek, 7),
                           ('month', raw.clock.dt.month, 12)):
        out[f'{name}_sin'] = np.sin(2 * np.pi * v / denom)
        out[f'{name}_cos'] = np.cos(2 * np.pi * v / denom)
    shares = months.div(months.sum(axis=1), axis=0)
    for lag in range(2, 8):
        lagged = shares.copy()
        lagged.index += lag
        lagged.columns = [f'{g}_lag{lag}' for g in GROUPS5]
        out = out.join(lagged, on='period')
    history = shares.copy()
    history['total'] = months.sum(axis=1)
    for p in out.period.unique():
        idx = (history.index <= p - 2) & (history.index >= p - 7)
        block = history.loc[idx]
        if len(block) == 0 or p - 2 not in block.index:
            continue
        mask = out.period.eq(p)
        out.loc[mask, 'thermal_mean'] = block.thermal.mean()
        out.loc[mask, 'thermal_std'] = block.thermal.std(ddof=0)
        out.loc[mask, 'thermal_change'] = block.thermal.iloc[-1] - block.thermal.iloc[0]
        totals = block.total.to_numpy(float)
        out.loc[mask, 'total_cv'] = totals.std() / totals.mean() if totals.mean() else 0.
        out.loc[mask, 'observations'] = len(block) / 6.
    for g in GROUPS5:
        for suffix in ('sin', 'cos'):
            out[f'{g}_hour_{suffix}'] = out[f'{g}_lag2'] * out[f'hour_{suffix}']
    out['year'] = raw.clock.dt.year.to_numpy()
    valid = out[list(FEATURES)].notna().all(axis=1).to_numpy()
    return out.loc[valid].reset_index(drop=True), months


def panel_labels(raw: pd.DataFrame, panel: pd.DataFrame) -> np.ndarray:
    """Read fixed-factor hourly responses only for training/calibration/scoring rows."""
    fuel = raw.iloc[panel.raw_row.to_numpy()][list(FINE)].to_numpy(float)
    return fuel @ FACTORS / fuel.sum(axis=1)


class SmallMLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, 64), nn.GELU(),
                                 nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def balanced_indices(groups: dict[str, np.ndarray], rng: np.random.Generator,
                     batch: int) -> dict[str, np.ndarray]:
    return {name: rng.choice(idx, size=batch, replace=len(idx) < batch)
            for name, idx in groups.items()}


def train_network(train: pd.DataFrame, labels: dict[str, np.ndarray],
                  kind: str, seed: int) -> tuple[SmallMLP, StandardScaler, float, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    groups = {c: np.flatnonzero(train.country.to_numpy() == c) for c in sorted(labels)}
    scaler = StandardScaler().fit(train[list(FEATURES)])
    x = torch.tensor(scaler.transform(train[list(FEATURES)]), dtype=torch.float32)
    yparts = np.concatenate([labels[c] for c in sorted(labels)])
    # training frames are concatenated in sorted country order; record offsets explicitly
    ymean, ystd = float(yparts.mean()), float(yparts.std())
    if ystd <= 0 or not np.isfinite(ystd):
        raise ValueError('invalid training label scale')
    y = torch.tensor((yparts - ymean) / ystd, dtype=torch.float32)
    model = SmallMLP(len(FEATURES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=OUTER_LR, weight_decay=1e-4)
    for step in range(STEPS):
        picked = balanced_indices(groups, rng, BATCH)
        if kind == 'erm':
            all_i = np.concatenate(list(picked.values()))
            loss = ((model(x[all_i]) - y[all_i]) ** 2).mean()
        elif kind == 'mldg':
            meta = sorted(groups)[step % len(groups)]
            train_i = np.concatenate([picked[c] for c in sorted(groups) if c != meta])
            test_i = picked[meta]
            base_params = dict(model.named_parameters())
            train_loss = ((functional_call(model, base_params, (x[train_i],)) - y[train_i]) ** 2).mean()
            grads = torch.autograd.grad(train_loss, tuple(base_params.values()), create_graph=True)
            adapted = {name: param - INNER_LR * grad
                       for (name, param), grad in zip(base_params.items(), grads)}
            test_loss = ((functional_call(model, adapted, (x[test_i],)) - y[test_i]) ** 2).mean()
            loss = train_loss + META_WEIGHT * test_loss
        else:
            raise ValueError(kind)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
    model.eval()
    return model, scaler, ymean, ystd


def infer_network(fitted: tuple[SmallMLP, StandardScaler, float, float],
                  frame: pd.DataFrame) -> np.ndarray:
    model, scaler, ymean, ystd = fitted
    with torch.no_grad():
        x = torch.tensor(scaler.transform(frame[list(FEATURES)]), dtype=torch.float32)
        return np.clip(model(x).numpy() * ystd + ymean, 0., 1000.)


def recompose(train: pd.DataFrame, yfuel: np.ndarray, query: pd.DataFrame) -> np.ndarray:
    weights = 1. / train.groupby('country').country.transform('size').to_numpy(float)
    thermal = yfuel[:, :4].sum(axis=1) + yfuel[:, -1]
    ef = float(np.sum(weights * (yfuel @ FACTORS)) / np.sum(weights * thermal))
    profile_rows = []
    c5 = np.column_stack([thermal, yfuel[:, 5], yfuel[:, 4], yfuel[:, 6], yfuel[:, 7]])
    for country in sorted(train.country.unique()):
        mask = train.country.to_numpy() == country
        block = pd.DataFrame(c5[mask], columns=GROUPS5)
        block['hour'] = train.loc[mask, 'clock'].dt.hour.to_numpy()
        profile = block.groupby('hour')[list(GROUPS5)].mean().reindex(range(24)).fillna(0.)
        profile_rows.append(profile.div(profile.mean().replace(0., np.nan)).fillna(1.).to_numpy())
    profile = np.mean(profile_rows, axis=0)
    coarse = query[[f'{g}_lag2' for g in GROUPS5]].to_numpy(float)
    hourly = profile[query.clock.dt.hour.to_numpy()] * coarse
    return np.clip(ef * hourly[:, 0] / hourly.sum(axis=1), 0., 1000.)


def score(y: np.ndarray, pred: np.ndarray, radius: float,
          clock: pd.Series) -> dict:
    error = y - pred
    tmp = pd.DataFrame({'day': clock.dt.strftime('%Y-%m-%d').to_numpy(), 'y': y, 'p': pred})
    centered = tmp[['y', 'p']] - tmp.groupby('day')[['y', 'p']].transform('mean')
    return {'hours': int(len(y)), 'mae': float(np.abs(error).mean()),
            'shape_mae': float(np.abs(centered.y - centered.p).mean()),
            'coverage90': float((np.abs(error) <= radius).mean()), 'width90': float(2 * radius),
            'interval_score90': float(np.mean(2 * radius + 20 * np.maximum(np.abs(error) - radius, 0.)))}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((DATA / 'acquisition_manifest.json').read_text())
    input_files = [DATA / f'{c}_2022_2024.json' for c in COUNTRIES]
    input_files += [DATA / 'acquisition_manifest.json', DATA / 'audit.json']
    code_files = [Path(__file__), PROTOCOL, ROOT / 'scripts/run_energy_charts_loco.py',
                  ROOT / 'src/transcif/research/multicountry.py',
                  ROOT / 'src/transcif/research/monthly_adaptation.py']
    hashes = {'inputs': {str(p): sha(p) for p in input_files},
              'code': {str(p): sha(p) for p in code_files}}
    raw, load_audit = {}, {}
    panels, months, labels = {}, {}, {}
    for country in COUNTRIES:
        raw[country.upper()], load_audit[country] = load_country(country, manifest)
        panel, months[country] = make_panel(raw[country.upper()])
        panel['country'] = country
        panels[country] = panel
    (out / 'startup_hashes.json').write_text(json.dumps(hashes, indent=2))
    (out / 'input_audit.json').write_text(json.dumps(load_audit, indent=2, default=str))

    rows, seals = [], []
    for target in COUNTRIES:
        sources = sorted(c for c in COUNTRIES if c != target)
        train_parts = [panels[c].loc[panels[c].year.eq(2022)].reset_index(drop=True) for c in sources]
        train = pd.concat(train_parts, ignore_index=True)
        train_labels = {}
        for c in sources:
            rows_2022 = panels[c].loc[panels[c].year.eq(2022)].reset_index(drop=True)
            train_labels[c] = panel_labels(raw[c.upper()], rows_2022)
        train_fuels = np.concatenate([
            raw[c.upper()].iloc[panels[c].loc[panels[c].year.eq(2022), 'raw_row'].to_numpy()][
                list(FINE)].to_numpy(float) for c in sources])
        if len(train) != len(train_fuels):
            raise ValueError('training feature/label row mismatch')
        calibration_parts = [panels[c].loc[panels[c].year.eq(2023) &
                                                   panels[c].clock.dt.month.le(6)].reset_index(drop=True)
                             for c in sources]
        calibration = pd.concat(calibration_parts, ignore_index=True)
        calibration_y = np.concatenate([
            panel_labels(raw[c.upper()], part)
            for c in sources for part in [panels[c].loc[
                panels[c].year.eq(2023) & panels[c].clock.dt.month.le(6)].reset_index(drop=True)]])
        predictions_cal: dict[str, list[np.ndarray]] = {'five_group_recomposition': []}
        predictions_test: dict[str, list[np.ndarray]] = {'five_group_recomposition': []}
        for method in ('erm', 'mldg'):
            predictions_cal[method], predictions_test[method] = [], []
            for seed in SEEDS:
                fitted = train_network(train, train_labels, method, seed)
                predictions_cal[method].append(infer_network(fitted, calibration))
                query = panels[target].loc[panels[target].year.eq(2024)].reset_index(drop=True)
                predictions_test[method].append(infer_network(fitted, query))
        # Same-feature conventional baselines.
        xtrain = train[list(FEATURES)]
        xcal, xtest = calibration[list(FEATURES)], panels[target].loc[
            panels[target].year.eq(2024), list(FEATURES)]
        ytrain = np.concatenate([train_labels[c] for c in sources])
        train_weights = np.zeros(len(train), float)
        for country in sources:
            mask = train.country.eq(country).to_numpy()
            train_weights[mask] = 1. / (len(sources) * mask.sum())
        ridge_scale = StandardScaler().fit(xtrain)
        ridge = Ridge(alpha=10.).fit(ridge_scale.transform(xtrain), ytrain,
                                     sample_weight=train_weights * len(train))
        hgb = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=15,
                                            min_samples_leaf=100, l2_regularization=10.,
                                            random_state=416, early_stopping=False).fit(
                                                xtrain, ytrain, sample_weight=train_weights * len(train))
        for method, model in (('ridge5', (ridge_scale, ridge)), ('hgb5', hgb)):
            if method == 'ridge5':
                sc, mdl = model
                predictions_cal[method] = [np.clip(mdl.predict(sc.transform(xcal)), 0., 1000.)]
                predictions_test[method] = [np.clip(mdl.predict(sc.transform(xtest)), 0., 1000.)]
            else:
                predictions_cal[method] = [np.clip(model.predict(xcal), 0., 1000.)]
                predictions_test[method] = [np.clip(model.predict(xtest), 0., 1000.)]
        predictions_cal['five_group_recomposition'] = [recompose(train, train_fuels, calibration)]
        query = panels[target].loc[panels[target].year.eq(2024)].reset_index(drop=True)
        predictions_test['five_group_recomposition'] = [recompose(train, train_fuels, query)]
        # Source-country-equal absolute residual quantiles calibrate all methods.
        cal_country = calibration.country.to_numpy()
        cal_weights = np.zeros(len(calibration), float)
        for c in sources:
            mask = cal_country == c
            cal_weights[mask] = 1. / (len(sources) * mask.sum())
        sealed = {'time': query.clock.astype(str).to_numpy()}
        radii = {}
        for method, parts in predictions_cal.items():
            pred_cal = np.mean(parts, axis=0)
            radii[method] = weighted_quantile(np.abs(calibration_y - pred_cal), cal_weights, .9)
            sealed[method] = np.mean(predictions_test[method], axis=0)
        path = out / f'{target.upper()}_predictions.csv.gz'
        pd.DataFrame(sealed).to_csv(path, index=False, compression='gzip')
        seals.append({'target': target.upper(), 'file': path.name, 'sha256': sha(path),
                      'predicted_hours': len(query), 'target_labels_scored': False})
        (out / f'{target.upper()}_radii.json').write_text(json.dumps(radii, indent=2))
        if any(sha(Path(p)) != digest for p, digest in hashes['inputs'].items()):
            raise RuntimeError('input changed during model fit')
        if any(sha(Path(p)) != digest for p, digest in hashes['code'].items()):
            raise RuntimeError('code/protocol changed during model fit')
    (out / 'prediction_seal.json').write_text(json.dumps(seals, indent=2))

    # Only after all five outer targets are sealed, score 2024 fixed-factor labels.
    for target in COUNTRIES:
        panel = panels[target]
        query = panel.loc[panel.year.eq(2024)].reset_index(drop=True)
        rows_2024 = panel.loc[panel.year.eq(2024)].reset_index(drop=True)
        y = panel_labels(raw[target.upper()], rows_2024)
        pred = pd.read_csv(out / f'{target.upper()}_predictions.csv.gz')
        radii = json.loads((out / f'{target.upper()}_radii.json').read_text())
        for method, radius in radii.items():
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                rows.append({'target_country': target.upper(), 'method': method, 'month': int(month),
                             **score(y[mask], pred.loc[mask, method].to_numpy(float),
                                     float(radius), query.loc[mask, 'clock'].reset_index(drop=True))})
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / 'monthly_metrics.csv', index=False)
    metrics = ['mae', 'shape_mae', 'coverage90', 'width90', 'interval_score90']
    country = monthly.groupby(['target_country', 'method'])[metrics].mean().reset_index()
    country.to_csv(out / 'country_metrics.csv', index=False)
    country.groupby('method')[metrics].mean().to_csv(out / 'macro_metrics.csv')
    status = {'status': 'DEVELOPMENT_FIVE_COUNTRY_EPISODIC_COLDSTART',
              'training_year': 2022, 'source_calibration': '2023H1', 'test_year': 2024,
              'target_budget': 'five technology monthly shares t-2 through t-7 + calendar',
              'target_hourly_inputs_used': False, 'target_labels_used_before_seal': False,
              'monthly_vintage': 'revised hourly reconstruction with simulated t-2 delay',
              'label': 'fixed-factor generation-side proxy', 'mldg_steps': STEPS,
              'seeds': list(SEEDS)}
    (out / 'status.json').write_text(json.dumps(status, indent=2))
    print(country.groupby('method')[metrics].mean().to_string())


if __name__ == '__main__':
    main()
