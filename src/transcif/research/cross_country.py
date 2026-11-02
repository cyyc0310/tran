"""Calendar + delayed coarse-month inputs for country-held-out development.

Labels are fixed-factor domestic-generation proxies. Monthly inputs emulate a
release delay on revised hourly-derived aggregates, not historical releases.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FUELS = ('coal', 'gas', 'oil', 'other', 'nuclear', 'hydro', 'wind', 'solar')
COARSE = ('thermal', 'hydro', 'nuclear', 'wind', 'solar')
EF = np.array([1000., 400., 800., 400., 0., 0., 0., 0.])
ZONES = {'BPAT': 'America/Los_Angeles', 'CISO': 'America/Los_Angeles',
         'ERCO': 'America/Chicago', 'FPL': 'America/New_York',
         'ISNE': 'America/New_York', 'MISO': 'America/Chicago',
         'NYIS': 'America/New_York'}
CALENDAR = tuple(f'{v}_{s}' for v in ('hour', 'dow', 'month') for s in ('sin', 'cos'))
FEATURES = (*CALENDAR, *COARSE,
            *(f'{f}_hour_{s}' for f in COARSE for s in ('sin', 'cos')))
METHODS = ('anchor700', 'source_thermal_ef', 'calendar_ridge', 'coarse_ridge',
           'coarse_boosting', 'coarse_recomposition')


def load_inputs(root: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Load existing source files; retain only nonnegative complete hourly rows."""
    frames, audit = {}, {}
    for region, zone in ZONES.items():
        path = root / f'results/research/data/{region}_hourly.csv'
        columns = ['time', *(f'generation_{f}' for f in FUELS), 'valid']
        raw = pd.read_csv(path, usecols=columns)
        stamp = pd.to_datetime(raw.time, utc=True, errors='raise')
        if stamp.duplicated().any():
            raise ValueError(f'duplicate EIA timestamps: {region}')
        values = raw[[f'generation_{f}' for f in FUELS]].to_numpy(float)
        valid = (raw.valid.astype(str).str.lower().eq('true').to_numpy()
                 & np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
                 & (values.sum(axis=1) > 0))
        frame = pd.DataFrame(values[valid], columns=FUELS)
        frame['time'] = stamp[valid].reset_index(drop=True)
        frame['clock'] = frame.time.dt.tz_convert(zone).dt.tz_localize(None)
        frame['expected_hours'] = frame.clock.map(
            lambda t: ((t.to_period('M') + 1).start_time.tz_localize(zone)
                       - t.to_period('M').start_time.tz_localize(zone)).total_seconds() / 3600)
        frame['country'], frame['region'] = 'US', region
        frames[region] = frame
        audit[region] = {'path': str(path.relative_to(root)), 'raw_rows': len(raw),
                         'valid_rows': int(valid.sum()), 'calendar': zone}
    path = root / 'data/neso-gb/df_fuel_ckan.csv'
    generation_columns = ['DATETIME','GAS','COAL','NUCLEAR','WIND','WIND_EMB',
                          'HYDRO','BIOMASS','OTHER','SOLAR']
    raw = pd.read_csv(path, usecols=generation_columns)
    raw['stamp'] = pd.to_datetime(raw.DATETIME, errors='raise')
    if raw.stamp.duplicated().any() or not raw.stamp.dt.minute.isin([0, 30]).all():
        raise ValueError('NESO requires unique half-hour timestamps')
    raw = raw.loc[raw.stamp.dt.year.between(2022, 2024)].copy()
    raw['coal'], raw['gas'], raw['oil'] = raw.COAL, raw.GAS, 0.
    raw['other'] = raw.BIOMASS + raw.OTHER
    raw['nuclear'], raw['hydro'] = raw.NUCLEAR, raw.HYDRO
    raw['wind'], raw['solar'] = raw.WIND + raw.WIND_EMB, raw.SOLAR
    values = raw[list(FUELS)].to_numpy(float)
    raw = raw.loc[np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)].copy()
    raw['clock'] = raw.stamp.dt.floor('h')
    grouped = raw.groupby('clock')[list(FUELS)].mean()
    grouped = grouped.loc[raw.groupby('clock').size().eq(2)].reset_index()
    grouped = grouped.loc[grouped[list(FUELS)].sum(axis=1).gt(0)].copy()
    grouped['time'] = grouped.clock
    grouped['expected_hours'] = grouped.clock.dt.days_in_month * 24
    grouped['country'], grouped['region'] = 'GB', 'GB'
    frames['GB'] = grouped
    audit['GB'] = {'path': str(path.relative_to(root)), 'valid_rows': len(grouped),
                   'calendar': 'source naive clock; timezone not inferred',
                   'hourly_energy': 'sum of two half-hour MW averages / 2'}
    return frames, audit


def monthly_inputs(frame: pd.DataFrame, lag: int = 2) -> pd.DataFrame:
    """Attach exact t-lag month features; never substitute a newer month.

    At least 99% of expected hourly observations must be present. Totals remain
    observed-subset totals; no rescaling or missing-hour imputation is performed.
    """
    if lag < 2:
        raise ValueError('main contract requires lag >= 2')
    out = frame[['time', 'clock', 'country', 'region']].copy()
    clock = pd.to_datetime(out.clock)
    out['period'] = clock.dt.to_period('M')
    source = frame[list(FUELS)].copy()
    source['period'] = out.period
    months = source.groupby('period')[list(FUELS)].sum()
    counts = source.groupby('period').size()
    expected = frame.assign(period=out.period).groupby('period').expected_hours.first()
    months['coverage'] = counts / expected
    months = months.loc[months.coverage >= .99].copy()
    months['thermal'] = months[['coal', 'gas', 'oil', 'other']].sum(axis=1)
    total = months[list(FUELS)].sum(axis=1)
    shares = months[list(COARSE)].div(total, axis=0)
    shares['observed_month'] = shares.index.astype(str)
    shares.index += lag
    out = out.join(shares, on='period')
    out['year'] = clock.dt.year
    for name, values, period in [('hour', clock.dt.hour, 24),
                                 ('dow', clock.dt.dayofweek, 7),
                                 ('month', clock.dt.month, 12)]:
        out[f'{name}_sin'] = np.sin(2 * np.pi * values / period)
        out[f'{name}_cos'] = np.cos(2 * np.pi * values / period)
    for fuel in COARSE:
        for suffix in ('sin', 'cos'):
            out[f'{fuel}_hour_{suffix}'] = out[fuel] * out[f'hour_{suffix}']
    return out


def labels(frame: pd.DataFrame) -> np.ndarray:
    fuel = frame[list(FUELS)].to_numpy(float)
    return fuel @ EF / fuel.sum(axis=1)


def fit_models(train: pd.DataFrame, fuel: np.ndarray) -> dict:
    """All training responses and auxiliary fuel labels come from source only."""
    y = fuel @ EF / fuel.sum(axis=1)
    weights = 1 / train.groupby('region').region.transform('size').to_numpy(float)
    weights *= len(weights) / weights.sum()
    models = {}
    for name, columns in [('calendar_ridge', CALENDAR), ('coarse_ridge', FEATURES)]:
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.))
        model.fit(train[list(columns)], y, ridge__sample_weight=weights)
        models[name] = model
    model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                                          min_samples_leaf=100, l2_regularization=10.,
                                          random_state=412)
    model.fit(train[list(FEATURES)], y, sample_weight=weights)
    models['coarse_boosting'] = model
    thermal = fuel[:, :4].sum(axis=1)
    models['thermal_ef'] = float(np.sum(weights * (fuel @ EF)) / np.sum(weights * thermal))
    # Equal-region average of within-coarse-category normalized hourly shapes.
    coarse = np.column_stack([thermal, fuel[:, 5], fuel[:, 4], fuel[:, 6], fuel[:, 7]])
    profiles = []
    for region in train.region.unique():
        mask = train.region.to_numpy() == region
        g = pd.DataFrame(coarse[mask], columns=COARSE)
        g['hour'] = train.loc[mask, 'clock'].dt.hour.to_numpy()
        p = g.groupby('hour')[list(COARSE)].mean().reindex(range(24)).fillna(0.)
        mean = p.mean()
        profiles.append(p.div(mean.replace(0., np.nan)).fillna(1.).to_numpy())
    models['profile'] = np.mean(profiles, axis=0)
    return models


def predict_models(models: dict, inputs: pd.DataFrame) -> dict[str, np.ndarray]:
    """Frozen inference receives no hourly generation or target CIF fields."""
    result = {'anchor700': 700. * inputs.thermal.to_numpy(),
              'source_thermal_ef': models['thermal_ef'] * inputs.thermal.to_numpy()}
    for name, cols in [('calendar_ridge', CALENDAR), ('coarse_ridge', FEATURES),
                       ('coarse_boosting', FEATURES)]:
        result[name] = models[name].predict(inputs[list(cols)])
    q = models['profile'][inputs.clock.dt.hour.to_numpy()] * inputs[list(COARSE)].to_numpy()
    result['coarse_recomposition'] = models['thermal_ef'] * q[:, 0] / q.sum(axis=1)
    return {name: np.clip(value, 0., 1000.) for name, value in result.items()}


def metrics(inputs: pd.DataFrame, y: np.ndarray, point: np.ndarray,
            bias: float, sigma: float) -> dict:
    """Source-calibrated Gaussian reference; no target coverage guarantee."""
    point = point + bias
    error = y - point
    z = error / sigma
    crps = sigma * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-z*z/2) / np.sqrt(2*np.pi)
                    - 1 / np.sqrt(np.pi))
    radius = 1.6448536269514722 * sigma
    tmp = pd.DataFrame({'day': inputs.clock.dt.strftime('%Y-%m-%d').to_numpy(),
                        'y': y, 'p': point})
    centered = tmp[['y', 'p']] - tmp.groupby('day')[['y', 'p']].transform('mean')
    return {'hours': len(y), 'mae': float(np.abs(error).mean()),
            'shape_mae': float(np.abs(centered.y - centered.p).mean()),
            'crps_gaussian': float(crps.mean()),
            'coverage90': float((np.abs(error) <= radius).mean()),
            'width90': float(2*radius),
            'interval_score90': float((2*radius + 20*np.maximum(np.abs(error)-radius, 0)).mean())}
