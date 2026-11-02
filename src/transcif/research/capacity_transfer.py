"""Public coal/gas capacity proxies as a separate cross-country information budget."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from transcif.research.multicountry import FACTORS, GROUPS, HISTORY, weights

METHODS = (
    'capacity_anchor_flat', 'capacity_anchor_shaped', 'capacity_source_rate',
    'capacity_ridge', 'capacity_direct', 'capacity_factor',
)
COLUMNS = (*HISTORY, 'capacity_coal_share')


def load_capacity(path: Path, earliest_query: str = '2022-01-01') -> dict:
    """Read only frozen public metadata, with a conservative availability gate."""
    table = json.loads(path.read_text())
    cutoff = pd.Timestamp(earliest_query, tz='UTC')
    for region, item in table['regions'].items():
        values = np.array([item['coal_mw'], item['gas_mw']], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
            raise ValueError(f'invalid public coal/gas capacity for {region}')
        available = pd.Timestamp(item['available_by_utc'])
        if available.tzinfo is None or available > cutoff:
            raise ValueError(f'capacity unavailable by earliest query: {region}')
    return table


def add_capacity(frame: pd.DataFrame, table: dict) -> pd.DataFrame:
    """A reported stock proportion is a predictor, never a hard dispatch bound."""
    out = frame.copy()
    regions = table['regions']
    missing = set(out.region.unique()) - set(regions)
    if missing:
        raise ValueError(f'missing capacity regions: {sorted(missing)}')
    for fuel in ('coal', 'gas'):
        out[f'capacity_{fuel}_mw'] = out.region.map({r: v[f'{fuel}_mw'] for r, v in regions.items()})
    out['capacity_coal_share'] = out.capacity_coal_mw / (out.capacity_coal_mw + out.capacity_gas_mw)
    return out


def nominal_rates(frame: pd.DataFrame, fuel: np.ndarray) -> list[float]:
    """Equal countries, then equal positive-stock regions, separately per fuel."""
    rates = []
    for index, name in enumerate(('coal', 'gas')):
        stock = frame[f'capacity_{name}_mw'].to_numpy()
        active = stock > 0
        if not active.any():
            raise ValueError(f'no positive source capacity for {name}')
        # Metadata populations differ from operating telemetry; this is a nominal
        # rate and may exceed one. No target generation is used in this estimate.
        active_weights = weights(frame.loc[active])
        rates.append(float(np.average(fuel[active, index] / stock[active], weights=active_weights)))
    return rates


def fit(frame: pd.DataFrame, fuel: np.ndarray, reference: dict) -> dict:
    w = weights(frame)
    y = fuel @ FACTORS / fuel.sum(axis=1)
    thermal = fuel[:, :4].sum(axis=1)
    thermal_y = np.divide(fuel @ FACTORS, thermal, out=np.full(len(fuel), 400.), where=thermal > 0)
    models = {'profile': reference['profile'], 'source_rates': nominal_rates(frame, fuel)}
    scaler = StandardScaler().fit(frame[list(COLUMNS)], sample_weight=w)
    ridge = Ridge(alpha=10.).fit(scaler.transform(frame[list(COLUMNS)]), y, sample_weight=w * len(w))
    models['capacity_ridge'] = (scaler, ridge)
    options = dict(max_iter=80, max_leaf_nodes=15, min_samples_leaf=100,
                   l2_regularization=10., random_state=416, early_stopping=False)
    for name, target in [('capacity_direct', y), ('capacity_factor', thermal_y)]:
        active = thermal > 0 if name == 'capacity_factor' else np.ones(len(frame), dtype=bool)
        model = HistGradientBoostingRegressor(**options)
        model.fit(frame.loc[active, list(COLUMNS)], target[active], sample_weight=w[active] * len(w))
        models[name] = model
    return models


def predict(models: dict, public: pd.DataFrame) -> dict:
    q = public[list(GROUPS)].to_numpy() * models['profile'][public.clock.dt.hour.to_numpy()]
    hourly_thermal = q[:, 0] / q.sum(axis=1)
    stock_share = public.capacity_coal_share.to_numpy()
    stock_ef = 400. + 600. * stock_share
    coal = public.capacity_coal_mw.to_numpy() * models['source_rates'][0]
    gas = public.capacity_gas_mw.to_numpy() * models['source_rates'][1]
    if ((coal + gas) <= 0).any():
        raise ValueError('degenerate source-rate anchor')
    rate_ef = 400. + 600. * coal / (coal + gas)
    scaler, ridge = models['capacity_ridge']
    return {
        'capacity_anchor_flat': stock_ef * public.thermal.to_numpy(),
        'capacity_anchor_shaped': stock_ef * hourly_thermal,
        'capacity_source_rate': rate_ef * hourly_thermal,
        'capacity_ridge': np.clip(ridge.predict(scaler.transform(public[list(COLUMNS)])), 0., 1000.),
        'capacity_direct': np.clip(models['capacity_direct'].predict(public[list(COLUMNS)]), 0., 1000.),
        'capacity_factor': np.clip(models['capacity_factor'].predict(public[list(COLUMNS)]), 400., 1000.) * hourly_thermal,
    }
