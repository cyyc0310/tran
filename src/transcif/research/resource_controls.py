"""Fixed saturated-power and bounded tabular amplitude controls; no novelty claim."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from transcif.research import resource_amplitudes as base
from transcif.research import coarse_trajectories as ct
from transcif.research.multicountry import weights

METHODS = ('joint_original', 'weather_solar', 'cube_weather_wind_solar',
           'saturated_weather_wind_solar', 'ridge_no_weather', 'ridge_weather',
           'bounded_ridge_weather', 'rf_no_weather', 'rf_weather',
           'hgb_no_weather', 'hgb_weather')
day_groups = base.day_groups


def wind_curve(speed: np.ndarray) -> np.ndarray:
    """Generic 100m 3/12/25 m/s curve, not a known local fleet power curve."""
    speed = np.asarray(speed, float)
    if not np.isfinite(speed).all() or (speed < 0).any():
        raise ValueError('finite nonnegative wind speed required')
    ramp = np.clip((speed**3-3.**3)/(12.**3-3.**3), 0., 1.)
    return np.where((speed >= 3.) & (speed < 25.), ramp, 0.)


def day_inputs(frame: pd.DataFrame, climate: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    daily, _, weather = base.day_inputs(frame, climate)
    reference = climate.set_index(['region', 'month'])
    curve_ratios = []
    for d, ix in enumerate(day_groups(frame)):
        row = frame.iloc[ix[0]]; old_month = pd.Period(row.latest_allowed_period, freq='M').month
        now_key, old_key = (row.region, row.clock.month), (row.region, old_month)
        before = float(reference.loc[old_key, 'wind_curve_resource']) if old_key in reference.index else np.nan
        now = float(reference.loc[now_key, 'wind_curve_resource']) if now_key in reference.index else np.nan
        supported = np.isfinite([before, now]).all() and 0 < before <= 1 and 0 < now <= 1
        season, forecast = (now/before, float(wind_curve(frame.iloc[ix].wind_speed_ms).mean())/before) if supported else (1., 1.)
        daily.loc[d, 'curve_wind_supported'] = float(supported)
        daily.loc[d, 'log_seasonal_curve_wind'] = np.log1p(season)
        daily.loc[d, 'log_weather_curve_wind'] = np.log1p(forecast)
        curve_ratios.append(forecast)
    return daily, weather, np.column_stack([curve_ratios, weather[:, 1]])


def design(daily: pd.DataFrame, weather: bool) -> np.ndarray:
    columns = ['log_seasonal_curve_wind', 'curve_wind_supported']
    if weather:
        columns.append('log_weather_curve_wind')
    return np.column_stack([base.design(daily, weather), daily[columns].to_numpy(float)])


def fit(frame: pd.DataFrame, coarse: np.ndarray, past_month_power: np.ndarray,
        climate: pd.DataFrame, *, allowed_countries: tuple) -> dict:
    """Same source cohort, targets and features for Ridge, RF and HGB controls."""
    # Reuse the prior validation/audit contract; its cube-only Ridge is not scored here.
    checked = base.fit(frame, coarse, past_month_power, climate, allowed_countries=allowed_countries)
    daily, _, _ = day_inputs(frame, climate)
    actual = np.stack([np.asarray(coarse)[ix].mean(axis=0) for ix in day_groups(frame)])
    past = np.asarray(past_month_power)
    amplitude = np.divide(actual, past, out=np.ones_like(actual), where=past > 0)
    result = {'models': {}, 'upper': np.ones(2), 'audit': {**checked['audit'],
        'controls': 'same cube+curve features for all three learners; no target-selected cap',
        'ridge_alpha': 10., 'rf_trees': 256, 'rf_min_leaf': 3,
        'hgb_max_iter': 200, 'hgb_max_leaf_nodes': 7, 'hgb_min_samples_leaf': 5,
        'hgb_learning_rate': .05, 'hgb_l2': 1., 'seed': 416,
        'bounded_ridge_upper': 'proper source maximum amplitude; not a physical target bound'}}
    for learner in ('ridge', 'rf', 'hgb'):
        for kind in ('B', 'W'):
            result['models'][learner+'_'+kind] = []
    for j, fuel in enumerate((3, 4)):
        mask = (past[:, fuel] > 0) & daily[base.RESOURCES[j]+'_supported'].eq(1).to_numpy()
        if j == 0:
            mask &= daily.curve_wind_supported.eq(1).to_numpy()
        if not mask.any():
            for models in result['models'].values():
                models.append(None)
            continue
        w = weights(daily.loc[mask]); target = np.log1p(amplitude[mask, fuel])
        result['upper'][j] = float(amplitude[mask, fuel].max())
        for kind in ('B', 'W'):
            x = design(daily, kind == 'W')[mask]
            scaler = StandardScaler().fit(x, sample_weight=w)
            ridge = Ridge(alpha=10.).fit(scaler.transform(x), target, sample_weight=w*len(w))
            rf = RandomForestRegressor(n_estimators=256, min_samples_leaf=3,
                max_features=1., n_jobs=1, random_state=416).fit(x, target, sample_weight=w*len(w))
            hgb = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=7, min_samples_leaf=5,
                learning_rate=.05, l2_regularization=1., early_stopping=False, random_state=416).fit(x, target, sample_weight=w*len(w))
            result['models']['ridge_'+kind].append((scaler, ridge))
            result['models']['rf_'+kind].append(rf)
            result['models']['hgb_'+kind].append(hgb)
    return result


def predict_amplitudes(fitted: dict, daily: pd.DataFrame, learner: str, kind: str,
                       *, bounded: bool = False) -> np.ndarray:
    out = np.ones((len(daily), 2)); x = design(daily, kind == 'W')
    for j, model in enumerate(fitted['models'][learner+'_'+kind]):
        if model is None:
            continue
        if learner == 'ridge':
            scaler, model = model; value = model.predict(scaler.transform(x))
        else:
            value = model.predict(x)
        with np.errstate(over='raise', invalid='raise'):
            amplitude = np.maximum(np.expm1(value), 0.)
        if bounded or learner == 'hgb':
            amplitude = np.minimum(amplitude, fitted['upper'][j])
        supported = daily[base.RESOURCES[j]+'_supported'].eq(1).to_numpy()
        if j == 0:
            supported = supported & daily.curve_wind_supported.eq(1).to_numpy()
        out[:, j] = np.where(supported, amplitude, 1.)
    return out


def predict_methods(bank: dict, fitted: dict, public: pd.DataFrame, climate: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    daily, cube, curve = day_inputs(public, climate)
    factors = {'weather_solar': np.column_stack([np.ones(len(daily)), cube[:, 1]]),
               'cube_weather_wind_solar': cube, 'saturated_weather_wind_solar': curve}
    for learner in ('ridge', 'rf', 'hgb'):
        for kind, name in [('B', 'no_weather'), ('W', 'weather')]:
            factors[learner+'_'+name] = predict_amplitudes(fitted, daily, learner, kind)
    factors['bounded_ridge_weather'] = predict_amplitudes(fitted, daily, 'ridge', 'W', bounded=True)
    out = {name: np.empty((len(public), 128)) for name in METHODS}; diagnostics = []
    for d, ix in enumerate(day_groups(public)):
        day = public.iloc[ix]
        original, _ = ct.predict(bank, day); out['joint_original'][ix] = original['joint']
        for method, multiplier in factors.items():
            changed = {**bank, 'q': base.scale_paths(bank['q'], bank['support'], multiplier[d])}
            values, _ = ct.predict(changed, day); out[method][ix] = values['joint']
            diagnostics.append(dict(region=day.region.iloc[0], origin=day.origin.iloc[0], method=method,
                wind_factor=float(multiplier[d, 0]), solar_factor=float(multiplier[d, 1]),
                wind_supported=bool(daily.curve_wind_supported.iloc[d] if method != 'cube_weather_wind_solar' else daily.wind_resource_supported.iloc[d]),
                solar_supported=bool(daily.solar_resource_supported.iloc[d])))
    return out, pd.DataFrame(diagnostics)
