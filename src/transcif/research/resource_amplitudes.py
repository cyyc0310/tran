"""Ordinary climate/weather ratio and Ridge amplitude references, not a new method."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from transcif.research import coarse_trajectories as ct
from transcif.research.coarse_weather_shapes import day_groups, physical_shapes
from transcif.research.multicountry import GROUPS, HISTORY, weights

METHODS = ('joint_original', 'seasonal_solar', 'seasonal_wind_solar', 'weather_solar',
           'weather_wind_solar', 'bias_weather_wind_solar', 'ridge_no_weather',
           'ridge_weather', 'weather_solar_shape', 'weather_wind_solar_shape')
RESOURCES = ('wind_resource', 'solar_resource')


def day_inputs(frame: pd.DataFrame, climate: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Daily public statistics, seasonal ratios and forecast ratios (wind,solar)."""
    if (climate.duplicated(['region', 'month']).any() or
            not climate.reference_years.eq(5).all() or climate.eligible_for_issued_public.any()):
        raise ValueError('fixed uncertified five-year climate reference required')
    reference = climate.set_index(['region', 'month'])
    rows, seasonal, weather = [], [], []
    for ix in day_groups(frame):
        day = frame.iloc[ix]; first = day.iloc[0]
        if day.latest_allowed_period.nunique() != 1:
            raise ValueError('one public reference month per day required')
        period = pd.Period(first.latest_allowed_period, freq='M')
        origin = pd.Period(pd.Timestamp(first.forecast_origin_local), freq='M')
        if period > origin-2:
            raise ValueError('reference month must precede forecast origin by at least two months')
        row = {name: float(day[name].mean()) for name in HISTORY}
        row.update(country=first.country, region=first.region, origin=first.origin,
                   day=str(first.clock.floor('D')), reference_period=str(period))
        sw, fw = np.ones(2), np.ones(2)
        missing = pd.Series(dict(temperature_c=0., wind_resource=np.nan, solar_resource=np.nan))
        now_key, old_key = (first.region, first.clock.month), (first.region, period.month)
        current = reference.loc[now_key] if now_key in reference.index else missing
        old = reference.loc[old_key] if old_key in reference.index else missing
        values = day[['temperature_c', 'wind_speed_ms', 'shortwave_w_per_m2']].to_numpy(float)
        if not np.isfinite(values).all() or (values[:, 1:] < 0).any():
            raise ValueError('finite issued weather with nonnegative resources required')
        forecast = np.array([(np.minimum(values[:, 1], 25.)**3).mean(), values[:, 2].mean()])
        for j, resource in enumerate(RESOURCES):
            before, now = float(old[resource]), float(current[resource])
            supported = np.isfinite([before, now]).all() and before > 0 and now > 0
            if supported:
                sw[j], fw[j] = now/before, forecast[j]/before
            row[resource+'_supported'] = float(supported)
            row['log_seasonal_'+resource] = float(np.log1p(sw[j]))
            row['log_weather_'+resource] = float(np.log1p(fw[j]))
        row.update(climate_temperature_now=float(current.temperature_c)/20.,
                   climate_temperature_lag=float(old.temperature_c)/20.,
                   forecast_temperature=float(values[:, 0].mean())/20.,
                   forecast_heating=float(np.maximum(18.-values[:, 0], 0.).mean())/20.,
                   forecast_cooling=float(np.maximum(values[:, 0]-22., 0.).mean())/20.)
        rows.append(row); seasonal.append(sw); weather.append(fw)
    daily = pd.DataFrame(rows)
    numeric = daily.select_dtypes(include='number').to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError('nonfinite daily public design')
    return daily, np.asarray(seasonal), np.asarray(weather)


def design(daily: pd.DataFrame, weather: bool) -> np.ndarray:
    columns = [*HISTORY, *(f'log_seasonal_{r}' for r in RESOURCES),
               *(r+'_supported' for r in RESOURCES), 'climate_temperature_now', 'climate_temperature_lag']
    if weather:
        columns += [*(f'log_weather_{r}' for r in RESOURCES),
                    'forecast_temperature', 'forecast_heating', 'forecast_cooling']
    return daily[columns].to_numpy(float)


def fit(frame: pd.DataFrame, coarse: np.ndarray, past_month_power: np.ndarray,
        climate: pd.DataFrame, *, allowed_countries: tuple) -> dict:
    """Fit daily amplitude against legal past monthly mean power, source 2022 only."""
    if (frame.empty or not set(frame.country) <= set(allowed_countries)
            or not frame.split.eq('train').all() or not frame.clock.dt.year.eq(2022).all()):
        raise ValueError('proper source-2022 training only')
    groups = day_groups(frame); coarse = np.asarray(coarse, float)
    past_month_power = np.asarray(past_month_power, float)
    if (coarse.shape != (len(frame), 6) or past_month_power.shape != (len(groups), 6)
            or not np.isfinite(coarse).all() or not np.isfinite(past_month_power).all()
            or (coarse < 0).any() or (past_month_power < 0).any()
            or (coarse.sum(axis=1) <= 0).any()):
        raise ValueError('aligned finite source generation and past monthly power required')
    for d, ix in enumerate(groups):
        past = past_month_power[d]
        if past.sum() <= 0 or not np.allclose(past/past.sum(), frame.iloc[ix[0]][list(GROUPS)].to_numpy(float), atol=1e-10, rtol=0.):
            raise ValueError('past power must match the visible monthly composition')
    daily, _, weather = day_inputs(frame, climate)
    actual = np.stack([coarse[ix].mean(axis=0) for ix in groups])
    amplitude = np.divide(actual, past_month_power, out=np.ones_like(actual), where=past_month_power > 0)
    result = {'B': [], 'W': [], 'bias': np.ones(2),
              'audit': dict(source_countries=sorted(set(frame.country)), source_hours=len(frame),
                            source_days=len(daily), alpha=10., source_year=2022,
                            target='daily generation / same-source legal past monthly mean power',
                            bias_interpretation='ordinary pooled amplitude correction, not pure weather-model bias',
                            active_days=[], target_calibration=False)}
    for j, fuel in enumerate((3, 4)):
        mask = (past_month_power[:, fuel] > 0) & daily[RESOURCES[j]+'_supported'].eq(1).to_numpy()
        result['audit']['active_days'].append(int(mask.sum()))
        if not mask.any():
            result['B'].append(None); result['W'].append(None)
            continue
        w = weights(daily.loc[mask]); y = amplitude[mask, fuel]
        denominator = float(w @ weather[mask, j])
        if denominator > 0:
            result['bias'][j] = float(w @ y)/denominator
        for kind in ('B', 'W'):
            x = design(daily, kind == 'W')[mask]
            scaler = StandardScaler().fit(x, sample_weight=w)
            ridge = Ridge(alpha=10.).fit(scaler.transform(x), np.log1p(y), sample_weight=w*len(w))
            result[kind].append((scaler, ridge))
    return result


def head_amplitudes(fitted: dict, daily: pd.DataFrame, kind: str) -> np.ndarray:
    values = np.ones((len(daily), 2)); x = design(daily, kind == 'W')
    for j, model in enumerate(fitted[kind]):
        if model is not None:
            scaler, ridge = model
            with np.errstate(over='raise', invalid='raise'):
                predicted = np.maximum(np.expm1(ridge.predict(scaler.transform(x))), 0.)
            values[:, j] = np.where(daily[RESOURCES[j]+'_supported'].eq(1), predicted, 1.)
    return values


def scale_paths(q: np.ndarray, support: np.ndarray, factors: np.ndarray,
                shape: np.ndarray | None = None) -> np.ndarray:
    """Change wind/solar amplitudes; keep unsupported source fuels at their q=1 prior."""
    original = np.asarray(q); supported = np.asarray(support)
    factors = np.asarray(factors, float)
    if (original.ndim != 3 or original.shape[1:] != (24, 6) or supported.shape != (len(q), 6)
            or not np.isin(supported, [False, True]).all() or factors.shape != (2,)
            or not np.isfinite(original).all() or (original < 0).any()
            or not np.isfinite(factors).all() or (factors < 0).any()
            or np.any((~supported.astype(bool))[:, None, :] & (original != 1.))):
        raise ValueError('valid nonnegative bank, factors and support required')
    if shape is not None:
        shape = np.asarray(shape, float)
        if shape.shape != (24, 2) or not np.isfinite(shape).all() or (shape < 0).any() or not np.allclose(shape.mean(axis=0), 1.):
            raise ValueError('mean-one resource shapes required')
    if shape is None and np.array_equal(factors, np.ones(2)):
        return original.copy()
    result = original.astype(float, copy=True)
    for j, fuel in enumerate((3, 4)):
        changed = supported[:, fuel].astype(bool)
        paths = original[changed, :, fuel]
        if shape is not None:
            paths = paths.mean(axis=1)[:, None]*shape[:, j]
        result[changed, :, fuel] = paths*factors[j]
    return result


def predict_methods(bank: dict, fitted: dict, public: pd.DataFrame, climate: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Use the original kernel and random donors for every amplitude reference."""
    daily, seasonal, weather = day_inputs(public, climate)
    groups = day_groups(public); physical = physical_shapes(public)
    no_weather = head_amplitudes(fitted, daily, 'B'); learned = head_amplitudes(fitted, daily, 'W')
    result = {method: np.empty((len(public), 128)) for method in METHODS}
    diagnostics = []
    for d, ix in enumerate(groups):
        day = public.iloc[ix]
        base, _ = ct.predict(bank, day)
        result['joint_original'][ix] = base['joint']
        options = {'seasonal_solar': [1., seasonal[d, 1]], 'seasonal_wind_solar': seasonal[d],
                   'weather_solar': [1., weather[d, 1]], 'weather_wind_solar': weather[d],
                   'bias_weather_wind_solar': weather[d]*fitted['bias'],
                   'ridge_no_weather': no_weather[d], 'ridge_weather': learned[d],
                   'weather_solar_shape': [1., weather[d, 1]], 'weather_wind_solar_shape': weather[d]}
        # Unsupported climate ratios must return the original q, including bias control.
        active = daily.iloc[d][[r+'_supported' for r in RESOURCES]].to_numpy(float).astype(bool)
        for method, factors in options.items():
            factors = np.where(active, np.asarray(factors), 1.)
            shape = None
            if method.endswith('_shape'):
                shape = physical[ix][:, [3, 4]].copy()
                if method == 'weather_solar_shape':
                    # A unit wind shape would replace its donor; restore below instead.
                    shape[:, 0] = 1.
            corrected = scale_paths(bank['q'], bank['support'], factors, shape)
            if shape is not None:
                for j, fuel in enumerate((3, 4)):
                    if not active[j] or (method == 'weather_solar_shape' and fuel == 3):
                        corrected[:, :, fuel] = bank['q'][:, :, fuel]
            changed = {**bank, 'q': corrected}
            predicted, _ = ct.predict(changed, day)
            result[method][ix] = predicted['joint']
            diagnostics.append(dict(region=day.region.iloc[0], origin=day.origin.iloc[0], method=method,
                                    wind_factor=float(factors[0]), solar_factor=float(factors[1]),
                                    wind_supported=bool(active[0]), solar_supported=bool(active[1])))
    return result, pd.DataFrame(diagnostics)
