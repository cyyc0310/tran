"""Six-category version of standard V11/V12 shape references, not a new method."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from transcif.research.multicountry import GROUPS, LATEST, weights
from transcif.research import coarse_trajectories as trajectory

METHODS = ('joint_original', 'learned_zero', 'physical_solar', 'physical_wind_solar',
           'head_no_weather', 'head_weather', 'delta_half', 'delta_full')


def day_groups(frame: pd.DataFrame) -> list[np.ndarray]:
    """Return ordered complete local-day positions without altering caller indices."""
    f = frame.reset_index(drop=True).copy()
    f['_position'] = np.arange(len(f)); f['_day'] = f.clock.dt.floor('D')
    groups = []
    for _, group in f.groupby(['country', 'region', '_day'], sort=True):
        group = group.sort_values('clock')
        if len(group) != 24 or group.clock.dt.hour.tolist() != list(range(24)):
            raise ValueError('unique complete 24-hour local days required')
        mix = group[list(GROUPS)].to_numpy(float)
        if (not np.isfinite(mix).all() or (mix < 0).any()
                or not np.allclose(mix.sum(axis=1), 1., atol=1e-12, rtol=0.)
                or not np.allclose(mix, mix[0], atol=1e-12, rtol=0.)):
            raise ValueError('fixed valid public monthly composition per day required')
        groups.append(group['_position'].to_numpy())
    if not groups:
        raise ValueError('nonempty complete local days required')
    return groups


def project_shapes(values: np.ndarray) -> np.ndarray:
    """Per-fuel projection to a nonnegative 24-hour shape with sum 24 (V11)."""
    values = np.asarray(values, float)
    if values.ndim < 2 or values.shape[-2:] != (24, 6) or not np.isfinite(values).all():
        raise ValueError('finite (...,24,6) shapes required')
    rows = np.moveaxis(values, -2, -1).reshape(-1, 24)
    ordered = np.sort(rows, axis=1)[:, ::-1]
    thresholds = (np.cumsum(ordered, axis=1)-24.)/np.arange(1, 25)
    rho = (ordered > thresholds).sum(axis=1)-1
    result = np.maximum(rows-thresholds[np.arange(len(rows)), rho, None], 0.)
    return np.moveaxis(result.reshape((*values.shape[:-2], 6, 24)), -1, -2)


def delta_transport(q: np.ndarray, query_w: np.ndarray, query_b: np.ndarray,
                    donor_w: np.ndarray, donor_b: np.ndarray, support: np.ndarray,
                    *, strength: float = 1.) -> np.ndarray:
    """V12 difference transport, preserving amplitudes and exact identity paths."""
    original = np.asarray(q); profiles = np.asarray(q, float)
    if (profiles.ndim < 2 or profiles.shape[-2:] != (24, 6)
            or not np.isfinite(profiles).all() or (profiles < 0).any()
            or strength not in (0., .5, 1.)):
        raise ValueError('nonnegative (...,24,6) q and fixed strength required')
    supported = np.asarray(support)
    if not np.isin(supported, [False, True]).all():
        raise ValueError('boolean support required')
    supported = np.broadcast_to(supported.astype(bool), profiles.shape[:-2]+(6,))
    predictions = []
    for value in (query_w, query_b, donor_w, donor_b):
        value = np.asarray(value, float)
        if value.ndim < 2 or value.shape[-2:] != (24, 6) or not np.isfinite(value).all():
            raise ValueError('finite (...,24,6) shape predictions required')
        predictions.append(np.broadcast_to(value, profiles.shape))
    if np.any(np.broadcast_to(~supported[..., None, :], profiles.shape) & (profiles != 1.)):
        raise ValueError('unsupported donor fuels must remain flat q=1')
    if strength == 0.:
        return original.copy()
    with np.errstate(over='ignore', invalid='ignore'):
        increment = (predictions[0]-predictions[1])-(predictions[2]-predictions[3])
    if not np.isfinite(increment).all():
        raise ValueError('finite weather increment required')
    if not increment.any():
        return original.copy()
    amplitude = profiles.mean(axis=-2)
    shape = np.divide(profiles, amplitude[..., None, :], out=np.ones_like(profiles),
                      where=amplitude[..., None, :] > 0)
    corrected = amplitude[..., None, :]*project_shapes(shape+strength*increment)
    changed = supported & (amplitude > 0) & np.any(increment != 0., axis=-2)
    return np.where(changed[..., None, :], corrected, original)


def physical_shapes(frame: pd.DataFrame) -> np.ndarray:
    """Solar and clipped-wind-cube shapes, with flat prior for zero weather mean."""
    weather = frame[['temperature_c', 'wind_speed_ms', 'shortwave_w_per_m2']].to_numpy(float)
    if not np.isfinite(weather).all() or (weather[:, 1:] < 0).any():
        raise ValueError('finite weather and nonnegative wind/radiation required')
    out = np.ones((len(frame), 6))
    for ix in day_groups(frame):
        for fuel, values in [(3, np.minimum(weather[ix, 1], 25.)**3), (4, weather[ix, 2])]:
            if values.mean() > 0:
                out[ix, fuel] = values/values.mean()
    return out


def design(frame: pd.DataFrame, weather: bool) -> np.ndarray:
    x = frame[list(LATEST)].to_numpy(float)
    if weather:
        raw = frame[['temperature_c', 'wind_speed_ms']].to_numpy(float)
        physical = physical_shapes(frame)
        extra = np.column_stack([raw[:, 0]/20., np.maximum(18.-raw[:, 0], 0.)/20.,
            np.maximum(raw[:, 0]-22., 0.)/20., raw[:, 1]/10., physical[:, 3], physical[:, 4]])
        for ix in day_groups(frame):
            extra[ix, :4] -= extra[ix, :4].mean(axis=0)
        x = np.column_stack([x, extra])
    if not np.isfinite(x).all():
        raise ValueError('finite public inputs required')
    return x


def fit_heads(frame: pd.DataFrame, coarse: np.ndarray, *, allowed_countries: tuple) -> dict:
    """Only proper source-2022 positive-amplitude fuel hours enter Ridge heads."""
    if (frame.empty or not set(frame.country) <= set(allowed_countries)
            or ('split' in frame and not frame.split.eq('train').all())):
        raise ValueError('proper source training rows required')
    if not frame.clock.dt.year.eq(2022).all():
        raise ValueError('source 2022 training rows required')
    coarse = np.asarray(coarse, float)
    if (coarse.shape != (len(frame), 6) or not np.isfinite(coarse).all()
            or (coarse < 0).any() or (coarse.sum(axis=1) <= 0).any()):
        raise ValueError('aligned nonnegative source generation required')
    active = np.zeros_like(coarse, dtype=bool); shapes = np.ones_like(coarse)
    for ix in day_groups(frame):
        amplitude = coarse[ix].mean(axis=0)
        active[ix] = amplitude > 0
        shapes[ix] = np.divide(coarse[ix], amplitude, out=np.ones_like(coarse[ix]), where=amplitude > 0)
    result = {'B': [], 'W': [], 'audit': {'source_countries': sorted(set(frame.country)),
              'training_hours': len(frame), 'active_hours_per_fuel': active.sum(axis=0).tolist()}}
    for kind in ('B', 'W'):
        x = design(frame, kind == 'W')
        for fuel in range(6):
            mask = active[:, fuel]
            if not mask.any():
                result[kind].append(None)
                continue
            w = weights(frame.loc[mask])
            scaler = StandardScaler().fit(x[mask], sample_weight=w)
            model = Ridge(alpha=10.).fit(scaler.transform(x[mask]), shapes[mask, fuel], sample_weight=w*len(w))
            result[kind].append((scaler, model))
    return result


def predict_heads(fitted: dict, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """No query generation or CIF accepted; predicted shapes preserve day means."""
    out = []
    for kind in ('B', 'W'):
        x = design(frame, kind == 'W'); values = np.ones((len(frame), 6))
        for fuel, model in enumerate(fitted[kind]):
            if model is not None:
                scaler, regression = model
                values[:, fuel] = regression.predict(scaler.transform(x))
        for ix in day_groups(frame):
            values[ix] = project_shapes(values[ix])
        out.append(values)
    return tuple(out)


def fit_crossfit(frame: pd.DataFrame, coarse: np.ndarray, *, allowed_countries: tuple) -> dict:
    """Crossfit all donor regions sharing a forecast origin in the same fold."""
    f = frame.reset_index(drop=True).copy(); coarse = np.asarray(coarse, float)
    if 'origin' not in f or f.origin.nunique() < 3:
        raise ValueError('at least three proper source forecast dates required')
    final = fit_heads(f, coarse, allowed_countries=allowed_countries)
    b, w = np.empty_like(coarse), np.empty_like(coarse)
    folds = []
    for origin in sorted(f.origin.unique()):
        held = f.origin.eq(origin).to_numpy(); proper = ~held
        fitted = fit_heads(f.loc[proper], coarse[proper], allowed_countries=allowed_countries)
        b[held], w[held] = predict_heads(fitted, f.loc[held])
        folds.append({'held_origin': origin, 'proper_origins': sorted(f.loc[proper, 'origin'].unique()),
                      'proper_hours': int(proper.sum()), 'held_hours': int(held.sum())})
    groups = day_groups(f); physical = physical_shapes(f)
    return {'final': final, 'donor_b': np.stack([b[ix] for ix in groups]),
            'donor_w': np.stack([w[ix] for ix in groups]),
            'donor_physical': np.stack([physical[ix] for ix in groups]),
            'donor_keys': [{'country': f.country.iloc[ix[0]], 'region': f.region.iloc[ix[0]],
                            'day': str(f.clock.iloc[ix[0]].floor('D'))} for ix in groups],
            'audit': {'folds': folds, 'proper_year': 2022, 'alpha': 10.,
                      'donor_crossfit': 'all regions of each forecast origin excluded together'}}


def predict_methods(bank: dict, fitted: dict, public: pd.DataFrame, *, methods: tuple = METHODS) -> dict:
    """Apply shape-only references to the exact original joint sampling kernel."""
    if len(bank['q']) != len(fitted['donor_keys']):
        raise ValueError('donor bank and weather rows differ')
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(METHODS):
        raise ValueError('unique registered methods required')
    result = {method: np.empty((len(public), 128)) for method in methods}
    for ix in day_groups(public):
        group = public.iloc[ix]
        values, _ = trajectory.predict(bank, group)
        if 'joint_original' in result:
            result['joint_original'][ix] = values['joint']
        if set(methods) <= {'joint_original', 'learned_zero'}:
            if 'learned_zero' in result:
                result['learned_zero'][ix] = values['joint']
            continue  # No query weather columns are required for this branch.
        b, w = predict_heads(fitted['final'], group)
        physical = physical_shapes(group)
        ones = np.ones((24, 6)); zeros = np.zeros((24, 6))
        for method in (m for m in methods if m != 'joint_original'):
            if method in ('physical_solar', 'physical_wind_solar'):
                channels = [4] if method == 'physical_solar' else [3, 4]
                query = ones.copy(); query[:, channels] = physical[:, channels]
                donor = np.ones_like(fitted['donor_physical'])
                donor[:, :, channels] = fitted['donor_physical'][:, :, channels]
                q = delta_transport(bank['q'], query, ones, donor, ones, bank['support'])
            elif method in ('head_no_weather', 'head_weather'):
                query = b if method == 'head_no_weather' else w
                donor = fitted['donor_b'] if method == 'head_no_weather' else fitted['donor_w']
                q = delta_transport(bank['q'], query, zeros, donor, zeros, bank['support'])
            else:
                strength = {'learned_zero': 0., 'delta_half': .5, 'delta_full': 1.}[method]
                q = delta_transport(bank['q'], w, b, fitted['donor_w'], fitted['donor_b'],
                                    bank['support'], strength=strength)
            changed_bank = {**bank, 'q': q}
            prediction, _ = trajectory.predict(changed_bank, group)
            result[method][ix] = prediction['joint']
        if 'learned_zero' in result:
            np.testing.assert_array_equal(values['joint'], result['learned_zero'][ix])
    return result
