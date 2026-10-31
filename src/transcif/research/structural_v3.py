"""Persistent coarse level and source-selected zero-mean shape transfer."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .dynamic_v2 import (EF, daily_profiles, read_year, observation_features,
                         calendar_features, _solve, _predict)
from .eia_proxy import FUELS

POINTS = ('latest', 'rolling', 'seasonal', 'ridge_level', 'ridge_shape',
          'static_shape', 'analog_shape', 'analog_distribution',
          'static_shape_half', 'analog_shape_half', 'level_shape_half')
METHODS = POINTS + ('source_policy',)


def delayed(monthly, days):
    result = monthly.copy(deep=True)
    result['available_at'] = pd.to_datetime(result.available_at, utc=True) + pd.Timedelta(days=days)
    return result


def features(origins, region, monthly, seasonal):
    origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
    x, mix, age, valid, ends = observation_features(origins, region, monthly, seasonal)
    latest = mix @ EF
    rolling = latest.copy()
    seasonal_level = latest.copy()
    history = monthly.loc[monthly.region == region].copy()
    for col in ('available_at', 'period_end', 'period_start'):
        history[col] = pd.to_datetime(history[col], utc=True)
    cols = [f'fuelshare_{f}' for f in FUELS]
    for i, origin in enumerate(origins):
        part = history.loc[(history.available_at <= origin) & (history.period_end <= origin)].sort_values('period_end')
        if len(part):
            values = part[cols].to_numpy(float) @ EF
            rolling[i] = np.median(values[-3:])
            prior = part.loc[part.period_start.dt.month == origin.month]
            if len(prior):
                seasonal_level[i] = float(prior.iloc[-1][cols].to_numpy(float) @ EF)
            else:
                # Source seasonal displacement, never target future monthly equality.
                last_month = int(part.iloc[-1].period_start.month)
                seasonal_level[i] += float((seasonal[origin.month-1] - seasonal[last_month-1]) @ EF)
    design = np.column_stack([mix, age / 90, valid, calendar_features(origins)])
    return design, latest, rolling, seasonal_level, age, valid


def zero_mean(value):
    return value - value.mean(axis=-1, keepdims=True)


@dataclass
class StructuralModel:
    template: np.ndarray
    seasonal: np.ndarray
    level_fit: tuple
    shape_fit: tuple
    analog_x: np.ndarray
    analog_shape: np.ndarray
    analog_level: np.ndarray
    source_regions: tuple
    delay_days: int

    def predict(self, origins, region, monthly):
        origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
        x, latest, rolling, seasonal, age, valid = features(origins, region, delayed(monthly, self.delay_days), self.seasonal)
        template = self.template[origins.month-1]
        shape = zero_mean(template)
        latest = np.where(valid, latest, template.mean(axis=1))
        rolling = np.where(valid, rolling, latest)
        seasonal = np.where(valid, seasonal, latest)
        correction = zero_mean(_predict(self.shape_fit, x))
        # Mean-zero correction survives both attenuation and mixing.
        dynamic = correction * np.exp(-age[:, None] / 90)
        scale = np.maximum(self.analog_x.std(axis=0), .05)
        distance = (((x[:, None, :] - self.analog_x[None, :, :]) / scale) ** 2).mean(axis=2)
        k = min(32, len(self.analog_x))
        indices = np.argpartition(distance, k-1, axis=1)[:, :k]
        weights = 1 / (np.take_along_axis(distance, indices, axis=1) + .1)
        weights /= weights.sum(axis=1, keepdims=True)
        analog_shape = zero_mean(np.sum(self.analog_shape[indices] * weights[:, :, None], axis=1))
        analog_level = np.sum(self.analog_level[indices] * weights, axis=1)
        result = {
            'latest': latest[:, None] + shape,
            'rolling': rolling[:, None] + shape,
            'seasonal': seasonal[:, None] + shape,
            'ridge_level': latest[:, None] + _predict(self.level_fit, x) + shape,
            'ridge_shape': latest[:, None] + shape + dynamic,
            'static_shape': latest[:, None] + shape + correction,
            'analog_shape': latest[:, None] + analog_shape,
            'analog_distribution': latest[:, None] + analog_level[:, None] + analog_shape,
        }
        result['static_shape_half'] = .5 * (result['latest'] + result['static_shape'])
        result['analog_shape_half'] = .5 * (result['latest'] + result['analog_shape'])
        result['level_shape_half'] = .5 * (result['ridge_level'] + result['analog_shape'])
        # Raw forecasts deliberately un-clipped: clipping could alter the structural mean.
        return result


def fit_model(frames, monthly, cutoff='2023-01-01T00:00:00Z', delay_days=30):
    cutoff = pd.Timestamp(cutoff)
    records = [(r, *daily_profiles(f, cutoff)) for r, f in sorted(frames.items())]
    all_y = np.concatenate([y for _, _, y in records])
    template = np.stack([np.concatenate([y[d.month == m] for _, d, y in records]).mean(axis=0)
                         if any(np.any(d.month == m) for _, d, y in records) else all_y.mean(axis=0)
                         for m in range(1, 13)])
    source = delayed(monthly.loc[monthly.region.isin(frames)], delay_days)
    source = source.loc[(pd.to_datetime(source.available_at, utc=True) < cutoff) & (pd.to_datetime(source.period_end, utc=True) <= cutoff)]
    cols = [f'fuelshare_{f}' for f in FUELS]
    global_mix = source[cols].mean().to_numpy(float) if len(source) else np.ones(len(FUELS)) / len(FUELS)
    seasonal = np.stack([source.loc[pd.to_datetime(source.period_start, utc=True).dt.month == m, cols].mean().to_numpy(float)
                         if np.any(pd.to_datetime(source.period_start, utc=True).dt.month == m) else global_mix for m in range(1,13)])
    xx, shape_y, level_y = [], [], []
    for region, days, y in records:
        x, latest, _, _, _, valid = features(days, region, delayed(monthly, delay_days), seasonal)
        if not valid.any():
            valid = np.ones(len(days), dtype=bool)
            latest = template[days.month-1].mean(axis=1)
        xx.append(x[valid]); shape_y.append(zero_mean(y[valid]) - zero_mean(template[days.month[valid]-1]))
        level_y.append((y[valid].mean(axis=1)-latest[valid])[:,None])
    x = np.concatenate(xx); shapes = np.concatenate(shape_y); levels = np.concatenate(level_y)
    # Analog bank uses source day shapes; no target identity or hourly input.
    full_shapes = []
    for region, days, y in records:
        *_, valid = features(days, region, delayed(monthly, delay_days), seasonal)
        if not valid.any():
            valid = np.ones(len(days), dtype=bool)
        full_shapes.append(zero_mean(y[valid]))
    return StructuralModel(template, seasonal, _solve(x, levels), _solve(x, shapes),
                           x, np.concatenate(full_shapes), levels[:,0], tuple(sorted(frames)), delay_days)


def source_policy(frames, monthly, delay_days=30, train_year=2022):
    """Leave-source-region AND time-out selection; target hourly frames excluded."""
    errors = {name: [] for name in POINTS}
    banks = {name: [] for name in POINTS}
    fold_records = []
    cutoff = f'{train_year}-07-01T00:00:00Z'
    for held, frame in sorted(frames.items()):
        sources = {r: f for r, f in frames.items() if r != held}
        model = fit_model(sources, monthly, cutoff, delay_days)
        days, truth = daily_profiles(frame)
        take = days >= pd.Timestamp(cutoff)
        days, truth = days[take], truth[take]
        preds = model.predict(days, held, monthly)
        for name in POINTS:
            residual = truth - preds[name]
            errors[name].extend(np.abs(residual).mean(axis=1).tolist())
            banks[name].extend(residual)
        fold_records.append({'held_source': held, 'fit_sources': list(model.source_regions), 'days': len(days)})
    ranking = sorted(POINTS, key=lambda name: np.mean(errors[name]))
    return ranking[0], {name: np.asarray(bank) for name, bank in banks.items()}, {
        'selection': 'source-only leave-region-out July–December MAE',
        'source_regions': sorted(frames), 'folds': fold_records,
        'ranking': [{'method': name, 'mae': float(np.mean(errors[name]))} for name in ranking]}
