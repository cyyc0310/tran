"""Exploratory source-only release/support/age conditioned daily proxy forecasts.

Monthly fuel mix is an observation of generation energy, never the arithmetic
mean of hourly CIF. This module learns its relationship to future daily CIF;
it does not impose historical energy totals on future generation.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import dct

from .eia_proxy import FUELS
from .references import proxy_values

EF = np.array([1000., 400., 800., 0., 0., 0., 0., 400.])
METHODS = ("calendar", "level_template", "ridge", "static", "age_global",
           "age_time", "age_dct", "age_haar", "quarterly_haar")


def read_year(path: Path, year: int) -> pd.DataFrame:
    """Filter CSV text before numeric parsing: never parse 2024 for this study."""
    with path.open(newline="") as handle:
        rows = csv.reader(handle)
        header = next(rows)
        selected = [row for row in rows if row[0].startswith(f"{year}-")]
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(selected)
    buffer.seek(0)
    frame = pd.read_csv(buffer)
    frame["time"] = pd.to_datetime(frame.time, utc=True)
    return frame


def basis(kind: str) -> np.ndarray:
    """Rows are an orthonormal 24-hour basis, including a complete Haar basis."""
    if kind == "time":
        return np.eye(24)
    if kind == "dct":
        return dct(np.eye(24), axis=0, norm="ortho")
    if kind != "haar":
        raise ValueError(kind)
    low = np.eye(24)
    details = []
    while len(low) % 2 == 0:
        details.append((low[::2] - low[1::2]) / np.sqrt(2))
        low = (low[::2] + low[1::2]) / np.sqrt(2)
    # 24 = 3 * 2**3: complete remaining coarse space by orthogonal DCT.
    return np.vstack([dct(np.eye(3), axis=0, norm="ortho") @ low,
                      *reversed(details)])


def quarterly(monthly: pd.DataFrame) -> pd.DataFrame:
    """Only three complete contiguous calendar months define a quarter."""
    rows = []
    for region, part in monthly.groupby("region"):
        part = part.copy()
        start = pd.to_datetime(part.period_start, utc=True)
        part["quarter"] = start.dt.year * 4 + (start.dt.month - 1) // 3
        for _, group in part.groupby("quarter"):
            group = group.sort_values("period_start")
            begins = pd.DatetimeIndex(pd.to_datetime(group.period_start, utc=True))
            ends = pd.DatetimeIndex(pd.to_datetime(group.period_end, utc=True))
            if (len(group) != 3 or begins[0].month not in (1, 4, 7, 10)
                    or not np.all(ends[:-1] == begins[1:])):
                continue
            weights = group.total_mwh.to_numpy(float)
            if not np.isfinite(weights).all() or np.any(weights <= 0):
                continue
            item = group.iloc[-1].to_dict()
            item.update(period_start=begins[0], period_end=ends[-1],
                        available_at=pd.to_datetime(group.available_at, utc=True).max(),
                        total_mwh=weights.sum(), coverage=float(group.coverage.min()))
            for fuel in FUELS:
                item[f"fuelshare_{fuel}"] = np.average(group[f"fuelshare_{fuel}"], weights=weights)
            rows.append(item)
    return pd.DataFrame(rows, columns=monthly.columns)


def observation_features(origins, region: str, history: pd.DataFrame,
                         seasonal_mix: np.ndarray):
    """Pure as-of feature builder; neither target hourly labels nor telemetry enter."""
    origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
    history = history[history.region == region].copy()
    for col in ("available_at", "period_start", "period_end"):
        history[col] = pd.to_datetime(history[col], utc=True)
    features = np.zeros((len(origins), len(FUELS) + 2))
    mixes = np.zeros((len(origins), len(FUELS)))
    ages = np.zeros(len(origins))
    permitted = np.zeros(len(origins), dtype=bool)
    selected_end = np.full(len(origins), np.datetime64("NaT"), dtype="datetime64[ns]")
    history = history.loc[history.period_start < history.period_end].sort_values(["period_end", "available_at"])
    if history.empty or len(origins) == 0:
        return features, mixes, ages, permitted, selected_end
    eligible = ((history.available_at.to_numpy()[None, :] <= origins.to_numpy()[:, None])
                & (history.period_end.to_numpy()[None, :] <= origins.to_numpy()[:, None]))
    indices = np.max(np.where(eligible, np.arange(len(history)), -1), axis=1)
    # Work once per selected observation, not once per forecast origin. Future
    # unselected values cannot affect either validation or numeric transforms.
    for selected in np.unique(indices[indices >= 0]):
        take = indices == selected
        row = history.iloc[selected]
        mix = row[[f"fuelshare_{fuel}" for fuel in FUELS]].to_numpy(float)
        if (not np.isfinite(mix).all() or np.any(mix < 0)
                or not np.isclose(mix.sum(), 1, atol=1e-5)):
            raise ValueError("Released historical fuel shares must be a simplex")
        support = (row.period_end - row.period_start).total_seconds() / 86400
        support_months = pd.date_range(row.period_start, row.period_end, freq="MS", inclusive="left")
        expected = seasonal_mix[support_months.month - 1].mean(axis=0)
        features[take] = np.r_[mix - expected, support / 90, float(row.coverage)]
        mixes[take] = mix
        ages[take] = (origins[take] - row.period_end).total_seconds() / 86400
        permitted[take] = True
        selected_end[take] = row.period_end.tz_localize(None).to_datetime64()
    return features, mixes, ages, permitted, selected_end


def daily_profiles(frame: pd.DataFrame, cutoff=None):
    times = pd.DatetimeIndex(pd.to_datetime(frame.time, utc=True))
    valid = frame.valid.to_numpy(bool).copy()
    if cutoff is not None:
        valid &= times < pd.Timestamp(cutoff)
    y = proxy_values(frame, EF)
    records, days = [], []
    for day, positions in pd.Series(np.arange(len(times)), index=times).groupby(times.floor("D")):
        indices = positions.to_numpy()
        if len(indices) == 24 and valid[indices].all() and np.isfinite(y[indices]).all():
            days.append(day)
            records.append(y[indices])
    return pd.DatetimeIndex(days), np.asarray(records).reshape(-1, 24)


def _solve(x, y):
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-5)
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    penalty = np.eye(design.shape[1]) * 30
    penalty[0, 0] = 0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return mean, scale, weights


def _predict(fit, x):
    mean, scale, weights = fit
    return np.column_stack([np.ones(len(x)), (x - mean) / scale]) @ weights


def calendar_features(origins):
    origins = pd.DatetimeIndex(origins)
    phase = 2 * np.pi * (origins.dayofyear.to_numpy() - 1) / 365.25
    return np.column_stack([np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase)])


@dataclass
class DynamicModel:
    template: np.ndarray
    seasonal_mix: np.ndarray
    fits: dict
    tau: dict

    def predict(self, origins, region, monthly):
        origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
        template = self.template[origins.month - 1]
        x, mix, age, valid, _ = observation_features(origins, region, monthly, self.seasonal_mix)
        base = {"calendar": template.copy(), "level_template": template.copy()}
        base["level_template"][valid] = (mix[valid] @ EF)[:, None] + (template[valid] - template[valid].mean(axis=1, keepdims=True))
        ridge_x = np.column_stack([mix, age / 90, valid, calendar_features(origins)])
        base["ridge"] = _predict(self.fits["ridge"], ridge_x)
        for method in METHODS[3:]:
            if method == "quarterly_haar":
                xx, _, aa, vv, _ = observation_features(origins, region, quarterly(monthly), self.seasonal_mix)
            else:
                xx, aa, vv = x, age, valid
            kind = method.removeprefix("age_") if method.startswith("age_") else "time"
            if method == "quarterly_haar":
                kind = "haar"
            if kind == "global":
                kind = "time"
            transform = basis(kind)
            correction = np.zeros_like(template)
            if method in self.fits:
                coefficients = np.empty((len(origins), 24))
                for j in range(24):
                    attenuation = np.exp(-aa / self.tau[method][j])
                    features = xx * attenuation[:, None]
                    coefficients[:, j] = _predict(self.fits[method][j], features)
                correction[vv] = coefficients[vv] @ transform
            base[method] = template + correction
        return {name: np.clip(value, 0, EF.max()) for name, value in base.items()}


def fit_model(frames, monthly, cutoff="2023-01-01T00:00:00Z"):
    cutoff = pd.Timestamp(cutoff)
    records = [(region, *daily_profiles(frame, cutoff)) for region, frame in sorted(frames.items())]
    all_y = np.concatenate([y for _, _, y in records])
    overall = all_y.mean(axis=0)
    template = np.stack([np.concatenate([y[days.month == month] for _, days, y in records]).mean(axis=0)
                         if any(np.any(days.month == month) for _, days, y in records) else overall
                         for month in range(1, 13)])
    source = monthly[monthly.region.isin(frames)].copy()
    eligible = (pd.to_datetime(source.available_at, utc=True) < cutoff) & (pd.to_datetime(source.period_end, utc=True) <= cutoff)
    source = source.loc[eligible]
    mix_cols = [f"fuelshare_{fuel}" for fuel in FUELS]
    global_mix = source[mix_cols].mean().to_numpy(float)
    seasonal = np.stack([source.loc[pd.to_datetime(source.period_start, utc=True).dt.month == month, mix_cols].mean().to_numpy(float)
                         if np.any(pd.to_datetime(source.period_start, utc=True).dt.month == month) else global_mix
                         for month in range(1, 13)])
    fits, tau = {}, {}
    monthly_features = {region: observation_features(days, region, monthly, seasonal)
                        for region, days, _ in records}
    quarter_history = quarterly(monthly)
    quarter_features = {region: observation_features(days, region, quarter_history, seasonal)
                        for region, days, _ in records}
    coefficient_cache = {kind: {region: (y - template[days.month - 1]) @ basis(kind).T
                                for region, days, y in records} for kind in ("time", "dct", "haar")}
    xs, ys = [], []
    for region, days, y in records:
        _, mix, age, valid, _ = monthly_features[region]
        xs.append(np.column_stack([mix, age / 90, valid, calendar_features(days)]))
        ys.append(y)
    fits["ridge"] = _solve(np.concatenate(xs), np.concatenate(ys))
    # Month-lag coefficient correlation estimates source persistence. This is
    # a heuristic transition estimate, not an identified physical state law.
    for method in METHODS[3:]:
        kind = "haar" if method == "quarterly_haar" else method.removeprefix("age_")
        if kind in ("static", "global"):
            kind = "time"
        transform = basis(kind)
        lag_left, lag_right = [], []
        for region, days, y in records:
            residual = coefficient_cache[kind][region]
            lookup = {day: i for i, day in enumerate(days)}
            for i, day in enumerate(days):
                before = lookup.get(day - pd.Timedelta(days=30))
                if before is not None:
                    lag_left.append(residual[before]); lag_right.append(residual[i])
        left, right = np.asarray(lag_left), np.asarray(lag_right)
        denominator = np.sqrt(np.sum(left ** 2, axis=0) * np.sum(right ** 2, axis=0))
        rho = np.divide(np.sum(left * right, axis=0), denominator, out=np.zeros(24), where=denominator > 0)
        persistence = np.clip(-30 / np.log(np.clip(rho, .01, .999)), 14, 365)
        if method == "static":
            persistence[:] = 1e12
        elif method == "age_global":
            persistence[:] = np.median(persistence)
        tau[method] = persistence
        xrows, crows, ages = [], [], []
        cached_features = quarter_features if method == "quarterly_haar" else monthly_features
        for region, days, y in records:
            x, _, age, valid, _ = cached_features[region]
            xrows.append(x[valid]); ages.append(age[valid])
            crows.append(coefficient_cache[kind][region][valid])
        x, age, coef = np.concatenate(xrows), np.concatenate(ages), np.concatenate(crows)
        if len(x) < 40:
            continue
        fits[method] = [_solve(x * np.exp(-age / persistence[j])[:, None], coef[:, j]) for j in range(24)]
    return DynamicModel(template, seasonal, fits, tau)
