"""Same-information direct references; fitting and calibration are source-only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .eia_proxy import FUELS
from .mechanisms import local_hours


def proxy_values(frame: pd.DataFrame, ef: np.ndarray) -> np.ndarray:
    generation = frame[[f"generation_{f}" for f in FUELS]].to_numpy(float)
    total = generation.sum(axis=1)
    return np.divide(generation @ ef, total, out=np.full(len(frame), np.nan), where=total > 0)


def features(times, region: str, monthly: pd.DataFrame):
    """As-of join uses availability, never the current/future month label."""
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    history = monthly[monthly.region == region].sort_values("available_at")
    released = pd.DatetimeIndex(pd.to_datetime(history.available_at, utc=True))
    # All forecasts in this protocol originate at 00:00 UTC. Freeze the
    # information set for the whole 24-hour window, even for intraday releases.
    origins = times.floor("D")
    indices = released.searchsorted(origins, side="right") - 1
    valid = indices >= 0
    if history.empty:
        return np.empty((len(times), 0)), np.zeros((len(times), len(FUELS))), valid
    observations = history.iloc[np.maximum(indices, 0)]
    mix = observations[[f"fuelshare_{f}" for f in FUELS]].to_numpy(float)
    ends = pd.DatetimeIndex(pd.to_datetime(observations.period_end, utc=True))
    valid &= ends <= origins
    age = np.asarray((origins - ends).total_seconds() / (86400 * 90))
    hour = np.asarray(local_hours(times, region), dtype=float)
    phase = 2 * np.pi * hour / 24
    season = 2 * np.pi * (times.dayofyear.to_numpy() - 1) / 365.25
    periodic = np.column_stack([np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase)])
    x = np.column_stack([mix, age, np.sin(season), np.cos(season), periodic,
                         (mix[:, :, None] * periodic[:, None, :]).reshape(len(times), -1)])
    return x, mix, valid


@dataclass
class References:
    template: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    ridge: np.ndarray
    tree: object
    ef: np.ndarray

    def predict(self, times, region: str, monthly: pd.DataFrame) -> dict[str, np.ndarray]:
        times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
        x, mix, valid = features(times, region, monthly)
        if not valid.all():
            raise ValueError("No eligible historical observation for prediction")
        hours = np.asarray(local_hours(times, region), dtype=int)
        template = self.template[times.month.to_numpy() - 1, hours]
        # Center each source seasonal day at its own 24-hour mean, not a future target mean.
        centered = template - self.template[times.month.to_numpy() - 1].mean(axis=1)
        level = mix @ self.ef
        design = np.column_stack([np.ones(len(x)), (x - self.feature_mean) / self.feature_scale])
        return {"source_template": template, "lagged_level": level,
                "level_template": np.clip(level + centered, 0, self.ef.max()),
                "ridge": np.clip(design @ self.ridge, 0, self.ef.max()),
                "extra_trees": np.clip(self.tree.predict(x), 0, self.ef.max())}


def fit_references(frames: dict[str, pd.DataFrame], monthly: pd.DataFrame,
                   ef: np.ndarray, max_year: int) -> References:
    from sklearn.ensemble import ExtraTreesRegressor

    xs, ys, balanced, profiles = [], [], [], []
    for region, frame in sorted(frames.items()):
        times = pd.DatetimeIndex(pd.to_datetime(frame.time, utc=True))
        take = frame.valid.to_numpy(bool) & (times.year <= max_year)
        part = frame.loc[take]
        times = times[take]
        y = proxy_values(part, ef)
        x, _, permitted = features(times, region, monthly)
        keep = permitted & np.isfinite(y)
        xs.append(x[keep]); ys.append(y[keep])
        balanced.append(np.full(keep.sum(), 1 / max(1, keep.sum())))
        p = pd.DataFrame({"month": times.month, "hour": local_hours(times, region), "y": y})
        profiles.append(p.groupby(["month", "hour"]).y.mean())
    x, y = np.concatenate(xs), np.concatenate(ys)
    weights = np.concatenate(balanced)
    weights *= len(weights) / weights.sum()
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.maximum(scale, 1e-6)
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    penalty = np.eye(design.shape[1]) * 100.0
    penalty[0, 0] = 0
    ridge = np.linalg.solve(design.T @ (weights[:, None] * design) + penalty,
                            design.T @ (weights * y))
    tree = ExtraTreesRegressor(n_estimators=64, min_samples_leaf=24, max_depth=18,
                              random_state=718, n_jobs=1)
    tree.fit(x, y, sample_weight=weights)
    seasonal = pd.concat(profiles, axis=1).mean(axis=1)
    template = seasonal.reindex(pd.MultiIndex.from_product([range(1, 13), range(24)])).to_numpy().reshape(12, 24)
    if not np.isfinite(template).all():
        raise ValueError("Source seasonal reference has an unobserved calendar cell")
    return References(template, mean, scale, ridge, tree, np.asarray(ef))


def calibrate_reference_errors(frames: dict[str, pd.DataFrame], monthly: pd.DataFrame,
                               ef: np.ndarray, count: int = 128):
    """2022-fit/2023 source temporal errors; never read an outer target frame.

    Error curves retain their bias. Adding them is explicit source residual
    calibration, with no target coverage guarantee.
    """
    model = fit_references(frames, monthly, ef, max_year=2022)
    records: dict[str, list[np.ndarray]] = {}
    for region, frame in sorted(frames.items()):
        times = pd.DatetimeIndex(pd.to_datetime(frame.time, utc=True))
        part = frame.loc[frame.valid.to_numpy(bool) & (times.year == 2023)].copy()
        part["day"] = pd.to_datetime(part.time, utc=True).dt.floor("D")
        predictions = model.predict(part.time, region, monthly)
        truths = proxy_values(part, ef)
        for positions in part.groupby("day").indices.values():
            if len(positions) != 24:
                continue
            for name, values in predictions.items():
                records.setdefault(name, []).append(truths[positions] - values[positions])
    rng = np.random.default_rng(901)
    result = {}
    for name, errors in records.items():
        bank = np.asarray(errors)
        result[name] = bank[rng.choice(len(bank), count, replace=len(bank) < count)]
    return result, {name: len(errors) for name, errors in records.items()}
