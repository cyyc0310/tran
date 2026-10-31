"""Legal, monthly-only fuel composition dynamics for V9 comparisons.

Every historical error is generated at a pseudo origin in its target month.
Its label enters an origin's calibration only after both the month ends and
the (optionally delayed) release occurs. No hourly observations are accepted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from transcif.research.coarse_forecast_v7 import COLUMNS, released_forecasts

METHODS = ("latest", "seasonal", "v7_mixture", "harmonic_ridge",
           "seasonal_kalman", "release_bridge")
_WINDOW = 24
_PSEUDO_WINDOW = 60


def _simplex(value: np.ndarray, active: np.ndarray | None = None) -> np.ndarray:
    """Euclidean simplex projection, restricted to observed fuel support."""
    x = np.asarray(value, float)
    if active is None:
        active = np.ones(len(COLUMNS), bool)
    if not active.any():
        active = np.ones(len(COLUMNS), bool)
    y = x[active]
    ordered = np.sort(y)[::-1]
    rho = np.flatnonzero(ordered - (np.cumsum(ordered) - 1) /
                         np.arange(1, len(y) + 1) > 0)[-1]
    threshold = (ordered[:rho + 1].sum() - 1) / (rho + 1)
    result = np.zeros(len(COLUMNS))
    result[active] = np.maximum(y - threshold, 0)
    return result / result.sum()


def _simplex_rows(values: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Project sampled joint residual vectors in one batch."""
    x = np.asarray(values, float)
    if not active.any():
        active = np.ones(len(COLUMNS), bool)
    y = x[:, active]
    ordered = np.sort(y, axis=1)[:, ::-1]
    threshold_candidates = (np.cumsum(ordered, axis=1) - 1) / np.arange(1, y.shape[1] + 1)
    rho = np.sum(ordered > threshold_candidates, axis=1) - 1
    threshold = threshold_candidates[np.arange(len(y)), rho]
    result = np.zeros_like(x)
    result[:, active] = np.maximum(y - threshold[:, None], 0)
    return result / result.sum(axis=1, keepdims=True)


def _support(history: pd.DataFrame) -> np.ndarray:
    return history[COLUMNS].to_numpy(float).max(axis=0) > 0 if len(history) else np.ones(len(COLUMNS), bool)


def _ordinal(timestamp: pd.Timestamp) -> int:
    return timestamp.year * 12 + timestamp.month - 1


def _horizon(history: pd.DataFrame, target: pd.Timestamp) -> int:
    return (_ordinal(target) - _ordinal(history.period_start.iloc[-1])) if len(history) else 99


def _pseudo_origin(label: pd.Series, day: int) -> pd.Timestamp:
    # The same calendar day is used in each historical month. Short months clip.
    end = label.period_end
    return label.period_start + pd.Timedelta(days=min(day, (end - label.period_start).days) - 1)


def _seasonal(history: pd.DataFrame, target: pd.Timestamp) -> np.ndarray:
    if history.empty:
        return np.full(len(COLUMNS), 1 / len(COLUMNS))
    same = history.loc[history.period_start.dt.month.eq(target.month)]
    return (same if len(same) else history)[COLUMNS].iloc[-1].to_numpy(float)


def _helmert() -> np.ndarray:
    basis = np.zeros((len(COLUMNS), len(COLUMNS) - 1))
    for k in range(1, len(COLUMNS)):
        basis[:k, k - 1] = 1 / np.sqrt(k * (k + 1))
        basis[k, k - 1] = -k / np.sqrt(k * (k + 1))
    return basis


_BASIS = _helmert()


def _ridge(history: pd.DataFrame, target: pd.Timestamp) -> np.ndarray:
    if len(history) < 12:
        return _seasonal(history, target)
    rows = history.tail(60)
    months = np.array([_ordinal(t) for t in rows.period_start], float)
    month = rows.period_start.dt.month.to_numpy(float)
    phase = 2 * np.pi * (month - 1) / 12
    design = np.column_stack([np.ones(len(rows)), (months - months[-1]) / 12,
                              np.sin(phase), np.cos(phase),
                              np.sin(2 * phase), np.cos(2 * phase)])
    response = (rows[COLUMNS].to_numpy(float) - 1 / len(COLUMNS)) @ _BASIS
    penalty = np.diag([0, 6, 2, 2, 4, 4])
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ response)
    target_phase = 2 * np.pi * (target.month - 1) / 12
    query = np.array([1, (_ordinal(target) - months[-1]) / 12,
                      np.sin(target_phase), np.cos(target_phase),
                      np.sin(2 * target_phase), np.cos(2 * target_phase)])
    return _simplex(1 / len(COLUMNS) + (query @ coefficient) @ _BASIS.T,
                    _support(history))


def _kalman(history: pd.DataFrame, target: pd.Timestamp) -> np.ndarray:
    if len(history) < 12:
        return _seasonal(history, target)
    rows = history.tail(60)
    values = rows[COLUMNS].to_numpy(float)
    center = values.mean(axis=0)
    months = rows.period_start.dt.month.to_numpy()
    templates = np.stack([(values[months == m].sum(axis=0) + 3 * center) /
                          (np.count_nonzero(months == m) + 3) for m in range(1, 13)])
    level = np.zeros(len(COLUMNS))
    variance = .02
    previous = _ordinal(rows.period_start.iloc[0])
    for t, value, month in zip(rows.period_start, values, months):
        current = _ordinal(t)
        variance += .0025 * max(current - previous, 0)
        gain = variance / (variance + .01)
        level += gain * (value - templates[month - 1] - level)
        variance *= 1 - gain
        previous = current
    return _simplex(templates[target.month - 1] + level, _support(history))


def _bias(records: list[dict], at: pd.Timestamp, horizon: int) -> tuple[np.ndarray, int]:
    eligible = [r for r in records if r["period_end"] <= at and r["available_at"] <= at]
    eligible = eligible[-_WINDOW:]
    if not eligible:
        return np.zeros(len(COLUMNS)), 0
    gaps = np.array([abs(r["horizon"] - horizon) for r in eligible], float)
    age = np.arange(len(eligible) - 1, -1, -1, dtype=float)
    weight = np.exp(-gaps / 1.5 - age / 12)
    residual = np.stack([r["truth"] - r["v7_mixture"] for r in eligible])
    effective = weight.sum() ** 2 / np.sum(weight ** 2)
    return (weight @ residual / weight.sum()) * effective / (effective + 4), len(eligible)


def _ordinary(history: pd.DataFrame, target: pd.Timestamp, v7_mix: np.ndarray) -> dict[str, np.ndarray]:
    if history.empty:
        uniform = np.full(len(COLUMNS), 1 / len(COLUMNS))
        return {name: uniform.copy() for name in METHODS if name != "release_bridge"}
    latest = history[COLUMNS].iloc[-1].to_numpy(float)
    return {"latest": latest, "seasonal": _seasonal(history, target),
            "v7_mixture": v7_mix, "harmonic_ridge": _ridge(history, target),
            "seasonal_kalman": _kalman(history, target)}


@dataclass
class _Forecast:
    points: dict[str, np.ndarray]
    records: list[dict]
    n_bias: int


def _one_origin(part: pd.DataFrame, region: str, origin: pd.Timestamp,
                day: int, extra_delay: int,
                point_cache: dict[tuple, dict[str, np.ndarray]]) -> _Forecast:
    prefix = part.loc[(part.period_end <= origin) & (part.delayed_at <= origin)]
    labels = prefix.tail(_PSEUDO_WINDOW)
    pseudo = [_pseudo_origin(row, day) for _, row in labels.iterrows()]
    def ordinary_at(history: pd.DataFrame, at: pd.Timestamp) -> dict[str, np.ndarray]:
        key = (tuple(history.index), at.year, at.month)
        if key not in point_cache:
            # V7's temperature-0.03 predictive *mean* is exact. Its original
            # three-expert distribution is deliberately replaced below by the
            # common prequential residual law used for every V9 comparison.
            vintage = history.drop(columns="delayed_at")
            v7 = released_forecasts(vintage, region, [at], extra_delay)
            v7_mix = np.einsum("ne,nef->nf", v7["weights"]["0.03"],
                               v7["experts"])[0]
            point_cache[key] = _ordinary(history, at, v7_mix)
        return point_cache[key]
    records: list[dict] = []
    for i, (_, label) in enumerate(labels.iterrows()):
        historical = prefix.loc[(prefix.period_end <= pseudo[i]) &
                                (prefix.delayed_at <= pseudo[i])]
        ordinary = ordinary_at(historical, pseudo[i]).copy()
        horizon = _horizon(historical, pseudo[i])
        bias, _ = _bias(records, pseudo[i], horizon)
        ordinary["release_bridge"] = _simplex(ordinary["v7_mixture"] + bias,
                                               _support(historical))
        records.append({"period_end": label.period_end, "available_at": label.delayed_at,
                        "horizon": horizon, "truth": label[COLUMNS].to_numpy(float),
                        **ordinary})
    horizon = _horizon(prefix, origin)
    ordinary = ordinary_at(prefix, origin).copy()
    bias, n_bias = _bias(records, origin, horizon)
    ordinary["release_bridge"] = _simplex(ordinary["v7_mixture"] + bias,
                                           _support(prefix))
    return _Forecast(ordinary, records[-_WINDOW:], n_bias)


def _draws(point: np.ndarray, records: list[dict], method: str,
           horizon: int, n_samples: int, seed: int, date: pd.Timestamp,
           active: np.ndarray) -> np.ndarray:
    # All methods use this identical prequential, joint-vector residual sampler.
    # A date-derived stream makes scalar, batched, and differently ordered calls
    # return the same samples. The same label indices are used across methods.
    key = [seed & 0xffffffff, (seed >> 32) & 0xffffffff,
           date.year, date.month, date.day]
    rng = np.random.default_rng(np.random.SeedSequence(key))
    if not records:
        return np.repeat(point[None, :], n_samples, axis=0)
    age = np.arange(len(records) - 1, -1, -1, dtype=float)
    gaps = np.array([abs(r["horizon"] - horizon) for r in records], float)
    weights = np.exp(-age / 12 - gaps / 1.5)
    weights /= weights.sum()
    residual = np.stack([r["truth"] - r[method] for r in records])
    choice = rng.choice(len(records), size=n_samples, p=weights)
    return _simplex_rows(point[None, :] + residual[choice], active)


def forecast_monthly(monthly: pd.DataFrame, region: str, origins,
                     extra_delay: int = 0, n_samples: int = 128,
                     seed: int = 17023) -> dict[str, dict]:
    """Forecast the fuel mix in each origin's calendar month.

    `mix` has shape (N, 8), `draws` has shape (N, S, 8). `audit` is a small,
    JSON-serializable account of released and prequential label support.
    """
    if not isinstance(extra_delay, (int, np.integer)) or extra_delay < 0:
        raise ValueError("extra_delay must be a nonnegative integer number of days")
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
        raise ValueError("n_samples must be positive")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    dates = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
    if dates.hasnans:
        raise ValueError("origins must have finite dates")
    part = monthly.loc[monthly.region.eq(region)].copy()
    for column in ("period_start", "period_end", "available_at"):
        part[column] = pd.to_datetime(part[column], utc=True)
    if len(part):
        if part[["period_start", "period_end", "available_at"]].isna().any().any():
            raise ValueError("Monthly timestamps must be finite")
        starts = part.period_start
        if (not starts.eq(starts.dt.normalize()).all() or
                not starts.dt.day.eq(1).all() or
                not part.period_end.eq(starts + pd.offsets.MonthBegin(1)).all()):
            raise ValueError("Monthly intervals must cover one UTC calendar month")
    part["delayed_at"] = part.available_at + pd.Timedelta(days=int(extra_delay))
    if len(dates):
        part = part.loc[(part.period_end <= dates.max()) & (part.delayed_at <= dates.max())]
    else:
        part = part.iloc[:0]
    part = part.sort_values(["period_end", "delayed_at"]).reset_index(drop=True)
    if part.period_start.duplicated().any():
        raise ValueError("One frozen vintage per region/month required")
    if len(part):
        values = part[COLUMNS].to_numpy(float)
        if (not np.isfinite(values).all() or np.any(values < 0) or
                not np.allclose(values.sum(axis=1), 1)):
            raise ValueError("Invalid released fuel composition")
    output = {name: {"mix": np.empty((len(dates), len(COLUMNS))),
                     "draws": np.empty((len(dates), n_samples, len(COLUMNS))),
                     "audit": {"draw_law": "shared_prequential_joint_residual",
                               "v7_point": "exact_temperature_0.03_expert_mean" if name == "v7_mixture" else None,
                               "released_months": [], "prequential_labels": [],
                               "bias_labels": [], "horizon_months": [],
                               "fallback": []}} for name in METHODS}
    cache: dict[tuple, _Forecast] = {}
    point_cache: dict[tuple, dict[str, np.ndarray]] = {}
    for i, origin in enumerate(dates):
        prefix = part.loc[(part.period_end <= origin) & (part.delayed_at <= origin)]
        key = (tuple(prefix.index), origin.year, origin.month, origin.day)
        if key not in cache:
            cache[key] = _one_origin(part, region, origin, origin.day,
                                     int(extra_delay), point_cache)
        result = cache[key]
        horizon = _horizon(prefix, origin)
        active = _support(prefix)
        for method in METHODS:
            entry = output[method]
            entry["mix"][i] = result.points[method]
            entry["draws"][i] = _draws(result.points[method], result.records,
                                        method, horizon, n_samples, int(seed), origin, active)
            audit = entry["audit"]
            audit["released_months"].append(int(len(prefix)))
            audit["prequential_labels"].append(int(len(result.records)))
            audit["bias_labels"].append(int(result.n_bias if method == "release_bridge" else 0))
            audit["horizon_months"].append(int(horizon))
            audit["fallback"].append("cold_start" if len(prefix) == 0 else
                                     "short_history" if method in ("harmonic_ridge", "seasonal_kalman")
                                     and len(prefix) < 12 else "none")
    return output
