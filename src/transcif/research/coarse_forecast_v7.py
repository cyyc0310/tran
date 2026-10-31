"""Released fuel-composition forecasts, with genuinely prequential experts.

These forecasts are not constraints on the unknown future month. Expert losses
use historical months only after their release and reconstruct their own
mid-month information set with the same release delay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import softmax

from .eia_proxy import FUELS
from .dynamic_v2 import EF

COLUMNS = [f"fuelshare_{fuel}" for fuel in FUELS]
EXPERTS = ("latest", "seasonal", "annual_drift")


def simplex(values):
    values = np.maximum(np.asarray(values, float), 0.)
    total = values.sum(axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(total <= 0):
        raise ValueError("Finite, positive composition mass required")
    return values / total


def expert_forecasts(history, origin):
    """Three composition forecasts from a pre-filtered released prefix."""
    if history.empty:
        return np.full((3, len(FUELS)), 1 / len(FUELS))
    latest = history[COLUMNS].iloc[-1].to_numpy(float)
    same = history[history.period_start.dt.month.eq(origin.month)]
    seasonal = same[COLUMNS].iloc[-1].to_numpy(float) if len(same) else latest
    deltas = []
    lookup = {(r.period_start.year, r.period_start.month): r for r in history.itertuples()}
    for row in list(history.itertuples())[-6:]:
        old = lookup.get((row.period_start.year - 1, row.period_start.month))
        if old is not None:
            deltas.append(np.array([getattr(row, c) - getattr(old, c) for c in COLUMNS]))
    drift = np.median(deltas, axis=0) if deltas else np.zeros(len(FUELS))
    return simplex(np.stack([latest, seasonal, seasonal + drift]))


def released_forecasts(monthly, region, origins, extra_delay=0):
    """Return expert mixes, past losses, and common baseline covariates.

    Baselines receive all expert vectors and both adaptive mixtures; candidates
    do not get an exclusive input. No target hourly argument is accepted.
    """
    if extra_delay < 0:
        raise ValueError("Negative publication delay")
    part = monthly[monthly.region.eq(region)].copy()
    for c in ("period_start", "period_end", "available_at"):
        part[c] = pd.to_datetime(part[c], utc=True)
    part["available_at"] += pd.Timedelta(days=extra_delay)
    part = part.sort_values(["period_end", "available_at"])
    if part.period_start.duplicated().any():
        raise ValueError("One frozen vintage per region/month required")
    if len(part):
        vals = part[COLUMNS].to_numpy(float)
        if not np.isfinite(vals).all() or np.any(vals < 0) or not np.allclose(vals.sum(1), 1):
            raise ValueError("Invalid released fuel composition")
    origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
    predictions, losses, extras, biases = [], [], [], []
    cache = {}
    for origin in origins:
        prefix = part[(part.period_end <= origin) & (part.available_at <= origin)]
        key = (tuple(prefix.index), origin.month)
        if key not in cache:
            forecast = expert_forecasts(prefix, origin)
            errors, signed_errors = [], []
            for _, label in prefix.tail(24).iterrows():
                pseudo = label.period_start + (label.period_end - label.period_start) / 2
                past = part[(part.period_end <= pseudo) & (part.available_at <= pseudo)]
                if len(past) < 12:
                    continue
                predicted = expert_forecasts(past, pseudo)
                truth = label[COLUMNS].to_numpy(float)
                errors.append(.5 * ((np.sqrt(predicted) - np.sqrt(truth))**2).sum(1))
                signed_errors.append((truth[None, :] - predicted) @ EF)
            loss = np.mean(errors, axis=0) if errors else np.zeros(3)
            bias = np.mean(signed_errors, axis=0) if errors else np.zeros(3)
            cache[key] = forecast, loss, len(errors), bias
        forecast, loss, n_backtests, bias = cache[key]
        age = (origin - prefix.period_end.iloc[-1]).total_seconds() / 86400 if len(prefix) else 365.
        predictions.append(forecast); losses.append(loss)
        biases.append(bias)
        extras.append([age / 90, min(len(prefix), 60) / 60, n_backtests / 24])
    forecasts, loss = np.asarray(predictions), np.asarray(losses)
    weights = {str(t): softmax(-loss / t, axis=1) for t in (.03, .1)}
    angles = 2 * np.pi * (origins.dayofyear.to_numpy() - 1) / 365.25
    calendar = np.column_stack([np.sin(angles), np.cos(angles), np.sin(2*angles),
        np.cos(2*angles), origins.dayofweek.to_numpy() / 6])
    features = np.column_stack([forecasts.reshape(len(origins), -1), loss, biases,
        *[np.einsum("ne,nef->nf", w, forecasts) for w in weights.values()], extras, calendar])
    return {"experts": forecasts, "losses": loss, "weights": weights,
            "features": features, "anchor": np.asarray(biases)[:, 1],
            "feature_count": features.shape[1]}
