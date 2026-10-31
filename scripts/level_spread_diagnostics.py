"""Stable moments and explicitly oracle affine perturbations of fixed CIF laws."""
from __future__ import annotations

import numpy as np

from calendar_cif_law import CalendarLaw
from v154_hourly_strong_baselines import CompressedLaw

NUMERICAL_VARIANCE = 1e-8


def moments(laws, clock: np.ndarray) -> tuple[float, float]:
    occupancy = np.bincount(clock, minlength=24) / len(clock)
    mean = float(occupancy @ np.asarray([law.mean for law in laws]))
    variance = float(sum(p * (law.w @ (law.x - mean)**2) for p, law in zip(occupancy, laws)))
    return mean, variance


def spread_ratio(actual_variance: float, predicted_variance: float) -> tuple[float, bool]:
    if predicted_variance <= NUMERICAL_VARIANCE:
        return 1., actual_variance > NUMERICAL_VARIANCE
    if actual_variance <= NUMERICAL_VARIANCE:
        return 0., False
    return float(np.sqrt(actual_variance / predicted_variance)), False


def affine_laws(laws, clock, delta: float, scale: float, bounds, phase: int = 0):
    shifted = [laws[(h + phase) % 24] for h in range(24)]
    mean, _ = moments(shifted, clock)
    epsilon = 64 * np.finfo(float).eps
    if abs(delta) <= epsilon * max(1., abs(mean), abs(mean + delta)):
        delta = 0.
    if abs(scale - 1) <= epsilon:
        scale = 1.
    if not np.isfinite(delta + scale) or scale < 0:
        raise ValueError('invalid oracle affine parameters')
    if delta == 0 and scale == 1:
        return shifted, 0., delta, scale
    occupancy = np.bincount(clock, minlength=24) / len(clock)
    result, clipping = [], 0.
    lo, hi = bounds
    for p, law in zip(occupancy, shifted):
        values = mean + delta + scale * (law.x - mean)
        tolerance = epsilon * np.maximum.reduce([np.ones(len(values)), np.abs(values),
                                                 np.full(len(values), abs(lo)), np.full(len(values), abs(hi))])
        clipped = (values < lo - tolerance) | (values > hi + tolerance)
        clipping += p * float(law.w @ clipped)
        result.append(CompressedLaw(np.clip(values, lo, hi), law.w))
    return result, clipping, delta, scale


def score_laws(laws, actual, clock, radius: float) -> dict:
    law = CalendarLaw(laws, clock)
    lo = np.maximum(0, law.quantile(.05) - radius)
    hi = np.minimum(1000, law.quantile(.95) + radius)
    if np.any(lo > hi):
        raise ValueError('inverted diagnostic interval')
    tolerance = 64 * np.finfo(float).eps * np.maximum.reduce([
        np.ones(len(actual)), np.abs(actual), np.abs(lo), np.abs(hi)])
    lower = actual < lo
    upper = actual > hi
    lower_ulp = actual < lo - tolerance
    upper_ulp = actual > hi + tolerance
    support_lo = np.asarray([distribution.x[0] for distribution in laws])[clock]
    support_hi = np.asarray([distribution.x[-1] for distribution in laws])[clock]
    mean, variance = moments(laws, clock)
    return {'crps': float(law.crps(actual).mean()), 'mae': float(np.abs(actual - law.mean).mean()),
            'coverage': float(np.mean(~lower & ~upper)), 'tolerant_coverage': float(np.mean(~lower_ulp & ~upper_ulp)),
            'lower_miss': float(lower.mean()), 'upper_miss': float(upper.mean()),
            'lower_miss_ulp64': float(lower_ulp.mean()), 'upper_miss_ulp64': float(upper_ulp.mean()),
            'width': float(np.mean(hi - lo)),
            'winkler': float(np.mean(hi - lo + 20 * np.maximum(lo - actual, 0) + 20 * np.maximum(actual - hi, 0))),
            'support_below': float(np.mean(actual < support_lo - tolerance)),
            'support_above': float(np.mean(actual > support_hi + tolerance)),
            'post_mean': mean, 'post_variance': variance}
