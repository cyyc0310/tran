"""Exact clock-assignment losses for an unchanged set of 24 predictive laws."""
from __future__ import annotations

import numpy as np


def phase_scores(laws, actual: np.ndarray, clock: np.ndarray, radius: float) -> dict:
    if len(laws) != 24 or len(actual) != len(clock) or not len(actual):
        raise ValueError('phase score dimensions')
    if not np.isfinite(actual).all() or np.any((clock < 0) | (clock >= 24)):
        raise ValueError('invalid phase observations')
    clock = np.asarray(clock, int)
    counts = np.bincount(clock, minlength=24)
    loss, absolute, covered, interval = [np.zeros((24, 24)) for _ in range(4)]
    width = np.zeros(24)
    marginal = np.zeros(24)
    for h, law in enumerate(laws):
        point_loss = law.crps(actual)
        loss[h] = np.bincount(clock, weights=point_loss, minlength=24)
        marginal[h] = point_loss.mean()
        absolute[h] = np.bincount(clock, weights=np.abs(actual - law.mean), minlength=24)
        lo = max(0., law.quantile(.05) - radius)
        hi = min(1000., law.quantile(.95) + radius)
        if not np.isfinite(lo + hi) or lo > hi:
            raise ValueError('invalid phase interval')
        width[h] = hi - lo
        covered[h] = np.bincount(clock, weights=(actual >= lo) & (actual <= hi), minlength=24)
        winkler = hi - lo + 20 * np.maximum(lo - actual, 0) + 20 * np.maximum(actual - hi, 0)
        interval[h] = np.bincount(clock, weights=winkler, minlength=24)
    observed_clock = np.arange(24)
    assigned = (observed_clock[None, :] + np.arange(24)[:, None]) % 24
    result = {name: values[assigned, observed_clock].sum(axis=1) / len(actual)
              for name, values in [('crps', loss), ('mae', absolute), ('coverage', covered), ('winkler', interval)]}
    result['width'] = (width[assigned] * counts).sum(axis=1) / len(actual)
    result['calendar_width'] = width[assigned].mean(axis=1)
    result['shuffle_crps'] = (marginal[assigned] * counts).sum(axis=1) / len(actual)
    result['clock_occupancy_discrepancy'] = np.full(24, np.max(np.abs(counts / len(actual) - 1 / 24)))
    if np.ptp(result['calendar_width']) > 1e-10:
        raise ValueError('calendar width changed under a permutation')
    return result


def signed_phase(phase: int) -> int:
    return phase if phase <= 12 else phase - 24


def choose_phase(phases: np.ndarray, crps: np.ndarray) -> int:
    candidates = np.asarray(phases, int)[np.asarray(crps) <= np.min(crps) + 1e-10]
    return int(min(candidates, key=lambda p: (abs(signed_phase(int(p))), signed_phase(int(p)))))
