"""Frozen-style coarse distribution scores with separate numerical coverage checks."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from v151_hourly_law import EF
from v154_hourly_strong_baselines import CompressedLaw

COARSE_METRICS = ('share_crps_mean', 'energy_score', 'same_EF_energy_score', 'anchor_crps',
                  'anchor_coverage', 'anchor_width', 'tolerant_anchor_coverage')
FUEL_METRICS = ('crps', 'coverage', 'width', 'zero_brier', 'predicted_zero', 'actual_zero', 'tolerant_coverage')
NONCONSTANT = ('CHPD', 'DOPD', 'GVL', 'SCL', 'SPA')


def same_ef(values):
    projected = np.zeros_like(values)
    for ids in ([3, 4, 5, 6], [1, 7]):
        projected[..., ids] = values[..., ids] - values[..., ids].mean(axis=-1, keepdims=True)
    return projected


def covered(lower, upper, actual):
    epsilon = 64 * np.finfo(float).eps * max(1., abs(lower), abs(upper), abs(actual))
    return float(lower <= actual <= upper), float(lower - epsilon <= actual <= upper + epsilon)


def score(values, probabilities, actual, cache):
    values, probabilities, actual = [np.asarray(v, float) for v in (values, probabilities, actual)]
    if (values.ndim != 2 or values.shape[1] != 8 or actual.shape != (8,) or probabilities.shape != (len(values),)
            or not np.isfinite(values).all() or not np.isfinite(probabilities).all() or not np.isfinite(actual).all()
            or (values < 0).any() or (probabilities <= 0).any() or abs(probabilities.sum() - 1) > 1e-12
            or max(abs(values.sum(axis=1) - 1)) > 1e-12):
        raise ValueError('invalid coarse scoring distribution')
    key = values.shape, hashlib.sha256(values.tobytes()).digest()
    if key not in cache:
        projected = same_ef(values)
        cache[key] = cdist(values, values), cdist(projected, projected), projected
    distance, projected_distance, projected = cache[key]
    energy = float(probabilities @ np.linalg.norm(values - actual, axis=1) - .5 * probabilities @ distance @ probabilities)
    projected_energy = float(probabilities @ np.linalg.norm(projected - same_ef(actual), axis=1)
                             - .5 * probabilities @ projected_distance @ probabilities)
    fuels, marginal = [], []
    for f in range(8):
        law = CompressedLaw(values[:, f], probabilities)
        lower, upper = law.quantile(.05), law.quantile(.95)
        strict, tolerant = covered(lower, upper, actual[f])
        crps = float(law.crps(np.asarray([actual[f]]))[0])
        zero = float(probabilities @ (values[:, f] == 0))
        marginal.append(crps)
        fuels.append({'fuel_index': f, 'crps': crps, 'coverage': strict, 'tolerant_coverage': tolerant,
                      'width': upper - lower, 'zero_brier': float((zero - (actual[f] == 0))**2),
                      'predicted_zero': zero, 'actual_zero': bool(actual[f] == 0)})
    anchor = CompressedLaw(values @ EF, probabilities)
    target = float(actual @ EF)
    lower, upper = anchor.quantile(.05), anchor.quantile(.95)
    strict, tolerant = covered(lower, upper, target)
    coarse = {'share_crps_mean': float(np.mean(marginal)), 'energy_score': energy,
              'same_EF_energy_score': projected_energy, 'anchor_crps': float(anchor.crps(np.asarray([target]))[0]),
              'anchor_coverage': strict, 'tolerant_anchor_coverage': tolerant, 'anchor_width': upper - lower}
    if not np.isfinite(list(coarse.values())).all() or min(energy, projected_energy) < -1e-10:
        raise ValueError('invalid finite coarse loss')
    return coarse, fuels


def aggregate(coarse, fuels):
    regions = coarse.groupby(['region', 'method'])[list(COARSE_METRICS)].mean().reset_index()
    fuel_regions = fuels.groupby(['region', 'method', 'fuel_index'])[list(FUEL_METRICS)].mean().reset_index()
    macro, fuel_macro = [], []
    for cohort in ('all29', 'nonconstant24'):
        subset = regions if cohort == 'all29' else regions.loc[~regions.region.isin(NONCONSTANT)]
        for method, group in subset.groupby('method'):
            macro.append({'cohort': cohort, 'method': method, 'regions': len(group),
                          **{m: float(group[m].mean()) for m in COARSE_METRICS},
                          'worst_anchor_coverage': float(group.anchor_coverage.min()),
                          'worst_tolerant_anchor_coverage': float(group.tolerant_anchor_coverage.min())})
        subset = fuel_regions if cohort == 'all29' else fuel_regions.loc[~fuel_regions.region.isin(NONCONSTANT)]
        for (method, fuel), group in subset.groupby(['method', 'fuel_index']):
            fuel_macro.append({'cohort': cohort, 'method': method, 'fuel_index': int(fuel), 'regions': len(group),
                               **{m: float(group[m].mean()) for m in FUEL_METRICS}})
    return regions, pd.DataFrame(macro), fuel_regions, pd.DataFrame(fuel_macro)
