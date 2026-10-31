"""Ordinary, atom-aware PIT recalibration with nested source-region supervision."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from coarse_distribution_lifting import CoarseLawForecast
from fuel_activity_forest import ActivityDonors, source_activity
from monthly_history_features import monthly_context
from scenario_reweighting import ScenarioPrediction
from v151_hourly_law import EF
from v154_hourly_strong_baselines import CompressedLaw, balanced

GRID = np.asarray([0, .005, .01, .025, .05, .1, .2, .3, .4, .5,
                   .6, .7, .8, .9, .95, .975, .99, .995, 1.])
REFERENCES = ('conditional_activity_recomposition', 'world_gaussian')
SETTINGS = {f'{prefix}_pit_{kind}_a{int(alpha * 100)}': (reference, kind, alpha)
            for prefix, reference in zip(('conditional', 'gaussian'), REFERENCES)
            for kind in ('global', 'conditional20', 'conditional43') for alpha in (.5, 1.)}
METHODS = REFERENCES + tuple(SETTINGS)


def inner_roles(regions):
    """Fixed region folds; no hourly values enter their construction."""
    ordered = sorted(regions, key=lambda r: hashlib.sha256(('v174/inner/' + r).encode()).hexdigest())
    return {r: i % 5 for i, r in enumerate(ordered)}


def source_samples(archive: Path, regions):
    frame = pd.read_csv(archive / 'source_sampling.csv')
    frame = frame.loc[frame.budget.eq('asof_30d')].set_index('region')
    result = {r: np.asarray(json.loads(frame.loc[r, 'sampled_raw_hour_indices']), int) for r in sorted(regions)}
    if any(len(ids) > 4096 or len(ids) == 0 or np.any(np.diff(ids) <= 0) for ids in result.values()):
        raise ValueError('fixed source sample inventory')
    return result


class NestedPhysical:
    """Rebuild the physical bank using only one outer-proper inner training set."""

    def __init__(self, data, archive, fit, queries):
        self.data, self.budget = data, 'asof_30d'
        self.fit = sorted(fit)
        self.allowed_queries = set(map(int, queries))
        if not self.fit or set(data.index.region.iloc[list(self.allowed_queries)]) & set(self.fit):
            raise ValueError('inner query/training overlap')
        samples = source_samples(archive, self.fit)
        proper = {r: data.source(r, self.budget, samples[r]) for r in self.fit}
        prior = np.mean([data.shares[proper[r][3]].mean(axis=0) for r in self.fit], axis=0) + 1e-6
        self.prior = prior / prior.sum()
        sources = {r: source_activity(data, r, self.budget, samples[r], self.prior) for r in self.fit}
        self.bank = {r: np.unique(sources[r]['profiles']) for r in self.fit}
        self.activity = ActivityDonors(data, sources)
        self.month = pd.to_datetime(data.index.month).dt.month.to_numpy()


class PhysicalReferences(ScenarioPrediction):
    """The two original physical laws, without reading query hourly labels."""

    def __init__(self, frozen, monthly):
        super().__init__(frozen, CoarseLawForecast(monthly, frozen.fit))

    def build(self, q):
        forecast, _, B, donors, prior, anchors = self.inputs(q)
        logits = np.log(prior) - .5 * ((anchors - forecast['anchor']) / forecast['sigma'])**2
        gaussian = np.exp(np.maximum(logits - logits.max(), -700))
        gaussian /= gaussian.sum()
        result = {}
        for name, weights in zip(REFERENCES, (prior, gaussian)):
            law, rejected = self.frozen.activity.laws(donors, weights, B)
            if max(rejected) != 0:
                raise ValueError('positive-carrier conditioning changed')
            result[name] = law
        return result, forecast


def pit_bounds(law, actual):
    """PIT interval with an explicit numerical-only nearest-atom snap."""
    y = np.asarray(actual, float)
    if not np.isfinite(y).all():
        raise ValueError('nonfinite PIT outcome')
    right = np.searchsorted(law.x, y)
    lo = np.clip(right - 1, 0, len(law.x) - 1)
    hi = np.clip(right, 0, len(law.x) - 1)
    nearest = np.where(abs(y - law.x[lo]) <= abs(y - law.x[hi]), lo, hi)
    tolerance = 64 * np.finfo(float).eps * np.maximum.reduce([np.ones(len(y)), abs(y), abs(law.x[nearest])])
    snapped = abs(y - law.x[nearest]) <= tolerance
    left = np.searchsorted(law.x, np.where(snapped, y - tolerance, y), side='left')
    right = np.searchsorted(law.x, np.where(snapped, y + tolerance, y), side='right')
    changed = snapped & ((y != law.x[nearest]) | (right - left > 1))
    cumulative = np.clip(np.r_[0., np.cumsum(law.w)], 0, 1)
    cumulative[-1] = 1.
    lower, upper = cumulative[left], cumulative[right]
    outside = (~snapped) & ((y < law.x[0]) | (y > law.x[-1]))
    return lower, upper, changed, outside


def pit_targets(lower, upper):
    """Integrate randomized PIT jitter analytically; return interior CDF values."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if (lower.shape != upper.shape or lower.ndim != 1 or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or (lower < 0).any() or (upper > 1).any()
            or (upper < lower).any()):
        raise ValueError('invalid PIT bounds')
    span = upper - lower
    target = np.broadcast_to(GRID[1:-1], (len(lower), len(GRID) - 2)) - lower[:, None]
    result = (target >= 0).astype(float)
    np.divide(target, span[:, None], out=result, where=span[:, None] > 0)
    return np.clip(result, 0, 1)


def check_maps(values):
    values = np.asarray(values, float)
    tolerance = 1e-12
    if (values.shape[-1] != len(GRID) or not np.isfinite(values).all()
            or (values < -tolerance).any() or (values > 1 + tolerance).any()
            or (np.diff(values, axis=-1) < -tolerance).any()
            or np.any(values[..., 0] != 0) or np.any(values[..., -1] != 1)):
        raise ValueError('invalid monotone PIT map')
    return np.clip(values, 0, 1)


def recalibrate(law, mapping, alpha):
    """Apply a piecewise-linear CDF map; retain positive original support."""
    mapping = check_maps(mapping)
    if mapping.ndim != 1 or alpha not in (.5, 1.):
        raise ValueError('map shape or fixed strength')
    if np.array_equal(mapping, GRID):
        return law
    cumulative = np.clip(np.r_[0., np.cumsum(law.w)], 0, 1)
    cumulative[-1] = 1.
    weights = np.diff(np.interp(cumulative, GRID, mapping))
    if np.min(weights) < -1e-12:
        raise ValueError('nonmonotone transformed mass')
    weights = (1 - .95 * alpha) * law.w + .95 * alpha * np.maximum(weights, 0)
    result = CompressedLaw(law.x, weights)
    if not np.array_equal(law.x, result.x) or not (result.w > 0).all():
        raise ValueError('physical support not preserved')
    return result


class PITRecalibration:
    """Source-only global and conditional empirical PIT-CDF controls."""

    def __init__(self, frozen, monthly, archive: Path):
        self.frozen, self.monthly = frozen, monthly
        self.physical = PhysicalReferences(frozen, monthly)
        with np.load(archive / 'oof.npz', allow_pickle=False) as saved:
            self.raw_ids = saved['raw_id'].copy()
            self.profiles = saved['profile_id'].copy()
            self.regions = saved['region'].copy()
            self.inner = saved['inner_fold'].copy()
            lower, upper = saved['lower'].copy(), saved['upper'].copy()
        samples = source_samples(archive.parent.parent / 'v159-fuel-activity-source-pilot-r1', frozen.fit)
        wanted = np.concatenate([samples[r] for r in sorted(frozen.fit)])
        roles = inner_roles(frozen.fit)
        if (not np.array_equal(self.raw_ids, wanted)
                or not np.array_equal(self.regions, frozen.data.index.region.to_numpy()[self.profiles])
                or not np.array_equal(self.inner, np.asarray([roles[r] for r in self.regions]))
                or lower.shape != (len(wanted), 2) or upper.shape != lower.shape):
            raise ValueError('OOF data identity')
        expected_profiles = np.searchsorted(frozen.data.index.stop.to_numpy(), self.raw_ids, side='right')
        if not np.array_equal(self.profiles, expected_profiles):
            raise ValueError('OOF raw/profile binding')
        contexts = {int(q): monthly_context(monthly, int(q)) for q in np.unique(self.profiles)}
        X20 = np.empty((len(self.raw_ids), 20))
        for q, context in contexts.items():
            if context is None:
                raise ValueError('OOF missing released prefix')
            take = self.profiles == q
            X20[take] = frozen.data.features(q, context['local'], frozen.data.states[self.raw_ids[take]])
        X43 = np.column_stack([X20, np.asarray([contexts[int(q)]['history'][13:] for q in self.profiles])])
        weight = balanced(self.regions)
        self.models, self.global_maps = {}, {}
        for j, reference in enumerate(REFERENCES):
            target = pit_targets(lower[:, j], upper[:, j])
            self.global_maps[reference] = check_maps(np.r_[0, np.average(target, axis=0, weights=weight), 1])
            for kind, X in [('conditional20', X20), ('conditional43', X43)]:
                model = ExtraTreesRegressor(n_estimators=32, min_samples_leaf=64, max_features=1.,
                                             random_state=20261028, n_jobs=2)
                model.fit(X, target, sample_weight=weight)
                model.n_jobs = 1
                self.models[(reference, kind)] = model

    def maps(self, q):
        if q not in self.frozen.allowed_queries or self.monthly.index.region.iloc[q] in self.frozen.fit:
            raise ValueError('query outside frozen evaluation roster')
        context = monthly_context(self.monthly, q)
        if context is None or context['local'] != self.frozen.data.local(q, 'asof_30d'):
            raise ValueError('query released prefix mismatch')
        X20 = self.frozen.data.features(q, context['local'], np.arange(48))
        X43 = np.column_stack([X20, np.tile(context['history'][13:], (48, 1))])
        result = {}
        for reference in REFERENCES:
            result[(reference, 'global')] = np.tile(self.global_maps[reference], (48, 1))
            for kind, X in [('conditional20', X20), ('conditional43', X43)]:
                interior = self.models[(reference, kind)].predict(X)
                result[(reference, kind)] = check_maps(np.column_stack([np.zeros(48), interior, np.ones(48)]))
        return result

    def build(self, q):
        laws, forecast = self.physical.build(q)
        laws = {name: cells + cells for name, cells in laws.items()}
        maps = self.maps(q)
        details = []
        for name, (reference, kind, alpha) in SETTINGS.items():
            laws[name] = [recalibrate(cell, mapping, alpha) for cell, mapping in zip(laws[reference], maps[(reference, kind)])]
            prior = laws[reference]
            details.append({'method': name, 'maps': json.dumps(maps[(reference, kind)].tolist()),
                            'state_mean_before': json.dumps([float(c.mean) for c in prior]),
                            'state_mean_after': json.dumps([float(c.mean) for c in laws[name]])})
        return laws, forecast, details
