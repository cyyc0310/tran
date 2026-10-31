"""Joint nonnegative fuel activity laws, with explicit CIF domain conditioning."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from v151_hourly_law import EF
from v154_hourly_strong_baselines import CompressedLaw


def carrier(shares: np.ndarray, budget: str, prior: np.ndarray) -> np.ndarray:
    if budget == 'same_month':
        return shares
    if budget != 'asof_30d':
        raise ValueError('unknown budget')
    return .99 * shares + .01 * prior


def response_values(base: np.ndarray, activity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(activity).all() or (activity < 0).any():
        raise ValueError('invalid activity')
    fuel = activity * base
    total = fuel.sum(axis=1)
    positive = total > 0
    values = np.divide(fuel @ EF, total, out=np.zeros(len(total)), where=positive)
    return values, positive


def activity_law(base: np.ndarray, activity: np.ndarray, weights: np.ndarray):
    values, positive = response_values(base, activity)
    kept = weights[positive].sum()
    if kept <= 0:
        raise ValueError('no positive-generation probability support')
    rejected = float(weights[~positive].sum() / weights.sum())
    return CompressedLaw(values[positive], weights[positive]), rejected


def source_activity(data, region: str, budget: str, allowed: np.ndarray, prior: np.ndarray) -> dict:
    X, y, _, inputs = data.source(region, budget, allowed)
    profiles = np.searchsorted(data.index.stop.to_numpy(), allowed, side='right')
    g = data.raw[allowed]
    mean = data.M[profiles] / data.index.hours.to_numpy()[profiles, None]
    shape = np.divide(g, mean, out=np.ones_like(g), where=mean > 0)
    if budget == 'same_month':
        u = shape
    else:
        local_by_profile = {int(q): data.local(int(q), budget) for q in np.unique(profiles)}
        local = np.asarray([local_by_profile[int(q)] for q in profiles], int)
        if np.any(data.release.iloc[local].to_numpy() > data.dates.iloc[profiles].to_numpy()):
            raise ValueError('activity input timing leakage')
        base = carrier(data.shares[local], budget, prior)
        mean_total = data.M[local].sum(axis=1) / data.index.hours.to_numpy()[profiles]
        u = g / mean_total[:, None] / base
    if not np.isfinite(u).all() or (u < 0).any() or (u.sum(axis=1) <= 0).any():
        raise ValueError('invalid source activity')
    return {'X': X, 'y': y, 'u': u, 'shape': shape, 'inputs': inputs,
            'allowed': allowed, 'profiles': profiles}


class IndexedLeafMixture:
    """Exact sample-index aggregation of a forest's empirical leaf mixture."""

    def __init__(self, X: np.ndarray, labels: np.ndarray, weights: np.ndarray):
        self.forest = ExtraTreesRegressor(
            n_estimators=32, min_samples_leaf=64, max_features=1.,
            random_state=20261028, n_jobs=2)
        self.forest.fit(X, labels, sample_weight=weights)
        leaves = self.forest.apply(X)
        self.maps = []
        self.weights = weights
        self.cache = {}
        for tree in range(32):
            order = np.argsort(leaves[:, tree], kind='stable')
            ids = leaves[order, tree]
            cuts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1, len(ids)]
            self.maps.append({int(ids[a]): order[a:b] for a, b in zip(cuts[:-1], cuts[1:])})

    def atoms(self, signature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = tuple(signature)
        if key not in self.cache:
            parts = [self.maps[t][int(leaf)] for t, leaf in enumerate(signature)]
            weights = np.concatenate([
                self.weights[ids] / self.weights[ids].sum() / 32 for ids in parts])
            ids, inverse = np.unique(np.concatenate(parts), return_inverse=True)
            probability = np.bincount(inverse, weights=weights, minlength=len(ids))
            if len(self.cache) >= 128:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = ids, probability
        return self.cache[key]


class ActivityDonors:
    """Only selected proper-source atoms can enter calendar donor laws."""

    def __init__(self, data, sources: dict, coordinate: str = 'u'):
        self.data = data
        self.sources = sources
        self.coordinate = coordinate
        self.cache = {}
        self.empty_calendar_cells = 0

    def donor(self, profile: int):
        if profile not in self.cache:
            region = self.data.index.region.iloc[profile]
            source = self.sources[region]
            take = np.flatnonzero(source['profiles'] == profile)
            if not len(take):
                raise ValueError('donor profile outside selected source sample')
            states = self.data.states[source['allowed'][take]] % 24
            cells = []
            for state in range(24):
                ids = take[states == state]
                if not len(ids):
                    ids = take
                    self.empty_calendar_cells += 1
                cells.append(source[self.coordinate][ids])
            self.cache[profile] = cells
        return self.cache[profile]

    def laws(self, profiles: np.ndarray, weights: np.ndarray, base: np.ndarray):
        donors = [self.donor(int(q)) for q in profiles]
        laws, rejected = [], []
        for state in range(24):
            activity = np.concatenate([d[state] for d in donors])
            probability = np.concatenate([
                np.full(len(d[state]), weight / len(d[state])) for d, weight in zip(donors, weights)])
            law, mass = activity_law(base, activity, probability)
            laws.append(law)
            rejected.append(mass)
        return laws, rejected
