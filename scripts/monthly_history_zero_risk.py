"""Same sampled-hour zero-event forests with legal short/extended monthly inputs."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from monthly_risk_features import NAMES, feature_matrices
from two_scale_zero import METHODS as OLD_METHODS, checked_probability, probability_forest

OLD_ALIASES = {name: ('short43' if name == 'direct43' else name) for name in OLD_METHODS}
METHODS = tuple(OLD_ALIASES.values()) + ('long43', 'long84')


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class MonthlyHistoryZeroRisk:
    """Four ordinary per-fuel forests; other V171 controls remain archived controls."""

    def __init__(self, frozen, short, long, old_archive: Path):
        self.frozen, self.short, self.long = frozen, short, long
        parts = [frozen.activity.sources[r] for r in frozen.fit]
        self.raw_ids = np.concatenate([s['allowed'] for s in parts])
        self.profiles = np.concatenate([s['profiles'] for s in parts])
        self.regions = frozen.data.index.region.to_numpy()[self.profiles]
        if not set(self.regions).issubset(frozen.fit):
            raise ValueError('source region leakage')
        self.contexts = {int(q): feature_matrices(short, long, int(q)) for q in np.unique(self.profiles)}
        states = frozen.data.states[self.raw_ids]
        self.X = {name: np.asarray([self.contexts[int(q)][0][name][state]
                                   for q, state in zip(self.profiles, states)]) for name in NAMES}
        if not np.array_equal(self.X['direct20'], np.concatenate([s['X'] for s in parts])):
            raise ValueError('original sampled-hour base features changed')
        self.zero = frozen.data.raw[self.raw_ids] == 0
        self.models = {name: [probability_forest(X, self.zero[:, f].astype(float), self.regions, 64)
                              for f in range(8)] for name, X in self.X.items()}
        with np.load(old_archive / 'probabilities.npz', allow_pickle=False) as saved:
            self.old_ids = saved['profile_id'].copy()
            self.old = saved['hourly'].copy()
        if (not np.array_equal(self.old_ids, frozen.queries.profile_id.to_numpy(int))
                or self.old.shape != (len(self.old_ids), 8, 48, 8)):
            raise ValueError('V171 probability inventory differs')
        self.lookup = {int(q): j for j, q in enumerate(self.old_ids)}
        self.provenance = []
        for name in NAMES:
            for fuel in range(8):
                self.provenance.append({'method': name, 'fuel_index': fuel, 'rows': len(self.raw_ids),
                                        'dimensions': self.X[name].shape[1], 'regions': frozen.fit,
                                        'profiles': np.unique(self.profiles).tolist(),
                                        'raw_sha256': array_hash(self.raw_ids), 'feature_sha256': array_hash(self.X[name]),
                                        'target_sha256': array_hash(self.zero[:, fuel])})

    def predict(self, q: int) -> dict[str, np.ndarray]:
        if q not in self.frozen.allowed_queries or self.short.index.region.iloc[q] in self.frozen.fit:
            raise ValueError('query outside nontraining roster')
        features, _, _ = feature_matrices(self.short, self.long, q)
        hourly = {OLD_ALIASES[name]: self.old[self.lookup[q], j].copy() for j, name in enumerate(OLD_METHODS)}
        for name in NAMES:
            probability = checked_probability(np.column_stack([model.predict(features[name]) for model in self.models[name]]))
            if name in ('direct20', 'short43') and not np.array_equal(probability, hourly[name]):
                raise ValueError('V171 direct probability not reproduced exactly')
            hourly[name] = probability
        return hourly

    def fingerprint(self) -> dict:
        fields = ('children_left', 'children_right', 'feature', 'threshold', 'value', 'weighted_n_node_samples')
        return {name: [[{key: array_hash(getattr(tree.tree_, key)) for key in fields}
                        for tree in model.estimators_] for model in models] for name, models in self.models.items()}


def check_inputs(predictor: MonthlyHistoryZeroRisk, q: int) -> None:
    """Check completed predictions against all unreleased query M and raw absence."""
    before = predictor.predict(q)
    data = predictor.frozen.data
    saved = []
    for container in (data, predictor.short, predictor.long):
        ids = container.groups[container.index.region.iloc[q]]
        ids = ids[(container.release.iloc[ids] > container.dates.iloc[q]).to_numpy()]
        saved.append((container, ids, container.M[ids].copy(), container.shares[ids].copy()))
    raw = data.raw
    try:
        for container, ids, M, _ in saved:
            changed = np.random.default_rng(q + 186).uniform(1, 1000, size=M.shape)
            container.M[ids] = changed
            container.shares[ids] = changed / changed.sum(axis=1, keepdims=True)
        data.raw = None
        after = predictor.predict(q)
    finally:
        data.raw = raw
        for container, ids, M, S in saved:
            container.M[ids], container.shares[ids] = M, S
    if any(not np.array_equal(before[name], after[name]) for name in METHODS):
        raise ValueError('unreleased monthly or query raw access affected predictions')
