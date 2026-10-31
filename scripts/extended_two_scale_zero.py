"""Ordinary long-history month-state and positive-month hourly-zero controls."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from monthly_history_features import monthly_context
from monthly_history_zero_risk import METHODS as DIRECT_METHODS, array_hash
from monthly_risk_features import feature_matrices, volatility_features
from two_scale_zero import (MONTH_METHODS as OLD_MONTH_METHODS, probability_forest,
                            checked_probability, hurdle_probability)
from v154_hourly_strong_baselines import balanced

MONTH_HEADS = {'long_month_base': 'base13', 'long_month_history': 'history36', 'long_month_volatility': 'history77'}
CONDITIONAL = ('clock', 'long_cond43', 'long_cond84')
HURDLES = {f'{month}_{component}': (month, component) for month in MONTH_HEADS for component in CONDITIONAL}
HURDLES |= {'long_persistence_cond43': ('month_persistence', 'long_cond43'),
            'long_persistence_cond84': ('month_persistence', 'long_cond84')}
METHODS = DIRECT_METHODS + tuple(HURDLES)
MONTH_METHODS = OLD_MONTH_METHODS + tuple(MONTH_HEADS)
COMPONENTS = ('clock', 'short_cond43', 'long_cond43', 'long_cond84')


class ExtendedTwoScaleZero:
    """Fit only proper monthly labels and the original positive-month source hours."""

    def __init__(self, frozen, short, long, monthly_ids: np.ndarray, direct_archive: Path, old_archive: Path):
        self.frozen, self.short, self.long = frozen, short, long
        self.month_ids = np.asarray(monthly_ids, int)
        self.month_regions = long.index.region.to_numpy()[self.month_ids]
        if (not len(self.month_ids) or len(np.unique(self.month_ids)) != len(self.month_ids)
                or set(self.month_regions) != set(frozen.fit)
                or not long.index.month.iloc[self.month_ids].str[:4].isin(['2022', '2023', '2024']).all()):
            raise ValueError('monthly long_all label role/year/inventory')
        self.month_contexts = {int(q): monthly_context(long, int(q)) for q in self.month_ids}
        if any(c is None for c in self.month_contexts.values()):
            raise ValueError('monthly supervision lacks released prefix')
        self.month_X = {'base13': np.asarray([self.month_contexts[int(q)]['base'] for q in self.month_ids]),
                        'history36': np.asarray([self.month_contexts[int(q)]['history'] for q in self.month_ids])}
        self.month_X['history77'] = np.column_stack([self.month_X['history36'],
            np.asarray([volatility_features(long, self.month_contexts[int(q)]['prefix']) for q in self.month_ids])])
        self.month_zero = long.M[self.month_ids] == 0
        parts = [frozen.activity.sources[r] for r in frozen.fit]
        self.raw_ids = np.concatenate([s['allowed'] for s in parts])
        self.profiles = np.concatenate([s['profiles'] for s in parts])
        self.regions = frozen.data.index.region.to_numpy()[self.profiles]
        self.contexts = {int(q): feature_matrices(short, long, int(q)) for q in np.unique(self.profiles)}
        states = frozen.data.states[self.raw_ids]
        self.X = {name: np.asarray([self.contexts[int(q)][0][feature][h] for q, h in zip(self.profiles, states)])
                  for name, feature in [('long_cond43', 'long43'), ('long_cond84', 'long84')]}
        self.zero = frozen.data.raw[self.raw_ids] == 0
        self.positive = long.M[self.profiles] > 0
        for name, feature in [('long_cond43', 'long43'), ('long_cond84', 'long84')]:
            with np.load(direct_archive / f'{feature}_training.npz', allow_pickle=False) as old:
                if (not np.array_equal(self.X[name], old['X']) or not np.array_equal(self.raw_ids, old['raw_id'])
                        or not np.array_equal(self.profiles, old['profile_id']) or not np.array_equal(self.zero, old['target'])):
                    raise ValueError('V186 sampled-hour/features contract changed')
        self.models = {name: [] for name in (*MONTH_HEADS, *self.X)}
        self.clock = np.empty((48, 8))
        self.provenance = []
        for fuel in range(8):
            take = self.positive[:, fuel]
            if not take.any():
                raise ValueError('no positive-month fine supervision')
            for name, X in self.X.items():
                self.models[name].append(probability_forest(X[take], self.zero[take, fuel].astype(float), self.regions[take], 64))
                self.provenance.append({'method': name, 'fuel_index': fuel, 'rows': int(take.sum()),
                    'dimensions': X.shape[1], 'regions': sorted(set(self.regions[take])),
                    'raw_index_sha256': array_hash(self.raw_ids[take]), 'feature_sha256': array_hash(X[take]),
                    'target_sha256': array_hash(self.zero[take, fuel])})
            ids = np.flatnonzero(take)
            weights = balanced(self.regions[take])
            for h in range(48):
                mask = states[ids] == h
                if not mask.any():
                    mask = np.ones(len(ids), bool)
                self.clock[h, fuel] = np.average(self.zero[ids[mask], fuel], weights=weights[mask])
            for name, feature in MONTH_HEADS.items():
                X = self.month_X[feature]
                self.models[name].append(probability_forest(X, self.month_zero[:, fuel].astype(float), self.month_regions, 4))
                self.provenance.append({'method': name, 'fuel_index': fuel, 'rows': len(self.month_ids),
                    'dimensions': X.shape[1], 'regions': frozen.fit,
                    'profile_index_sha256': array_hash(self.month_ids), 'feature_sha256': array_hash(X),
                    'target_sha256': array_hash(self.month_zero[:, fuel])})
        with np.load(direct_archive / 'probabilities.npz', allow_pickle=False) as old:
            self.old_ids, self.old_hourly = old['profile_id'].copy(), old['hourly'].copy()
        with np.load(old_archive / 'probabilities.npz', allow_pickle=False) as old:
            if not np.array_equal(old['profile_id'], self.old_ids):
                raise ValueError('old direct/month query IDs differ')
            self.old_monthly, self.old_components = old['monthly'].copy(), old['components'].copy()
        if (not np.array_equal(self.old_ids, frozen.queries.profile_id.to_numpy(int))
                or self.old_hourly.shape != (len(self.old_ids), 10, 48, 8)
                or self.old_monthly.shape != (len(self.old_ids), 5, 8)
                or self.old_components.shape != (len(self.old_ids), 2, 48, 8)
                or not np.array_equal(self.old_components[:, 0], np.broadcast_to(self.clock, (len(self.old_ids), 48, 8)))):
            raise ValueError('archived controls/positive clock do not reproduce')
        self.lookup = {int(q): j for j, q in enumerate(self.old_ids)}

    def predict(self, q: int):
        if q not in self.frozen.allowed_queries or self.long.index.region.iloc[q] in self.frozen.fit:
            raise ValueError('query outside frozen nontraining roster')
        features, contexts, extra = feature_matrices(self.short, self.long, q)
        context = contexts[1]
        X = {'base13': context['base'][None], 'history36': context['history'][None],
             'history77': np.r_[context['history'], extra][None]}
        j = self.lookup[q]
        hourly = {name: self.old_hourly[j, k].copy() for k, name in enumerate(DIRECT_METHODS)}
        monthly = {name: self.old_monthly[j, k].copy() for k, name in enumerate(OLD_MONTH_METHODS)}
        components = {'clock': self.clock.copy(), 'short_cond43': self.old_components[j, 1].copy()}
        for name, feature in [('long_cond43', 'long43'), ('long_cond84', 'long84')]:
            components[name] = checked_probability(np.column_stack([m.predict(features[feature]) for m in self.models[name]]))
        for name, feature in MONTH_HEADS.items():
            monthly[name] = checked_probability(np.asarray([m.predict(X[feature])[0] for m in self.models[name]]))
        persistence = (self.long.M[context['local']] == 0).astype(float)
        if not np.array_equal(persistence, monthly['month_persistence']):
            raise ValueError('unchanged monthly persistence differs')
        for method, (month, conditional) in HURDLES.items():
            hourly[method] = checked_probability(hurdle_probability(monthly[month], components[conditional]))
        return hourly, monthly, components, context

    def fingerprint(self) -> dict:
        fields = ('children_left', 'children_right', 'feature', 'threshold', 'value', 'weighted_n_node_samples')
        return {name: [[{key: array_hash(getattr(t.tree_, key)) for key in fields}
                        for t in model.estimators_] for model in models] for name, models in self.models.items()}


def check_inputs(predictor: ExtendedTwoScaleZero, q: int) -> None:
    before = predictor.predict(q)[:3]
    data = predictor.frozen.data
    saved = []
    for container in (data, predictor.short, predictor.long):
        ids = container.groups[container.index.region.iloc[q]]
        ids = ids[(container.release.iloc[ids] > container.dates.iloc[q]).to_numpy()]
        saved.append((container, ids, container.M[ids].copy(), container.shares[ids].copy()))
    raw = data.raw
    try:
        for container, ids, M, _ in saved:
            changed = np.random.default_rng(q + 187).uniform(1, 1000, size=M.shape)
            container.M[ids] = changed
            container.shares[ids] = changed / changed.sum(axis=1, keepdims=True)
        data.raw = None
        after = predictor.predict(q)[:3]
    finally:
        data.raw = raw
        for container, ids, M, S in saved:
            container.M[ids], container.shares[ids] = M, S
    if any(not np.array_equal(v, b[name]) for a, b in zip(before, after) for name, v in a.items()):
        raise ValueError('unreleased query data changed a hurdle forecast')
