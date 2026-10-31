"""Ordinary direct and two-level zero-probability controls with legal prefixes."""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from monthly_history_features import monthly_context
from v154_hourly_strong_baselines import balanced

REFERENCES = ('conditional_activity_recomposition', 'world_gaussian')
NEW_METHODS = ('direct20', 'direct43', 'month_history_clock', 'month_history_conditional',
               'month_persistence_conditional', 'month_base_conditional')
METHODS = REFERENCES + NEW_METHODS
MONTH_METHODS = REFERENCES + ('month_persistence', 'month_base', 'month_history')


def checked_probability(values):
    values = np.asarray(values, float)
    tolerance = 64 * np.finfo(float).eps
    if not np.isfinite(values).all() or (values < -tolerance).any() or (values > 1 + tolerance).any():
        raise ValueError('invalid zero-event probability')
    return np.clip(values, 0, 1)


def hurdle_probability(monthly, conditional):
    return monthly[None] + (1 - monthly[None]) * conditional


def probability_forest(X, target, regions, leaf):
    model = ExtraTreesRegressor(n_estimators=32, min_samples_leaf=leaf, max_features=1.,
                                random_state=20261028, n_jobs=2)
    model.fit(X, target, sample_weight=balanced(np.asarray(regions)))
    model.n_jobs = 1
    return model


class TwoScaleZero:
    """A source-only hurdle baseline, without target hourly observations."""

    def __init__(self, frozen, monthly):
        self.frozen, self.monthly = frozen, monthly
        self.provenance = []
        self.contexts = {}
        for q, row in monthly.index.iterrows():
            if row.region in frozen.fit and row.month.startswith('2024'):
                context = monthly_context(monthly, int(q))
                if context is not None:
                    self.contexts[int(q)] = context
        self.month_ids = np.asarray(list(self.contexts), int)
        self.month_regions = monthly.index.region.to_numpy()[self.month_ids]
        self.month_base = np.asarray([self.contexts[q]['base'] for q in self.month_ids])
        self.month_history = np.asarray([self.contexts[q]['history'] for q in self.month_ids])
        parts = [frozen.activity.sources[r] for r in frozen.fit]
        self.raw_ids = np.concatenate([s['allowed'] for s in parts])
        self.profiles = np.concatenate([s['profiles'] for s in parts])
        self.regions = frozen.data.index.region.to_numpy()[self.profiles]
        self.X20 = np.concatenate([s['X'] for s in parts])
        self.X43 = np.column_stack([self.X20, np.asarray([self.contexts[int(q)]['history'][13:] for q in self.profiles])])
        self.zero = frozen.data.raw[self.raw_ids] == 0
        self.positive_month = monthly.M[self.profiles] > 0
        if (self.X20.shape[1] != 20 or self.X43.shape[1] != 43
                or self.month_base.shape[1] != 13 or self.month_history.shape[1] != 36
                or not set(self.regions).issubset(frozen.fit)):
            raise ValueError('proper role or feature dimension')
        self.models = {name: [] for name in ('direct20', 'direct43', 'conditional43', 'month_base', 'month_history')}
        self.clock = np.empty((48, 8))
        self.empty_clock_cells = 0
        states = frozen.data.states[self.raw_ids]
        for j in range(8):
            for name, X in [('direct20', self.X20), ('direct43', self.X43)]:
                self.models[name].append(probability_forest(X, self.zero[:, j].astype(float), self.regions, 64))
                self.record(name, j, np.ones(len(self.raw_ids), bool), X.shape[1], False)
            take = self.positive_month[:, j]
            if not take.any():
                raise ValueError('no proper positive-month fuel supervision')
            self.models['conditional43'].append(probability_forest(
                self.X43[take], self.zero[take, j].astype(float), self.regions[take], 64))
            self.record('conditional43', j, take, 43, False)
            ids = np.flatnonzero(take)
            weights = balanced(self.regions[take])
            for h in range(48):
                mask = states[ids] == h
                if not mask.any():
                    mask = np.ones(len(ids), bool)
                    self.empty_clock_cells += 1
                self.clock[h, j] = np.average(self.zero[ids[mask], j], weights=weights[mask])
            target = (monthly.M[self.month_ids, j] == 0).astype(float)
            for name, X in [('month_base', self.month_base), ('month_history', self.month_history)]:
                self.models[name].append(probability_forest(X, target, self.month_regions, 4))
                self.record(name, j, np.ones(len(self.month_ids), bool), X.shape[1], True)

    def record(self, name, fuel, take, dimensions, monthly):
        ids = self.month_ids[take] if monthly else self.raw_ids[take]
        profiles = ids if monthly else self.profiles[take]
        regions = self.month_regions[take] if monthly else self.regions[take]
        self.provenance.append({'model': name, 'fuel_index': fuel, 'rows': len(ids), 'dimensions': dimensions,
                                'regions': sorted(set(regions)), 'profiles': np.unique(profiles).tolist(),
                                'training_index_sha256': hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()})

    def inputs(self, q):
        if q not in self.frozen.allowed_queries or self.monthly.index.region.iloc[q] in self.frozen.fit:
            raise ValueError('query outside frozen evaluation roster')
        context = monthly_context(self.monthly, q)
        if context is None:
            raise ValueError('no legal query prefix')
        local = self.frozen.data.local(q, 'asof_30d')
        if local != context['local']:
            raise ValueError('monthly inputs disagree')
        X20 = self.frozen.data.features(q, local, np.arange(48))
        X43 = np.column_stack([X20, np.tile(context['history'][13:], (48, 1))])
        return context, X20, X43

    def predict(self, q):
        context, X20, X43 = self.inputs(q)
        direct20 = np.column_stack([m.predict(X20) for m in self.models['direct20']])
        direct43 = np.column_stack([m.predict(X43) for m in self.models['direct43']])
        conditional = np.column_stack([m.predict(X43) for m in self.models['conditional43']])
        monthly = {'month_persistence': (self.monthly.M[context['local']] == 0).astype(float),
                   'month_base': np.asarray([m.predict(context['base'][None])[0] for m in self.models['month_base']]),
                   'month_history': np.asarray([m.predict(context['history'][None])[0] for m in self.models['month_history']])}
        hourly = {'direct20': direct20, 'direct43': direct43,
                  'month_history_clock': hurdle_probability(monthly['month_history'], self.clock),
                  'month_history_conditional': hurdle_probability(monthly['month_history'], conditional),
                  'month_persistence_conditional': hurdle_probability(monthly['month_persistence'], conditional),
                  'month_base_conditional': hurdle_probability(monthly['month_base'], conditional)}
        components = {'clock': self.clock.copy(), 'conditional43': conditional}
        hourly = {k: checked_probability(v) for k, v in hourly.items()}
        monthly = {k: checked_probability(v) for k, v in monthly.items()}
        components = {k: checked_probability(v) for k, v in components.items()}
        return hourly, monthly, components, context


def reference_probabilities(frozen, worlds, q):
    selected = worlds.loc[worlds.profile_id.eq(q)]
    donors = selected.source_profile_id.to_numpy(int)
    if len(donors) != 2 * len(frozen.fit) or not set(frozen.data.index.region.iloc[donors]).issubset(frozen.fit):
        raise ValueError('reference donor inventory')
    arrays = [frozen.activity.donor(int(d)) for d in donors]
    hourly, monthly = {}, {}
    for method, column in zip(REFERENCES, ('conditional_weight', 'gaussian_weight')):
        weights = selected[column].to_numpy(float)
        if (weights <= 0).any() or abs(weights.sum() - 1) > 1e-12:
            raise ValueError('reference world probabilities')
        cells = np.asarray([sum(w * (a[h] == 0).mean(axis=0) for w, a in zip(weights, arrays)) for h in range(24)])
        hourly[method] = checked_probability(np.tile(cells, (2, 1)))
        monthly[method] = checked_probability(weights @ (frozen.data.M[donors] == 0))
    return hourly, monthly
