"""Ordinary additive coarse scenarios recomposed with sampled within-fuel shapes."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cif_pit_recalibration import PhysicalReferences, REFERENCES
from v151_hourly_law import EF
from v154_hourly_strong_baselines import CompressedLaw
from v179_increment_support_diagnostic import project_simplex

SETTINGS = {
    'rebased_persistence': ('past', 'allmonth_hellinger'),
    'rebased_point_mix': ('point', 'allmonth_hellinger'),
    'rebased_increment_hellinger': ('increment', 'allmonth_hellinger'),
    'rebased_increment_gaussian_frozen': ('increment', 'allmonth_gaussian'),
    'rebased_increment_gaussian': ('increment', 'recomputed_gaussian'),
    **{f'rebased_increment_history_{key}': ('increment', f'forest_history_{key}')
       for key in ('scalar8', 'delta8', 'logratio8', 'joint_mmd80')},
}
METHODS = REFERENCES + tuple(SETTINGS)


class RebasedCoarseLift:
    def __init__(self, frozen, monthly, archive):
        self.frozen, self.monthly = frozen, monthly
        self.physical = PhysicalReferences(frozen, monthly)
        self.records = pd.read_csv(archive / 'world_probabilities.csv')
        if self.records.duplicated(['profile_id', 'method']).any():
            raise ValueError('duplicate frozen world probabilities')
        self.lookup = self.records.set_index(['profile_id', 'method'])
        ids = sorted({int(q) for r in frozen.fit for q in frozen.bank[r]})
        self.profiles = np.asarray(ids, int)
        self.inputs = np.asarray([monthly.local(q, 'asof_30d') for q in ids], int)
        self.delta = monthly.shares[self.profiles] - monthly.shares[self.inputs]
        self.delta[abs(self.delta) <= 64 * np.finfo(float).eps] = 0
        self.cells, self.sample_checks = [], 0
        data = frozen.data
        parts = [[] for _ in range(24)]
        for j, q in enumerate(ids):
            region = data.index.region.iloc[q]
            source = frozen.activity.sources[region]
            mask = source['profiles'] == q
            raw_ids = source['allowed'][mask]
            if region not in frozen.fit or not data.index.month.iloc[q].startswith('2024') or not len(raw_ids):
                raise ValueError('invalid source shape role')
            expected = data.raw[raw_ids]
            mean = monthly.M[q] / data.index.hours.iloc[q]
            shape = np.divide(expected, mean, out=np.ones_like(expected), where=mean > 0)
            if not np.array_equal(shape, source['shape'][mask]):
                raise ValueError('source shapes differ from original sampled bank')
            self.sample_checks += len(raw_ids)
            for h in range(24):
                take = np.flatnonzero(data.states[raw_ids] % 24 == h)
                if not len(take):
                    take = np.arange(len(raw_ids))
                parts[h].append((shape[take], np.full(len(take), j, int), np.full(len(take), 1 / len(take)), raw_ids[take]))
        for part in parts:
            shape, component, within, raw_ids = [np.concatenate([p[k] for p in part]) for k in range(4)]
            if (not np.isfinite(shape).all() or (shape < 0).any()
                    or max(abs(np.bincount(component, weights=within, minlength=len(ids)) - 1)) > 1e-12):
                raise ValueError('invalid sampled shape probabilities')
            self.cells.append((shape, component, within, raw_ids))

    def weights(self, q, key):
        record = self.lookup.loc[(q, key)]
        if json.loads(record.source_profiles) != self.profiles.tolist():
            raise ValueError('frozen coarse world bank differs')
        w = np.asarray(json.loads(record.weights))
        if w.shape != (len(self.profiles),) or not np.isfinite(w).all() or (w <= 0).any() or abs(w.sum() - 1) > 1e-12:
            raise ValueError('invalid frozen world probabilities')
        return w

    def compose(self, values, weights):
        laws, rejected = [], []
        for shape, component, within, _ in self.cells:
            fuel = values[component] * shape
            total = fuel.sum(axis=1)
            probability = weights[component] * within
            positive = total > 0
            rejected.append(float(probability[~positive].sum()))
            if not positive.any() or probability[positive].sum() <= 0:
                raise ValueError('no positive-generation support')
            y = fuel[positive] @ EF / total[positive]
            if not np.isfinite(y).all() or min(y) < -1e-10 or max(y) > 1000 + 1e-10:
                raise ValueError('invalid physical CIF ratio')
            laws.append(CompressedLaw(y, probability[positive]))
        return laws, rejected

    def build(self, q):
        if q not in self.frozen.allowed_queries:
            raise ValueError('query not in frozen roster')
        laws, forecast = self.physical.build(q)
        local = self.monthly.local(q, 'asof_30d')
        if local != forecast['local'] or self.monthly.index.region.iloc[q] in self.frozen.fit:
            raise ValueError('invalid query origin/role')
        S = self.monthly.shares[local]
        increment = project_simplex(S + self.delta)
        past = np.broadcast_to(S, increment.shape)
        point = np.broadcast_to(forecast['shares'], increment.shape)
        supports = {'increment': increment, 'past': past, 'point': point}
        pH = self.weights(q, 'allmonth_hellinger')
        logits = np.log(pH) - .5 * ((increment @ EF - forecast['anchor']) / forecast['sigma'])**2
        pG = np.exp(np.maximum(logits - logits.max(), -700))
        pG /= pG.sum()
        details = []
        for name, (kind, key) in SETTINGS.items():
            w = pG if key == 'recomputed_gaussian' else self.weights(q, key)
            values = supports[kind]
            laws[name], rejected = self.compose(values, w)
            missing = (self.monthly.shares[self.profiles] == 0) * values
            details.append({'method': name, 'source_profiles': json.dumps(self.profiles.tolist()),
                            'weights': json.dumps(w.tolist()), 'coarse_support': json.dumps(values.tolist()),
                            'rejected_mass': json.dumps(rejected), 'max_rejected_mass': max(rejected),
                            'unknown_shape_energy_mass': float(w @ missing.sum(axis=1))})
        return laws, forecast, details


def check_query_inputs(predictor, q):
    """The completed predictor cannot use query raw or unreleased query M."""
    data, monthly = predictor.frozen.data, predictor.monthly
    before, _, _ = predictor.build(q)
    ids = monthly.groups[monthly.index.region.iloc[q]]
    ids = ids[(monthly.release.iloc[ids] > monthly.dates.iloc[q]).to_numpy()]
    original = monthly.M[ids].copy(), monthly.shares[ids].copy()
    changed = np.random.default_rng(q + 180).uniform(1, 1000, size=original[0].shape)
    raw = data.raw
    try:
        monthly.M[ids] = changed
        monthly.shares[ids] = changed / changed.sum(axis=1, keepdims=True)
        data.raw = None
        after, _, _ = predictor.build(q)
    finally:
        data.raw = raw
        monthly.M[ids], monthly.shares[ids] = original
    for method in METHODS:
        for a, b in zip(before[method], after[method]):
            if not np.array_equal(a.x, b.x) or not np.array_equal(a.w, b.w):
                raise ValueError('query raw/unreleased M changed prediction')
