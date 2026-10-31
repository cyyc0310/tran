"""Source-world conditional completion diagnostics with hidden-label protection."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from fuel_activity_forest import IndexedLeafMixture
from v151_hourly_law import EF
from v154_hourly_strong_baselines import CompressedLaw, balanced

METHODS = ('flat', 'clock_independent', 'coarse_joint', 'conditional_mean',
           'conditional_independent', 'conditional_joint', 'conditional_joint_log', 'conditional_joint_imputed')
PARTICLES = 1024


def masks_for(shares):
    emitting = np.flatnonzero((shares > 0) & (EF > 0))
    clean = np.flatnonzero((shares > 0) & (EF == 0))
    masks = []
    if len(emitting):
        masks.append((int(emitting[np.argmax(shares[emitting])]),))
    if len(clean):
        masks.append((int(clean[np.argmax(shares[clean])]),))
    if len(masks) == 2:
        masks.append(tuple(sorted(masks[0] + masks[1])))
    return masks


class ShapePanel:
    def __init__(self, data, sampling):
        self.data = data
        selected = sampling.loc[sampling.budget.eq('asof_30d')]
        self.raw_ids = np.sort(np.concatenate([np.asarray(json.loads(s), int)
                                               for s in selected.sampled_raw_hour_indices]))
        if len(np.unique(self.raw_ids)) != len(self.raw_ids):
            raise ValueError('source hour overlap')
        self.profile = np.searchsorted(data.index.stop.to_numpy(), self.raw_ids, side='right')
        self.regions = data.index.region.to_numpy()[self.profile]
        self.present = data.M[self.profile] > 0
        mean = data.M[self.profile] / data.index.hours.to_numpy()[self.profile, None]
        self.q = np.divide(data.raw[self.raw_ids], mean, out=np.ones_like(mean), where=mean > 0)
        self.base = np.empty((len(self.raw_ids), 20))
        for profile in np.unique(self.profile):
            ids = np.flatnonzero(self.profile == profile)
            self.base[ids] = data.features(int(profile), int(profile), data.states[self.raw_ids[ids]])
        self.hour = data.states[self.raw_ids] % 24
        if not data.index.month.iloc[self.profile].str.startswith('2024').all():
            raise ValueError('nondevelopment source year')

    def features(self, ids, mask):
        visible = self.present[ids].copy()
        visible[:, mask] = False
        q = np.where(visible, self.q[ids], 1.)
        return np.column_stack([self.base[ids], np.log1p(q), visible.astype(float)])

    def queries(self, q):
        ids = np.flatnonzero(self.profile == q)
        if not len(ids):
            raise ValueError('test month has no selected source hours')
        positions = np.rint(np.linspace(0, len(ids) - 1, min(64, len(ids)))).astype(int)
        return ids[positions]


def joint_particles(values, probability, count=PARTICLES):
    cumulative = np.cumsum(probability / probability.sum())
    positions = (np.arange(count) + .5) / count
    selected = np.minimum(np.searchsorted(cumulative, positions, side='left'), len(probability) - 1)
    return values[selected]


def independent_particles(joint, key):
    if joint.shape[1] == 1:
        return joint
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(4):
        parts.append(np.column_stack([joint[rng.permutation(len(joint)), j] for j in range(joint.shape[1])]))
    return np.concatenate(parts)


def proposals_for(distributions, key, count=PARTICLES):
    clock, probability = distributions['clock']
    positions = (np.arange(count) + .5) / count
    columns = []
    for j in range(clock.shape[1]):
        marginal = CompressedLaw(clock[:, j], probability)
        take = np.minimum(np.searchsorted(marginal.c, positions, side='left'), len(marginal.x) - 1)
        columns.append(marginal.x[take])
    clock_draws = np.column_stack(columns)
    values, probability = distributions['conditional']
    joint = joint_particles(values, probability, count)
    mean = np.average(values, axis=0, weights=probability)
    result = {'flat': np.ones((1, clock.shape[1])),
              'clock_independent': independent_particles(clock_draws, key + '/clock'),
              'conditional_mean': mean[None], 'conditional_joint': joint,
              'conditional_independent': independent_particles(joint, key + '/conditional')}
    for method, bank in [('coarse_joint', 'coarse'), ('conditional_joint_log', 'log'),
                         ('conditional_joint_imputed', 'imputed')]:
        values, probability = distributions[bank]
        result[method] = joint_particles(values, probability, count)
    if set(result) != set(METHODS):
        raise ValueError('completion methods missing')
    return result, float(np.max(abs(joint.mean(axis=0) - mean)))


class CompletionForest:
    def __init__(self, panel, proper, mask):
        self.panel, self.mask = panel, tuple(mask)
        self.ids = np.flatnonzero(np.isin(panel.regions, proper) & panel.present[:, mask].all(axis=1))
        if not len(self.ids):
            raise ValueError('no observable proper completion responses')
        self.weights = balanced(panel.regions[self.ids])
        self.labels = panel.q[self.ids][:, mask]
        self.fallback = len(self.ids) < 128 or len(np.unique(panel.regions[self.ids])) < 2
        self.models = {}
        self.label_banks = {}
        self.training_ids = {}
        if not self.fallback:
            full = panel.features(self.ids, mask)
            for name, X, y in [('coarse', panel.base[self.ids], self.labels),
                                ('conditional', full, self.labels),
                                ('log', full, np.log1p(self.labels))]:
                response = y[:, 0] if y.shape[1] == 1 else y
                self.models[name] = IndexedLeafMixture(X, response, self.weights)
                self.label_banks[name] = self.labels
                self.training_ids[name] = self.ids
            all_proper = np.flatnonzero(np.isin(panel.regions, proper))
            imputed = panel.q[all_proper][:, mask]
            response = imputed[:, 0] if imputed.shape[1] == 1 else imputed
            self.models['imputed'] = IndexedLeafMixture(panel.features(all_proper, mask), response,
                                                       balanced(panel.regions[all_proper]))
            self.label_banks['imputed'] = imputed
            self.training_ids['imputed'] = all_proper
        self.clock = {}
        for hour in range(24):
            take = np.flatnonzero(panel.hour[self.ids] == hour)
            if not len(take):
                take = np.arange(len(self.ids))
            p = self.weights[take]
            self.clock[hour] = (self.labels[take], p / p.sum())

    def distributions(self, ids):
        panel = self.panel
        full = panel.features(ids, self.mask)
        leaves = {name: model.forest.apply(panel.base[ids] if name == 'coarse' else full)
                  for name, model in self.models.items()}
        result = []
        for position, index in enumerate(ids):
            clock_values, clock_probability = self.clock[int(panel.hour[index])]
            row = {'clock': (clock_values, clock_probability)}
            for name in ['coarse', 'conditional', 'log', 'imputed']:
                if self.fallback:
                    row[name] = row['clock']
                else:
                    atom_ids, probability = self.models[name].atoms(leaves[name][position])
                    row[name] = (self.label_banks[name][atom_ids], probability)
            result.append(row)
        return result


def score_completion(shares, observed_q, mask, proposal, actual_q, actual_cif):
    mask = np.asarray(mask, int)
    weights = np.full(len(proposal), 1 / len(proposal))
    q_crps = float(np.mean([CompressedLaw(proposal[:, j], weights).crps(np.array([actual_q[k]]))[0]
                            for j, k in enumerate(mask)]))
    visible = observed_q.copy()
    visible[mask] = 0
    known_generation = float(shares @ visible)
    known_emissions = float((shares * EF) @ visible)
    total = known_generation + proposal @ shares[mask]
    emissions = known_emissions + proposal @ (shares[mask] * EF[mask])
    positive = total > 0
    rejected = float(np.mean(~positive))
    fallback = not positive.any()
    if fallback:
        total = np.array([known_generation + shares[mask].sum()])
        emissions = np.array([known_emissions + (shares[mask] * EF[mask]).sum()])
        positive = total > 0
    active = shares > 0
    lower_bound, upper_bound = float(EF[active].min()), float(EF[active].max())
    cif = np.clip(emissions[positive] / total[positive], lower_bound, upper_bound)
    law = CompressedLaw(cif, np.ones(len(cif)))
    lo, hi = law.quantile(.05), law.quantile(.95)
    tolerance = 64 * np.finfo(float).eps * max(1., abs(actual_cif), abs(lo), abs(hi))
    return {'q_crps': q_crps, 'crps': float(law.crps(np.array([actual_cif]))[0]),
            'mae': abs(law.mean - actual_cif), 'coverage': float(lo <= actual_cif <= hi),
            'tolerant_coverage': float(lo - tolerance <= actual_cif <= hi + tolerance),
            'width': hi - lo, 'winkler': hi - lo + 20 * max(lo - actual_cif, 0) + 20 * max(actual_cif - hi, 0),
            'rejected_mass': rejected, 'flat_fallback': float(fallback)}
