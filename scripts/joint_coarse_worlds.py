"""Ordinary coarse-only forest kernels over the frozen sampled monthly worlds."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist
from sklearn.ensemble import ExtraTreesRegressor

from cif_pit_recalibration import PhysicalReferences, REFERENCES
from fuel_activity_forest import carrier
from monthly_history_features import monthly_context
from portfolio_activity import Geometry
from v151_hourly_law import EF
from v154_hourly_strong_baselines import balanced, CompressedLaw

RESPONSES = ('scalar8', 'delta8', 'logratio8', 'scalar_mmd80', 'joint_mmd80')
FORESTS = tuple(f'{feature}_{response}' for feature in ('base', 'history') for response in RESPONSES)
SETTINGS = {'allmonth_hellinger': ('world_gaussian', 'control', 'hellinger'),
            'allmonth_gaussian': ('world_gaussian', 'control', 'gaussian'),
            **{f'forest_{key}': ('world_gaussian', 'forest', key) for key in FORESTS}}
METHODS = REFERENCES + tuple(SETTINGS)
BACKOFF = .05


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def standardize(values, weights):
    mean = np.average(values, axis=0, weights=weights)
    scale = np.maximum(np.sqrt(np.average((values - mean)**2, axis=0, weights=weights)), 1e-3)
    return (values - mean) / scale, mean, scale


def rff(values):
    distances = pdist(values)
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.
    frequency = np.random.default_rng(20261029).normal(size=(values.shape[1], 40)) / bandwidth
    phase = values @ frequency
    result = np.column_stack([np.cos(phase), np.sin(phase)]) / np.sqrt(40)
    return result, bandwidth, frequency


def count_matches_calendar(index, q):
    start = pd.Timestamp(index.month.iloc[q])
    expected = int(((start + pd.offsets.MonthBegin(1)) - start).total_seconds() / 3600)
    return bool(index.hours.iloc[q] == expected)


class MonthlyWorldForests:
    """Accept monthly information only; no hourly observations enter fitting."""

    def __init__(self, monthly, profiles, proper, prior):
        self.monthly, self.proper = monthly, sorted(proper)
        self.ids = np.asarray(profiles, int)
        if len(np.unique(self.ids)) != len(self.ids) or not len(self.ids):
            raise ValueError('monthly world identifiers')
        self.regions = monthly.index.region.to_numpy()[self.ids]
        if not set(self.regions).issubset(proper) or not monthly.index.month.iloc[self.ids].str.startswith('2024').all():
            raise ValueError('coarse training role/year leakage')
        self.contexts = {int(q): monthly_context(monthly, int(q)) for q in self.ids}
        if any(c is None for c in self.contexts.values()):
            raise ValueError('source world without legal prefix')
        self.inputs = np.asarray([self.contexts[int(q)]['local'] for q in self.ids], int)
        self.X = {feature: np.asarray([self.contexts[int(q)][feature] for q in self.ids]) for feature in ('base', 'history')}
        if self.X['base'].shape[1] != 13 or self.X['history'].shape[1] != 36:
            raise ValueError('coarse feature dimensions')
        self.weight = balanced(self.regions)
        delta = monthly.shares[self.ids] - monthly.shares[self.inputs]
        tolerance = 64 * np.finfo(float).eps
        delta[abs(delta) <= tolerance] = 0.
        anchors = np.asarray([monthly.shares[q] @ EF for q in self.ids])
        past_anchors = np.asarray([self.contexts[int(q)]['anchor'] for q in self.ids])
        scalar = anchors - past_anchors
        scalar[abs(scalar) <= tolerance * np.maximum.reduce([np.ones(len(scalar)), abs(anchors), abs(past_anchors)])] = 0.
        scalar = scalar[:, None] / 1000
        source_B = np.asarray([carrier(S, 'asof_30d', prior) for S in monthly.shares[self.inputs]])
        logratio = np.log1p(monthly.shares[self.ids] / source_B)
        normalized, transforms = {}, {}
        for label, values in [('scalar', scalar), ('delta', delta), ('logratio', logratio)]:
            normalized[label], mean, scale = standardize(values, self.weight)
            transforms[label] = {'mean': mean.tolist(), 'scale': scale.tolist()}
        targets = {'scalar8': np.repeat(normalized['scalar'], 8, axis=1),
                   'delta8': normalized['delta'], 'logratio8': normalized['logratio']}
        for label, key in [('scalar', 'scalar_mmd80'), ('delta', 'joint_mmd80')]:
            targets[key], bandwidth, frequency = rff(normalized[label])
            transforms[key] = {'bandwidth': bandwidth, 'frequency': frequency.tolist()}
        self.targets, self.transforms = targets, transforms
        self.models, self.leaves, self.metadata = {}, {}, {}
        for feature in ('base', 'history'):
            for response in RESPONSES:
                key = f'{feature}_{response}'
                model = ExtraTreesRegressor(n_estimators=32, min_samples_leaf=4, max_features=1.,
                                             random_state=20261028, n_jobs=2)
                model.fit(self.X[feature], targets[response], sample_weight=self.weight)
                model.n_jobs = 1
                leaves = model.apply(self.X[feature])
                self.models[key] = model
                self.leaves[key] = [{int(node): np.flatnonzero(leaves[:, t] == node) for node in np.unique(leaves[:, t])}
                                    for t in range(32)]
                self.metadata[key] = {'feature_dimensions': self.X[feature].shape[1], 'response_dimensions': targets[response].shape[1],
                                      'training_profiles': self.ids.tolist(), 'proper_regions': self.proper,
                                      'feature_sha256': digest(self.X[feature]), 'target_sha256': digest(targets[response]),
                                      'transforms': transforms, 'sample_weight_sha256': digest(self.weight)}

    def probabilities(self, q):
        if self.monthly.index.region.iloc[q] in self.proper:
            raise ValueError('query overlaps coarse training region')
        context = monthly_context(self.monthly, q)
        if context is None:
            raise ValueError('query coarse prefix unavailable')
        probabilities = {}
        for key, model in self.models.items():
            feature = 'history' if key.startswith('history_') else 'base'
            signature = model.apply(context[feature][None])[0]
            p = np.zeros(len(self.ids))
            for t, node in enumerate(signature):
                take = self.leaves[key][t][int(node)]
                p[take] += self.weight[take] / self.weight[take].sum() / 32
            if not np.isfinite(p).all() or (p < 0).any() or abs(p.sum() - 1) > 1e-12:
                raise ValueError('invalid empirical leaf probabilities')
            probabilities[key] = p
        return probabilities, context

    def source_rows(self):
        records = []
        for q, p in zip(self.ids, self.inputs):
            c = self.contexts[int(q)]
            records.append({'profile_id': int(q), 'region': self.monthly.index.region.iloc[q],
                            'month': self.monthly.index.month.iloc[q], 'input_profile_id': int(p),
                            'prefix': json.dumps(c['prefix']), 'base': json.dumps(c['base'].tolist()),
                            'history': json.dumps(c['history'].tolist()),
                            'current_shares': json.dumps(self.monthly.shares[q].tolist()),
                            'input_shares': json.dumps(self.monthly.shares[p].tolist()),
                            'current_hour_count_matches_calendar': count_matches_calendar(self.monthly.index, int(q)),
                            'input_hour_count_matches_calendar': count_matches_calendar(self.monthly.index, int(p))})
        return records


def same_ef_projection(values):
    result = np.zeros_like(values)
    for level in np.unique(EF):
        ids = np.flatnonzero(EF == level)
        if len(ids) > 1:
            result[..., ids] = values[..., ids] - values[..., ids].mean(axis=-1, keepdims=True)
    return result


class JointCoarsePrediction:
    """Change source-world probabilities, retaining the observed hourly activity."""

    def __init__(self, frozen, monthly):
        self.frozen, self.monthly = frozen, monthly
        self.physical = PhysicalReferences(frozen, monthly)
        profiles = [int(q) for r in frozen.fit for q in frozen.bank[r]]
        self.month_models = MonthlyWorldForests(monthly, profiles, frozen.fit, frozen.prior)
        inputs = self.month_models.inputs
        self.source_B = np.asarray([carrier(S, 'asof_30d', frozen.prior) for S in monthly.shares[inputs]])
        # Preserve V177's arithmetic as well as its region/month-balanced law.
        prior = np.asarray([1 / len(frozen.fit) / len(frozen.bank[frozen.data.index.region.iloc[q]]) for q in profiles])
        prepared = [frozen.activity.donor(q) for q in profiles]
        cells = [(np.concatenate([a[h] for a in prepared]),
                  np.concatenate([np.full(len(a[h]), j, int) for j, a in enumerate(prepared)]),
                  np.concatenate([np.full(len(a[h]), 1 / len(a[h])) for a in prepared])) for h in range(24)]
        self.geometry = Geometry(profiles, monthly.shares[inputs], monthly.shares[profiles], self.source_B, prior, cells)

    def build(self, q):
        forecast, S, B, donors, prior, anchors = self.physical.inputs(q)
        logits = np.log(prior) - .5 * ((anchors - forecast['anchor']) / forecast['sigma'])**2
        gaussian = np.exp(np.maximum(logits - max(logits), -700))
        gaussian /= gaussian.sum()
        leaves, context = self.month_models.probabilities(q)
        if context['local'] != forecast['local']:
            raise ValueError('coarse input vintages disagree')
        laws, details = {}, []
        for method, weights in zip(REFERENCES, (prior, gaussian)):
            laws[method], rejected = self.frozen.activity.laws(donors, weights, B)
            if max(rejected) != 0:
                raise ValueError('reference physical domain changed')
            details.append(self.detail(method, donors, weights, B, []))
        pH = self.geometry.weights(S, B, forecast, False)
        pG = self.geometry.weights(S, B, forecast, True)
        for method, (_, kind, key) in SETTINGS.items():
            leaf = leaves[key] if kind == 'forest' else None
            weights = ((1 - BACKOFF) * leaf + BACKOFF * pH) if leaf is not None else (pG if key == 'gaussian' else pH)
            if (weights <= 0).any() or not np.isfinite(weights).all() or abs(weights.sum() - 1) > 1e-12:
                raise ValueError('background probability support not retained')
            laws[method], _ = self.geometry.predict(B, weights)
            details.append(self.detail(method, self.geometry.identifiers, weights, B, [] if leaf is None else leaf.tolist()))
        return laws, forecast, details

    def support(self, profiles, B):
        profiles = np.asarray(profiles, int)
        past = np.asarray([self.monthly.local(int(q), 'asof_30d') for q in profiles], int)
        denominator = np.asarray([carrier(S, 'asof_30d', self.frozen.prior) for S in self.monthly.shares[past]])
        values = B * self.monthly.shares[profiles] / denominator
        values /= values.sum(axis=1, keepdims=True)
        if not np.isfinite(values).all() or (values < 0).any() or max(abs(values.sum(axis=1) - 1)) > 1e-12:
            raise ValueError('invalid propagated coarse composition')
        return values

    def detail(self, method, profiles, weights, B, leaf):
        values = self.support(profiles, B)
        return {'method': method, 'source_profiles': json.dumps(list(map(int, profiles))),
                'weights': json.dumps(weights.tolist()), 'leaf_weights': json.dumps(leaf),
                'coarse_mean': json.dumps((weights @ values).tolist()),
                'coarse_anchor_mean': float(weights @ (values @ EF)),
                'effective_worlds': float(1 / (weights @ weights))}

    def score_coarse(self, q, details, forecast):
        """Evaluation only: caller must finish predictive construction first."""
        actual = self.monthly.shares[q]
        B = carrier(self.monthly.shares[forecast['local']], 'asof_30d', self.frozen.prior)
        distributions = [(d['method'], self.support(json.loads(d['source_profiles']), B), np.asarray(json.loads(d['weights'])))
                         for d in details]
        distributions += [('coarse_persistence', self.monthly.shares[forecast['local']][None], np.ones(1)),
                          ('coarse_point_mix', forecast['shares'][None], np.ones(1))]
        support_cache = {}
        rows, fuels = [], []
        projected_actual = same_ef_projection(actual)
        for method, values, weights in distributions:
            key = digest(values)
            if key not in support_cache:
                projected = same_ef_projection(values)
                support_cache[key] = (cdist(values, values), cdist(projected, projected), projected)
            distance, projected_distance, projected = support_cache[key]
            energy = float(weights @ np.linalg.norm(values - actual, axis=1) - .5 * weights @ distance @ weights)
            projected_energy = float(weights @ np.linalg.norm(projected - projected_actual, axis=1)
                                     - .5 * weights @ projected_distance @ weights)
            anchor = CompressedLaw(values @ EF, weights)
            truth_anchor = float(actual @ EF)
            crps = []
            for f in range(8):
                law = CompressedLaw(values[:, f], weights)
                zero = float(weights @ (values[:, f] == 0))
                score = float(law.crps(np.asarray([actual[f]]))[0])
                lower, upper = float(law.quantile(.05)), float(law.quantile(.95))
                crps.append(score)
                fuels.append({'method': method, 'fuel_index': f, 'crps': score,
                              'coverage': float(lower <= actual[f] <= upper), 'width': upper - lower,
                              'zero_brier': float((zero - (actual[f] == 0))**2),
                              'predicted_zero': zero, 'actual_zero': bool(actual[f] == 0)})
            lower, upper = float(anchor.quantile(.05)), float(anchor.quantile(.95))
            rows.append({'method': method, 'share_crps_mean': float(np.mean(crps)), 'energy_score': energy,
                         'same_EF_energy_score': projected_energy,
                         'anchor_crps': float(anchor.crps(np.asarray([truth_anchor]))[0]),
                         'anchor_coverage': float(lower <= truth_anchor <= upper), 'anchor_width': upper - lower,
                         'current_hour_count_matches_calendar': count_matches_calendar(self.monthly.index, q),
                         'input_hour_count_matches_calendar': count_matches_calendar(self.monthly.index, forecast['local'])})
        return rows, fuels
