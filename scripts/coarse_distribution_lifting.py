"""Lift legal coarse forecast distributions into existing hourly scenario laws."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from monthly_history_features import monthly_context
from scenario_reweighting import CoarseForecast, ScenarioPrediction, constant_anchor, fit_forest
from v151_hourly_law import EF
from v154_hourly_strong_baselines import CompressedLaw, balanced

METHODS = ('positive_recomposition', 'conditional_activity_recomposition', 'world_gaussian',
           'global_residual_nearest', 'global_residual_barycentric', 'local_residual_barycentric',
           'qrf_barycentric', 'hetero_history_barycentric', 'local_gaussian')


def lift_probabilities(anchors, prior, values, probabilities, nearest=False):
    """Project a one-dimensional coarse law; keep prior conditionals within ties."""
    order = np.argsort(anchors, kind='stable')
    groups = np.cumsum(np.r_[True, np.diff(anchors[order]) > 1e-10]) - 1
    inverse = np.empty(len(anchors), int)
    inverse[order] = groups
    group_prior = np.bincount(inverse, weights=prior)
    unique = np.bincount(inverse, weights=prior * anchors) / group_prior
    mass = np.zeros(len(unique))
    clipped = np.clip(values, unique[0], unique[-1])
    if len(unique) == 1:
        mass[0] = 1.
    else:
        right = np.searchsorted(unique, clipped, side='right')
        right = np.clip(right, 1, len(unique) - 1)
        left = right - 1
        fraction = (clipped - unique[left]) / (unique[right] - unique[left])
        if nearest:
            fraction = (fraction > .5).astype(float)
        mass += np.bincount(left, weights=probabilities * (1 - fraction), minlength=len(unique))
        mass += np.bincount(right, weights=probabilities * fraction, minlength=len(unique))
    posterior = prior * mass[inverse] / group_prior[inverse]
    posterior = (1 - 1e-12) * posterior + 1e-12 * prior
    posterior /= posterior.sum()
    return posterior, {'coarse_mean': float(probabilities @ values),
                       'clipped_coarse_mean': float(probabilities @ clipped),
                       'achieved': float(posterior @ anchors),
                       'outside_hull_mass': float(probabilities @ ((values < unique[0]) | (values > unique[-1]))),
                       'effective_scenarios': float(1 / (posterior @ posterior)),
                       'anchor_min': float(unique[0]), 'anchor_max': float(unique[-1])}


class CoarseLawForecast(CoarseForecast):
    """OOF residual, local-backtest, QRF, and heteroscedastic coarse controls."""

    def __init__(self, data, proper_regions):
        super().__init__(data, proper_regions)
        # Fixed reduction order keeps future-input mutation checks bit-exact.
        self.anchor_model.n_jobs = 1
        self.mix_model.n_jobs = 1
        regions = data.index.region.to_numpy()[self.ids]
        X = np.asarray([self.contexts[q]['base'] for q in self.ids])
        anchors = np.asarray([self.contexts[q]['anchor'] for q in self.ids])
        truth = np.asarray([data.shares[q] @ EF for q in self.ids])
        target = truth - anchors
        ordered = sorted(proper_regions, key=lambda r: hashlib.sha256(('v164/coarse/' + r).encode()).hexdigest())
        group = {r: j % 5 for j, r in enumerate(ordered)}
        oof = np.empty(len(self.ids))
        # OOF errors train another forest, so their reduction order matters too.
        for fold in range(5):
            test = np.asarray([group[r] == fold for r in regions])
            model = fit_forest(X[~test], target[~test], regions[~test])
            model.n_jobs = 1
            value = np.clip(anchors[test] + model.predict(X[test]), 0, 1000)
            for j, q in enumerate(self.ids[test]):
                if constant_anchor(data.shares[self.contexts[q]['local']]):
                    value[j] = anchors[test][j]
            oof[test] = value - truth[test]
        self.oof = [{'profile_id': int(q), 'region': str(r), 'fold': group[r], 'error': float(e)}
                    for q, r, e in zip(self.ids, regions, oof)]
        self.sigma = max(1., float(np.sqrt(np.average(oof**2, weights=balanced(regions)))))
        self.weights = balanced(regions)
        self.weights /= self.weights.sum()
        self.residual = -np.asarray([item['error'] for item in self.oof])
        history = np.asarray([self.contexts[q]['history'] for q in self.ids])
        self.risk_model = fit_forest(history, self.residual**2, regions)
        self.risk_model.n_jobs = 1
        leaves = self.anchor_model.apply(X)
        self.leaf_ids = [{int(leaf): np.flatnonzero(leaves[:, tree] == leaf)
                          for leaf in np.unique(leaves[:, tree])} for tree in range(32)]
        self.increments = np.asarray([data.shares[q] @ EF - self.contexts[q]['anchor'] for q in self.ids])
        self.local_cache = {}

    def local_residuals(self, q):
        if q not in self.local_cache:
            context = monthly_context(self.data, q)
            residuals, identities = [], []
            for p in context['prefix']:
                old_context = monthly_context(self.data, p)
                if old_context is None:
                    continue
                if self.data.release.iloc[p] > self.data.dates.iloc[q]:
                    raise ValueError('unreleased local backtest outcome')
                if any(self.data.release.iloc[i] > self.data.dates.iloc[p] for i in old_context['prefix']):
                    raise ValueError('historical backtest feature leakage')
                value = float(self.data.shares[p] @ EF - self.predict(p)['anchor'])
                residuals.append(value)
                identities.append({'observed_profile_id': int(p), 'prediction_input_profile_id': old_context['local'],
                                   'feature_prefix': old_context['prefix'], 'residual': value})
            self.local_cache[q] = np.asarray(residuals), identities
        return self.local_cache[q]

    def laws(self, q):
        forecast = self.predict(q)
        anchor = forecast['anchor']
        residuals, identities = self.local_residuals(q)
        n = len(residuals)
        fraction = n / (n + 5)
        local_values = np.r_[self.residual, residuals]
        local_weights = np.r_[(1 - fraction) * self.weights, np.full(n, fraction / n) if n else []]
        local_sigma = max(1., float(np.sqrt(local_weights @ local_values**2)))
        context = monthly_context(self.data, q)
        risk = float(np.sqrt(max(1., self.risk_model.predict(context['history'][None])[0])))
        leaf = self.anchor_model.apply(context['base'][None])[0]
        qrf_weights = np.zeros(len(self.ids))
        for tree, node in enumerate(leaf):
            ids = self.leaf_ids[tree][int(node)]
            qrf_weights[ids] += self.weights[ids] / self.weights[ids].sum() / 32
        result = {
            'global': CompressedLaw(np.clip(anchor + self.residual, 0, 1000), self.weights),
            'local': CompressedLaw(np.clip(anchor + local_values, 0, 1000), local_weights),
            'qrf': CompressedLaw(np.clip(anchor + self.increments - qrf_weights @ self.increments, 0, 1000), qrf_weights),
            'hetero': CompressedLaw(np.clip(anchor + self.residual * risk / self.sigma, 0, 1000), self.weights)}
        return result, {'local_backtests': n, 'local_fraction': fraction, 'local_sigma': local_sigma,
                        'hetero_sigma': risk, 'local_backtest_identities': json.dumps(identities)}


class DistributionScenarioPrediction(ScenarioPrediction):
    def build(self, q):
        forecast, S, B, donors, prior, anchors = self.inputs(q)
        frozen = self.frozen
        distance = .5 * ((np.sqrt(frozen.data.shares[donors]) - np.sqrt(S))**2).sum(axis=1)
        weight = np.exp(-distance / .3**2)
        weight /= weight.sum()
        shape, _ = frozen.shape.laws(donors, weight, S)
        dynamic, _ = frozen.activity.laws(donors, prior, B)
        laws = {'positive_recomposition': shape, 'conditional_activity_recomposition': dynamic}
        coarse, local = self.coarse.laws(q)
        records = []
        for name, mode, nearest in [('global_residual_nearest', 'global', True),
                                     ('global_residual_barycentric', 'global', False),
                                     ('local_residual_barycentric', 'local', False),
                                     ('qrf_barycentric', 'qrf', False),
                                     ('hetero_history_barycentric', 'hetero', False)]:
            law = coarse[mode]
            posterior, info = lift_probabilities(anchors, prior, law.x, law.w, nearest)
            laws[name], _ = frozen.activity.laws(donors, posterior, B)
            records.append({'method': name, **info, **local,
                            'coarse_values': json.dumps(law.x.tolist()),
                            'coarse_weights': json.dumps(law.w.tolist())})
        for name, sigma in [('world_gaussian', forecast['sigma']), ('local_gaussian', local['local_sigma'])]:
            logits = np.log(prior) - .5 * ((anchors - forecast['anchor']) / sigma)**2
            posterior = np.exp(np.maximum(logits - logits.max(), -700))
            posterior /= posterior.sum()
            laws[name], _ = frozen.activity.laws(donors, posterior, B)
            records.append({'method': name, **local, 'achieved': float(posterior @ anchors),
                            'effective_scenarios': float(1 / (posterior @ posterior)),
                            'coarse_values': '[]', 'coarse_weights': '[]'})
        if set(laws) != set(METHODS):
            raise ValueError('missing distribution lifting arm')
        return laws, forecast, records
