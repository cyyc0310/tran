"""Coarse-only forecasts and support-preserving month-scenario updates."""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from fuel_activity_forest import carrier
from level_spread_diagnostics import affine_laws
from monthly_history_features import monthly_context
from v151_hourly_law import EF
from v154_hourly_strong_baselines import balanced

METHODS = ('positive_recomposition', 'conditional_activity_recomposition',
           'forecast_affine', 'forecast_composition', 'world_entropy',
           'world_entropy_cap', 'world_gaussian')


def constant_anchor(shares: np.ndarray) -> bool:
    """A past-support point-forecast fallback, not a future support restriction."""
    return len(np.unique(EF[shares > 0])) == 1


def fit_forest(X, target, regions):
    return ExtraTreesRegressor(n_estimators=32, min_samples_leaf=4,
                               max_features=1., random_state=20261028, n_jobs=2).fit(
                                   X, target, sample_weight=balanced(np.asarray(regions)))


class CoarseForecast:
    """Training and inference have access to monthly information only."""

    def __init__(self, data, proper_regions):
        self.data = data
        self.contexts = {}
        for q, row in data.index.iterrows():
            if row.region in proper_regions and row.month.startswith('2024'):
                context = monthly_context(data, int(q))
                if context is not None:
                    self.contexts[int(q)] = context
        self.ids = np.asarray(list(self.contexts), int)
        regions = data.index.region.to_numpy()[self.ids]
        X = np.asarray([self.contexts[q]['base'] for q in self.ids])
        anchor = np.asarray([self.contexts[q]['anchor'] for q in self.ids])
        # Preserve V161's per-row arithmetic: tiny BLAS rounding differences
        # can resolve equal-gain ExtraTrees split ties differently.
        target = np.asarray([data.shares[q] @ EF - self.contexts[q]['anchor'] for q in self.ids])
        old = np.asarray([self.contexts[q]['local'] for q in self.ids])
        self.anchor_model = fit_forest(X, target, regions)
        self.mix_model = fit_forest(X, data.shares[self.ids] - data.shares[old], regions)
        # Coarse uncertainty is estimated without in-sample fitted residuals.
        ordered = sorted(proper_regions, key=lambda r: hashlib.sha256(('v164/coarse/' + r).encode()).hexdigest())
        group = {region: j % 5 for j, region in enumerate(ordered)}
        oof = np.empty(len(self.ids))
        for fold in range(5):
            test = np.asarray([group[r] == fold for r in regions])
            model = fit_forest(X[~test], target[~test], regions[~test])
            value = np.clip(anchor[test] + model.predict(X[test]), 0, 1000)
            for j, q in enumerate(self.ids[test]):
                if constant_anchor(data.shares[self.contexts[q]['local']]):
                    value[j] = anchor[test][j]
            oof[test] = value - data.shares[self.ids[test]] @ EF
        weights = balanced(regions)
        self.sigma = max(1., float(np.sqrt(np.average(oof**2, weights=weights))))
        self.oof = [{'profile_id': int(q), 'region': str(r), 'fold': group[r], 'error': float(e)}
                    for q, r, e in zip(self.ids, regions, oof)]

    def predict(self, q: int) -> dict:
        context = monthly_context(self.data, q)
        if context is None:
            raise ValueError('no released monthly input')
        S = self.data.shares[context['local']]
        X = context['base'][None]
        raw = float(np.clip(context['anchor'] + self.anchor_model.predict(X)[0], 0, 1000))
        mix = np.maximum(0, S + self.mix_model.predict(X)[0])
        mix = mix / mix.sum() if mix.sum() > 0 else S.copy()
        fallback = constant_anchor(S)
        return {'raw_anchor': raw, 'anchor': context['anchor'] if fallback else raw,
                'past_anchor': context['anchor'], 'shares': S.copy() if fallback else mix,
                'fallback': fallback, 'sigma': self.sigma, 'local': context['local']}


def exponential_weights(prior: np.ndarray, z: np.ndarray, coefficient: float) -> np.ndarray:
    logits = np.log(prior) + coefficient * z
    weights = np.exp(np.maximum(logits - logits.max(), -700))
    return weights / weights.sum()


def entropy_update(prior, anchors, target: float, cap: float | None = None):
    """Finite exponential tilt with recorded feasibility and optional KL cap."""
    span = float(np.ptp(anchors))
    if span <= 1e-10:
        weights = prior.copy()
        return weights, {'lambda': 0., 'kl': 0., 'achieved': float(weights @ anchors),
                         'requested': target, 'feasible_target': float(anchors[0]),
                         'convex_hull_clipped': not abs(target - anchors[0]) <= 1e-8}
    z = (anchors - anchors.min()) / span
    feasible = float(np.clip(target, anchors.min(), anchors.max()))
    desired = (feasible - anchors.min()) / span
    lo, hi = -512., 512.
    for _ in range(70):
        mid = (lo + hi) / 2
        if exponential_weights(prior, z, mid) @ z < desired:
            lo = mid
        else:
            hi = mid
    coefficient = (lo + hi) / 2
    weights = exponential_weights(prior, z, coefficient)

    def divergence(p):
        return max(0., float(p @ (np.log(p) - np.log(prior))))

    if cap is not None and divergence(weights) > cap:
        lo, hi = 0., 1.
        for _ in range(70):
            middle = (lo + hi) / 2
            if divergence(exponential_weights(prior, z, coefficient * middle)) > cap:
                hi = middle
            else:
                lo = middle
        coefficient *= (lo + hi) / 2
        weights = exponential_weights(prior, z, coefficient)
    return weights, {'lambda': coefficient, 'kl': divergence(weights),
                     'achieved': float(weights @ anchors), 'requested': target,
                     'feasible_target': feasible, 'convex_hull_clipped': feasible != target}


class ScenarioPrediction:
    """Keep V159 physical atoms and change only their donor-month probabilities."""

    def __init__(self, frozen, coarse):
        if frozen.budget != 'asof_30d':
            raise ValueError('asof-only pilot')
        self.frozen, self.coarse = frozen, coarse

    def inputs(self, q):
        frozen = self.frozen
        if q not in frozen.allowed_queries or frozen.data.index.region.iloc[q] in frozen.fit:
            raise ValueError('held-out source query roster')
        forecast = self.coarse.predict(q)
        S = frozen.data.shares[forecast['local']]
        B = carrier(S, 'asof_30d', frozen.prior)
        donors = []
        for region in frozen.fit:
            ids = frozen.bank[region]
            distance = np.abs(frozen.month[ids] - frozen.month[q])
            distance = np.minimum(distance, 12 - distance)
            donors.extend(ids[np.lexsort((ids, distance))[:2]])
        donors = np.asarray(donors, int)
        if not set(frozen.data.index.region.iloc[donors]).issubset(frozen.fit):
            raise ValueError('nonproper donor')
        past = np.asarray([frozen.data.local(int(d), 'asof_30d') for d in donors], int)
        distance = .5 * ((np.sqrt(frozen.data.shares[past]) - np.sqrt(S))**2).sum(axis=1)
        weights = np.exp(-distance / .3**2)
        weights /= weights.sum()
        a = B * frozen.data.shares[donors] / carrier(frozen.data.shares[past], 'asof_30d', frozen.prior)
        anchors = (a @ EF) / a.sum(axis=1)
        return forecast, S, B, donors, weights, anchors

    def build(self, q):
        forecast, S, B, donors, prior, anchors = self.inputs(q)
        frozen = self.frozen
        current_distance = .5 * ((np.sqrt(frozen.data.shares[donors]) - np.sqrt(S))**2).sum(axis=1)
        weight = np.exp(-current_distance / .3**2)
        weight /= weight.sum()
        shape, _ = frozen.shape.laws(donors, weight, S)
        dynamic, rejected = frozen.activity.laws(donors, prior, B)
        if max(rejected) != 0:
            raise ValueError('positive carrier rejected activity')
        row = frozen.data.index.iloc[q]
        clock = frozen.data.states[int(row.offset):int(row.stop)] % 24
        affine, _, _, _ = affine_laws(dynamic, clock, forecast['anchor'] - forecast['past_anchor'], 1., (0, 1000))
        mix = forecast['shares']
        distance = .5 * ((np.sqrt(frozen.data.shares[donors]) - np.sqrt(mix))**2).sum(axis=1)
        weight = np.exp(-distance / .3**2)
        weight /= weight.sum()
        composed, _ = frozen.shape.laws(donors, weight, mix)
        laws = dict(zip(METHODS[:4], [shape, dynamic, affine, composed]))
        records = []
        for name, cap in [('world_entropy', None), ('world_entropy_cap', .1)]:
            updated, info = entropy_update(prior, anchors, forecast['anchor'], cap)
            laws[name], rejected = frozen.activity.laws(donors, updated, B)
            if max(rejected) != 0:
                raise ValueError('tilt changed positive domain')
            records.append({'method': name, **info, 'effective_scenarios': float(1 / (updated @ updated)),
                            'prior_anchor': float(prior @ anchors), 'anchor_min': float(anchors.min()),
                            'anchor_max': float(anchors.max())})
        logits = np.log(prior) - .5 * ((anchors - forecast['anchor']) / forecast['sigma'])**2
        gaussian = np.exp(np.maximum(logits - logits.max(), -700))
        gaussian /= gaussian.sum()
        laws['world_gaussian'], _ = frozen.activity.laws(donors, gaussian, B)
        return laws, forecast, records
