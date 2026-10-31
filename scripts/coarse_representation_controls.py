"""Ordinary monthly response/support factorial; no hourly data interface."""
from __future__ import annotations

import numpy as np

from monthly_history_features import monthly_context
from monthly_history_value_models import MonthlyValueModels, FORESTS, forest, constant_anchor
from v151_hourly_law import EF
from v179_increment_support_diagnostic import project_simplex
from v184_coarse_history_value import fingerprint, array_hash

LEVEL = {f'forest_{feature}_level8': feature for feature in ('base', 'history')}
NAMES = (*FORESTS, *LEVEL, 'hellinger', 'gaussian')
SUPPORTS = ('increment', 'absolute')


class RepresentationControls:
    """Fix all data and capacity while crossing response labels and support."""

    def __init__(self, data, proper, profiles):
        self.base = MonthlyValueModels(data, proper, profiles)
        b = self.base
        self.mean = b.prior @ b.current
        self.scale = np.maximum(np.sqrt(b.prior @ (b.current - self.mean)**2), 1e-3)
        self.target = (b.current - self.mean) / self.scale
        self.models = dict(b.models)
        self.models.update({name: forest(b.X[feature], self.target, b.regions) for name, feature in LEVEL.items()})
        self.features = {name: spec[0] for name, spec in FORESTS.items()} | LEVEL
        self.leaves = {name: model.apply(b.X[self.features[name]]) for name, model in self.models.items()}

    def metadata(self):
        """Bind both existing fits and new level response forests."""
        keys = ('children_left', 'children_right', 'feature', 'threshold', 'value', 'weighted_n_node_samples')
        return {'base': fingerprint(self.base), 'level_mean': self.mean.tolist(), 'level_scale': self.scale.tolist(),
                'level_target_sha256': array_hash(self.target),
                'level_trees': {name: [{key: array_hash(getattr(t.tree_, key)) for key in keys}
                                      for t in self.models[name].estimators_] for name in LEVEL}}

    def build(self, q):
        """Construct complete coarse laws using only the query's legal prefix."""
        b = self.base
        if b.data.index.region.iloc[q] in b.proper:
            raise ValueError('query/proper region overlap')
        context = monthly_context(b.data, q)
        if context is None:
            raise ValueError('missing query legal prefix')
        S = b.data.shares[context['local']]
        supports = {'increment': project_simplex(S + b.delta), 'absolute': b.current.copy()}
        logits = np.log(b.prior) - .5 * ((np.sqrt(b.previous) - np.sqrt(S))**2).sum(axis=1) / .3**2
        pH = np.exp(np.maximum(logits - max(logits), -700)); pH /= pH.sum()
        weights = {'hellinger': pH}
        for name, fit in self.models.items():
            signature = fit.apply(context[self.features[name]][None])[0]
            p = np.zeros(len(b.ids))
            for tree, node in enumerate(signature):
                ids = np.flatnonzero(self.leaves[name][:, tree] == node)
                p[ids] += b.raw_weight[ids] / b.raw_weight[ids].sum() / 32
            weights[name] = .95 * p + .05 * pH
        anchor = S @ EF if constant_anchor(S) else float(np.clip(S @ EF + b.anchor_model.predict(context['base'][None])[0], 0, 1000))
        laws = {}
        for support, values in supports.items():
            logp = np.log(pH) - .5 * ((values @ EF - anchor) / b.sigma)**2
            p = np.exp(np.maximum(logp - max(logp), -700)); p /= p.sum()
            for name in NAMES:
                laws[f'{support}__{name}'] = values, (p.copy() if name == 'gaussian' else weights[name].copy())
        if len(laws) != 16:
            raise ValueError('factorial inventory')
        for values, p in laws.values():
            if (values.shape != (len(b.ids), 8) or not np.isfinite(values).all() or (values < 0).any()
                    or np.max(abs(values.sum(axis=1) - 1)) > 1e-12 or p.shape != (len(b.ids),)
                    or not np.isfinite(p).all() or (p <= 0).any() or abs(p.sum() - 1) > 1e-12):
                raise ValueError('invalid factorial coarse law')
        for name in (*self.models, 'hellinger'):
            if not np.array_equal(laws[f'increment__{name}'][1], laws[f'absolute__{name}'][1]):
                raise ValueError('fixed-weight support control differs')
        # Preserve original predictors exactly, including probability arithmetic.
        old = b.build(q)
        for name in FORESTS:
            if any(not np.array_equal(a, c) for a, c in zip(laws[f'increment__{name}'], old[name])):
                raise ValueError('V184 forest negative control differs')
        if any(not np.array_equal(a, c) for a, c in zip(laws['increment__gaussian'], old['gaussian_increment'])):
            raise ValueError('V184 Gaussian negative control differs')
        return laws

    def future_guard(self, q):
        """Mutate all unreleased local monthly values without changing fits."""
        before = self.build(q)
        data = self.base.data
        ids = data.groups[data.index.region.iloc[q]]
        ids = ids[(data.release.iloc[ids] > data.dates.iloc[q]).to_numpy()]
        saved = data.M[ids].copy(), data.shares[ids].copy()
        try:
            changed = np.random.default_rng(185 + q).uniform(1, 1000, size=saved[0].shape)
            data.M[ids] = changed
            data.shares[ids] = changed / changed.sum(axis=1, keepdims=True)
            after = self.build(q)
        finally:
            data.M[ids], data.shares[ids] = saved
        if any(not np.array_equal(a, c) for name in before for a, c in zip(before[name], after[name])):
            raise ValueError('unreleased monthly values changed prediction')

