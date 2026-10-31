"""Ordinary monthly distribution baselines, without any hourly data interface."""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from monthly_history_features import monthly_context, month_number
from v151_hourly_law import EF
from v179_increment_support_diagnostic import project_simplex

FORESTS = {f'forest_{feature}_{response}': (feature, response)
           for feature in ('base', 'history') for response in ('scalar8', 'delta8')}
RESIDUAL = ('persistence_residual', 'seasonal_residual', 'trend_residual', 'pooled_linear_residual')
METHODS = RESIDUAL + ('gaussian_increment',) + tuple(FORESTS)
EPS = 64 * np.finfo(float).eps


def region_weights(regions):
    unique, counts = np.unique(regions, return_counts=True)
    count = dict(zip(unique, counts))
    return np.asarray([1 / count[r] for r in regions])


def forest(X, y, regions):
    model = ExtraTreesRegressor(n_estimators=32, min_samples_leaf=4, max_features=1.,
                                random_state=20261028, n_jobs=2).fit(X, y, sample_weight=region_weights(regions))
    model.n_jobs = 1
    return model


def fit_linear(X, y, regions):
    raw = region_weights(regions)
    w = raw / raw.sum()
    mean = w @ X
    scale = np.maximum(np.sqrt(w @ (X - mean)**2), 1e-3)
    model = Ridge(alpha=1., fit_intercept=True).fit((X - mean) / scale, y, sample_weight=w * len(X))
    return mean, scale, model


def linear_predict(fitted, X):
    mean, scale, model = fitted
    return model.predict((X - mean) / scale)


def constant_anchor(S):
    return len(np.unique(EF[S > 0])) == 1


class MonthlyValueModels:
    """Proper-only pooled models and local legal-prefix dynamic controls."""

    def __init__(self, data, proper_regions, profiles):
        self.data, self.proper = data, set(proper_regions)
        self.ids = np.asarray(profiles, int)
        if not len(self.ids) or len(np.unique(self.ids)) != len(self.ids):
            raise ValueError('invalid monthly training identities')
        self.regions = data.index.region.to_numpy()[self.ids]
        if (not set(self.regions).issubset(self.proper)
                or not data.index.month.iloc[self.ids].str[:4].isin(['2022', '2023', '2024']).all()):
            raise ValueError('monthly training role/year leakage')
        self.contexts = {int(q): monthly_context(data, int(q)) for q in self.ids}
        if any(c is None for c in self.contexts.values()):
            raise ValueError('training row lacks legal monthly prefix')
        self.inputs = np.asarray([self.contexts[int(q)]['local'] for q in self.ids], int)
        self.X = {key: np.asarray([self.contexts[int(q)][key] for q in self.ids]) for key in ('base', 'history')}
        self.previous, self.current = data.shares[self.inputs], data.shares[self.ids]
        self.delta = self.current - self.previous
        self.delta[abs(self.delta) <= EPS] = 0
        self.raw_weight = region_weights(self.regions)
        self.prior = self.raw_weight / self.raw_weight.sum()
        scalar = np.asarray([data.shares[q] @ EF - data.shares[p] @ EF for q, p in zip(self.ids, self.inputs)])
        magnitude = np.asarray([max(1., abs(data.shares[q] @ EF), abs(data.shares[p] @ EF)) for q, p in zip(self.ids, self.inputs)])
        scalar[abs(scalar) <= EPS * magnitude] = 0
        self.transforms, targets = {}, {}
        for key, value in (('scalar8', np.repeat(scalar[:, None] / 1000, 8, axis=1)), ('delta8', self.delta)):
            mean = self.prior @ value
            scale = np.maximum(np.sqrt(self.prior @ (value - mean)**2), 1e-3)
            targets[key] = (value - mean) / scale
            self.transforms[key] = {'mean': mean.tolist(), 'scale': scale.tolist()}
        self.models, self.leaf_maps = {}, {}
        for name, (feature, response) in FORESTS.items():
            model = forest(self.X[feature], targets[response], self.regions)
            leaves = model.apply(self.X[feature])
            self.models[name] = model
            self.leaf_maps[name] = [{int(node): np.flatnonzero(leaves[:, tree] == node)
                                     for node in np.unique(leaves[:, tree])} for tree in range(32)]
        self.linear = fit_linear(self.X['base'], self.delta, self.regions)
        self.anchor_model = forest(self.X['base'], scalar, self.regions)
        ordered = sorted(self.proper, key=lambda r: hashlib.sha256(('v184/coarse/' + r).encode()).hexdigest())
        self.region_fold = {r: i % 5 for i, r in enumerate(ordered)}
        linear_errors = np.empty_like(self.current)
        anchor_errors = np.empty(len(self.ids))
        for fold in range(5):
            selected = np.asarray([self.region_fold[r] == fold for r in self.regions])
            if not selected.any() or selected.all():
                raise ValueError('empty monthly proper OOF split')
            linear = fit_linear(self.X['base'][~selected], self.delta[~selected], self.regions[~selected])
            points = project_simplex(self.previous[selected] + linear_predict(linear, self.X['base'][selected]))
            linear_errors[selected] = self.current[selected] - points
            anchor = forest(self.X['base'][~selected], scalar[~selected], self.regions[~selected])
            predicted = np.clip(self.previous[selected] @ EF + anchor.predict(self.X['base'][selected]), 0, 1000)
            for j, S in enumerate(self.previous[selected]):
                if constant_anchor(S):
                    predicted[j] = S @ EF
            anchor_errors[selected] = predicted - self.current[selected] @ EF
        self.sigma = max(1., float(np.sqrt(self.prior @ anchor_errors**2)))
        self.residuals = {'persistence_residual': self.delta.copy(), 'pooled_linear_residual': linear_errors}
        for method in ('seasonal_residual', 'trend_residual'):
            points = np.asarray([self.local_point(method, int(q), self.contexts[int(q)]) for q in self.ids])
            self.residuals[method] = self.current - points
        for values in self.residuals.values():
            values[abs(values) <= EPS] = 0
        self.metadata = {'profiles': self.ids.tolist(), 'inputs': self.inputs.tolist(), 'regions': sorted(self.proper),
                         'region_fold': self.region_fold, 'transforms': self.transforms, 'sigma': self.sigma,
                         'OOF_anchor_error_sha256': hashlib.sha256(anchor_errors.tobytes()).hexdigest(),
                         'OOF_linear_error_sha256': hashlib.sha256(linear_errors.tobytes()).hexdigest()}

    def local_point(self, method, q, context):
        S = self.data.shares[context['local']]
        prefix = context['prefix']
        if method == 'persistence_residual':
            return S.copy()
        if method == 'seasonal_residual':
            origin = self.data.dates.iloc[q]
            same = [p for p in prefix if self.data.dates.iloc[p].month == origin.month
                    and self.data.dates.iloc[p].year == origin.year - 1]
            return self.data.shares[same[-1]].copy() if same else S.copy()
        if method == 'trend_residual':
            ids = prefix[-12:]
            if len(ids) < 4:
                return S.copy()
            last = month_number(self.data.dates.iloc[int(ids[-1])])
            t = np.asarray([month_number(self.data.dates.iloc[int(p)]) - last for p in ids])[:, None]
            origin = np.asarray([[month_number(self.data.dates.iloc[q]) - last]])
            point = Ridge(alpha=12., fit_intercept=True).fit(t, self.data.shares[ids]).predict(origin)[0]
            return project_simplex(point[None])[0]
        raise ValueError('unknown local monthly method')

    def build(self, q):
        if self.data.index.region.iloc[q] in self.proper:
            raise ValueError('query belongs to proper training region')
        context = monthly_context(self.data, q)
        if context is None:
            raise ValueError('query lacks legal monthly prefix')
        S = self.data.shares[context['local']]
        support = project_simplex(S + self.delta)
        logits = np.log(self.prior) - .5 * ((np.sqrt(self.previous) - np.sqrt(S))**2).sum(axis=1) / .3**2
        pH = np.exp(np.maximum(logits - max(logits), -700)); pH /= pH.sum()
        laws = {}
        for method in RESIDUAL:
            point = project_simplex((S + linear_predict(self.linear, context['base'][None])[0])[None])[0] if method == 'pooled_linear_residual' else self.local_point(method, q, context)
            laws[method] = (project_simplex(point + self.residuals[method]), self.prior.copy())
        A = S @ EF if constant_anchor(S) else float(np.clip(S @ EF + self.anchor_model.predict(context['base'][None])[0], 0, 1000))
        logp = np.log(pH) - .5 * ((support @ EF - A) / self.sigma)**2
        p = np.exp(np.maximum(logp - max(logp), -700)); p /= p.sum()
        laws['gaussian_increment'] = support, p
        for method, (feature, _) in FORESTS.items():
            signature = self.models[method].apply(context[feature][None])[0]
            p = np.zeros(len(self.ids))
            for tree, node in enumerate(signature):
                ids = self.leaf_maps[method][tree][int(node)]
                p[ids] += self.raw_weight[ids] / self.raw_weight[ids].sum() / 32
            laws[method] = support, .95 * p + .05 * pH
        if set(laws) != set(METHODS):
            raise ValueError('monthly method inventory')
        for values, probability in laws.values():
            if (values.shape != (len(self.ids), 8) or not np.isfinite(values).all() or (values < 0).any()
                    or max(abs(values.sum(axis=1) - 1)) > 1e-12 or not np.isfinite(probability).all()
                    or (probability <= 0).any() or abs(probability.sum() - 1) > 1e-12):
                raise ValueError('invalid monthly predictive law')
        return laws
