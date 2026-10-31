"""Source-only cross-region monthly mean controls for frozen V192 development."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from extended_monthly_information import ExtendedMonthlyInformation
from monthly_history_features import monthly_context, month_number
from v151_hourly_law import EF

ROOT = Path(__file__).resolve().parents[1]
ORIGINS = ('2024-04-01', '2024-05-01', '2024-06-01')
ALLOCATIONS = ('spread', 'recent')
METHODS = ('anchor', 'seasonal', 'direct_ridge36', 'direct_et36',
           'decomp_ridge36', 'decomp_et36', 'dynamic_pooled', 'dynamic_random')


def region_weights(regions: np.ndarray) -> np.ndarray:
    """Equal total mass per region, equal profile mass within region."""
    unique, counts = np.unique(regions, return_counts=True)
    lookup = dict(zip(unique, counts))
    return np.asarray([1 / len(unique) / lookup[r] for r in regions])


class StandardRidge:
    """Proper-weighted standardization and fixed Ridge, retaining numeric artifacts."""

    def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        self.center = w @ X / w.sum()
        self.scale = np.sqrt(w @ ((X - self.center) ** 2) / w.sum())
        self.scale[self.scale < 1e-12] = 1.
        self.model = Ridge(alpha=1).fit((X - self.center) / self.scale, y,
                                        sample_weight=w / w.mean())

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict((np.asarray(X) - self.center) / self.scale)


def forest(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> ExtraTreesRegressor:
    """Fixed ordinary forest, deterministic single-thread prediction."""
    return ExtraTreesRegressor(n_estimators=64, min_samples_leaf=3,
                               max_features=1., random_state=192, n_jobs=1).fit(
                                   X, y, sample_weight=w / w.mean())


class Context:
    """Coarse-only data plus explicit selected-source fine access."""

    def __init__(self):
        self.data = ExtendedMonthlyInformation(
            ROOT, ROOT / 'results/v183-coarse-history-export-r1/monthly_history.csv')
        self.index = pd.read_csv(ROOT / 'results/v144-canonical-profile-cache-r1/profile_index.csv')
        with np.load(ROOT / 'results/v153-calendar-times-r1/timestamps.npz', allow_pickle=False) as p:
            self.seconds = p['unix_seconds'].copy()
        with np.load(ROOT / 'results/v188-forward-zero-risk-r1/fold_0/training_matrices.npz', allow_pickle=False) as p:
            regions, counts = np.unique(p['region'], return_counts=True)
        self.budgets = dict(zip(regions.tolist(), counts.tolist()))
        self.roster = sorted(json.loads((ROOT / 'results/v188-forward-zero-risk-r1/fold_0/training_provenance.json').read_text())['proper_regions'])
        if set(self.budgets) != set(self.roster) or len(self.roster) != 19:
            raise ValueError('fixed proper roster/budget')
        if not self.index.profile_id.eq(np.arange(len(self.index))).all() or self.index.duplicated(['region', 'month']).any():
            raise ValueError('canonical index identity')
        if not np.isfinite(self.data.M).all() or (self.data.M < 0).any() or (self.data.M.sum(axis=1) <= 0).any():
            raise ValueError('invalid monthly input')
        self.features: dict[int, dict] = {}
        self.raw: np.ndarray | None = None

    def feature(self, q: int) -> dict:
        if q not in self.features:
            ctx = monthly_context(self.data, q)
            if ctx is None or not np.all(self.data.release.iloc[ctx['prefix']] <= self.data.dates.iloc[q]):
                raise ValueError('historical prefix unavailable')
            self.features[q] = ctx
        return self.features[q]

    def clear_features(self) -> None:
        self.features.clear()

    def profile(self, region: str, month: str) -> int:
        rows = self.index.loc[self.index.region.eq(region) & self.index.month.eq(month)]
        if len(rows) != 1:
            raise ValueError('missing/duplicate canonical query')
        return int(rows.profile_id.iloc[0])

    def selection(self, target: str, origin: str, allocation: str) -> list[dict]:
        if target not in self.roster or origin not in ORIGINS or allocation not in ALLOCATIONS:
            raise ValueError('unregistered case')
        result = []
        cutoff = int(pd.Timestamp(origin, tz='UTC').timestamp())
        for region in self.roster:
            if region == target:
                continue
            ids = [int(q) for q in self.data.groups[region]
                   if q < self.data.canonical_count and self.index.month.iloc[q].startswith('2024')
                   and self.data.release.iloc[q] <= pd.Timestamp(origin)]
            ids.sort(key=lambda q: self.index.month.iloc[q])
            if allocation == 'recent':
                ids = ids[-2:]
            if len(ids) < 2:
                raise ValueError('insufficient released source states')
            n, remainder = divmod(self.budgets[region], len(ids))
            for j, q in enumerate(ids):
                row = self.index.iloc[q]
                count = n + (j < remainder)
                if count > row.hours:
                    raise ValueError('budget exceeds distinct valid hours')
                rng = np.random.default_rng(np.random.SeedSequence([192, q]))
                raw = np.sort(rng.permutation(np.arange(int(row.offset), int(row.stop)))[:count])
                if not len(raw) or not np.all(self.seconds[raw] < cutoff):
                    raise ValueError('source raw time violation')
                result.append({'region': region, 'profile_id': q, 'raw_ids': raw,
                               'prefix': self.feature(q)['prefix']})
        if sum(len(r['raw_ids']) for r in result) != sum(self.budgets.values()) - self.budgets[target]:
            raise ValueError('fine budget changed')
        return result

    def source_values(self, records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read labels only for explicit proper selections; no target access."""
        if self.raw is None:
            with np.load(ROOT / 'results/v144-canonical-profile-cache-r1/profiles.npz', allow_pickle=False) as p:
                self.raw = p['generation'].copy()
        anchors, means = [], []
        for rec in records:
            g = self.raw[rec['raw_ids']]
            if not np.isfinite(g).all() or (g < 0).any() or (g.sum(axis=1) <= 0).any():
                raise ValueError('invalid sampled positive generation')
            anchors.append(float(self.data.shares[rec['profile_id']] @ EF))
            means.append(float(np.mean((g @ EF) / g.sum(axis=1))))
        a, mu = np.asarray(anchors), np.asarray(means)
        return a, mu, a - mu


class Dynamic:
    """Ordinary pooled/random-intercept VAR and coarse-observation Kalman filter."""

    def __init__(self, context: Context, records: list[dict], states: np.ndarray,
                 kappa_model: StandardRidge, random_effect: bool):
        regions = np.asarray([r['region'] for r in records])
        pairs = [(i, j) for i in range(len(records)) for j in range(len(records))
                 if regions[i] == regions[j]
                 and month_number(context.data.dates.iloc[records[j]['profile_id']])
                 - month_number(context.data.dates.iloc[records[i]['profile_id']]) == 1]
        if not pairs:
            raise ValueError('no legal dynamic transitions')
        src, dst = np.asarray(pairs).T
        self.pairs = np.asarray(pairs)
        X, Y = states[src] / 1000, states[dst] / 1000
        pair_regions = regions[src]
        w = region_weights(pair_regions)
        names = np.unique(pair_regions)
        Z = (pair_regions[:, None] == names).astype(float)
        Z -= w @ Z
        D = np.column_stack([np.ones(len(X)), X, Z]) if random_effect else np.column_stack([np.ones(len(X)), X])
        penalty = np.ones(D.shape[1]); penalty[0] = 0
        wn = w / w.mean()
        coef = np.linalg.solve(D.T @ (wn[:, None] * D) + np.diag(penalty), D.T @ (wn[:, None] * Y))
        U, s, V = np.linalg.svd(coef[1:3].T, full_matrices=False)
        self.A = (U * np.minimum(s, .98)) @ V
        self.b = coef[0]
        # Retain trained source effects only in process covariance estimation.
        source_b = D @ coef - X @ coef[1:3]
        residual = Y - X @ self.A.T - source_b
        self.Q = (residual.T * w) @ residual + np.eye(2) * 1e-8
        Xf = np.asarray([context.feature(r['profile_id'])['history'] for r in records])
        kr = states[:, 1] - kappa_model.predict(Xf)
        initial_var = float(region_weights(regions) @ (kr / 1000) ** 2)
        self.P0 = np.diag([max(float(np.var(states[:, 0] / 1000)), 1e-8), max(initial_var, 1e-8)])
        self.coefficient = coef
        self.kappa_model = kappa_model
        self.context = context

    def predict(self, q: int) -> tuple[float, float]:
        data = self.context.data
        ids = [int(p) for p in data.groups[data.index.region.iloc[q]]
               if p < data.canonical_count and data.index.month.iloc[p].startswith('2024')
               and data.release.iloc[p] <= data.dates.iloc[q]]
        ids.sort(key=lambda p: data.index.month.iloc[p])
        first = ids[0]
        x = np.asarray([float(data.shares[first] @ EF),
                        float(self.kappa_model.predict([self.context.feature(first)['history']])[0])]) / 1000
        P = self.P0.copy()
        previous = first
        for p in ids[1:]:
            gap = month_number(data.dates.iloc[p]) - month_number(data.dates.iloc[previous])
            for _ in range(gap):
                x = self.A @ x + self.b
                P = self.A @ P @ self.A.T + self.Q
            if P[0, 0] <= 0 or not np.isfinite(P).all():
                raise ValueError('invalid conditional covariance')
            gain = P[:, 0] / P[0, 0]
            x += gain * (float(data.shares[p] @ EF) / 1000 - x[0])
            P -= np.outer(gain, P[0].copy())
            P = (P + P.T) / 2
            previous = p
        gap = month_number(data.dates.iloc[q]) - month_number(data.dates.iloc[previous])
        for _ in range(gap):
            x = self.A @ x + self.b
        return float(x[0] * 1000), float(x[1] * 1000)


class Predictor:
    """One target/origin/allocation: all parameters exclude target region."""

    def __init__(self, context: Context, target: str, origin: str, allocation: str):
        self.context, self.target, self.origin, self.allocation = context, target, origin, allocation
        self.records = context.selection(target, origin, allocation)
        self.profile_ids = np.asarray([r['profile_id'] for r in self.records])
        self.regions = np.asarray([r['region'] for r in self.records])
        self.X = np.asarray([context.feature(q)['history'] for q in self.profile_ids])
        self.a, self.mu, self.kappa = context.source_values(self.records)
        self.w = region_weights(self.regions)
        self.direct_ridge = StandardRidge(self.X, self.mu, self.w)
        self.direct_et = forest(self.X, self.mu, self.w)
        self.kappa_ridge = StandardRidge(self.X, self.kappa, self.w)
        self.kappa_et = forest(self.X, self.kappa, self.w)
        data = context.data
        self.coarse_ids = np.asarray([int(p) for region in context.roster if region != target
                                      for p in data.groups[region] if data.release.iloc[p] <= pd.Timestamp(origin)
                                      and monthly_context(data, int(p)) is not None])
        if (target in set(data.index.region.to_numpy()[self.coarse_ids])
                or not np.all(data.release.iloc[self.coarse_ids] <= pd.Timestamp(origin))
                or set(self.regions) != set(context.roster) - {target}):
            raise ValueError('coarse/fine role or release violation')
        self.coarse_X = np.asarray([context.feature(p)['history'] for p in self.coarse_ids])
        self.coarse_y = np.asarray([float(data.shares[p] @ EF) for p in self.coarse_ids])
        cw = region_weights(data.index.region.to_numpy()[self.coarse_ids])
        self.coarse_model = forest(self.coarse_X, self.coarse_y, cw)
        states = np.column_stack([self.a, self.kappa])
        self.dynamic = {name: Dynamic(context, self.records, states, self.kappa_ridge, name == 'dynamic_random')
                        for name in ('dynamic_pooled', 'dynamic_random')}

    def predict(self) -> list[dict]:
        q = self.context.profile(self.target, self.origin)
        ctx = self.context.feature(q)
        X = [ctx['history']]
        a_hat = float(self.coarse_model.predict(X)[0])
        year = pd.Timestamp(self.origin).year - 1
        previous = [int(p) for p in ctx['prefix'] if self.context.data.dates.iloc[p].year == year
                    and self.context.data.dates.iloc[p].month == pd.Timestamp(self.origin).month]
        seasonal = float(self.context.data.shares[previous[0]] @ EF) if previous else ctx['anchor']
        values = {'anchor': (ctx['anchor'], 0.), 'seasonal': (seasonal, 0.)}
        for name, model in [('direct_ridge36', self.direct_ridge), ('direct_et36', self.direct_et)]:
            mu = float(model.predict(X)[0]);values[name] = (a_hat, a_hat - mu)
        values['decomp_ridge36'] = (a_hat, float(self.kappa_ridge.predict(X)[0]))
        values['decomp_et36'] = (a_hat, float(self.kappa_et.predict(X)[0]))
        values.update({name: model.predict(q) for name, model in self.dynamic.items()})
        rows = [{'target': self.target, 'origin': self.origin, 'allocation': self.allocation,
                 'profile_id': q, 'method': name, 'predicted_a': values[name][0],
                 'predicted_kappa': values[name][1], 'mu_unclipped': values[name][0] - values[name][1],
                 'predicted_mu': float(np.clip(values[name][0] - values[name][1], 0, 1000)),
                 'common_coarse_a': a_hat} for name in METHODS]
        if not np.isfinite([[r[k] for k in ('predicted_a', 'predicted_kappa', 'predicted_mu')] for r in rows]).all():
            raise ValueError('nonfinite prediction')
        return rows

    def arrays(self) -> dict:
        result = {'profile_ids': self.profile_ids, 'regions': self.regions, 'X': self.X,
                  'a': self.a, 'mu': self.mu, 'kappa': self.kappa, 'w': self.w,
                  'coarse_ids': self.coarse_ids, 'coarse_X': self.coarse_X, 'coarse_y': self.coarse_y}
        for name, model in self.dynamic.items():
            for key in ('A', 'b', 'Q', 'P0', 'coefficient', 'pairs'):
                result[f'{name}_{key}'] = getattr(model, key)
        return result
