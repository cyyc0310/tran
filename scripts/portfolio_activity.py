"""Fixed-budget source portfolios and ordinary matched physical-law controls."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cif_pit_recalibration import PhysicalReferences, REFERENCES
from fuel_activity_forest import carrier
from v151_hourly_law import EF
from v154_hourly_strong_baselines import CompressedLaw

MODES = ('expanded_original', 'selected_parent', 'pair_mixture',
         'pair_sync', 'pair_permute1', 'pair_permute2')
SETTINGS = {f'{prefix}_{mode}': (reference, mode)
            for prefix, reference in zip(('conditional', 'gaussian'), REFERENCES) for mode in MODES}
METHODS = REFERENCES + tuple(SETTINGS)
MATCHED = ('pair_sync', 'pair_permute1', 'pair_permute2', 'pair_mixture')


def digest_array(array):
    return hashlib.sha256(np.asarray(array, np.int64).tobytes()).hexdigest()


def permutation(parents, month, states, seed):
    """Move full fuel rows only within identical UTC hour/weekend cells."""
    order = np.arange(len(states))
    for state in np.unique(states):
        ids = np.flatnonzero(states == state)
        key = f'v177/perm/{",".join(parents)}/{month}/{int(state)}/{seed}'
        number = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
        order[ids] = np.random.default_rng(number).permutation(ids)
    if not np.array_equal(np.sort(order), np.arange(len(order))) or not np.array_equal(states[order], states):
        raise ValueError('invalid joint calendar permutation')
    return order


def portfolio_cells(g, past, states, prior, parents, month):
    """Construct portfolios with explicit within-component probabilities."""
    if g.shape != (len(states), 2, 8) or past.shape != (2, 8):
        raise ValueError('portfolio axes')
    if not np.isfinite(g).all() or not np.isfinite(past).all() or (g < 0).any() or (past < 0).any():
        raise ValueError('invalid nonnegative portfolio inputs')
    total = past.sum(axis=1)
    if min(total) <= 0 or (g.sum(axis=2) <= 0).any() or not set(states % 24) == set(range(24)):
        raise ValueError('portfolio positive generation or calendar support')
    # past has units of mean MW over the CURRENT complete calendar month.
    B = carrier(past.sum(axis=0) / total.sum(), 'asof_30d', prior)
    beta = total / total.sum()
    u_sync = g.sum(axis=1) / total.sum() / B
    orders = [permutation(parents, month, states, seed) for seed in (1, 2)]
    values = {'pair_sync': u_sync,
              **{f'pair_permute{j + 1}': (g[:, 0] + g[order, 1]) / total.sum() / B
                 for j, order in enumerate(orders)}}
    pair_parent = [g[:, j] / total[j] / B for j in range(2)]
    identity = [g[:, j] / total[j] / carrier(past[j] / total[j], 'asof_30d', prior) for j in range(2)]
    cells = {mode: [] for mode in MODES if mode != 'expanded_original'}
    max_error = 0.
    for state in range(24):
        take = np.flatnonzero(states % 24 == state)
        n = len(take)
        expected = u_sync[take].mean(axis=0)
        for name, array in values.items():
            cells[name].append((array[take], np.zeros(n, int), np.full(n, 1 / n)))
            max_error = max(max_error, float(max(abs(array[take].mean(axis=0) - expected))) / expected.sum())
        mixed = np.concatenate([a[take] for a in pair_parent])
        weights = np.r_[np.full(n, beta[0] / n), np.full(n, beta[1] / n)]
        cells['pair_mixture'].append((mixed, np.zeros(2 * n, int), weights))
        cells['selected_parent'].append((np.concatenate([a[take] for a in identity]),
                                         np.repeat(np.arange(2), n), np.full(2 * n, 1 / n)))
        max_error = max(max_error, float(max(abs(weights @ mixed - expected))) / expected.sum())
    if max_error > 2e-12:
        raise ValueError('matched portfolio fuel means differ')
    return cells, max_error, [digest_array(order) for order in orders]


class Geometry:
    """Immutable sample-index geometry; query weights are separate."""

    def __init__(self, identifiers, past, current, source_B, prior, cells):
        self.identifiers = identifiers
        self.past, self.current, self.source_B, self.prior = [np.asarray(a, float) for a in (past, current, source_B, prior)]
        self.cells = cells
        if (self.past.shape != self.current.shape or self.past.shape != self.source_B.shape
                or self.past.shape != (len(prior), 8) or len(cells) != 24 or not (self.prior > 0).all()
                or abs(self.prior.sum() - 1) > 1e-12):
            raise ValueError('geometry metadata')
        for activity, component, within in self.cells:
            if (activity.shape != (len(component), 8) or len(within) != len(component)
                    or not np.isfinite(activity).all() or (activity < 0).any() or (activity.sum(axis=1) <= 0).any()
                    or not (within > 0).all() or component.min() < 0 or component.max() >= len(prior)
                    or max(abs(np.bincount(component, weights=within, minlength=len(prior)) - 1)) > 1e-12):
                raise ValueError('geometry particles/probabilities')

    def weights(self, shares, base, forecast, gaussian):
        distance = .5 * ((np.sqrt(self.past) - np.sqrt(shares))**2).sum(axis=1)
        logits = np.log(self.prior) - distance / .3**2
        a = base * self.current / self.source_B
        anchors = a @ EF / a.sum(axis=1)
        if gaussian:
            logits -= .5 * ((anchors - forecast['anchor']) / forecast['sigma'])**2
        weights = np.exp(np.maximum(logits - logits.max(), -700))
        weights /= weights.sum()
        return weights

    def predict(self, base, weights):
        laws, means = [], []
        for activity, component, within in self.cells:
            probability = weights[component] * within
            fuel = activity * base
            total = fuel.sum(axis=1)
            if (total <= 0).any() or abs(probability.sum() - 1) > 1e-12:
                raise ValueError('physical portfolio domain/probability')
            values = fuel @ EF / total
            if min(values) < min(EF) - 1e-10 or max(values) > max(EF) + 1e-10:
                raise ValueError('portfolio CIF outside physical EF range')
            laws.append(CompressedLaw(values, probability))
            means.append(probability @ fuel)
        return laws, np.asarray(means)


class PortfolioPrediction:
    """Ordinary source-bank augmentation without a new learner or target hours."""

    def __init__(self, frozen, monthly, inventory: Path, root: Path):
        self.frozen, self.monthly = frozen, monthly
        self.physical = PhysicalReferences(frozen, monthly)
        self.index = frozen.data.index
        with np.load(root / 'results/v153-calendar-times-r1/timestamps.npz', allow_pickle=False) as saved:
            self.seconds = saved['unix_seconds'].copy()
        frame = pd.read_csv(inventory / 'portfolio_month_inventory.csv')
        # Fold is unambiguous from the frozen roster and query set; supplied by
        # caller as an explicit attribute to avoid guessing from region names.
        self.worlds, self.source_proofs = [], []
        self.frame = frame
        self.full_cache = {}
        self.sample_by_profile = {}
        for region in frozen.fit:
            source = frozen.activity.sources[region]
            for q in np.unique(source['profiles']):
                pick = source['allowed'][source['profiles'] == q]
                self.sample_by_profile[int(q)] = pick
        self.scale_error = self.check_scaling()
        self.cache = OrderedDict()
        self.expanded = None

    def full_month(self, q):
        if q not in self.full_cache:
            start = self.frozen.data.dates.iloc[q].tz_localize('UTC')
            end = start + pd.offsets.MonthBegin(1)
            expected = np.arange(int(start.timestamp()), int(end.timestamp()), 3600, dtype=np.int64)
            row = self.index.iloc[q]
            self.full_cache[q] = row.hours == len(expected) and np.array_equal(self.seconds[int(row.offset):int(row.stop)], expected)
        return bool(self.full_cache[q])

    def check_scaling(self):
        error = 0.
        data = self.frozen.data
        for q, pick in self.sample_by_profile.items():
            local = data.local(q, 'asof_30d')
            M = data.M[local]
            g = data.raw[pick]
            T = self.index.hours.iloc[q]
            u = g / (M.sum() / T) / carrier(M / M.sum(), 'asof_30d', self.frozen.prior)
            for k in (2., 4.):
                scaled = k * M
                other = (k * g) / (scaled.sum() / T) / carrier(scaled / scaled.sum(), 'asof_30d', self.frozen.prior)
                error = max(error, float(np.max(abs(other - u) / np.maximum(1., abs(u)))))
        if error > 1e-12:
            raise ValueError('proportional activity invariance failed')
        return error

    def prepare(self, fold):
        if self.worlds:
            raise ValueError('portfolio bank already prepared')
        data = self.frozen.data
        frame = self.frame.loc[self.frame.fold.eq(fold) & self.frame.k.eq(2)
                               & self.frame.legal_aggregate_month & self.frame.clock_cells.eq(24)]
        for row in frame.itertuples(index=False):
            parents = json.loads(row.parents)
            qs, locals_ = json.loads(row.profiles), json.loads(row.input_profiles)
            if (len(parents) != 2 or not set(parents).issubset(self.frozen.fit)
                    or [self.index.region.iloc[q] for q in qs] != parents
                    or any(self.index.month.iloc[q] != row.month or not row.month.startswith('2024') for q in qs)
                    or any(data.local(q, 'asof_30d') != p for q, p in zip(qs, locals_))
                    or len(set(self.index.month.iloc[p] for p in locals_)) != 1
                    or not all(self.full_month(q) for q in qs + locals_)):
                raise ValueError('portfolio proper-source/vintage/calendar identity')
            picks = [self.sample_by_profile[q] for q in qs]
            common = np.intersect1d(self.seconds[picks[0]], self.seconds[picks[1]], assume_unique=True)
            raw = np.column_stack([pick[np.searchsorted(self.seconds[pick], common)] for pick in picks])
            if (len(common) != row.shared_sampled_hours or digest_array(raw) != row.intersection_raw_sha256
                    or any(not np.isin(raw[:, j], picks[j]).all() for j in range(2))
                    or not np.array_equal(self.seconds[raw[:, 0]], self.seconds[raw[:, 1]])):
                raise ValueError('portfolio original-hour budget mismatch')
            states = data.states[raw[:, 0]]
            if not np.array_equal(states, data.states[raw[:, 1]]):
                raise ValueError('portfolio calendar states differ')
            T = int(self.index.hours.iloc[qs[0]])
            past = data.M[locals_] / T
            g = data.raw[raw]
            cells, error, perm_hashes = portfolio_cells(g, past, states, self.frozen.prior, parents, row.month)
            Spast = data.M[locals_] / data.M[locals_].sum(axis=1, keepdims=True)
            Scurrent = data.shares[qs]
            aggregate_past = past.sum(axis=0) / past.sum()
            aggregate_current = data.M[qs].sum(axis=0) / data.M[qs].sum()
            self.worlds.append({'parents': tuple(parents), 'month': row.month, 'month_number': int(row.month[5:7]),
                                'profiles': qs, 'inputs': locals_, 'past': aggregate_past, 'current': aggregate_current,
                                'parent_past': Spast, 'parent_current': Scurrent, 'cells': cells})
            self.source_proofs.append({'fold': fold, 'parents': row.parents, 'month': row.month,
                                       'profiles': row.profiles, 'input_profiles': row.input_profiles,
                                       'shared_hours': len(common), 'clock_cells': len(np.unique(states % 24)),
                                       'raw_sha256': digest_array(raw), 'permutation1_sha256': perm_hashes[0],
                                       'permutation2_sha256': perm_hashes[1], 'matched_moment_error': error})
        if not self.worlds:
            raise ValueError('empty legal source portfolio bank')
        self.groups = {}
        for j, world in enumerate(self.worlds):
            self.groups.setdefault(world['parents'], []).append(j)
        del self.frame
        return self

    def geometry(self, month):
        if month in self.cache:
            self.cache.move_to_end(month)
            return self.cache[month]
        if len(self.cache) >= 2:
            self.cache.popitem(last=False)
        selected, prior = [], []
        for _, ids in sorted(self.groups.items()):
            values = np.asarray(ids, int)
            distance = np.abs(np.asarray([self.worlds[j]['month_number'] for j in ids]) - month)
            distance = np.minimum(distance, 12 - distance)
            take = values[np.lexsort((values, distance))[:2]]
            selected.extend(take.tolist())
            prior.extend([1 / len(self.groups) / len(take)] * len(take))
        worlds = [self.worlds[j] for j in selected]
        result = {}
        for mode in MODES[1:]:
            identity = mode == 'selected_parent'
            past = np.concatenate([w['parent_past'] for w in worlds]) if identity else np.asarray([w['past'] for w in worlds])
            current = np.concatenate([w['parent_current'] for w in worlds]) if identity else np.asarray([w['current'] for w in worlds])
            source_B = np.asarray([carrier(S, 'asof_30d', self.frozen.prior) for S in past])
            probability = np.repeat(np.asarray(prior) / 2, 2) if identity else np.asarray(prior)
            ids = [f'{j}/parent{p}' for j in selected for p in range(2)] if identity else selected
            cells = []
            for h in range(24):
                arrays = [w['cells'][mode][h] for w in worlds]
                activity = np.concatenate([a[0] for a in arrays])
                component = np.concatenate([a[1] + (2 * k if identity else k) for k, a in enumerate(arrays)])
                within = np.concatenate([a[2] for a in arrays])
                cells.append((activity, component, within))
            result[mode] = Geometry(ids, past, current, source_B, probability, cells)
        self.cache[month] = result
        return result

    def expanded_geometry(self):
        if self.expanded is None:
            data = self.frozen.data
            profiles = [int(q) for r in self.frozen.fit for q in self.frozen.bank[r]]
            inputs = [data.local(q, 'asof_30d') for q in profiles]
            prior = [1 / len(self.frozen.fit) / len(self.frozen.bank[data.index.region.iloc[q]]) for q in profiles]
            past = data.shares[inputs]
            B = np.asarray([carrier(S, 'asof_30d', self.frozen.prior) for S in past])
            prepared = [self.frozen.activity.donor(q) for q in profiles]
            cells = [(np.concatenate([a[h] for a in prepared]),
                      np.concatenate([np.full(len(a[h]), j, int) for j, a in enumerate(prepared)]),
                      np.concatenate([np.full(len(a[h]), 1 / len(a[h])) for a in prepared])) for h in range(24)]
            self.expanded = Geometry(profiles, past, data.shares[profiles], B, prior, cells)
        return self.expanded

    def build(self, q):
        forecast, S, B, donors, prior, anchors = self.physical.inputs(q)
        logits = np.log(prior) - .5 * ((anchors - forecast['anchor']) / forecast['sigma'])**2
        gaussian = np.exp(np.maximum(logits - logits.max(), -700))
        gaussian /= gaussian.sum()
        laws, moments = {}, {}
        prepared = [self.frozen.activity.donor(int(d)) for d in donors]
        for reference, weights in zip(REFERENCES, (prior, gaussian)):
            laws[reference], rejected = self.frozen.activity.laws(donors, weights, B)
            if max(rejected) != 0:
                raise ValueError('original physical reference domain')
            moments[reference] = np.asarray([sum(w * a[h].mean(axis=0) for w, a in zip(weights, prepared)) * B for h in range(24)])
        geometry = self.geometry(int(self.frozen.month[q])) | {'expanded_original': self.expanded_geometry()}
        row = self.index.iloc[q]
        occupancy = np.bincount(self.frozen.data.states[int(row.offset):int(row.stop)] % 24, minlength=24) / row.hours
        details = []
        for name, (reference, mode) in SETTINGS.items():
            bank = geometry[mode]
            weights = bank.weights(S, B, forecast, reference == REFERENCES[1])
            cells, fuel = bank.predict(B, weights)
            if mode == 'expanded_original':
                laws[name], moments[name] = cells, fuel
            else:
                laws[name] = [CompressedLaw(np.r_[old.x, new.x], np.r_[.5 * old.w, .5 * new.w])
                              for old, new in zip(laws[reference], cells)]
                moments[name] = .5 * moments[reference] + .5 * fuel
            details.append({'method': name, 'worlds': json.dumps(bank.identifiers), 'weights': json.dumps(weights.tolist()),
                            'fuel_mean': json.dumps(moments[name].tolist()),
                            'generator_energy_anchor': float((occupancy @ (moments[name] @ EF)) /
                                                             (occupancy @ moments[name].sum(axis=1)))})
        max_error = 0.
        for prefix in ('conditional', 'gaussian'):
            expected = moments[f'{prefix}_pair_sync']
            for mode in MATCHED:
                actual = moments[f'{prefix}_{mode}']
                max_error = max(max_error, float(np.max(abs(actual - expected) / expected.sum(axis=1, keepdims=True))))
        if max_error > 2e-12:
            raise ValueError('query matched first moments differ')
        for detail in details:
            detail['matched_moment_error'] = max_error
        return laws, forecast, details
