"""Rebuild V159's sampled source laws without fitting or reading held-out labels."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from fuel_activity_forest import ActivityDonors, carrier, source_activity
from v151_hourly_law import EF


class FrozenRecomposition:
    def __init__(self, data, archive: Path, fold: int, budget: str):
        self.data, self.budget = data, budget
        block = archive / f'fold_{fold}_{budget}'
        meta = json.loads((block / 'summary.json').read_text())
        self.fit = sorted(meta['source_regions'])
        self.queries = pd.read_csv(block / 'query_inputs.csv')
        self.allowed_queries = set(self.queries.profile_id)
        radii = pd.read_csv(block / 'calibration_radii.csv')
        self.radii = {(r.method, r.calibration): float(r.radius) for r in radii.itertuples(index=False)}
        samples = pd.read_csv(archive / 'source_sampling.csv')
        samples = samples.loc[samples.budget.eq(budget)].set_index('region')
        allowed = {r: np.asarray(json.loads(samples.loc[r, 'sampled_raw_hour_indices']), int) for r in self.fit}
        proper = {r: data.source(r, budget, allowed[r]) for r in self.fit}
        prior = np.mean([data.shares[proper[r][3]].mean(axis=0) for r in self.fit], axis=0) + 1e-6
        self.prior = prior / prior.sum()
        if not np.array_equal(self.prior, np.asarray(meta['carrier_prior'])):
            raise ValueError('frozen source prior differs')
        sources = {r: source_activity(data, r, budget, allowed[r], self.prior) for r in self.fit}
        self.bank = {r: np.unique(sources[r]['profiles']) for r in self.fit}
        self.shape = ActivityDonors(data, sources, 'shape')
        self.activity = ActivityDonors(data, sources)
        self.month = pd.to_datetime(data.index.month).dt.month.to_numpy()

    @staticmethod
    def _generation_moments(bank, donors, weights, base):
        prepared = [bank.donor(int(d)) for d in donors]
        generation, emissions = [], []
        for state in range(24):
            activity = np.concatenate([d[state] for d in prepared])
            probability = np.concatenate([np.full(len(d[state]), w / len(d[state]))
                                          for d, w in zip(prepared, weights)])
            fuel = activity * base
            total = fuel.sum(axis=1)
            positive = total > 0
            mass = probability[positive].sum()
            if mass <= 0:
                raise ValueError('no positive pseudo-generation support')
            generation.append(float(probability[positive] @ total[positive] / mass))
            emissions.append(float(probability[positive] @ (fuel[positive] @ EF) / mass))
        return np.asarray(generation), np.asarray(emissions)

    def build(self, q: int):
        if q not in self.allowed_queries or self.data.index.region.iloc[q] in self.fit:
            raise ValueError('query outside frozen nontraining roster')
        if not self.data.index.month.iloc[q].startswith('2024'):
            raise ValueError('source-only query year')
        local = self.data.local(q, self.budget)
        if local is None:
            raise ValueError('unavailable local statistic')
        S = self.data.shares[local]
        donors = []
        for region in self.fit:
            ids = self.bank[region]
            if len(ids) < 2:
                raise ValueError('insufficient sampled donor months')
            distance = np.abs(self.month[ids] - self.month[q])
            distance = np.minimum(distance, 12 - distance)
            donors.extend(ids[np.lexsort((ids, distance))[:2]])
        donors = np.asarray(donors, int)
        current = .5 * ((np.sqrt(self.data.shares[donors]) - np.sqrt(S))**2).sum(axis=1)
        weight = np.exp(-current / .3**2)
        weight /= weight.sum()
        shape, _ = self.shape.laws(donors, weight, S)
        shape_moments = self._generation_moments(self.shape, donors, weight, S)
        if self.budget == 'same_month':
            dynamic = shape
            dynamic_moments = shape_moments
        else:
            input_ids = np.asarray([self.data.local(int(d), self.budget) for d in donors], int)
            distance = .5 * ((np.sqrt(self.data.shares[input_ids]) - np.sqrt(S))**2).sum(axis=1)
            weight = np.exp(-distance / .3**2)
            weight /= weight.sum()
            base = carrier(S, self.budget, self.prior)
            dynamic, _ = self.activity.laws(donors, weight, base)
            dynamic_moments = self._generation_moments(self.activity, donors, weight, base)
        return ({'positive_recomposition': shape, 'conditional_activity_recomposition': dynamic},
                {'positive_recomposition': shape_moments, 'conditional_activity_recomposition': dynamic_moments})

    def laws(self, q: int) -> dict:
        return self.build(q)[0]
