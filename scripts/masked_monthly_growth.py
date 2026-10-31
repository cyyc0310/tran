"""Separate unavailable source responses from observed positive-to-zero output."""
from __future__ import annotations

import json
import numpy as np

from coarse_distribution_lifting import CoarseLawForecast
from coarse_oracle_zero_audit import CoarseOracleDiagnostic
from v151_hourly_law import EF
from v154_hourly_strong_baselines import CompressedLaw

METHODS = ('positive_recomposition', 'conditional_activity_recomposition', 'world_gaussian',
           'mask_reweighted', 'neutral_share', 'neutral_energy',
           'neutral_share_gaussian', 'neutral_energy_gaussian')
REFERENCE = {name: 'conditional_activity_recomposition' for name in METHODS[3:]}


def scenario_shape_laws(frozen, donors, probability, composition):
    prepared = [frozen.shape.donor(int(d)) for d in donors]
    laws = []
    for h in range(24):
        fuel = np.concatenate([cells[h] * a for cells, a in zip(prepared, composition)])
        weight = np.concatenate([np.full(len(cells[h]), p / len(cells[h])) for cells, p in zip(prepared, probability)])
        total = fuel.sum(axis=1)
        if (total <= 0).any():
            raise ValueError('missing growth produced undefined CIF')
        laws.append(CompressedLaw(fuel @ EF / total, weight))
    return laws


def gaussian_weights(probability, anchors, prediction, sigma):
    logits = np.log(probability) - .5 * ((anchors - prediction) / sigma)**2
    weight = np.exp(np.maximum(logits - logits.max(), -700))
    return weight / weight.sum()


class MaskedMonthlyGrowth(CoarseOracleDiagnostic):
    def __init__(self, frozen, monthly):
        super().__init__(frozen)
        self.monthly = monthly
        self.coarse = CoarseLawForecast(monthly, frozen.fit)

    def build(self, q):
        f = self.frozen
        local, S, donors, past, previous, current, p = self.inputs(q)
        forecast = self.coarse.predict(q)
        if forecast['local'] != local:
            raise ValueError('coarse and hourly inputs differ')
        B = .99 * S + .01 * f.prior
        native = B * current / (.99 * previous + .01 * f.prior)
        mask = (previous == 0) & (current == 0) & (S[None] > 0)
        exposure = (mask * S).sum(axis=1)
        total_growth = f.data.M[donors].sum(axis=1) / f.data.M[past].sum(axis=1)
        share = np.where(mask, B, native)
        energy = np.where(mask, B / total_growth[:, None], native)
        native_anchor = (native @ EF) / native.sum(axis=1)
        share_anchor = (share @ EF) / share.sum(axis=1)
        energy_anchor = (energy @ EF) / energy.sum(axis=1)
        old_distance = .5 * ((np.sqrt(current) - np.sqrt(S))**2).sum(axis=1)
        old_weight = np.exp(-old_distance / .3**2)
        old_weight /= old_weight.sum()
        native_gaussian = gaussian_weights(p, native_anchor, forecast['anchor'], forecast['sigma'])
        share_gaussian = gaussian_weights(p, share_anchor, forecast['anchor'], forecast['sigma'])
        energy_gaussian = gaussian_weights(p, energy_anchor, forecast['anchor'], forecast['sigma'])
        filtered = p * np.exp(-exposure / .1)
        filtered /= filtered.sum()
        laws = {'positive_recomposition': f.shape.laws(donors, old_weight, S)[0],
                'conditional_activity_recomposition': f.activity.laws(donors, p, B)[0],
                'world_gaussian': f.activity.laws(donors, native_gaussian, B)[0],
                'mask_reweighted': f.activity.laws(donors, filtered, B)[0],
                'neutral_share': scenario_shape_laws(f, donors, p, share),
                'neutral_energy': scenario_shape_laws(f, donors, p, energy),
                'neutral_share_gaussian': scenario_shape_laws(f, donors, share_gaussian, share),
                'neutral_energy_gaussian': scenario_shape_laws(f, donors, energy_gaussian, energy)}
        context = {'input_profile_id': local, 'past_anchor': float(S @ EF),
                   'coarse_forecast_anchor': forecast['anchor'], 'coarse_forecast_sigma': forecast['sigma'],
                   'missing_energy_proxy': float(p @ exposure),
                   'native_world_anchor_mean': float(p @ native_anchor),
                   'share_world_anchor_mean': float(p @ share_anchor),
                   'energy_world_anchor_mean': float(p @ energy_anchor)}
        worlds = []
        for j, d in enumerate(donors):
            worlds.append({'donor_profile_id': int(d), 'donor_input_profile_id': int(past[j]),
                           'conditional_probability': float(p[j]), 'mask_probability': float(filtered[j]),
                           'native_gaussian_probability': float(native_gaussian[j]),
                           'share_gaussian_probability': float(share_gaussian[j]),
                           'energy_gaussian_probability': float(energy_gaussian[j]),
                           'missing_energy_proxy': float(exposure[j]), 'total_monthly_growth': float(total_growth[j]),
                           'missing_mask': json.dumps(mask[j].tolist()),
                           'native_composition': json.dumps(native[j].tolist()),
                           'share_composition': json.dumps(share[j].tolist()),
                           'energy_composition': json.dumps(energy[j].tolist())})
        sensitivity = [{'method': name, 'world_anchor_mean': float(probability @ anchors)}
                       for name, probability, anchors in [('native', p, native_anchor), ('share', p, share_anchor),
                                                         ('energy', p, energy_anchor)]]
        return laws, context, worlds, sensitivity
