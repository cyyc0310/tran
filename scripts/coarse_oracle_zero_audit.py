"""Fixed-source oracle coarse information and positive-carrier diagnostics."""
from __future__ import annotations

import json

import numpy as np

from coarse_distribution_lifting import lift_probabilities
from scenario_reweighting import entropy_update
from v151_hourly_law import EF
from v154_hourly_strong_baselines import CompressedLaw

METHODS = ('positive_recomposition', 'conditional_activity_recomposition', 'carrier_001', 'carrier_100',
           'oracle_anchor_entropy', 'oracle_anchor_barycentric',
           'oracle_composition_fixed_weights', 'oracle_composition_hellinger')
REFERENCE = {name: ('positive_recomposition' if name.startswith('oracle_composition')
                    else 'conditional_activity_recomposition') for name in METHODS[2:]}


def carrier_laws(frozen, donors, probability, shares, epsilon):
    """Change both carriers without changing any source-hour selection."""
    base = (1 - epsilon) * shares + epsilon * frozen.prior
    if epsilon == .01:
        return frozen.activity.laws(donors, probability, base)[0]
    factors = []
    for d in donors:
        local = frozen.data.local(int(d), 'asof_30d')
        S = frozen.data.shares[local]
        old = .99 * S + .01 * frozen.prior
        new = (1 - epsilon) * S + epsilon * frozen.prior
        factors.append(base * old / new)
    prepared = [frozen.activity.donor(int(d)) for d in donors]
    result = []
    for h in range(24):
        fuel = np.concatenate([cells[h] * factor for cells, factor in zip(prepared, factors)])
        p = np.concatenate([np.full(len(cells[h]), mass / len(cells[h]))
                            for cells, mass in zip(prepared, probability)])
        total = fuel.sum(axis=1)
        if (total <= 0).any():
            raise ValueError('carrier diagnostic produced undefined CIF')
        result.append(CompressedLaw(fuel @ EF / total, p))
    return result


class CoarseOracleDiagnostic:
    def __init__(self, frozen):
        if frozen.budget != 'asof_30d':
            raise ValueError('asof source diagnostic')
        self.frozen = frozen

    def inputs(self, q):
        f = self.frozen
        if q not in f.allowed_queries or f.data.index.region.iloc[q] in f.fit:
            raise ValueError('source role leakage')
        local = f.data.local(q, 'asof_30d')
        S = f.data.shares[local]
        donors = []
        for region in f.fit:
            ids = f.bank[region]
            distance = np.abs(f.month[ids] - f.month[q])
            distance = np.minimum(distance, 12 - distance)
            donors.extend(ids[np.lexsort((ids, distance))[:2]])
        donors = np.asarray(donors, int)
        past = np.asarray([f.data.local(int(d), 'asof_30d') for d in donors], int)
        previous, current = f.data.shares[past], f.data.shares[donors]
        distance = .5 * ((np.sqrt(previous) - np.sqrt(S))**2).sum(axis=1)
        p = np.exp(-distance / .3**2)
        p /= p.sum()
        if not set(f.data.index.region.iloc[donors]).issubset(f.fit):
            raise ValueError('nonproper donor')
        return local, S, donors, past, previous, current, p

    def build(self, q):
        f = self.frozen
        local, S, donors, past, previous, current, p = self.inputs(q)
        # Current query composition is deliberately oracle-only.
        truth_S = f.data.shares[q]
        truth_A = float(truth_S @ EF)
        B = .99 * S + .01 * f.prior
        a = B * current / (.99 * previous + .01 * f.prior)
        anchors = (a @ EF) / a.sum(axis=1)
        old_distance = .5 * ((np.sqrt(current) - np.sqrt(S))**2).sum(axis=1)
        old_weight = np.exp(-old_distance / .3**2)
        old_weight /= old_weight.sum()
        oracle_distance = .5 * ((np.sqrt(current) - np.sqrt(truth_S))**2).sum(axis=1)
        oracle_weight = np.exp(-oracle_distance / .3**2)
        oracle_weight /= oracle_weight.sum()
        laws = {'positive_recomposition': f.shape.laws(donors, old_weight, S)[0],
                'conditional_activity_recomposition': f.activity.laws(donors, p, B)[0],
                'carrier_001': carrier_laws(f, donors, p, S, .001),
                'carrier_100': carrier_laws(f, donors, p, S, .1),
                'oracle_composition_fixed_weights': f.shape.laws(donors, old_weight, truth_S)[0],
                'oracle_composition_hellinger': f.shape.laws(donors, oracle_weight, truth_S)[0]}
        entropy, entropy_info = entropy_update(p, anchors, truth_A)
        projected, projection_info = lift_probabilities(anchors, p, np.array([truth_A]), np.ones(1))
        laws['oracle_anchor_entropy'] = f.activity.laws(donors, entropy, B)[0]
        laws['oracle_anchor_barycentric'] = f.activity.laws(donors, projected, B)[0]
        birth = (previous == 0) & (S[None] > 0) & (current > 0)
        birth_fraction = (a * birth).sum(axis=1) / a.sum(axis=1)
        missing = ((current == 0) * S).sum(axis=1)
        missing_true = ((current == 0) * truth_S).sum(axis=1)
        context = {'input_profile_id': local, 'truth_anchor': truth_A, 'past_anchor': float(S @ EF),
                   'flat_imputed_past_energy_proxy': float(old_weight @ missing),
                   'flat_imputed_true_energy_proxy_fixed_weights': float(old_weight @ missing_true),
                   'flat_imputed_true_energy_proxy_hellinger': float(oracle_weight @ missing_true),
                   'zero_past_amplified_energy_proxy': float(p @ birth_fraction),
                   'zero_past_world_probability': float(p @ birth.any(axis=1)),
                   'oracle_entropy_achieved': entropy_info['achieved'],
                   'oracle_entropy_requested': truth_A,
                   'oracle_entropy_kl': entropy_info['kl'],
                   'oracle_projection_achieved': projection_info['achieved'],
                   'oracle_projection_outside_hull_mass': projection_info['outside_hull_mass']}
        worlds, sensitivity = [], []
        for j, d in enumerate(donors):
            worlds.append({'donor_profile_id': int(d), 'donor_input_profile_id': int(past[j]),
                           'conditional_probability': float(p[j]), 'shape_probability': float(old_weight[j]),
                           'oracle_shape_probability': float(oracle_weight[j]),
                           'source_previous_shares': json.dumps(previous[j].tolist()),
                           'source_current_shares': json.dumps(current[j].tolist()),
                           'world_anchor': float(anchors[j]), 'zero_past_energy_proxy': float(birth_fraction[j]),
                           'flat_past_energy_proxy': float(missing[j]), 'flat_true_energy_proxy': float(missing_true[j])})
        for epsilon in [.001, .01, .1]:
            base = (1 - epsilon) * S + epsilon * f.prior
            fuel = base * current / ((1 - epsilon) * previous + epsilon * f.prior)
            values = (fuel @ EF) / fuel.sum(axis=1)
            mean = float(p @ values)
            sensitivity.append({'epsilon': epsilon, 'world_anchor_mean': mean,
                                'world_anchor_sd': float(np.sqrt(p @ (values - mean)**2)),
                                'world_anchor_abs_drift': float(p @ np.abs(values - anchors))})
        if set(laws) != set(METHODS):
            raise ValueError('missing diagnostic arm')
        return laws, context, worlds, sensitivity
