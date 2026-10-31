"""Ordinary tempered odds updates of existing physical CIF atoms."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from coarse_distribution_lifting import CoarseLawForecast
from fuel_activity_forest import response_values
from scenario_reweighting import ScenarioPrediction
from two_scale_zero import METHODS as ZERO_METHODS
from v154_hourly_strong_baselines import CompressedLaw

REFERENCES = ('conditional_activity_recomposition', 'world_gaussian')
ZERO_INPUTS = ('direct20', 'direct43', 'month_history_conditional', 'month_persistence_conditional')
SETTINGS = {f'{short}_odds_{zero}_a{int(alpha * 100)}': (reference, zero, alpha)
            for short, reference in zip(('conditional', 'gaussian'), REFERENCES)
            for zero in ZERO_INPUTS for alpha in (.5, 1.)}
METHODS = REFERENCES + tuple(SETTINGS)


def odds_update(prior, zero, target, alpha):
    """One-pass marginal odds product; correlated margins need not match target."""
    prior, target = np.asarray(prior, float), np.asarray(target, float)
    zero = np.asarray(zero, bool)
    if (prior.ndim != 1 or zero.shape != (len(prior), 8) or target.shape != (8,)
            or not np.isfinite(prior).all() or (prior <= 0).any() or abs(prior.sum() - 1) > 1e-12
            or not np.isfinite(target).all() or (target < 0).any() or (target > 1).any()
            or alpha not in (0., .5, 1.)):
        raise ValueError('invalid odds update')
    if alpha == 0:
        return prior
    varying = zero.any(axis=0) & (~zero).any(axis=0)
    if not varying.any():
        return prior
    # Compute both classes directly: 1-pzero loses a rare positive class.
    pzero = prior @ zero[:, varying]
    ppositive = prior @ (~zero[:, varying])
    tzero = .95 * target[varying] + .05 * pzero
    tpositive = .95 * (1 - target[varying]) + .05 * ppositive
    log_zero = np.log(tzero) - np.log(pzero)
    log_positive = np.log(tpositive) - np.log(ppositive)
    adjustment = zero[:, varying] @ (log_zero - log_positive) + log_positive.sum()
    logits = np.log(prior) + alpha * adjustment
    posterior = np.exp(np.maximum(logits - logits.max(), -700))
    posterior /= posterior.sum()
    if not np.isfinite(posterior).all() or (posterior <= 0).any():
        raise ValueError('invalid positive posterior')
    return posterior


class ZeroOddsCIF(ScenarioPrediction):
    def __init__(self, frozen, monthly, zero_archive: Path):
        super().__init__(frozen, CoarseLawForecast(monthly, frozen.fit))
        saved = json.loads((zero_archive / 'summary.json').read_text())
        if saved['proper_regions'] != frozen.fit:
            raise ValueError('zero model proper cohort differs')
        with np.load(zero_archive / 'probabilities.npz', allow_pickle=False) as packed:
            self.profiles = packed['profile_id'].copy()
            self.probabilities = packed['hourly'].copy()
        if (not np.array_equal(self.profiles, frozen.queries.profile_id.to_numpy(int))
                or self.probabilities.shape != (len(self.profiles), len(ZERO_METHODS), 48, 8)):
            raise ValueError('zero forecast identity/dimensions')
        self.lookup = {int(q): i for i, q in enumerate(self.profiles)}

    def build(self, q):
        forecast, _, base, donors, prior, anchors = self.inputs(q)
        logits = np.log(prior) - .5 * ((anchors - forecast['anchor']) / forecast['sigma'])**2
        gaussian = np.exp(np.maximum(logits - logits.max(), -700))
        gaussian /= gaussian.sum()
        priors = dict(zip(REFERENCES, (prior, gaussian)))
        laws = {}
        for name, weights in priors.items():
            original, rejected = self.frozen.activity.laws(donors, weights, base)
            if max(rejected) != 0:
                raise ValueError('original positive source domain rejected')
            laws[name] = original + original
        for method in SETTINGS:
            laws[method] = []
        prepared = [self.frozen.activity.donor(int(d)) for d in donors]
        targets = {name: self.probabilities[self.lookup[q], ZERO_METHODS.index(name)] for name in ZERO_INPUTS}
        row = self.frozen.data.index.iloc[q]
        states = self.frozen.data.states[int(row.offset):int(row.stop)]
        if states.max() != 47:
            raise ValueError('48-state frozen scorer occupancy contract not met')
        occupancy = np.bincount(states, minlength=48) / len(states)
        # Occupancy is used for diagnostics only; every law is constructed for all states.
        diagnostic = {name: {'mean_kl': 0., 'mean_ess_ratio': 0., 'min_ess_ratio': float('inf'),
                              'mean_unreachable_zero_difference': 0.,
                              **{key: np.zeros(8) for key in ('prior_zero', 'target_zero', 'achieved_zero', 'fuel_before', 'fuel_after')}}
                      for name in SETTINGS}
        cache = {}
        for state in range(48):
            h = state % 24
            if h not in cache:
                activity = np.concatenate([d[h] for d in prepared])
                values, positive = response_values(base, activity)
                if not positive.all():
                    raise ValueError('source atom outside positive generation domain')
                weights = {name: np.concatenate([np.full(len(d[h]), p / len(d[h])) for d, p in zip(prepared, world_probability)])
                           for name, world_probability in priors.items()}
                cache[h] = (activity, values, weights)
            activity, values, weights = cache[h]
            zero = activity == 0
            fuel = activity * base
            varying = zero.any(axis=0) & (~zero).any(axis=0)
            for method, (reference, zero_model, alpha) in SETTINGS.items():
                probability = weights[reference]
                target = targets[zero_model][state]
                posterior = odds_update(probability, zero, target, alpha)
                laws[method].append(CompressedLaw(values, posterior))
                old_zero, new_zero = probability @ zero, posterior @ zero
                ess_ratio = float((probability @ probability) / (posterior @ posterior))
                info, mass = diagnostic[method], occupancy[state]
                info['mean_kl'] += mass * max(0., float(posterior @ (np.log(posterior) - np.log(probability))))
                info['mean_ess_ratio'] += mass * ess_ratio
                info['min_ess_ratio'] = min(info['min_ess_ratio'], ess_ratio)
                info['mean_unreachable_zero_difference'] += mass * float(abs(target[~varying] - old_zero[~varying]).sum())
                for key, value in [('prior_zero', old_zero), ('target_zero', target), ('achieved_zero', new_zero),
                                    ('fuel_before', probability @ fuel), ('fuel_after', posterior @ fuel)]:
                    info[key] += mass * value
        records = []
        for name, info in diagnostic.items():
            record = {'method': name}
            for key, value in info.items():
                record[key] = json.dumps(value.tolist()) if isinstance(value, np.ndarray) else float(value)
            records.append(record)
        return laws, forecast, records
