"""Standard moment-protected controls for V172 physical-atom odds updates."""
from __future__ import annotations

import json

import numpy as np

from fuel_activity_forest import response_values
from moment_protection import MomentGeometry
from two_scale_zero import METHODS as ZERO_METHODS
from v151_hourly_law import EF
from v154_hourly_strong_baselines import CompressedLaw
from zero_odds_cif import ZeroOddsCIF, REFERENCES, odds_update

ZERO_INPUTS = ('direct20', 'direct43')
SETTINGS = {f'{short}_protect_{mode}_{zero}_a{int(alpha * 100)}': (reference, zero, alpha, mode)
            for short, reference in zip(('conditional', 'gaussian'), REFERENCES)
            for zero in ZERO_INPUTS for alpha in (.5, 1.) for mode in ('GE', 'fuel8')}
METHODS = REFERENCES + tuple(SETTINGS)


def protected_features(fuel, mode):
    aggregate = np.column_stack([fuel.sum(axis=1), fuel @ EF / 1000])
    if mode == 'GE':
        return aggregate
    if mode == 'fuel8':
        # Redundant aggregates are retained in the numerical certificate;
        # their affine redundancy is removed from the optimization basis.
        return np.column_stack([fuel, aggregate])
    raise ValueError('unknown moment protection')


class MomentProtectedCIF(ZeroOddsCIF):
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
                raise ValueError('positive physical source domain rejected')
            laws[name] = original + original
        for method in SETTINGS:
            laws[method] = []
        prepared = [self.frozen.activity.donor(int(d)) for d in donors]
        targets = {name: self.probabilities[self.lookup[q], ZERO_METHODS.index(name)] for name in ZERO_INPUTS}
        row = self.frozen.data.index.iloc[q]
        states = self.frozen.data.states[int(row.offset):int(row.stop)]
        if states.max() != 47:
            raise ValueError('48-state scoring occupancy contract')
        occupancy = np.bincount(states, minlength=48) / len(states)
        diagnostics = {method: {'projection_calls': 0, 'projection_fallbacks': 0, 'projection_iterations': 0,
                                'max_moment_error': 0., 'max_generation_error': 0., 'max_emissions_error': 0.,
                                'mean_kl': 0., 'mean_projection_kl': 0., 'mean_ess_ratio': 0.,
                                **{key: np.zeros(8) for key in ('prior_zero', 'target_zero', 'achieved_zero', 'fuel_before', 'fuel_after')}}
                       for method in SETTINGS}
        certificates, cache = [], {}
        for state in range(48):
            h = state % 24
            if h not in cache:
                activity = np.concatenate([d[h] for d in prepared])
                values, positive = response_values(base, activity)
                if not positive.all():
                    raise ValueError('source zero total generation')
                fuel = activity * base
                weights = {name: np.concatenate([np.full(len(d[h]), p / len(d[h])) for d, p in zip(prepared, probabilities)])
                           for name, probabilities in priors.items()}
                geometry = {(reference, mode): MomentGeometry(p, protected_features(fuel, mode), float(p @ fuel.sum(axis=1)))
                            for reference, p in weights.items() for mode in ('GE', 'fuel8')}
                cache[h] = (activity, fuel, values, weights, geometry)
            activity, fuel, values, weights, geometry = cache[h]
            zero = activity == 0
            proposed_cache = {}
            for method, (reference, zero_model, alpha, mode) in SETTINGS.items():
                key = (reference, zero_model, alpha)
                if key not in proposed_cache:
                    proposed_cache[key] = odds_update(weights[reference], zero, targets[zero_model][state], alpha)
                proposed = proposed_cache[key]
                g = geometry[(reference, mode)]
                posterior, certificate = g.project(proposed)
                before, after = g.prior @ fuel, posterior @ fuel
                denominator = before.sum()
                generation_error = float(abs(after.sum() - denominator) / denominator)
                emissions_error = float(abs((after - before) @ EF / 1000) / denominator)
                if max(generation_error, emissions_error) > 2e-10:
                    raise ValueError('protected aggregate moment certificate failed')
                laws[method].append(CompressedLaw(values, posterior))
                certificates.append({'state': state, 'method': method, **certificate,
                                     'generation_error': generation_error, 'emissions_error': emissions_error})
                info, mass = diagnostics[method], occupancy[state]
                info['projection_calls'] += 1
                info['projection_fallbacks'] += int(certificate['used_prior_fallback'])
                info['projection_iterations'] += int(certificate['iterations'])
                for key, value in [('max_moment_error', certificate['moment_error']),
                                   ('max_generation_error', generation_error), ('max_emissions_error', emissions_error)]:
                    info[key] = max(info[key], value)
                info['mean_kl'] += mass * max(0., float(posterior @ (np.log(posterior) - np.log(g.prior))))
                info['mean_projection_kl'] += mass * max(0., float(posterior @ (np.log(posterior) - np.log(proposed))))
                info['mean_ess_ratio'] += mass * float((g.prior @ g.prior) / (posterior @ posterior))
                for key, value in [('prior_zero', g.prior @ zero), ('target_zero', targets[zero_model][state]),
                                    ('achieved_zero', posterior @ zero), ('fuel_before', before), ('fuel_after', after)]:
                    info[key] += mass * value
        records = []
        for method, values in diagnostics.items():
            records.append({'method': method, **{key: json.dumps(value.tolist()) if isinstance(value, np.ndarray) else value
                                                  for key, value in values.items()}})
        return laws, forecast, records, certificates
