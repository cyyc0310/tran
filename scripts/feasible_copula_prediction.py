"""Change only proper-source within-hour coupling, keeping fuel margins fixed."""
from __future__ import annotations

import numpy as np

from feasible_fuel_copula import FeasibleCopulaCache
from fuel_activity_forest import activity_law
from scenario_reweighting import ScenarioPrediction
from v151_hourly_law import EF
from v154_hourly_strong_baselines import CompressedLaw

REFERENCES = ('conditional_activity_recomposition', 'world_gaussian')
SETTINGS = {
    f'{mode}_swap{int(alpha * 100)}_seed{seed}': (reference, seed, alpha)
    for mode, reference in zip(('conditional', 'gaussian'), REFERENCES)
    for seed in (41, 83) for alpha in (.5, 1.)
}
METHODS = REFERENCES + tuple(SETTINGS)


class FeasibleCopulaPrediction(ScenarioPrediction):
    """An admissible shuffle control, without an independence or novelty claim."""

    def __init__(self, frozen, coarse):
        super().__init__(frozen, coarse)
        self.shuffle = FeasibleCopulaCache()
        self.source_records = {}

    def shuffled(self, profile, hour, seed):
        old = self.frozen.activity.donor(profile)[hour]
        values, metadata = self.shuffle.transform(old, profile, hour, seed)
        key = (profile, hour, seed)
        if key not in self.source_records:
            relative = float(np.max(abs(values.mean(axis=0) - old.mean(axis=0))
                                    / np.maximum(1., abs(old.mean(axis=0)))))
            if relative > 1e-12:
                raise ValueError('fuel expectation changed')
            old_centered = old - old.mean(axis=0)
            new_centered = values - values.mean(axis=0)
            covariance_delta = new_centered.T @ new_centered / len(values) - old_centered.T @ old_centered / len(old)
            self.source_records[key] = {
                'source_profile_id': profile, 'hour': hour, 'seed': seed, **metadata,
                'max_relative_mean_error': relative,
                'covariance_change_frobenius': float(np.linalg.norm(covariance_delta)),
            }
        return values

    def build(self, q):
        forecast, _, base, donors, prior, anchors = self.inputs(q)
        if (base <= 0).any():
            raise ValueError('positive carrier required for support preservation')
        logits = np.log(prior) - .5 * ((anchors - forecast['anchor']) / forecast['sigma'])**2
        gaussian = np.exp(np.maximum(logits - logits.max(), -700))
        gaussian /= gaussian.sum()
        probabilities = dict(zip(REFERENCES, (prior, gaussian)))
        laws = {}
        for reference, probability in probabilities.items():
            laws[reference], rejected = self.frozen.activity.laws(donors, probability, base)
            if max(rejected) != 0:
                raise ValueError('original law rejected positive source rows')
        # The zero-shuffle branch is exactly the original object, never a replica.
        for seed in (41, 83):
            cells = {reference: [] for reference in REFERENCES}
            for hour in range(24):
                prepared = [self.shuffled(int(d), hour, seed) for d in donors]
                activity = np.concatenate(prepared)
                for reference, probability in probabilities.items():
                    weights = np.concatenate([np.full(len(a), w / len(a))
                                              for a, w in zip(prepared, probability)])
                    law, rejected = activity_law(base, activity, weights)
                    if rejected != 0:
                        raise ValueError('shuffle changed positive domain')
                    cells[reference].append(law)
            for method, (reference, method_seed, alpha) in SETTINGS.items():
                if method_seed != seed:
                    continue
                if alpha == 1.:
                    laws[method] = cells[reference]
                else:
                    laws[method] = [CompressedLaw(np.r_[a.x, b.x], np.r_[(1 - alpha) * a.w, alpha * b.w])
                                    for a, b in zip(laws[reference], cells[reference])]
        if set(laws) != set(METHODS):
            raise ValueError('incomplete predeclared coupling controls')
        details = [{'source_profile_id': int(d), 'conditional_weight': float(p),
                    'gaussian_weight': float(g), 'world_anchor': float(a)}
                   for d, p, g, a in zip(donors, prior, gaussian, anchors)]
        return laws, forecast, details
