"""Budgeted, two-stage importance sampling under a standard normal prior.

The adaptive proposal is frozen after the pilot. Only second-stage samples
enter the estimate; pilot evaluations still count against the decoder budget.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

Evaluator = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class InferenceResult:
    z: np.ndarray
    weights: np.ndarray
    predictions: np.ndarray
    evaluations: int
    ess: float
    elapsed_seconds: float
    log_likelihood: np.ndarray
    log_proposal: np.ndarray
    pilot_evaluations: int


def _normalize(log_weights: np.ndarray) -> np.ndarray:
    if not np.isfinite(log_weights).all():
        raise ValueError("Importance log weights must all be finite.")
    weights = np.exp(log_weights - np.max(log_weights))
    return weights / weights.sum()


def _standard_log_density(z: np.ndarray) -> np.ndarray:
    return -0.5 * (z.shape[1] * np.log(2.0 * np.pi) + np.square(z).sum(axis=1))


def _normal_log_density(z: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    chol = np.linalg.cholesky(covariance)
    standardized = np.linalg.solve(chol, (z - mean).T).T
    return -0.5 * (
        z.shape[1] * np.log(2.0 * np.pi)
        + 2.0 * np.log(np.diag(chol)).sum()
        + np.square(standardized).sum(axis=1)
    )


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True)), axis=axis)


@dataclass(frozen=True)
class _Proposal:
    means: np.ndarray
    covariance: np.ndarray
    component_weights: np.ndarray
    prior_weight: float = 0.25

    def log_density(self, z: np.ndarray) -> np.ndarray:
        components = [np.log(self.prior_weight) + _standard_log_density(z)]
        components.extend(
            np.log((1.0 - self.prior_weight) * weight)
            + _normal_log_density(z, mean, self.covariance)
            for mean, weight in zip(self.means, self.component_weights)
        )
        return _logsumexp(np.stack(components), axis=0)

    def sample(self, rng: np.random.Generator, count: int) -> np.ndarray:
        probabilities = np.r_[self.prior_weight, (1.0 - self.prior_weight) * self.component_weights]
        components = rng.choice(len(probabilities), size=count, p=probabilities)
        z = rng.normal(size=(count, self.means.shape[1]))
        chol = np.linalg.cholesky(self.covariance)
        for component, mean in enumerate(self.means, start=1):
            selected = components == component
            z[selected] = z[selected] @ chol.T + mean
        return z


def _evaluate(evaluate: Evaluator, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log_likelihood, predictions = map(np.asarray, evaluate(z))
    if log_likelihood.shape != (len(z),):
        raise ValueError("Evaluator likelihood must have shape (particles,).")
    if predictions.ndim != 2 or predictions.shape[0] != len(z) or predictions.shape[1] == 0:
        raise ValueError("Evaluator predictions must have shape (particles, nonempty horizon).")
    if not np.isfinite(log_likelihood).all() or not np.isfinite(predictions).all():
        raise ValueError("Evaluator returned nonfinite likelihoods or predictions.")
    return log_likelihood.astype(float), predictions.astype(float)


def _fit_proposal(z: np.ndarray, weights: np.ndarray, predictions: np.ndarray, method: str) -> _Proposal:
    dimension = z.shape[1]
    if method == "gaussian":
        mean = weights @ z
        centered = z - mean
        covariance = (centered * weights[:, None]).T @ centered + 0.15 * np.eye(dimension)
        return _Proposal(mean[None, :], covariance, np.ones(1))

    # Greedy, posterior-mass-weighted coverage in the *predicted* trajectory.
    # These model predictions use no held-out target observations.
    centers = [int(np.argmax(weights))]
    distances = np.mean(np.square(predictions - predictions[centers[0]]), axis=1)
    for _ in range(min(4, len(z)) - 1):
        scores = weights * distances
        scores[centers] = 0.0
        if float(scores.max()) <= 1e-12:
            break
        center = int(np.argmax(scores))
        centers.append(center)
        distances = np.minimum(distances, np.mean(np.square(predictions - predictions[center]), axis=1))
    distance_to_centers = np.mean(np.square(predictions[:, None, :] - predictions[centers][None, :, :]), axis=2)
    assignments = np.argmin(distance_to_centers, axis=1)
    mass = np.bincount(assignments, weights=weights, minlength=len(centers))
    retained = mass > 0.0
    mass = mass[retained]
    return _Proposal(z[np.asarray(centers)[retained]], 0.7**2 * np.eye(dimension), mass / mass.sum())


def infer(evaluate: Evaluator, dimension: int, budget: int, seed: int, method: str) -> InferenceResult:
    """Return a self-normalized importance estimate and an auditable cost count.

    ``prior`` uses all ``budget`` draws. Adaptive methods use max(8, B//4)
    pilot draws, fit a defensive proposal, then spend the remaining budget.
    Every evaluated particle is charged once, including the discarded pilot.
    """
    if method not in {"prior", "gaussian", "diverse"}:
        raise ValueError(f"Unknown inference method: {method}")
    if not isinstance(dimension, (int, np.integer)) or dimension < 1:
        raise ValueError("dimension must be a positive integer.")
    if not isinstance(budget, (int, np.integer)) or budget < 1:
        raise ValueError("budget must be a positive integer.")
    pilot_count = 0 if method == "prior" else max(8, budget // 4)
    if pilot_count >= budget:
        raise ValueError("Adaptive inference needs budget >= 9.")
    started = perf_counter()
    rng = np.random.default_rng(seed)
    if method == "prior":
        z = rng.normal(size=(budget, dimension))
        log_proposal = _standard_log_density(z)
    else:
        pilot_z = rng.normal(size=(pilot_count, dimension))
        pilot_likelihood, pilot_predictions = _evaluate(evaluate, pilot_z)
        proposal = _fit_proposal(pilot_z, _normalize(pilot_likelihood), pilot_predictions, method)
        z = proposal.sample(rng, budget - pilot_count)
        log_proposal = proposal.log_density(z)
    log_likelihood, predictions = _evaluate(evaluate, z)
    weights = _normalize(_standard_log_density(z) + log_likelihood - log_proposal)
    return InferenceResult(
        z=z, weights=weights, predictions=predictions, evaluations=budget,
        ess=float(1.0 / np.square(weights).sum()), elapsed_seconds=perf_counter() - started,
        log_likelihood=log_likelihood, log_proposal=log_proposal, pilot_evaluations=pilot_count,
    )
