"""Exploratory budgeted posterior samplers, preserving frozen v1 algorithms.

Goal allocation is a research hypothesis: approximate each discovered component's
mass times predictive standard deviation, as suggested by stratified integration
variance. Laplace masses and local risk are approximations, never extra labels.
All adaptive SNIS estimates discard their pilot and use the complete frozen q.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from .inference import (
    Evaluator, InferenceResult, _evaluate, _logsumexp, _normal_log_density,
    _normalize, _standard_log_density, infer,
)
from .representations import scale_weights, transform_predictions

METHODS = (
    "prior", "sobol", "gaussian", "diverse", "local_mixture", "local_mixture_qmc",
    "goal_time", "goal_dct", "goal_haar", "goal_cdf_time", "goal_cdf_dct", "goal_cdf_haar", "smc",
)


@dataclass(frozen=True)
class InferenceV2Result(InferenceResult):
    diagnostics: dict[str, Any]


def _uniforms(count: int, dimension: int, rng: np.random.Generator, quasi: bool) -> np.ndarray:
    if quasi:
        engine = qmc.Sobol(dimension, scramble=True, seed=int(rng.integers(2**32)))
        values = engine.random_base2(int(np.ceil(np.log2(max(1, count)))))[:count]
    else:
        values = rng.random((count, dimension))
    return np.clip(values, np.finfo(float).eps, 1.0 - np.finfo(float).eps)


@dataclass(frozen=True)
class LocalMixture:
    means: np.ndarray
    covariances: np.ndarray
    masses: np.ndarray
    prior_weight: float = 0.25

    def log_density(self, z: np.ndarray) -> np.ndarray:
        components = [np.log(self.prior_weight) + _standard_log_density(z)]
        components.extend(
            np.log((1 - self.prior_weight) * mass) + _normal_log_density(z, mean, covariance)
            for mean, covariance, mass in zip(self.means, self.covariances, self.masses)
        )
        return _logsumexp(np.stack(components), axis=0)

    def sample(self, count: int, rng: np.random.Generator, quasi: bool) -> np.ndarray:
        uniforms = _uniforms(count, self.means.shape[1] + 1, rng, quasi)
        probabilities = np.r_[self.prior_weight, (1 - self.prior_weight) * self.masses]
        assignments = np.searchsorted(np.cumsum(probabilities), uniforms[:, 0], side="right")
        assignments = np.minimum(assignments, len(probabilities) - 1)
        z = ndtri(uniforms[:, 1:])
        for component, (mean, covariance) in enumerate(zip(self.means, self.covariances), start=1):
            chosen = assignments == component
            z[chosen] = z[chosen] @ np.linalg.cholesky(covariance).T + mean
        return z


def _cdf_risk(means: np.ndarray, variances: np.ndarray, masses: np.ndarray,
              thresholds: np.ndarray, utility: np.ndarray) -> np.ndarray:
    """Delta-Gaussian E_k[(1{Y<=t}-F(t))²] averaged over fixed quantiles.

    This approximates marginal CDF SNIS integration variance, not a bound on
    joint prediction error. Threshold values use model pilot predictions only.
    """
    standard_deviation = np.sqrt(np.maximum(variances, 0.0))[:, None, :]
    difference = thresholds[None, :, :] - means[:, None, :]
    probabilities = ndtr(difference / np.maximum(standard_deviation, 1e-14))
    probabilities = np.where(standard_deviation > 1e-14, probabilities, difference >= 0)
    mixture_cdf = np.einsum("k,ktj->tj", masses, probabilities)
    squared_error = (probabilities * (1.0 - mixture_cdf)**2
                     + (1.0 - probabilities) * mixture_cdf**2)
    return np.mean(squared_error * utility[None, None, :], axis=(1, 2))


def _fit_local(evaluate: Evaluator, dimension: int, budget: int, rng: np.random.Generator,
               representation: str | None, *, cdf_goal: bool = False) -> tuple[LocalMixture, int, dict[str, Any]]:
    pilot_count = max(8, budget // 4)
    # Select separated likelihood modes without reading evaluator internals.
    geometry_per_center = 2 * dimension + 2
    max_centers = min(4, max(1, (budget // 2 - pilot_count) // geometry_per_center))
    if pilot_count + geometry_per_center >= budget:
        raise ValueError("Budget too small for local geometry at this dimension.")
    z = ndtri(_uniforms(pilot_count, dimension, rng, True))
    ll, pilot_y = _evaluate(evaluate, z)
    centers = []
    for candidate in np.argsort(-ll):
        if not centers or np.min(np.linalg.norm(z[centers] - z[candidate], axis=1)) > 0.65:
            centers.append(int(candidate))
        if len(centers) == max_centers:
            break
    means, covariances, predictions, local_variances, log_masses = [], [], [], [], []
    output_variances = []
    h = 0.1
    used = pilot_count
    for center in centers:
        origin = z[center]
        shifts = h * np.eye(dimension)
        stencil = np.concatenate([origin[None, :] + shifts, origin[None, :] - shifts])
        stencil_ll, stencil_y = _evaluate(evaluate, stencil)
        used += len(stencil)
        gradient = (stencil_ll[:dimension] - stencil_ll[dimension:]) / (2 * h) - origin
        curvature = (stencil_ll[:dimension] - 2 * ll[center] + stencil_ll[dimension:]) / h**2
        precision = np.clip(1.0 - curvature, 1.0, 1e6)
        covariance = np.diag(1.0 / precision)
        delta = np.clip(gradient / precision, -0.5, 0.5)
        mean = origin + delta
        center_ll, center_y = _evaluate(evaluate, mean[None, :])
        used += 1
        derivative = (stencil_y[:dimension] - stencil_y[dimension:]) / (2 * h)
        if representation is not None:
            derivative = transform_predictions(derivative, representation, weighted=not cdf_goal)
        output_variances.append(np.sum(np.square(derivative) / precision[:, None], axis=0))
        variance = float(np.sum(np.square(derivative) / precision[:, None]) / derivative.shape[1])
        means.append(mean)
        covariances.append(covariance)
        predictions.append(center_y[0])
        local_variances.append(variance)
        log_masses.append(float(center_ll[0] + _standard_log_density(mean[None, :])[0]
                                - 0.5 * np.log(precision).sum()))
    posterior_mass = _normalize(np.asarray(log_masses))
    masses = posterior_mass.copy()
    risks = np.ones(len(means))
    if representation is not None:
        goals = transform_predictions(np.asarray(predictions), representation, weighted=not cdf_goal)
        if cdf_goal:
            pilot_goals = transform_predictions(pilot_y, representation, weighted=False)
            thresholds = np.quantile(pilot_goals, [0.1, 0.3, 0.5, 0.7, 0.9], axis=0)
            risks = _cdf_risk(goals, np.asarray(output_variances), posterior_mass,
                              thresholds, scale_weights(goals.shape[1], representation))
        else:
            goal_mean = posterior_mass @ goals
            risks = np.asarray(local_variances) + np.mean(np.square(goals - goal_mean), axis=1)
        # Regularization retains every discovered mode and prevents vanishing q.
        floor = max(float(posterior_mass @ risks) * 0.05, 1e-12)
        masses = posterior_mass * np.sqrt(risks + floor)
        masses = 0.9 * masses / masses.sum() + 0.1 * posterior_mass
    return LocalMixture(np.asarray(means), np.asarray(covariances), masses), used, {
        "geometry": "diagonal finite-difference negative log posterior curvature",
        "pilot_rows": pilot_count, "geometry_rows": used - pilot_count,
        "centers": len(means), "laplace_mass": posterior_mass.tolist(),
        "proposal_mass": masses.tolist(), "predictive_risk": risks.tolist(),
        "representation": representation, "prior_weight": 0.25,
        "risk_objective": "integrated marginal CDF SNIS variance approximation" if cdf_goal else "mean trajectory variance",
        "cdf_threshold_quantiles": [0.1, 0.3, 0.5, 0.7, 0.9] if cdf_goal else [],
    }


def _smc(evaluate: Evaluator, dimension: int, budget: int, seed: int) -> InferenceV2Result:
    started = perf_counter()
    rng = np.random.default_rng(seed)
    count = budget // 5
    if count < 4:
        raise ValueError("SMC needs budget >= 20.")
    z = rng.normal(size=(count, dimension))
    ll, y = _evaluate(evaluate, z)
    weights = np.full(count, 1 / count)
    ancestors = np.arange(count)
    beta = 0.0
    accepted = 0
    stage_ess = []
    for next_beta in (0.05, 0.2, 0.5, 1.0):
        weights = _normalize(np.log(np.maximum(weights, 1e-300)) + (next_beta - beta) * ll)
        stage_ess.append(float(1 / np.square(weights).sum()))
        # Multinomial resampling is conditionally unbiased; moves preserve pi_beta.
        selected = rng.choice(count, size=count, p=weights)
        z, ll, y, ancestors = z[selected], ll[selected], y[selected], ancestors[selected]
        weights = np.full(count, 1 / count)
        # A deterministic symmetric random-walk kernel, independent of this
        # particle population: adaptation to its own state would need care.
        scale = (0.8 * (1 - next_beta) + 0.08) / np.sqrt(dimension)
        proposed = z + scale * rng.normal(size=z.shape)
        proposal_ll, proposal_y = _evaluate(evaluate, proposed)
        ratio = (_standard_log_density(proposed) - _standard_log_density(z)
                 + next_beta * (proposal_ll - ll))
        move = np.log(rng.random(count)) < np.minimum(0, ratio)
        z[move], ll[move], y[move] = proposed[move], proposal_ll[move], proposal_y[move]
        accepted += int(move.sum())
        beta = next_beta
    return InferenceV2Result(z, weights, y, count * 5, float(count), perf_counter() - started,
                             ll, np.full(count, np.nan), 0, {
        "beta": beta, "stages": 4, "stage_ess": stage_ess,
        "pilot_rows": 0, "geometry_rows": 0, "final_rows": count,
        "initial_rows": count, "move_rows": 4 * count, "counted_rows": 5 * count,
        "acceptance_rate": accepted / (4 * count), "unique_ancestors": len(np.unique(ancestors)),
        "ess_warning": "Weight ESS does not measure genealogical dependence; log_proposal undefined for SMC.",
    })


def infer_v2(evaluate: Evaluator, dimension: int, budget: int, seed: int, method: str) -> InferenceV2Result:
    """Use at most budget evaluated rows, including pilots and local stencils."""
    if method not in METHODS:
        raise ValueError(f"Unknown inference method: {method}")
    if not isinstance(dimension, (int, np.integer)) or isinstance(dimension, bool) or dimension < 1:
        raise ValueError("dimension must be a positive integer.")
    if not isinstance(budget, (int, np.integer)) or isinstance(budget, bool) or budget < 1:
        raise ValueError("budget must be a positive integer.")
    if method in {"prior", "gaussian", "diverse"}:
        result = infer(evaluate, dimension, budget, seed, method)
        return InferenceV2Result(**result.__dict__, diagnostics={"implementation": "frozen v1",
            "pilot_rows": result.pilot_evaluations, "geometry_rows": 0,
            "final_rows": len(result.z), "counted_rows": result.evaluations})
    if method == "smc":
        return _smc(evaluate, dimension, budget, seed)
    started = perf_counter()
    rng = np.random.default_rng(seed)
    if method == "sobol":
        z = ndtri(_uniforms(budget, dimension, rng, True))
        log_q = _standard_log_density(z)
        used, diagnostics = 0, {"implementation": "scrambled Sobol standard-normal prior"}
    else:
        cdf_goal = method.startswith("goal_cdf_")
        representation = method.removeprefix("goal_cdf_" if cdf_goal else "goal_") if method.startswith("goal_") else None
        proposal, used, diagnostics = _fit_local(evaluate, dimension, budget, rng, representation, cdf_goal=cdf_goal)
        z = proposal.sample(budget - used, rng, method != "local_mixture")
        log_q = proposal.log_density(z)
    diagnostics.setdefault("pilot_rows", 0)
    diagnostics.setdefault("geometry_rows", 0)
    diagnostics["final_rows"] = len(z)
    diagnostics["counted_rows"] = used + len(z)
    ll, predictions = _evaluate(evaluate, z)
    weights = _normalize(_standard_log_density(z) + ll - log_q)
    return InferenceV2Result(z, weights, predictions, used + len(z), float(1 / np.square(weights).sum()),
                             perf_counter() - started, ll, log_q, used, diagnostics)
