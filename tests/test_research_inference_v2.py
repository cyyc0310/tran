"""Contract and probability checks independent of the benchmark case family."""

import numpy as np
import pytest

from transcif.research.inference_v2 import METHODS, LocalMixture, infer_v2
from transcif.research.representations import basis_matrix, prediction_metric, transform_predictions


class CountingGaussian:
    def __init__(self):
        self.rows = 0

    def __call__(self, z):
        self.rows += len(z)
        return -0.5 * ((z[:, 0] - 0.8) / 0.4) ** 2, np.tile(z[:, :1], (1, 24))


@pytest.mark.parametrize("method", METHODS)
def test_budget_and_seed(method):
    first_evaluator = CountingGaussian()
    first = infer_v2(first_evaluator, 3, 128, 22, method)
    second_evaluator = CountingGaussian()
    second = infer_v2(second_evaluator, 3, 128, 22, method)
    assert first.evaluations == first_evaluator.rows <= 128
    assert second.evaluations == second_evaluator.rows
    assert first.pilot_evaluations + len(first.z) <= first.evaluations
    assert np.array_equal(first.z, second.z)
    assert np.array_equal(first.weights, second.weights)
    assert np.isclose(first.weights.sum(), 1)
    assert np.isfinite(first.predictions).all()
    if method == "smc":
        assert first.diagnostics["beta"] == 1.0
        assert np.isnan(first.log_proposal).all()
    else:
        assert np.isfinite(first.log_proposal).all()


@pytest.mark.parametrize("method", ["sobol", "local_mixture", "local_mixture_qmc", "goal_haar", "smc"])
def test_known_posterior_and_unobserved_nuisance(method):
    result = infer_v2(CountingGaussian(), 3, 16384, 5, method)
    mean = result.weights @ result.z
    variance = result.weights @ (result.z - mean)**2
    assert mean[0] == pytest.approx(0.8 / 1.16, abs=0.06)
    assert variance[0] == pytest.approx(0.16 / 1.16, abs=0.05)
    assert np.max(np.abs(mean[1:])) < 0.13
    assert np.max(np.abs(variance[1:] - 1)) < 0.18


def test_full_mixture_density_matches_manual_and_integrates():
    proposal = LocalMixture(np.array([[-1.0], [1.5]]), np.array([[[0.2]], [[0.7]]]), np.array([0.3, 0.7]))
    z = np.linspace(-10, 10, 100001)[:, None]
    normal = lambda mu, var: np.exp(-(z[:, 0] - mu)**2 / (2 * var)) / np.sqrt(2 * np.pi * var)
    manual = 0.25 * normal(0, 1) + 0.75 * (0.3 * normal(-1, 0.2) + 0.7 * normal(1.5, 0.7))
    np.testing.assert_allclose(np.exp(proposal.log_density(z)), manual, rtol=2e-14, atol=1e-16)
    assert np.trapz(np.exp(proposal.log_density(z)), z[:, 0]) == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize("kind", ["time", "dct", "haar"])
@pytest.mark.parametrize("length", [1, 3, 24, 32])
def test_orthogonal_invariance_and_weighted_time_kernel(kind, length):
    rng = np.random.default_rng(20)
    y = rng.normal(size=(9, length))
    np.testing.assert_allclose(basis_matrix(length, kind) @ basis_matrix(length, kind).T, np.eye(length), atol=2e-14)
    coefficients = transform_predictions(y, kind, weighted=False)
    np.testing.assert_allclose(np.sum((coefficients[1:] - coefficients[0])**2, axis=1),
                               np.sum((y[1:] - y[0])**2, axis=1), atol=1e-12)
    difference = y[1:] - y[0]
    transformed = transform_predictions(difference, kind)
    metric = prediction_metric(length, kind)
    np.testing.assert_allclose(np.sum(transformed**2, axis=1),
                               np.einsum("ni,ij,nj->n", difference, metric, difference), atol=1e-12)


@pytest.mark.parametrize("dimension,budget,method", [(0, 128, "sobol"), (2, 0, "sobol"),
    (True, 128, "sobol"), (2, 128, "unknown"), (12, 20, "local_mixture"), (2, 19, "smc")])
def test_invalid_contract(dimension, budget, method):
    with pytest.raises(ValueError):
        infer_v2(CountingGaussian(), dimension, budget, 0, method)


def test_cdf_risk_matches_discrete_indicator_formula_and_zero_variance():
    from transcif.research.inference_v2 import _cdf_risk

    # Two point-mass components, threshold between them: F=0.8.
    means = np.array([[-1.0], [1.0]])
    masses = np.array([0.8, 0.2])
    risks = _cdf_risk(means, np.zeros_like(means), masses, np.array([[0.0]]), np.ones(1))
    np.testing.assert_allclose(risks, [0.2**2, 0.8**2])
    # Continuous symmetric Gaussians have P=F=1/2 at their common mean.
    risks = _cdf_risk(np.zeros_like(means), np.ones_like(means), masses, np.array([[0.0]]), np.array([2.0]))
    np.testing.assert_allclose(risks, [0.5, 0.5])


@pytest.mark.parametrize("kind", ["time", "dct", "haar"])
def test_cdf_goal_mass_is_finite_and_nontrivial(kind):
    def evaluate(z):
        likelihood = -0.5 * ((z[:, 0]**2 - 1) / 0.2)**2
        # Asymmetric trajectory risk, with unknown mode determined by pilot.
        prediction = np.tile((z[:, 0] + 0.3 * z[:, 0]**2)[:, None], (1, 24))
        return likelihood, prediction

    result = infer_v2(evaluate, 2, 256, 6, f"goal_cdf_{kind}")
    mass = np.asarray(result.diagnostics["proposal_mass"])
    assert np.isfinite(mass).all()
    assert np.all(mass > 0)
    assert np.isclose(mass.sum(), 1)
    assert result.diagnostics["cdf_threshold_quantiles"] == [0.1, 0.3, 0.5, 0.7, 0.9]
