import numpy as np
import pytest

from transcif.research.inference import _Proposal, _standard_log_density, infer
from transcif.research.toy import ToyMechanism
from transcif.research.weighted_metrics import PreparedDistribution, distribution_metrics, weighted_quantile


@pytest.mark.parametrize("method", ["prior", "gaussian", "diverse"])
def test_budget_weights_and_reproducibility(method):
    evaluated = []

    def evaluate(z):
        evaluated.append(len(z))
        return -0.5 * np.square((z[:, 0] - 0.8) / 0.6), z[:, :1]

    first = infer(evaluate, 2, 128, 42, method)
    assert sum(evaluated) == first.evaluations == 128
    assert len(first.weights) + first.pilot_evaluations == 128
    expected = _standard_log_density(first.z) + first.log_likelihood - first.log_proposal
    expected = np.exp(expected - expected.max())
    np.testing.assert_allclose(first.weights, expected / expected.sum())
    assert first.ess == pytest.approx(1 / np.square(first.weights).sum())
    second = infer(evaluate, 2, 128, 42, method)
    np.testing.assert_array_equal(first.z, second.z)
    np.testing.assert_array_equal(first.weights, second.weights)


def test_defensive_mixture_density_and_samples():
    proposal = _Proposal(np.array([[-1.2], [1.7]]), np.array([[0.49]]), np.array([0.35, 0.65]))
    axis = np.linspace(-8, 8, 50001)[:, None]
    density = np.exp(proposal.log_density(axis))
    assert np.trapz(density, axis[:, 0]) == pytest.approx(1.0, abs=1e-10)
    draws = proposal.sample(np.random.default_rng(41), 100000)
    expected_mean = 0.75 * (-1.2 * 0.35 + 1.7 * 0.65)
    assert draws.mean() == pytest.approx(expected_mean, abs=0.02)
    assert np.all(proposal.log_density(axis) >= np.log(0.25) + _standard_log_density(axis) - 1e-12)


@pytest.mark.parametrize("method", ["prior", "gaussian", "diverse"])
def test_weighted_posterior_matches_analytic_normal(method):
    def evaluate(z):
        return -0.5 * np.square((z[:, 0] - 1.2) / 0.5), z[:, :1]

    result = infer(evaluate, 1, 30000, 101, method)
    mean = result.weights @ result.z[:, 0]
    variance = result.weights @ np.square(result.z[:, 0] - mean)
    assert mean == pytest.approx(1.2 / 1.25, abs=0.02)
    assert variance == pytest.approx(0.25 / 1.25, abs=0.02)


def test_metrics_match_explicit_crps_and_ignore_zero_mass():
    predictions = np.array([[0.0, 1.0], [2.0, 3.0], [7.0, 4.0]])
    weights = np.array([0.2, 0.5, 0.3])
    truth = np.array([1.5, 2.0])
    pair_distance = np.abs(predictions[:, None] - predictions[None, :])
    expected = (weights[:, None] * np.abs(predictions - truth)).sum(axis=0) - 0.5 * (weights[:, None, None] * weights[None, :, None] * pair_distance).sum(axis=(0, 1))
    assert distribution_metrics(predictions, weights, truth)["crps"] == pytest.approx(expected.mean())
    np.testing.assert_array_equal(weighted_quantile(np.array([-100.0, 2.0, 5.0]), np.array([0.0, 0.8, 0.2]), np.array([0.0, 0.5, 1.0])), [2.0, 2.0, 5.0])


def test_prepared_distribution_preserves_scores_and_isolates_inputs():
    predictions = np.array([[0.0, 1.0], [2.0, 3.0], [7.0, 4.0]])
    weights = np.array([0.2, 0.5, 0.3])
    prepared = PreparedDistribution(predictions, weights)
    for truth in (np.array([1.5, 2.0]), np.array([5.0, 4.0])):
        mean = weights @ predictions
        residual = mean - truth
        lower, upper = weighted_quantile(predictions, weights, np.array([0.05, 0.95]))
        pair_distance = np.abs(predictions[:, None] - predictions[None, :])
        crps = (weights[:, None] * np.abs(predictions - truth)).sum(axis=0) - 0.5 * (weights[:, None, None] * weights[None, :, None] * pair_distance).sum(axis=(0, 1))
        expected = {
            "mae": float(np.abs(residual).mean()),
            "rmse": float(np.sqrt(np.square(residual).mean())),
            "crps": float(crps.mean()),
            "coverage90": float(np.mean((truth >= lower) & (truth <= upper))),
            "width90": float(np.mean(upper - lower)),
            "shape_mae": float(np.abs((mean - mean.mean()) - (truth - truth.mean())).mean()),
            "level_mae": float(abs(mean.mean() - truth.mean())),
        }
        assert prepared.score(truth) == pytest.approx(expected)
        assert prepared.score(truth) == distribution_metrics(predictions, weights, truth)
    before = prepared.score(np.array([1.5, 2.0]))
    predictions[:] = -100.0
    weights[:] = 0.0
    assert prepared.score(np.array([1.5, 2.0])) == before
    with pytest.raises(ValueError, match="Truth"):
        prepared.score(np.array([np.nan, 0.0]))


def test_toy_ambiguity_and_nuisance_invariance():
    models = {case: ToyMechanism(case) for case in ("harmful", "harmless", "resolved")}
    widths = {}
    for case, model in models.items():
        first = model.evaluate(np.array([[1.6, 0.0], [-1.6, 0.0]]))[1]
        shifted = model.evaluate(np.array([[1.6, -100.0], [-1.6, 100.0]]))[1]
        np.testing.assert_allclose(first, shifted, atol=1e-11)
        reference = model.reference()
        lower, upper = weighted_quantile(reference["predictions"], reference["weights"], np.array([0.05, 0.95]))
        widths[case] = (upper - lower).mean()
        if case == "harmless":
            np.testing.assert_allclose(first[0], first[1])
        if case == "resolved":
            assert reference["weights"][reference["z"][:, 0] < 0].sum() < 1e-10
    assert widths["harmful"] > 3 * widths["harmless"]
    assert widths["resolved"] < widths["harmful"] / 2


def test_invalid_evaluations_fail_explicitly():
    with pytest.raises(ValueError, match="nonfinite"):
        infer(lambda z: (np.full(len(z), np.nan), z), 1, 32, 0, "prior")
    with pytest.raises(ValueError, match="budget"):
        infer(ToyMechanism("harmful").evaluate, 2, 8, 0, "diverse")
    with pytest.raises(ValueError, match="Unknown"):
        infer(ToyMechanism("harmful").evaluate, 2, 32, 0, "mystery")
def test_toy_observation_is_an_explicit_generation_aggregate():
    from transcif.research.toy import ToyMechanism
    model = ToyMechanism("harmful")
    z = np.array([[-1.6, 0], [1.6, 4]])
    observed = model.historical_observations(z)
    np.testing.assert_allclose(observed["normalized_daily_total_minus_one"], z[:, 0] ** 2)
    np.testing.assert_allclose(observed["peak_contrast"], z[:, 0])
    np.testing.assert_allclose(observed["daily_fuel_mwh"][0], observed["daily_fuel_mwh"][1])
    assert np.mean(np.abs(np.diff(model.evaluate(z)[1], axis=0))) > 100

