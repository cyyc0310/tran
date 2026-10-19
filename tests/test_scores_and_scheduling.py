import numpy as np
import pytest
import torch

from transcif.evaluation.metrics import ensemble_crps, forecast_metrics
from transcif.evaluation.scheduling import choose_block, scheduling_metrics
from transcif.training.losses import energy_score, fair_crps


def test_empirical_and_fair_scores_match_pairwise_definitions():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(3, 5, 4))
    truth = rng.normal(size=(3, 4))
    fit = np.abs(samples - truth[:, None]).mean()
    pairwise = np.abs(samples[:, :, None] - samples[:, None, :])
    empirical = fit - pairwise.mean() / 2
    fair = fit - pairwise.sum() / (2 * 3 * 5 * 4 * 4)
    assert ensemble_crps(truth, samples) == pytest.approx(empirical)
    tensor, label = torch.tensor(samples), torch.tensor(truth)
    assert fair_crps(tensor, label).item() == pytest.approx(fair)
    norm = np.linalg.norm(samples - truth[:, None], axis=-1).mean()
    distances = np.linalg.norm(samples[:, :, None] - samples[:, None, :], axis=-1)
    expected_energy = (norm - distances.sum() / (2 * 3 * 5 * 4)) / 2
    assert energy_score(tensor, label).item() == pytest.approx(expected_energy)


def test_coverage_is_empirical_and_energy_has_zero_degenerate_case():
    truth = np.ones((2, 4))
    draws = np.ones((2, 8, 4))
    metrics = forecast_metrics(truth, draws)
    assert metrics["crps"] == metrics["energy_score"] == 0
    assert metrics["pointwise_90_coverage"] == 1
    assert metrics["path_coverage_of_pointwise_90_band"] == 1
    tensor = torch.tensor(draws, requires_grad=True)
    loss = energy_score(tensor, torch.tensor(truth))
    loss.backward()
    assert torch.isfinite(tensor.grad).all()


def test_mean_and_cvar_make_different_decisions():
    # Hour 0 has low mean and rare high cost; hour 1 is safer.
    samples = np.tile([0., 3.], (10, 1))
    samples[-1, 0] = 20
    assert choose_block(samples, 1, "mean") == 0
    assert choose_block(samples, 1, "cvar") == 1
    truth = np.array([[20., 3.]])
    assert scheduling_metrics(truth, samples[None], 1, "cvar")["oracle_regret_g_per_kwh"] == 0


def test_fractional_discrete_cvar_tail():
    samples = np.array([[0., 5.], [4., 5.], [10., 5.]])
    # Tail mass 1.5: weighted tail at hour0=(10 + .5*4)/1.5=8.
    assert choose_block(samples, 1, "cvar", alpha=.5) == 1


@pytest.mark.parametrize("samples,truth", [
    (np.ones((2, 1, 3)), np.ones((2, 3))),
    (np.full((2, 4, 3), np.nan), np.ones((2, 3))),
    (np.ones((2, 4, 3)), np.ones((2, 2))),
])
def test_invalid_scores_fail(samples, truth):
    with pytest.raises(ValueError):
        ensemble_crps(truth, samples)
