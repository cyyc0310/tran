import numpy as np
import pytest
import torch

from transcif.evaluation.metrics import source_validation_metrics
from transcif.training.losses import energy_score


def test_source_selection_preserves_macro_weighting_and_explicit_point():
    truth = np.zeros((3, 2))
    samples = np.repeat(np.array([[0., 0.], [6., 6.], [6., 6.]])[:, None], 2, axis=1)
    point = np.array([[2., 2.], [4., 4.], [4., 4.]])
    pooled, macro = source_validation_metrics(
        truth, {"cif": samples, "point_cif": point}, ["small", "large", "large"])
    assert pooled == pytest.approx({"mae": 10 / 3, "rmse": 12 ** .5, "crps": 4.})
    assert macro == pytest.approx({"mae": 3., "rmse": 3., "crps": 3.})


def test_energy_score_is_translation_invariant_for_float32_many_scenarios():
    generator = torch.Generator().manual_seed(5)
    samples = torch.randint(-100, 100, (2, 64, 6), generator=generator).float().requires_grad_()
    truth = torch.randint(-100, 100, (2, 6), generator=generator).float()
    score = energy_score(samples, truth)
    shifted = energy_score(samples + 1e6, truth + 1e6)
    torch.testing.assert_close(score, shifted, rtol=1e-6, atol=1e-4)
    score.backward()
    assert torch.isfinite(samples.grad).all()
