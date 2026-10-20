import torch

from transcif.training.losses import energy_score


def test_energy_score_is_translation_invariant_for_float32_many_scenarios():
    generator = torch.Generator().manual_seed(5)
    samples = torch.randint(-100, 100, (2, 64, 6), generator=generator).float().requires_grad_()
    truth = torch.randint(-100, 100, (2, 6), generator=generator).float()
    score = energy_score(samples, truth)
    shifted = energy_score(samples + 1e6, truth + 1e6)
    torch.testing.assert_close(score, shifted, rtol=1e-6, atol=1e-4)
    score.backward()
    assert torch.isfinite(samples.grad).all()
