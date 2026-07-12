import torch
from transcif.models.cv_dwcc import gaussian_window_weights, local_weighted_r2_and_dominant


def test_gaussian_window_weights_sums_to_one_and_peaks_at_center():
    weights = gaussian_window_weights(window=11, bandwidth=3.0)
    assert weights.shape == (11,)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.argmax(weights).item() == 5


def test_local_weighted_r2_identifies_dominant_predictor():
    torch.manual_seed(0)
    batch, seq_len = 2, 80
    dominant_signal = torch.sin(torch.linspace(0, 8 * torch.pi, seq_len)).unsqueeze(0).repeat(batch, 1)
    noise_signal = torch.randn(batch, seq_len) * 5.0

    target = dominant_signal + 0.01 * torch.randn(batch, seq_len)
    predictors = torch.stack([dominant_signal, noise_signal], dim=-1)

    r2, dominant_idx = local_weighted_r2_and_dominant(target, predictors, window=25, bandwidth=6.0)

    assert r2.shape[0] == batch
    assert torch.all(r2 >= -1e-4) and torch.all(r2 <= 1.0 + 1e-4)
    center = r2.shape[1] // 2
    assert dominant_idx[0, center].item() == 0
    assert r2[0, center].item() > 0.9


def test_local_weighted_r2_valid_length():
    target = torch.randn(1, 60)
    predictors = torch.randn(1, 60, 3)
    r2, dominant_idx = local_weighted_r2_and_dominant(target, predictors, window=25, bandwidth=6.0)
    expected_len = 60 - 25 + 1
    assert r2.shape == (1, expected_len)
    assert dominant_idx.shape == (1, expected_len)
    assert torch.all(dominant_idx >= 0) and torch.all(dominant_idx < 3)


from transcif.models.cv_dwcc import CVDWCC


def test_cvdwcc_output_shapes():
    model = CVDWCC(num_variables=3, scales=((15, 3.0), (21, 5.0)), feature_dim=8)
    x = torch.randn(2, 60, 3)
    fused, dominant_idx = model(x)

    min_valid_len = 60 - 21 + 1
    assert fused.shape == (2, 8, 2, 3, min_valid_len)
    assert dominant_idx.shape == (2, 2, 3, min_valid_len)
    assert torch.all(dominant_idx >= 0) and torch.all(dominant_idx < 2)


def test_cvdwcc_gradients_flow_to_input():
    model = CVDWCC(num_variables=3, scales=((15, 3.0),), feature_dim=4)
    x = torch.randn(1, 40, 3, requires_grad=True)
    fused, _ = model(x)
    fused.pow(2).mean().backward()
    assert x.grad is not None
    assert torch.any(x.grad != 0)
