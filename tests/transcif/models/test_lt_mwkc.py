import torch
from transcif.models.lt_mwkc import MultiWaveletConv1D, LTMWKC


def test_multi_wavelet_conv1d_output_shape():
    conv = MultiWaveletConv1D(in_channels=3, kernel_size=5, out_channels_per_wavelet=4)
    x = torch.randn(2, 3, 50)
    out = conv(x)
    assert out.shape == (2, 3 * 4, 50 - 5 + 1)


def test_multi_wavelet_conv1d_alpha_softmax_sums_to_one():
    conv = MultiWaveletConv1D(in_channels=3, kernel_size=5)
    weights = torch.softmax(conv.alpha, dim=0)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_multi_wavelet_conv1d_gradients_flow_to_alpha():
    conv = MultiWaveletConv1D(in_channels=2, kernel_size=3)
    x = torch.randn(4, 2, 20, requires_grad=False)
    out = conv(x)
    loss = out.pow(2).mean()
    loss.backward()
    assert conv.alpha.grad is not None
    assert torch.any(conv.alpha.grad != 0)


def test_lt_mwkc_output_shape():
    model = LTMWKC(in_channels=3, kernel_sizes=(2, 3, 5, 7), out_channels_per_wavelet=4, feature_dim=32)
    x = torch.randn(2, 3, 48)
    out = model(x)
    expected_len = 48 - max((2, 3, 5, 7)) + 1
    assert out.shape == (2, 32, expected_len)


def test_lt_mwkc_output_is_not_constant_across_batch():
    model = LTMWKC(in_channels=3, feature_dim=16)
    x = torch.randn(2, 3, 30)
    out = model(x)
    assert not torch.allclose(out[0], out[1])
