import torch
from transcif.models.wavelets import morlet_kernel, mexican_hat_kernel, build_wavelet_bank


def test_morlet_kernel_shape_and_unit_norm():
    kernel = morlet_kernel(length=15)
    assert kernel.shape == (15,)
    assert torch.isclose(kernel.norm(), torch.tensor(1.0), atol=1e-5)


def test_mexican_hat_kernel_shape_and_unit_norm():
    kernel = mexican_hat_kernel(length=15)
    assert kernel.shape == (15,)
    assert torch.isclose(kernel.norm(), torch.tensor(1.0), atol=1e-5)


def test_mexican_hat_kernel_is_symmetric():
    kernel = mexican_hat_kernel(length=15)
    flipped = torch.flip(kernel, dims=[0])
    torch.testing.assert_close(kernel, flipped, atol=1e-5, rtol=1e-5)


def test_build_wavelet_bank_shape():
    bank = build_wavelet_bank(length=9)
    assert bank.shape == (2, 9)
