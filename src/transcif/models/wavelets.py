"""Wavelet basis functions used to seed LT-MWKC's multi-wavelet convolution kernels."""

import math

import torch

MORLET_SUPPORT_RADIUS = 2 * math.pi
MEXICAN_HAT_SUPPORT_RADIUS = 4.0


def morlet_kernel(length: int, w0: float = 5.0) -> torch.Tensor:
    """Real-valued Morlet wavelet: cos(w0 * t) * exp(-t^2 / 2), L2-normalized."""
    t = torch.linspace(-MORLET_SUPPORT_RADIUS, MORLET_SUPPORT_RADIUS, steps=length)
    envelope = torch.exp(-t ** 2 / 2)
    kernel = envelope * torch.cos(w0 * t)
    return kernel / kernel.norm()


def mexican_hat_kernel(length: int) -> torch.Tensor:
    """Mexican hat (Ricker) wavelet: (1 - t^2) * exp(-t^2 / 2), L2-normalized."""
    t = torch.linspace(-MEXICAN_HAT_SUPPORT_RADIUS, MEXICAN_HAT_SUPPORT_RADIUS, steps=length)
    kernel = (1 - t ** 2) * torch.exp(-t ** 2 / 2)
    return kernel / kernel.norm()


def build_wavelet_bank(length: int) -> torch.Tensor:
    """Stack all supported wavelet families into a (num_families, length) tensor."""
    return torch.stack([morlet_kernel(length), mexican_hat_kernel(length)], dim=0)
