"""Stage 1: Local-Temporal Multi-Wavelet Kernel Convolution module (LT-MWKC)."""

import torch
import torch.nn as nn

from transcif.models.wavelets import build_wavelet_bank


class MultiWaveletConv1D(nn.Module):
    """Applies one 1D conv per wavelet family (seeded from the wavelet shape) at a fixed
    kernel size, then fuses the family outputs with a learnable softmax-weighted sum
    (Eq. F'^(k) = sum_m alpha_m * (Psi_m * X^tr))."""

    def __init__(self, in_channels: int, kernel_size: int, out_channels_per_wavelet: int = 4):
        super().__init__()
        wavelet_bank = build_wavelet_bank(kernel_size)
        self.num_wavelets = wavelet_bank.shape[0]
        self.kernel_size = kernel_size

        self.convs = nn.ModuleList()
        for wavelet_idx in range(self.num_wavelets):
            conv = nn.Conv1d(
                in_channels,
                in_channels * out_channels_per_wavelet,
                kernel_size,
                stride=1,
                padding=0,
                groups=in_channels,
                bias=False,
            )
            init_kernel = (
                wavelet_bank[wavelet_idx]
                .view(1, 1, kernel_size)
                .repeat(in_channels * out_channels_per_wavelet, 1, 1)
            )
            with torch.no_grad():
                conv.weight.copy_(init_kernel)
            self.convs.append(conv)

        self.alpha = nn.Parameter(torch.ones(self.num_wavelets) / self.num_wavelets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [conv(x) for conv in self.convs]
        weights = torch.softmax(self.alpha, dim=0)
        fused = sum(weight * output for weight, output in zip(weights, branch_outputs))
        return fused


class LTMWKC(nn.Module):
    """Runs MultiWaveletConv1D at several kernel sizes in parallel, truncates all branches
    to the shortest output length, fuses them with a learnable softmax-weighted sum
    (mirroring MultiWaveletConv1D's intra-branch fusion), and projects to a fixed feature
    dimension."""

    def __init__(
        self,
        in_channels: int,
        kernel_sizes: tuple = (2, 3, 5, 7),
        out_channels_per_wavelet: int = 4,
        feature_dim: int = 32,
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.branches = nn.ModuleList(
            [
                MultiWaveletConv1D(in_channels, kernel_size, out_channels_per_wavelet)
                for kernel_size in kernel_sizes
            ]
        )
        branch_channels = in_channels * out_channels_per_wavelet
        self.project = nn.Conv1d(branch_channels, feature_dim, kernel_size=1)
        self.branch_alpha = nn.Parameter(torch.ones(len(kernel_sizes)) / len(kernel_sizes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        min_len = x.shape[-1] - max(self.kernel_sizes) + 1
        branch_outputs = [branch(x)[..., :min_len] for branch in self.branches]
        weights = torch.softmax(self.branch_alpha, dim=0)
        fused = sum(weight * output for weight, output in zip(weights, branch_outputs))
        return self.project(fused)
