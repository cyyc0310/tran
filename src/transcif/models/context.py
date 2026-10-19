"""Forecast-horizon residual TCN conditioned on public configuration."""

import torch
from torch import nn


class ContextEncoder(nn.Module):
    """Gated dilated residual blocks, local field 31h plus full-horizon mean context.

    Symmetric convolutions see issued FUTURE covariates, never future labels.
    Configuration FiLM is shared across a horizon and does not use region IDs.
    """

    def __init__(self, features, hidden):
        super().__init__()
        self.input = nn.Linear(features, hidden)
        self.global_context = nn.Linear(hidden, hidden)
        self.blocks = nn.ModuleList([nn.ModuleDict({
            "norm": nn.LayerNorm(hidden),
            "conv": nn.Conv1d(hidden, 2 * hidden, 3, dilation=d, padding=d),
            "film": nn.Linear(21, 2 * hidden), "mix": nn.Linear(hidden, hidden)})
            for d in (1, 2, 4, 8)])
        self.output = nn.LayerNorm(hidden)

    def forward(self, features, config):
        x = self.input(features)
        x = x + self.global_context(x.mean(1, keepdim=True))
        for block in self.blocks:
            residual = block["norm"](x)
            value, gate = block["conv"](residual.transpose(1, 2)).transpose(1, 2).chunk(2, -1)
            scale, shift = block["film"](config).chunk(2, -1)
            residual = torch.tanh(value) * torch.sigmoid(gate)
            residual = residual * (1 + .1 * torch.tanh(scale[:, None])) + .1 * shift[:, None]
            x = x + block["mix"](residual)
        return self.output(x)
