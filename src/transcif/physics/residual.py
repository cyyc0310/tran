"""Stage 2: learnable residual correction head Delta_t (Innovation 2), fit ONLY on the
target-domain calibration set to absorb systematic bias the pure physics formula misses
(cross-border trade, transmission losses, sub-fuel-mix differences)."""

import torch
import torch.nn as nn


class ResidualCorrectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def fit_residual_head(
    head: ResidualCorrectionHead,
    calibration_features: torch.Tensor,
    calibration_targets: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-2,
) -> ResidualCorrectionHead:
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = head(calibration_features)
        loss = loss_fn(predictions, calibration_targets)
        loss.backward()
        optimizer.step()
    return head
