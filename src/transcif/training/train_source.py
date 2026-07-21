"""Source-domain training loop: main RenewShare-prediction loss + Innovation 3's
consistency regularization, trained once on the single source region."""

import torch
import torch.nn as nn

from transcif.training.consistency import consistency_loss


def train_source_domain(
    encoder: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 150,
    lr: float = 5e-3,
    consistency_weight: float = 0.05,
) -> list:
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    mse_loss = torch.nn.MSELoss()
    losses = []

    for _ in range(epochs):
        optimizer.zero_grad()
        renew_share_pred, _ = encoder(x_train)
        main_loss = mse_loss(renew_share_pred, y_train)
        regularizer = consistency_loss(encoder, x_train)
        total_loss = main_loss + consistency_weight * regularizer
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())

    return losses
