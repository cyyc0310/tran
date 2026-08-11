"""Adversarial-persistence loss (Task 8.2).

Explicitly rewards beating the persistence baseline by a relative margin.
Formula (per window):

    L_w = ReLU(pred_mae_w - persistence_mae_w * (1 - margin))
    L   = mean(L_w)

A *relative* margin (not absolute) is used so the bar scales with the
baseline difficulty:

  - In regions where persistence is already small (e.g. QLD1 ~33 gCO2/kWh),
    the model only needs to be marginally better in absolute terms.
  - In regions where persistence is huge (e.g. VIC1 ~117 gCO2/kWh), the
    model has more absolute room but the same relative threshold.

Margin > 0 makes the loss exactly zero once the model beats persistence by
the required relative amount, so the model is not pushed to over-optimize
past the bar. The asymmetry (no penalty for being much better, linear
penalty for being worse) keeps gradients stable and avoids the model
collapsing onto persistence.

Tensors may be 1-D ``(N,)`` (already per-window MAE) or 2-D ``(N, H)``
(raw per-horizon errors). 2-D inputs are reduced to per-window MAE along
axis=1 before the margin check.
"""

from __future__ import annotations

import torch


def adversarial_persistence_loss(
    pred: torch.Tensor,
    persistence: torch.Tensor,
    margin: float = 0.10,
) -> torch.Tensor:
    """Compute the adversarial-persistence hinge loss.

    Args:
        pred: Model predictions. Shape ``(N,)`` for per-window MAE or
            ``(N, H)`` for per-horizon predictions.
        persistence: Persistence baseline. Same shape as ``pred``.
        margin: Relative margin the model must beat persistence by.
            ``margin=0.10`` means the model's per-window MAE must be
            below ``persistence_mae * 0.90`` for the loss to vanish on
            that window.

    Returns:
        Scalar loss tensor (mean over windows).

    Raises:
        ValueError: If shapes do not match or ``margin`` is out of [0, 1).
    """
    if pred.shape != persistence.shape:
        raise ValueError(
            f"Shape mismatch: pred {tuple(pred.shape)} vs "
            f"persistence {tuple(persistence.shape)}"
        )
    if not (0.0 <= margin < 1.0):
        raise ValueError(f"margin must be in [0, 1), got {margin}")

    pred_mae = pred.abs().mean(dim=-1) if pred.dim() >= 2 else pred.abs()
    persistence_mae = (
        persistence.abs().mean(dim=-1)
        if persistence.dim() >= 2
        else persistence.abs()
    )

    threshold = persistence_mae * (1.0 - margin)
    hinge = torch.relu(pred_mae - threshold)
    return hinge.mean()


__all__ = ["adversarial_persistence_loss"]
