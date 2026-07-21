"""Calibration handle for PersistenceSkipEncoder's persistence-skip gate. The gate is
learned once during source-domain training and, per real AEMO 2023 cross-region
diagnostics, converges to nearly the same value (~0.66-0.69) regardless of which region it
trained on -- it never adapts to how strongly the DEPLOYMENT target region favors
persistence. VIC1, for instance, has a real 2023 within-region naive-persistence MAE about
half of the other three regions, so the gate should sit much closer to 1.0 there, but a
frozen source-trained gate has no way to know this. A calibration set is already reused for
dominant-variable reweighting (see dominant_reweight.py) and residual-head fitting; this
reuses it once more to fine-tune ONLY the scalar gate_logit, leaving every other trained
weight (LT-MWKC, CV-DWCC, predict head) untouched."""

import torch
import torch.nn as nn


def recalibrate_persistence_gate(
    encoder: nn.Module,
    calibration_x: torch.Tensor,
    calibration_y: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-2,
) -> float:
    """Fine-tunes `encoder.gate_logit` in place against the target region's calibration
    set. Only the gate scalar is passed to the optimizer, so no other parameter is updated
    even though gradients flow through the full forward pass. Returns the recalibrated
    gate value (post-sigmoid, in (0, 1))."""
    if not hasattr(encoder, "gate_logit"):
        raise ValueError("encoder has no gate_logit -- expected a PersistenceSkipEncoder")

    optimizer = torch.optim.Adam([encoder.gate_logit], lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        renew_share_pred, _ = encoder(calibration_x)
        loss = loss_fn(renew_share_pred, calibration_y)
        loss.backward()
        optimizer.step()
    return torch.sigmoid(encoder.gate_logit).item()
