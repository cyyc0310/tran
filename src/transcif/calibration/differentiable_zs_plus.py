"""Differentiable ZS+ calibration (Task 8.1).

This module re-implements the ZS+ test-time calibration as a torch.nn.Module
so gradient can flow from the final CIF prediction back through the model's
share predictions to the model parameters.

Design overview
---------------
Original ZS+ (``calibration/zs_plus.py``) computes 6 branches per origin:

    0: ``cif_from_shares(s, ef_r, ef_nr) + delta``
       where ``s = clip(s_raw - mean(s_raw) + mean(rs_anchor), 0, 1)``
       and ``delta = mean(cif_recent - cif_from_shares(rs_recent))``
    1: daily lag (yesterday's CIF for today)
    2: weekly lag (last week's CIF for today)
    3: mean of last 7 daily-lag windows
    4: mean of weekly lags
    5: ``cif_from_shares(clip(s_raw, 0, 1), ef_r, ef_nr)``  (raw model output)

Branches 1-4 are deterministic functions of the CIF history — they do NOT
depend on the model and carry no gradient. Branches 0 and 5 carry gradient
through ``share_fn`` back to the model.

The original ZS+ picks a hard subset of branches (FUSION_MENU) and a fixed
gamma, then weights by ``ratio ** (-gamma)`` based on per-branch backtest
error. The differentiable version drops the hard subset selection and
replaces it with soft attention:

    weights = softmax(branch_gate + log(1 / (mean_err + eps)) * inv_temp)

where ``branch_gate`` is a learnable (6,) global preference and ``inv_temp``
controls how sharply the attention favors low-error branches. Gradient
through ``mean_err`` is stopped (``.detach()``) for two reasons:

    1. Stability: backprop through 28-day backtest history creates a long
       computation graph and meta-gradient noise.
    2. Correctness: the model should be optimized for the *live* prediction
       at t0, not for producing predictions at past origins that just happen
       to score well in the attention.

The persistence branches (1-4) and the delta anchor in branch 0 are
constants w.r.t. model parameters — they're functions of the CIF history
only — so gradient through them is naturally zero.

Parity with non-differentiable ZS+
----------------------------------
Default init (branch_gate = 0, inv_temp derived from gamma=2.0) does NOT
exactly reproduce non-differentiable DEFAULT ZS+ because the architectures
differ (no hard FUSION_MENU selection). The DoD requires the output to be
in a sane MAE range on a real region, not exact parity.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from transcif.config import HORIZON, SEQ_LEN


# Hyperparameters matched to original ZS+ where applicable
ANCHOR_WIN = 24
RESID_WIN = 48
WEEKLY_LAG = 168
K_BACKTEST_DEFAULT = 7


class DifferentiableZSPlus(nn.Module):
    """Differentiable test-time calibration.

    Args:
        horizon: Forecast horizon (default ``HORIZON``).
        k_backtest: Number of past days used for per-branch error backtest.
        gamma_init: Initial inverse-temperature for attention softmax. The
            module parameterizes ``log_inv_temp`` so ``inv_temp = exp(...)``;
            at init ``inv_temp = gamma_init`` to match original ZS+ default.
        err_eps: Numerical stability epsilon for the error-to-weight map.
            Errors below this are clamped to avoid blowing up the weight.
        err_scale: Multiplier on the error-to-weight logit. Higher values
            make attention sharper.
    """

    def __init__(
        self,
        horizon: int = HORIZON,
        k_backtest: int = K_BACKTEST_DEFAULT,
        gamma_init: float = 2.0,
        err_eps: float = 1.0,
        err_scale: float = 1.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.k_backtest = k_backtest
        self.err_eps = err_eps
        self.err_scale = err_scale

        # Learnable parameters
        self.branch_gate = nn.Parameter(torch.zeros(6))
        # inv_temp = exp(log_inv_temp); init so inv_temp = gamma_init
        self.log_inv_temp = nn.Parameter(torch.tensor(math.log(gamma_init)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        rs: torch.Tensor,
        cif: torch.Tensor,
        ef_r: float,
        ef_nr: float,
        origins: Sequence[int],
        share_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Compute calibrated CIF predictions for each origin.

        Args:
            rs: (T,) renewable share history as a 1-D torch tensor.
            cif: (T,) true CIF history as a 1-D torch tensor.
            ef_r, ef_nr: Emission factors (scalars).
            origins: Sequence of origin times ``t0`` (each ``>= SEQ_LEN``).
            share_fn: Callable taking a ``(SEQ_LEN,)`` torch tensor window
                and returning a ``(horizon,)`` torch tensor of predicted
                renewable shares in [0, 1]. Gradient is propagated through
                this callable.

        Returns:
            (n_origins, horizon) torch tensor of calibrated CIF predictions.
        """
        preds = []
        for t0 in origins:
            branches = self.compute_branches(
                t0, rs, cif, ef_r, ef_nr, share_fn
            )  # (6, horizon)
            weights = self.compute_attention_weights(
                t0, rs, cif, ef_r, ef_nr, share_fn
            )  # (6,)
            pred = (weights[:, None] * branches).sum(dim=0)
            preds.append(pred)
        return torch.stack(preds, dim=0)

    # ------------------------------------------------------------------
    # Internals (exposed for testing)
    # ------------------------------------------------------------------

    def compute_branches(
        self,
        t0: int,
        rs: torch.Tensor,
        cif: torch.Tensor,
        ef_r: float,
        ef_nr: float,
        share_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Compute all 6 branches at origin ``t0``.

        Returns:
            (6, horizon) torch tensor. Branches 1-4 are constants w.r.t.
            model parameters. Branches 0 and 5 carry gradient through
            ``share_fn``.
        """
        h = self.horizon
        x_win = rs[t0 - SEQ_LEN : t0]
        s_raw = share_fn(x_win)  # (horizon,)

        # --- Branch 0: anchored model output + residual delta ---
        # s = clip(s_raw - mean(s_raw) + mean(rs_anchor), 0, 1)
        # delta = mean(cif_recent - cif_from_shares(rs_recent))  [constant]
        s_anchor_mean = rs[t0 - ANCHOR_WIN : t0].mean()
        s = torch.clamp(s_raw - s_raw.mean() + s_anchor_mean, 0.0, 1.0)
        # delta is a constant (data-dependent but not model-dependent)
        rs_recent = rs[t0 - RESID_WIN : t0]
        cif_recent = cif[t0 - RESID_WIN : t0]
        # Inline cif_from_shares for differentiability (the version in
        # physics/decompose.py detaches tensors)
        delta = (cif_recent - (rs_recent * ef_r + (1 - rs_recent) * ef_nr)).mean()
        branch_0 = s * ef_r + (1 - s) * ef_nr + delta

        # --- Branch 1: daily lag (yesterday's CIF for today) ---
        branch_1 = cif[t0 - 24 : t0 - 24 + h]

        # --- Branch 2: weekly lag ---
        branch_2 = cif[t0 - WEEKLY_LAG : t0 - WEEKLY_LAG + h]

        # --- Branch 3: mean of last 7 daily-lag windows ---
        branch_3 = torch.stack(
            [cif[t0 - j * 24 : t0 - j * 24 + h] for j in range(1, 8)],
            dim=0,
        ).mean(dim=0)

        # --- Branch 4: mean of weekly lags (j=1..4 if available) ---
        weekly_lags = [j * WEEKLY_LAG for j in range(1, 5) if t0 - j * WEEKLY_LAG >= 0]
        if not weekly_lags:  # t0 too small for any weekly lag
            branch_4 = branch_3.detach().clone()  # graceful fallback
        else:
            branch_4 = torch.stack(
                [cif[t0 - lag : t0 - lag + h] for lag in weekly_lags],
                dim=0,
            ).mean(dim=0)

        # --- Branch 5: raw model output without delta anchor ---
        s_raw_clamped = torch.clamp(s_raw, 0.0, 1.0)
        branch_5 = s_raw_clamped * ef_r + (1 - s_raw_clamped) * ef_nr

        return torch.stack(
            [branch_0, branch_1, branch_2, branch_3, branch_4, branch_5],
            dim=0,
        )

    def compute_attention_weights(
        self,
        t0: int,
        rs: torch.Tensor,
        cif: torch.Tensor,
        ef_r: float,
        ef_nr: float,
        share_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Compute soft attention weights over the 6 branches.

        Weights are computed as:

            weights = softmax(branch_gate + err_scale * log(1 / (mean_err + eps)) * inv_temp)

        where ``mean_err[b]`` is the mean absolute error of branch ``b``
        over the past ``k_backtest`` days. Gradient through ``mean_err`` is
        stopped to avoid meta-gradient noise from the backtest history.

        Returns:
            (6,) torch tensor that sums to 1.
        """
        max_k = min(self.k_backtest, max(0, (t0 - SEQ_LEN) // 24))
        if max_k < 1:
            # Not enough history for backtest; fall back to branch_gate only
            return F.softmax(self.branch_gate, dim=0)

        errors = torch.zeros(6, dtype=torch.float32)
        n_valid = 0
        for k in range(1, max_k + 1):
            o = t0 - k * 24
            if o - SEQ_LEN < 0 or o + self.horizon > t0:
                continue
            branches_o = self.compute_branches(
                o, rs, cif, ef_r, ef_nr, share_fn
            )
            # Detach past branches — gradient through backtest history would
            # create a meta-gradient that's noisy and unstable. The model
            # is optimized for live prediction only.
            branches_o = branches_o.detach()
            true_cif = cif[o : o + self.horizon]
            err = (branches_o - true_cif[None, :]).abs().mean(dim=1)
            errors = errors + err
            n_valid += 1

        if n_valid == 0:
            return F.softmax(self.branch_gate, dim=0)

        mean_err = errors / n_valid  # (6,)
        # Robust attention: log(1/(err+eps)) is bounded above by log(1/eps)
        # and grows slowly with err. This avoids the degenerate softmax
        # that arises from -err/temperature when err is large.
        err_logit = torch.log(1.0 / (mean_err + self.err_eps))
        inv_temp = torch.exp(self.log_inv_temp)
        logits = self.branch_gate + self.err_scale * err_logit * inv_temp
        return F.softmax(logits, dim=0)


__all__ = ["DifferentiableZSPlus"]
