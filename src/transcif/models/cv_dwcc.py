"""Stage 1: Cross-Variable Dynamic Wavelet Correlation Convolution module (CV-DWCC).

Simplified, fully-differentiable re-implementation of the paper's wavelet local
multivariate regression: instead of a continuous wavelet transform followed by local
regression, we run local *weighted* least squares directly at several (window, bandwidth)
"scale" settings, which plays the same role (a coarser window/bandwidth approximates a
lower-frequency wavelet scale).
"""

import torch
import torch.nn as nn


def gaussian_window_weights(window: int, bandwidth: float) -> torch.Tensor:
    """theta(t - s): a Gaussian kernel over window offsets, normalized to sum to 1."""
    offsets = torch.arange(-(window // 2), window // 2 + 1, dtype=torch.float32)
    weights = torch.exp(-0.5 * (offsets / bandwidth) ** 2)
    return weights / weights.sum()


def _sliding_windows(x: torch.Tensor, window: int) -> torch.Tensor:
    """(batch, T) -> (batch, T - window + 1, window) via a sliding view."""
    return x.unfold(dimension=1, size=window, step=1)


def local_weighted_r2_and_dominant(
    target: torch.Tensor,
    predictors: torch.Tensor,
    window: int = 25,
    bandwidth: float = 6.0,
) -> tuple:
    """Local weighted multivariate regression (Eq. L_s, R_s^2, varphi_{X,s}(j)).

    Returns (r2_joint, dominant_idx), both of shape (batch, T - window + 1):
      - r2_joint: locally-weighted R^2 of `target` regressed on ALL of `predictors` jointly.
      - dominant_idx: index into `predictors`' last axis of the single predictor that alone
        achieves the highest locally-weighted R^2 (i.e. i*_{j,s} in the paper).
    """
    batch, seq_len = target.shape
    num_predictors = predictors.shape[-1]
    weights = gaussian_window_weights(window, bandwidth)
    sqrt_weights = torch.sqrt(weights)

    target_windows = _sliding_windows(target, window)
    valid_len = target_windows.shape[1]
    predictor_windows = torch.stack(
        [_sliding_windows(predictors[..., p], window) for p in range(num_predictors)], dim=-1
    )

    ones = torch.ones(batch, valid_len, window, 1, dtype=target.dtype)
    design = torch.cat([ones, predictor_windows], dim=-1)

    weighted_design = design * sqrt_weights.view(1, 1, window, 1)
    weighted_target = target_windows * sqrt_weights.view(1, 1, window)

    design_flat = weighted_design.reshape(batch * valid_len, window, num_predictors + 1)
    target_flat = weighted_target.reshape(batch * valid_len, window, 1)

    solution = torch.linalg.lstsq(design_flat, target_flat).solution
    prediction_flat = torch.bmm(design_flat, solution).squeeze(-1)
    residual = target_flat.squeeze(-1) - prediction_flat
    ss_res = (residual ** 2).sum(dim=-1)

    weighted_mean = (target_windows * weights.view(1, 1, window)).sum(dim=-1)
    centered = target_windows - weighted_mean.unsqueeze(-1)
    ss_tot = ((sqrt_weights.view(1, 1, window) * centered) ** 2).sum(dim=-1).clamp_min(1e-8)
    ss_tot_flat = ss_tot.reshape(-1)

    r2_joint = (1 - ss_res / ss_tot_flat).reshape(batch, valid_len)

    r2_single = torch.zeros(batch * valid_len, num_predictors)
    for predictor_idx in range(num_predictors):
        single_design = design_flat[..., [0, predictor_idx + 1]]
        single_solution = torch.linalg.lstsq(single_design, target_flat).solution
        single_prediction = torch.bmm(single_design, single_solution).squeeze(-1)
        single_residual = target_flat.squeeze(-1) - single_prediction
        ss_res_single = (single_residual ** 2).sum(dim=-1)
        r2_single[:, predictor_idx] = 1 - ss_res_single / ss_tot_flat

    dominant_idx = r2_single.argmax(dim=-1).reshape(batch, valid_len)
    return r2_joint.clamp(min=0.0, max=1.0), dominant_idx


class CVDWCC(nn.Module):
    """Computes multi-scale cross-variable correlation + dominant-variable maps for every
    variable (regressed against all others), then fuses them with a 2D conv over the
    (scale, correlation||dominant) representation (Eq. F_P = Conv2D([C_P || D_P], K_CV))."""

    def __init__(self, num_variables: int, scales: tuple = ((15, 3.0), (25, 6.0)), feature_dim: int = 16):
        super().__init__()
        self.num_variables = num_variables
        self.scales = scales
        self.fuse = nn.Conv2d(in_channels=2, out_channels=feature_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple:
        batch, _, num_variables = x.shape
        corr_per_scale = []
        dominant_per_scale = []

        for window, bandwidth in self.scales:
            r2_per_variable = []
            dominant_per_variable = []
            for variable_idx in range(num_variables):
                target = x[..., variable_idx]
                predictors = torch.cat([x[..., :variable_idx], x[..., variable_idx + 1:]], dim=-1)
                r2, dominant_idx = local_weighted_r2_and_dominant(target, predictors, window, bandwidth)
                r2_per_variable.append(r2)
                dominant_per_variable.append(dominant_idx)
            corr_per_scale.append(torch.stack(r2_per_variable, dim=1))
            dominant_per_scale.append(torch.stack(dominant_per_variable, dim=1))

        min_len = min(corr.shape[-1] for corr in corr_per_scale)
        corr = torch.stack([corr[..., :min_len] for corr in corr_per_scale], dim=1)
        dominant = torch.stack([dom[..., :min_len] for dom in dominant_per_scale], dim=1)

        two_channel = torch.stack([corr, dominant.float()], dim=1)
        batch_size, channels, num_scales, num_vars, valid_len = two_channel.shape
        conv_input = two_channel.reshape(batch_size, channels, num_scales, num_vars * valid_len)
        fused = self.fuse(conv_input)
        fused = fused.reshape(batch_size, -1, num_scales, num_vars, valid_len)
        return fused, dominant
