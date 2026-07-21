"""Stage 1: domain-invariant encoder that fuses LT-MWKC and CV-DWCC to predict the
future RenewShare trajectory (the reparameterized, transferable prediction target)."""

import torch
import torch.nn as nn

from transcif.models.cv_dwcc import CVDWCC
from transcif.models.lt_mwkc import LTMWKC


def instance_normalize(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RevIN/Dish-TS-style per-instance, per-channel standardization over the time axis
    (dim=1). Returns (x_norm, mean, std) so callers can reinject the removed level/scale
    as an explicit calibration signal elsewhere in the network."""
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp_min(eps)
    return (x - mean) / std, mean, std


class DomainInvariantEncoder(nn.Module):
    """LT-MWKC's wavelet convolutions run directly on raw input values, so their learned
    kernels implicitly calibrate to whatever absolute RenewShare level the source region
    happened to have (e.g. QLD1's ~0.18 mean vs SA1's ~0.69 mean) -- a real cross-region gap
    confirmed on 2023 AEMO/NEMED data. Following the RevIN/Dish-TS instance-normalization
    paradigm for distribution shift in time-series forecasting, each window is normalized by
    its own per-channel mean/std before LT-MWKC sees it, so the wavelet backbone learns
    relative temporal dynamics rather than an absolute level. CV-DWCC is left on raw `x`:
    its dominant-variable/correlation features are locally-weighted R^2 statistics, which
    are already invariant to per-channel affine rescaling. The level information stripped
    out by normalization is reinjected as an explicit (mean, std) side-channel into the
    final prediction head, so it can still recover the correct absolute share."""

    def __init__(
        self,
        num_variables: int = 2,
        horizon: int = 24,
        lt_feature_dim: int = 32,
        cv_feature_dim: int = 16,
        renew_share_channel_idx: int = 0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.lt_mwkc = LTMWKC(in_channels=num_variables, feature_dim=lt_feature_dim)
        self.cv_dwcc = CVDWCC(num_variables=num_variables, feature_dim=cv_feature_dim)
        self.renew_share_channel_idx = renew_share_channel_idx
        self.norm_eps = norm_eps

        fused_dim = lt_feature_dim + cv_feature_dim * num_variables + 2
        self.predict = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Linear(64, horizon),
            nn.Sigmoid(),
        )

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the pooled `fused` feature vector fed into the prediction head, plus
        the CV-DWCC dominant-variable index. Exposed as its own method (rather than only
        inline inside `forward`) so domain-adaptation code -- e.g. Deep CORAL covariance
        alignment between source-region and target-region feature distributions -- can tap
        the same representation the prediction head sees, without duplicating this logic."""
        x_norm, window_mean, window_std = instance_normalize(x, eps=self.norm_eps)

        lt_input = x_norm.permute(0, 2, 1)
        lt_features = self.lt_mwkc(lt_input).mean(dim=-1)

        cv_features, dominant_idx = self.cv_dwcc(x)
        cv_pooled = cv_features.mean(dim=(2, 4))
        cv_pooled = cv_pooled.reshape(cv_pooled.shape[0], -1)

        channel = self.renew_share_channel_idx
        level_context = torch.cat(
            [window_mean[:, 0, channel : channel + 1], window_std[:, 0, channel : channel + 1]],
            dim=-1,
        )

        fused = torch.cat([lt_features, cv_pooled, level_context], dim=-1)
        return fused, dominant_idx

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fused, dominant_idx = self.forward_features(x)
        renew_share_pred = self.predict(fused)
        return renew_share_pred, dominant_idx


class PersistenceSkipEncoder(nn.Module):
    """Wraps DomainInvariantEncoder with a learnable persistence-skip gate: predicts
    gate * last_observed_renew_share + (1 - gate) * network_output. RenewShare is a
    slowly-varying real-world signal, so "repeat the last observed value" is a strong
    baseline the raw encoder failed to beat on real AEMO data at its original training
    configuration. This lets training default toward that baseline and learn only a
    correction, rather than the full near-identity mapping from scratch through a deep
    nonlinear path. `gate_logit` initializes to 1.5 (sigmoid ~= 0.82), favoring
    persistence at the start of training.

    The gate is conditioned on each window's own recent RenewShare volatility (std of the
    last `volatility_window` observed steps): `effective_gate_logit = gate_logit -
    softplus(volatility_gain_raw) * recent_volatility`. The persistence prior is only
    justified when the signal is actually slowly-varying; a high-volatility region (e.g.
    SA1's real 2023 series swings far more than QLD1's) should trust the network
    correction more and the stale last-observed value less. The softplus keeps the learned
    gain non-negative so volatility can only ever push the gate toward the network (never
    toward more persistence), matching that hypothesis as an inductive bias rather than
    something the optimizer has to discover unconstrained. `volatility_gain_raw` initializes
    to -6.0 (softplus ~= 0.0025), so training starts at essentially the original
    volatility-blind gate and only conditions on volatility once gradients favor it.

    Exposes `.cv_dwcc` and `.lt_mwkc` pointing at the wrapped encoder's own submodules,
    so consistency_loss and the calibration-stage helpers that reach into these attributes
    directly keep working unmodified against the wrapper. These are plain @property
    accessors rather than aliased submodule assignments -- registering the same submodule
    object under two attribute paths (`cv_dwcc` and `base_encoder.cv_dwcc`) breaks
    torch.func.functional_call's higher-order (create_graph=True) double-backward across
    repeated calls (confirmed via MLDG multi-epoch training on real AEMO data: identical
    single-epoch calls succeed, but the second epoch's grad() call fails with "Trying to
    backward through the graph a second time" once the tied submodule is functional_call'd
    twice per epoch across epochs). A read-only property keeps the module registered once."""

    def __init__(
        self,
        base_encoder: DomainInvariantEncoder,
        renew_share_channel_idx: int = 0,
        volatility_window: int = 24,
    ):
        super().__init__()
        self.base_encoder = base_encoder
        self.renew_share_channel_idx = renew_share_channel_idx
        self.volatility_window = volatility_window
        self.gate_logit = nn.Parameter(torch.tensor(1.5))
        self.volatility_gain_raw = nn.Parameter(torch.tensor(-6.0))

    @property
    def cv_dwcc(self):
        return self.base_encoder.cv_dwcc

    @property
    def lt_mwkc(self):
        return self.base_encoder.lt_mwkc

    def recent_volatility(self, x: torch.Tensor) -> torch.Tensor:
        """Std of the last `volatility_window` observed RenewShare steps, per sample."""
        channel = self.renew_share_channel_idx
        window = min(self.volatility_window, x.shape[1])
        return x[:, -window:, channel].std(dim=1, unbiased=False, keepdim=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        network_pred, dominant_idx = self.base_encoder(x)
        channel = self.renew_share_channel_idx
        last_observed = x[:, -1, channel : channel + 1].expand(-1, network_pred.shape[-1])

        volatility_gain = torch.nn.functional.softplus(self.volatility_gain_raw)
        effective_gate_logit = self.gate_logit - volatility_gain * self.recent_volatility(x)
        gate = torch.sigmoid(effective_gate_logit)

        renew_share_pred = gate * last_observed + (1 - gate) * network_pred
        return renew_share_pred, dominant_idx
