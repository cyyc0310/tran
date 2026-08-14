"""Config hypernetwork for FuelDecompNet (Phase FD-2).

HN-MVTS-style weight generation: a small hypernetwork maps the FD config
vector (fuel-mix statistics — exactly what a telemetry-free region
publishes) to the weights of every per-hour dynamic head of
``FuelDecompNet``.  The model's dynamics "morph" per region instead of
being shared with only bias-level conditioning.

Generated heads (111 weights total):

    solar_mod   Linear(5 -> 1)      wind_mod    Linear(5 -> 1)
    base_delta  Linear(10 -> 5)     therm_dyn   Linear(10 -> 3)
    rs_exog     Linear(10 -> 1)

The hypernet's final layer is zero-initialised so a fresh model starts at
exactly the FD-1 physics prior (generated weights = 0) and training only
deviates from it as far as the data supports — the same warm-start
discipline as the base model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# name -> (out_features, in_features) of each generated head
GENERATED_HEADS = {
    "solar_mod": (1, 5),
    "wind_mod": (1, 5),
    "base_delta": (5, 10),
    "therm_dyn": (3, 10),
    "rs_exog": (1, 10),
}
_OFFSETS = {}
_o = 0
for _k, (_out, _in) in GENERATED_HEADS.items():
    _OFFSETS[_k] = (_o, _o + _out * _in, _o + _out * _in + _out)  # w0, w1, b1
    _o += _out * _in + _out
TOTAL_PARAMS = _o


class ConfigHyperNet(nn.Module):
    """Generates per-head (weight, bias) tensors from the config vector."""

    def __init__(self, config_dim=16, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, TOTAL_PARAMS),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, config):
        """config (B, D) -> {head: (weight (B,out,in), bias (B,out))}."""
        theta = self.mlp(config)
        out = {}
        for name, (n_out, n_in) in GENERATED_HEADS.items():
            w0, w1, b1 = _OFFSETS[name]
            w = theta[:, w0:w1].reshape(-1, n_out, n_in)
            b = theta[:, w1:b1]
            out[name] = (w, b)
        return out


def apply_generated_head(x, name, gen):
    """F.linear with per-sample generated weights.

    x : (B, ..., in) ; gen : dict from ConfigHyperNet.forward.
    Returns (B, ..., out).
    """
    w, b = gen[name]
    # einsum with per-sample weight matrices: (B, in) x (B, out, in) -> (B, out)
    # For (B, H, in) inputs broadcast over H.
    if x.dim() == 2:
        return torch.einsum("bi,boi->bo", x, w) + b
    return torch.einsum("bhi,boi->bho", x, w) + b.unsqueeze(1)
