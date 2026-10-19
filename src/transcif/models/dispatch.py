"""Config-conditioned joint dispatch scenarios without local telemetry inputs."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from transcif.data.dataset import ModelInputs, DISPATCHABLE
from transcif.physics.dispatch import capped_dispatch


@dataclass
class DispatchForecast:
    cif: torch.Tensor            # B,S,H, gCO2/kWh
    generation: torch.Tensor     # B,S,H,F, MW
    demand: torch.Tensor         # B,S,H, MW
    unserved: torch.Tensor       # B,S,H, MW
    curtailment: torch.Tensor    # B,S,H, MW, discarded non-dispatchable potential


def wind_cf(speed):
    ramp = ((speed.pow(3) - 3**3) / (12**3 - 3**3)).clamp(0, 1)
    return torch.where(speed > 25, torch.zeros_like(ramp), ramp)


class ProbabilisticDispatchNet(nn.Module):
    """Small future-covariate TCN + correlated latent state + convex dispatch.

    The six-fuel, single-bus production model conditions on published
    capacity/mix/EF/cost and issued weather/load forecasts. External EFs are
    fixed. No region ID, target label fit, additive CIF residual or anchor.
    """

    def __init__(self, hidden=32, dispatch_temperature=0.3):
        super().__init__()
        self.hidden = hidden
        self.dispatch_temperature = dispatch_temperature
        # weather(3), load(1), calendar(6), capacity/mix/EF(18), costs(3).
        self.encoder = nn.Sequential(
            nn.Conv1d(31, hidden, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU())
        # Six availability logits, three merit-order costs, one load state.
        self.location = nn.Linear(hidden, 10)
        self.scale = nn.Linear(hidden, 10)
        self.raw_rho = nn.Parameter(torch.tensor(math.log(4.0)))
        nn.init.zeros_(self.location.weight)
        nn.init.zeros_(self.location.bias)
        nn.init.zeros_(self.scale.weight)
        nn.init.constant_(self.scale.bias, -1.5)

    def forward(self, inputs: ModelInputs, num_scenarios=32, generator=None):
        if num_scenarios < 2:
            raise ValueError("at least two scenarios required")
        b, m, h, _ = inputs.weather.shape
        device, dtype = inputs.capacity.device, inputs.capacity.dtype
        scale_mw = inputs.capacity.sum(-1).clamp_min(1)
        cfg = torch.cat([inputs.capacity / scale_mw[:, None], inputs.mix,
                         inputs.ef / 1000, inputs.cost / 100], -1)
        member = torch.randint(m, (b, num_scenarios), device=device, generator=generator)
        row = torch.arange(b, device=device)[:, None]
        wx, load_forecast = inputs.weather[row, member], inputs.demand[row, member]
        cfg_seq = cfg[:, None, None].expand(-1, num_scenarios, h, -1)
        calendar = inputs.calendar[:, None].expand(-1, num_scenarios, -1, -1)
        wx_scaled = wx / wx.new_tensor([30., 15., 1000.])
        features = torch.cat([wx_scaled, (load_forecast / scale_mw[:, None, None])[..., None],
                              calendar, cfg_seq], -1)
        context = self.encoder(features.reshape(b * num_scenarios, h, 31).transpose(1, 2))
        context = context.transpose(1, 2).reshape(b, num_scenarios, h, self.hidden)
        loc, sigma = 2 * torch.tanh(self.location(context)), F.softplus(self.scale(context)) + 0.03
        rho = 0.98 * torch.sigmoid(self.raw_rho)
        eps = torch.randn(b, num_scenarios, h, 10, device=device, dtype=dtype, generator=generator)
        day_state = torch.randn(b, num_scenarios, 1, 10, device=device, dtype=dtype, generator=generator)
        history = [eps[:, :, 0]]
        for t in range(1, h):
            history.append(rho * history[-1] + (1 - rho.square()).sqrt() * eps[:, :, t])
        correlated = 0.6 * day_state + 0.8 * torch.stack(history, dim=2)
        state = loc + sigma * correlated
        availability = torch.sigmoid(2 + state[..., :6])
        capacity = inputs.capacity[:, None, None] / scale_mw[:, None, None, None]
        available = capacity * availability
        nuclear = available[..., 2]
        wind = available[..., 4] * wind_cf(wx[..., 1])
        solar = available[..., 5] * (0.85 * wx[..., 2] / 1000).clamp(0, 1)
        potential = torch.stack([nuclear, wind, solar], -1)
        demand = load_forecast / scale_mw[:, None, None] * torch.exp(
            (0.1 * state[..., 9]).clamp(-0.3, 0.3))
        potential_total = potential.sum(-1)
        served_nondisp = potential * torch.minimum(
            torch.ones_like(demand), demand / potential_total.clamp_min(1e-12))[..., None]
        residual_load = (demand - served_nondisp.sum(-1)).clamp_min(0)
        cost = inputs.cost[:, None, None] / 100 + 0.25 * state[..., 6:9]
        dispatchable, unserved = capped_dispatch(available[..., list(DISPATCHABLE)], cost,
                                                residual_load, self.dispatch_temperature)
        generation = torch.stack([dispatchable[..., 0], dispatchable[..., 1],
            served_nondisp[..., 0], dispatchable[..., 2], served_nondisp[..., 1],
            served_nondisp[..., 2]], -1)
        total = generation.sum(-1)
        cif = (generation * inputs.ef[:, None, None]).sum(-1) / total.clamp_min(1e-12)
        factor = scale_mw[:, None, None]
        return DispatchForecast(cif, generation * factor[..., None], demand * factor,
                                unserved * factor,
                                (potential_total - served_nondisp.sum(-1)).clamp_min(0) * factor)
