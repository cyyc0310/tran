"""Config-conditioned joint dispatch scenarios without local telemetry inputs."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from transcif.data.dataset import ModelInputs, DISPATCHABLE
from transcif.physics.dispatch import capped_dispatch
from transcif.models.context import ContextEncoder


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

    def __init__(self, hidden=32, dispatch_temperature=0.3, backbone="tcn", point_forecast=False,
                 point_estimator=None, encode_members=True):
        super().__init__()
        if backbone not in ("tcn", "context", "shared") or hidden < 1 or dispatch_temperature <= .01:
            raise ValueError("valid backbone, positive hidden and temperature>.01 required")
        self.encode_members = encode_members
        self.hidden = hidden
        self.dispatch_temperature = dispatch_temperature
        self.backbone = backbone
        self.point_forecast = point_forecast
        self.point_estimator = point_estimator or ("physical" if point_forecast else "mean")
        if self.point_estimator not in ("mean", "median", "physical"):
            raise ValueError("point_estimator must be mean, median or physical")
        # weather(3), load(1), calendar(6), capacity/mix/EF(18), costs(3).
        if backbone == "shared":
            # A 22-parameter transfer diagnostic: grid differences enter only
            # the physical decoder, not a configuration-conditioned network.
            self.encoder = None
            self.location = nn.Linear(1, 10, bias=False)
            self.scale = nn.Linear(1, 10, bias=False)
        else:
            self.encoder = nn.Sequential(
                nn.Conv1d(31, hidden, 3, padding=1), nn.GELU(),
                nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU()) if backbone == "tcn" else ContextEncoder(38, hidden)
            self.location = nn.Linear(hidden, 10)
            self.scale = nn.Linear(hidden, 10)
            nn.init.zeros_(self.location.bias)
            nn.init.constant_(self.scale.bias, -1.5)
        if backbone == "context":
            self.availability_bias = nn.Parameter(torch.full((6,), 2.))
        if backbone in ("context", "shared"):
            self.raw_temperature = nn.Parameter(torch.tensor(math.log(math.expm1(dispatch_temperature - .01))))
        # Six availability logits, three merit-order costs, one load state.
        self.raw_rho = nn.Parameter(torch.tensor(math.log(4.0)))
        nn.init.zeros_(self.location.weight)
        nn.init.constant_(self.scale.weight, -1.5 if backbone == "shared" else 0.)

    @property
    def model_config(self):
        return {"hidden": self.hidden, "dispatch_temperature": self.dispatch_temperature,
                "backbone": self.backbone, "point_forecast": self.point_forecast,
                "point_estimator": self.point_estimator, "encode_members": self.encode_members}

    def forward(self, inputs: ModelInputs, num_scenarios=32, generator=None,
                latent_scale=1., deterministic=False):
        if num_scenarios < 2:
            raise ValueError("at least two scenarios required")
        if not math.isfinite(latent_scale) or latent_scale <= 0:
            raise ValueError("finite positive latent_scale required")
        b, m, h, _ = inputs.weather.shape
        device, dtype = inputs.capacity.device, inputs.capacity.dtype
        scale_mw = inputs.capacity.sum(-1).clamp_min(1)
        cfg = torch.cat([inputs.capacity / scale_mw[:, None], inputs.mix,
                         inputs.ef / 1000, inputs.cost / 100], -1)
        row = torch.arange(b, device=device)[:, None]
        member = None if deterministic else torch.randint(
            m, (b, num_scenarios), device=device, generator=generator)
        # Encode each issued member once when it is cheaper than encoding the
        # sampled scenarios. The reference path supports equivalence checks.
        reuse = self.encode_members and (deterministic or m < num_scenarios)
        if deterministic:
            n = 1 if reuse else num_scenarios
            wx = inputs.weather.mean(1)[:, None].expand(-1, n, -1, -1)
            load_forecast = inputs.demand.mean(1)[:, None].expand(-1, n, -1)
        elif reuse:
            wx, load_forecast = inputs.weather, inputs.demand
        else:
            wx, load_forecast = inputs.weather[row, member], inputs.demand[row, member]
        wind_factor = wind_cf(wx[..., 1])
        solar_factor = (0.85 * wx[..., 2] / 1000).clamp(0, 1)
        if deterministic and self.backbone in ("context", "shared"):
            # E[power(weather)] differs from power(E[weather]).
            wind_factor = wind_cf(inputs.weather[..., 1]).mean(1)[:, None].expand(-1, wx.shape[1], -1)
            solar_factor = (.85 * inputs.weather[..., 2] / 1000).clamp(0, 1).mean(1)[:, None].expand(
                -1, wx.shape[1], -1)
        loc, sigma = self._latent_parameters(inputs, cfg, scale_mw, wx, load_forecast,
                                              wind_factor, solar_factor)
        if reuse:
            def select(value):
                return value.expand(-1, num_scenarios, *value.shape[2:]) if deterministic else value[row, member]
            loc, sigma = select(loc), select(sigma)
            wind_factor, solar_factor = select(wind_factor), select(solar_factor)
            load_forecast = select(load_forecast)
        state = loc
        if not deterministic:
            rho = 0.98 * torch.sigmoid(self.raw_rho)
            eps = torch.randn(b, num_scenarios, h, 10, device=device, dtype=dtype, generator=generator)
            day_state = torch.randn(b, num_scenarios, 1, 10, device=device, dtype=dtype, generator=generator)
            history = [eps[:, :, 0]]
            for t in range(1, h):
                history.append(rho * history[-1] + (1 - rho.square()).sqrt() * eps[:, :, t])
            correlated = 0.6 * day_state + 0.8 * torch.stack(history, dim=2)
            state = loc + latent_scale * sigma * correlated
        bias = self.availability_bias if self.backbone == "context" else 2
        availability = torch.sigmoid(bias + state[..., :6])
        capacity = inputs.capacity[:, None, None] / scale_mw[:, None, None, None]
        available = capacity * availability
        nuclear = available[..., 2]
        wind = available[..., 4] * wind_factor
        solar = available[..., 5] * solar_factor
        potential = torch.stack([nuclear, wind, solar], -1)
        demand = load_forecast / scale_mw[:, None, None] * torch.exp(
            (0.1 * state[..., 9]).clamp(-0.3, 0.3))
        potential_total = potential.sum(-1)
        served_nondisp = potential * torch.minimum(
            torch.ones_like(demand), demand / potential_total.clamp_min(1e-12))[..., None]
        residual_load = (demand - served_nondisp.sum(-1)).clamp_min(0)
        cost = inputs.cost[:, None, None] / 100 + 0.25 * state[..., 6:9]
        dispatchable, unserved = capped_dispatch(available[..., list(DISPATCHABLE)], cost,
            residual_load, .01 + F.softplus(self.raw_temperature)
            if self.backbone in ("context", "shared") else self.dispatch_temperature)
        generation = torch.stack([dispatchable[..., 0], dispatchable[..., 1],
            served_nondisp[..., 0], dispatchable[..., 2], served_nondisp[..., 1],
            served_nondisp[..., 2]], -1)
        total = generation.sum(-1)
        cif = (generation * inputs.ef[:, None, None]).sum(-1) / total.clamp_min(1e-12)
        factor = scale_mw[:, None, None]
        return DispatchForecast(cif, generation * factor[..., None], demand * factor,
                                unserved * factor,
                                (potential_total - served_nondisp.sum(-1)).clamp_min(0) * factor)

    def _latent_parameters(self, inputs, cfg, scale_mw, wx, load_forecast, wind_factor, solar_factor):
        b, n, h, _ = wx.shape
        if self.backbone == "shared":
            context = wx.new_ones(b, n, h, 1)
        else:
            cfg_seq = cfg[:, None, None].expand(-1, n, h, -1)
            calendar = inputs.calendar[:, None].expand(-1, n, -1, -1)
            load = load_forecast / scale_mw[:, None, None]
            features = torch.cat([wx / wx.new_tensor([30., 15., 1000.]), load[..., None],
                                  calendar, cfg_seq], -1)
            if self.backbone == "context":
                fractions = inputs.capacity / scale_mw[:, None]
                wind_potential = fractions[:, None, None, 4] * wind_factor
                solar_potential = fractions[:, None, None, 5] * solar_factor
                residual = load - fractions[:, None, None, 2] - wind_potential - solar_potential
                thermal = fractions[:, list(DISPATCHABLE)].sum(-1)[:, None, None].clamp_min(1e-4)
                ramp = F.pad(load[..., 1:] - load[..., :-1], (1, 0))
                physical = torch.stack([wind_factor, solar_factor, wind_potential, solar_potential,
                                        residual, (residual / thermal).clamp(-5, 5), ramp], -1)
                features = torch.cat([features, physical], -1)
                context = self.encoder(features.reshape(b * n, h, 38),
                    cfg[:, None].expand(-1, n, -1).reshape(b * n, 21))
            else:
                context = self.encoder(features.reshape(b * n, h, 31).transpose(1, 2)).transpose(1, 2)
            context = context.reshape(b, n, h, self.hidden)
        loc = self.location(context)
        if self.backbone == "tcn":
            loc = 2 * torch.tanh(loc)
        return loc, F.softplus(self.scale(context)) + 0.03
