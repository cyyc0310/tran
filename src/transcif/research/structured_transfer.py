"""Coarse-structure-preserving increments around the frozen recomposition."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from transcif.research.budget_transfer import INPUTS
from transcif.research.cross_country import COARSE, EF

HEADS = ('residual', 'factor', 'shape', 'joint', 'joint_aux')


class StructuredNet(nn.Module):
    def __init__(self, head: str, profile: np.ndarray, thermal_ef: float, prior: np.ndarray):
        super().__init__()
        if head not in HEADS:
            raise ValueError(head)
        self.head = head
        self.backbone = nn.Sequential(nn.Linear(len(INPUTS), 32), nn.Tanh(),
                                      nn.Linear(32, 32), nn.Tanh())
        self.output = nn.Linear(32, 9)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer('profile', torch.tensor(profile, dtype=torch.float32))
        self.register_buffer('thermal_ef', torch.tensor(thermal_ef/1000., dtype=torch.float32))
        self.register_buffer('prior_logits', torch.tensor(np.log(np.maximum(prior, 1e-8)), dtype=torch.float32))
        self.register_buffer('fuel_ef', torch.tensor(EF[:4]/1000., dtype=torch.float32))

    def forward(self, x: torch.Tensor, mix: torch.Tensor, hour: torch.Tensor):
        delta = self.output(self.backbone(x)).tanh()
        q0 = mix * self.profile[hour]
        base_share = q0[:, 0]/q0.sum(dim=1)
        base = self.thermal_ef * base_share
        internal = (self.prior_logits + 1.5*delta[:, 5:9]).softmax(dim=1)
        if self.head == 'residual':
            return (base + .15*delta[:, 0]).clamp(0., 1.), internal
        q = q0 * (torch.exp(.5*delta[:, :5]) if self.head in ('shape', 'joint', 'joint_aux') else 1.)
        factor = self.thermal_ef
        if self.head in ('factor', 'joint', 'joint_aux'):
            correction = internal @ self.fuel_ef - self.prior_logits.softmax(dim=0) @ self.fuel_ef
            factor = (factor + correction).clamp(.4, 1.)
        return factor*q[:, 0]/q.sum(dim=1), internal


def train_structured(frame: pd.DataFrame, fuel: np.ndarray, head: str,
                     seed: int, legacy: dict, steps: int = 600, batch: int = 1024) -> dict:
    if len(frame) != len(fuel) or not np.isfinite(fuel).all() or (fuel < 0).any():
        raise ValueError('invalid source supervision')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    values = frame[list(INPUTS)].to_numpy(float)
    weights = 1/frame.groupby('region').region.transform('size').to_numpy(float)
    weights /= weights.sum()
    mean = np.sum(values*weights[:, None], axis=0)
    scale = np.sqrt(np.sum((values-mean)**2*weights[:, None], axis=0))
    scale[scale < 1e-8] = 1.
    prior = np.sum(fuel[:, :4]*weights[:, None], axis=0)
    prior /= prior.sum()
    model = StructuredNet(head, legacy['profile'], legacy['thermal_ef'], prior)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    x = torch.tensor((values-mean)/scale, dtype=torch.float32)
    mix = torch.tensor(frame[list(COARSE)].to_numpy(), dtype=torch.float32)
    hour = torch.tensor(frame.clock.dt.hour.to_numpy(), dtype=torch.int64)
    y = torch.tensor(fuel@EF/fuel.sum(axis=1)/1000., dtype=torch.float32)
    thermal = fuel[:, :4].sum(axis=1)
    internal = torch.tensor(fuel[:, :4]/np.maximum(thermal[:, None], 1e-12), dtype=torch.float32)
    active = torch.tensor((thermal > 0).astype(float), dtype=torch.float32)
    indices = rng.choice(len(frame), size=(steps, batch), replace=True, p=weights)
    losses = []
    for ix in indices:
        pred, shares = model(x[ix], mix[ix], hour[ix])
        loss = (pred-y[ix]).square().mean()
        if head == 'joint_aux':
            loss = loss + .1*((shares-internal[ix]).square().mean(dim=1)*active[ix]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return {'model': model, 'mean': mean, 'scale': scale, 'head': head,
            'legacy': legacy, 'seed': seed, 'steps': steps, 'batch': batch,
            'parameters': sum(p.numel() for p in model.parameters()),
            'last_loss': float(np.mean(losses[-20:]))}


def predict_structured(fitted: dict, features: pd.DataFrame, enabled: bool = True) -> np.ndarray:
    if not enabled:
        legacy = fitted['legacy']
        q = legacy['profile'][features.clock.dt.hour.to_numpy()] * features[list(COARSE)].to_numpy()
        return np.clip(legacy['thermal_ef']*q[:, 0]/q.sum(axis=1), 0., 1000.)
    x = torch.tensor((features[list(INPUTS)].to_numpy()-fitted['mean'])/fitted['scale'], dtype=torch.float32)
    mix = torch.tensor(features[list(COARSE)].to_numpy(), dtype=torch.float32)
    hour = torch.tensor(features.clock.dt.hour.to_numpy(), dtype=torch.int64)
    with torch.no_grad():
        pred, _ = fitted['model'](x, mix, hour)
    return pred.numpy().astype(float)*1000.


def save_structured(fitted: dict, path) -> None:
    torch.save({'state_dict': fitted['model'].state_dict(), 'head': fitted['head'],
                'mean': torch.tensor(fitted['mean']), 'scale': torch.tensor(fitted['scale']),
                'native_profile': torch.tensor(fitted['legacy']['profile'], dtype=torch.float64),
                'native_thermal_ef': float(fitted['legacy']['thermal_ef']),
                'inputs': list(INPUTS), 'seed': fitted['seed']}, path)


def load_structured(path) -> dict:
    saved = torch.load(path, weights_only=True, map_location='cpu')
    if saved['inputs'] != list(INPUTS):
        raise ValueError('checkpoint input contract differs')
    state = saved['state_dict']
    prior = state['prior_logits'].softmax(dim=0).numpy()
    model = StructuredNet(saved['head'], state['profile'].numpy(),
                          saved['native_thermal_ef'], prior)
    model.load_state_dict(state)
    model.eval()
    return {'model': model, 'mean': saved['mean'].numpy(), 'scale': saved['scale'].numpy(),
            'head': saved['head'], 'seed': saved['seed'],
            'legacy': {'profile': saved['native_profile'].numpy(),
                       'thermal_ef': saved['native_thermal_ef']}}
