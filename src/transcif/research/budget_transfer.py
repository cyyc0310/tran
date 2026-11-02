"""Controlled source-only tests of delayed-input augmentation and fuel supervision."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from transcif.research.cross_country import EF, FEATURES

INPUTS = (*FEATURES, 'age_months')
HEADS = ('direct', 'mixture', 'fuel_aux')


def budget_views(views: dict[int, pd.DataFrame], year: int, augment: bool) -> pd.DataFrame:
    """Identical response rows, three input views in both control and treatment."""
    common = None
    for lag in (2, 3, 4):
        frame = views[lag]
        valid = frame.index[frame.year.eq(year) & frame[list(FEATURES)].notna().all(axis=1)]
        common = valid if common is None else common.intersection(valid)
    if common is None or len(common) == 0:
        raise ValueError('no common source hours for lag2/3/4')
    copies = []
    for lag in (2, 3, 4):
        used = lag if augment else 2
        frame = views[used].loc[common].copy()
        frame['raw_row'] = common
        frame['age_months'] = float(used)
        copies.append(frame)
    return pd.concat(copies, ignore_index=True)


class BudgetNet(nn.Module):
    """Same small backbone; mixture heads use fixed-factor recomposition."""

    def __init__(self, head: str):
        super().__init__()
        if head not in HEADS:
            raise ValueError(head)
        self.head = head
        self.backbone = nn.Sequential(nn.Linear(len(INPUTS), 32), nn.Tanh(),
                                      nn.Linear(32, 32), nn.Tanh())
        self.output = nn.Linear(32, 1 if head == 'direct' else 8)
        self.register_buffer('ef', torch.tensor(EF / 1000., dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        output = self.output(self.backbone(x))
        if self.head == 'direct':
            return output[:, 0].sigmoid(), None
        shares = output.softmax(dim=1)
        return shares @ self.ef, shares


def train_model(frame: pd.DataFrame, fuel: np.ndarray, head: str,
                seed: int, steps: int = 600, batch: int = 1024) -> dict:
    """Fixed optimizer budget; country-held-out labels are never arguments."""
    if len(frame) != len(fuel) or not np.isfinite(fuel).all() or (fuel < 0).any():
        raise ValueError('invalid source supervision')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    values = frame[list(INPUTS)].to_numpy(float)
    weights = 1 / frame.groupby('region').region.transform('size').to_numpy(float)
    weights /= weights.sum()
    mean = np.sum(values * weights[:, None], axis=0)
    scale = np.sqrt(np.sum((values-mean)**2 * weights[:, None], axis=0))
    scale[scale < 1e-8] = 1.
    x = torch.tensor((values-mean)/scale, dtype=torch.float32)
    shares = fuel / fuel.sum(axis=1, keepdims=True)
    target_shares = torch.tensor(shares, dtype=torch.float32)
    y = torch.tensor(shares @ EF / 1000., dtype=torch.float32)
    model = BudgetNet(head)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    indices = rng.choice(len(frame), size=(steps, batch), replace=True, p=weights)
    losses = []
    model.train()
    for ix in indices:
        pred, predicted_shares = model(x[ix])
        loss = (pred-y[ix]).square().mean()
        if head == 'fuel_aux':
            loss = loss + (predicted_shares-target_shares[ix]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return {'model': model, 'mean': mean, 'scale': scale, 'head': head,
            'seed': seed, 'steps': steps, 'batch': batch,
            'parameters': sum(p.numel() for p in model.parameters()),
            'last_loss': float(np.mean(losses[-20:]))}


def predict_model(fitted: dict, features: pd.DataFrame) -> np.ndarray:
    """Predict from the public-feature contract only; no response centering."""
    x = (features[list(INPUTS)].to_numpy(float)-fitted['mean'])/fitted['scale']
    with torch.no_grad():
        point, _ = fitted['model'](torch.tensor(x, dtype=torch.float32))
    return point.numpy().astype(float) * 1000.


def save_model(fitted: dict, path) -> None:
    """Save generated weights and feature transform, loadable with weights_only."""
    torch.save({'state_dict': fitted['model'].state_dict(), 'head': fitted['head'],
                'mean': torch.tensor(fitted['mean']), 'scale': torch.tensor(fitted['scale']),
                'inputs': list(INPUTS), 'seed': fitted['seed']}, path)
