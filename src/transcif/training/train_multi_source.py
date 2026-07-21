"""Proposal 4 (MLDG, AAAI 2018): multi-source domain-generalization training.
train_source_domain fits a single source region's own dynamics; on real AEMO 2023 data this
converges to a persistence-skip gate nearly identical across all four source regions and a
network that only beats the persistence baseline on 7/12 real cross-region pairs (see the
single-source validation matrix), with every remaining failure landing on SA1 -- the one
region whose renewable-share regime (mean ~0.69) sits far outside the other three (~0.18-0.36).
MLDG instead trains across several SOURCE regions at once with an explicit meta-objective:
each iteration, one source region is held out as "meta-test" and the rest are pooled as
"meta-train"; a virtual one-step gradient update on the meta-train loss is taken (kept
differentiable via create_graph=True) and the meta-test loss is evaluated against those
*virtual* post-update parameters. Backpropagating that meta-test loss into the real
parameters rewards updates that keep working on a region they didn't directly fit, which is
exactly the property a single source region can't teach on its own -- without ever training
on the actual deployment target.

The meta-train pool itself is domain-weighted (not a flat concatenation): each meta-train
region's contribution is scaled by its own target volatility/skew (see
`compute_domain_weight`), so a majority of low-difficulty, low-volatility source regions
(e.g. QLD1/NSW1's fairly steady real 2023 RenewShare) can't numerically outweigh the one
harder, more-volatile region carrying most of the transfer-relevant signal just because
concatenated-batch MSE otherwise weights every region by its raw sample count."""

import os
import random

import torch
import torch.nn as nn
from torch.func import functional_call

from transcif.training.consistency import consistency_loss


def compute_domain_weight(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Domain-adaptive weight for one source region's contribution to the pooled meta-train
    loss: std(y) + |skew(y)|. A region whose target distribution is more volatile or more
    asymmetric (e.g. SA1's real 2023 RenewShare vs. QLD1/NSW1's comparatively steady,
    symmetric series) carries more of the signal a single-region fit can't teach, so it
    should not be drowned out purely because other regions contribute more/easier samples."""
    flat = y.reshape(-1).detach()
    mean = flat.mean()
    std = flat.std(unbiased=False).clamp_min(eps)
    skew = ((flat - mean) / std).pow(3).mean()
    return std + skew.abs()


def train_multi_source_erm(
    encoder: nn.Module,
    source_windows: dict,
    epochs: int = 150,
    lr: float = 5e-3,
    consistency_weight: float = 0.05,
    checkpoint_path: str = None,
    checkpoint_every: int = 10,
) -> list:
    """Pure ERM baseline (DomainBed, Gulrajani & Lopez-Paz, ICLR 2021 "In Search of Lost Domain
    Generalization"): pools every source region's (x, y) into one flat batch and minimizes plain
    MSE, with no domain-generalization machinery at all -- no meta-learning bi-level
    optimization, no held-out meta-test region, no domain-adaptive reweighting (contrast with
    `train_multi_source_mldg` above). Tests DomainBed's empirical finding that a
    carefully-implemented ERM baseline often matches complex DG algorithms within ~1 percentage
    point; `consistency_weight` is kept at the same value as the MLDG variants (it is a
    semi-supervised regularizer, not itself a domain-generalization technique) so this isolates
    the meta-learning/domain-weighting machinery as the only variable removed.

    `checkpoint_path` mirrors `train_multi_source_mldg_coral`'s resumability escape hatch: if
    the file exists, training resumes from its saved epoch/optimizer/loss-history state instead
    of starting over, and the file is updated every `checkpoint_every` epochs."""
    x_all = torch.cat([x for x, _ in source_windows.values()], dim=0)
    y_all = torch.cat([y for _, y in source_windows.values()], dim=0)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    losses = []
    start_epoch = 0

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        encoder.load_state_dict(checkpoint["encoder_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        losses = checkpoint["losses"]
        start_epoch = checkpoint["epoch"]

    for epoch in range(start_epoch, epochs):
        pred, _ = encoder(x_all)
        loss = mse_loss(pred, y_all) + consistency_weight * consistency_loss(encoder, x_all)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if checkpoint_path is not None and (epoch + 1) % checkpoint_every == 0:
            torch.save(
                {
                    "encoder_state": encoder.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "losses": losses,
                    "epoch": epoch + 1,
                },
                checkpoint_path,
            )

    return losses


def train_multi_source_mldg(
    encoder: nn.Module,
    source_windows: dict,
    epochs: int = 150,
    outer_lr: float = 5e-3,
    inner_lr: float = 1e-2,
    meta_test_weight: float = 1.0,
    consistency_weight: float = 0.05,
    checkpoint_path: str = None,
    checkpoint_every: int = 10,
) -> list:
    """`source_windows` maps region name -> (x, y) tensors for >= 2 source regions. Each
    epoch, one region is randomly held out as the meta-test domain and the rest are pooled,
    domain-weighted by `compute_domain_weight`, as meta-train; returns the per-epoch total
    (meta-train + meta-test) loss.

    `checkpoint_path` mirrors `train_multi_source_mldg_coral`'s resumability escape hatch: if
    the file exists, training resumes from its saved epoch/optimizer/loss-history state instead
    of starting over, and the file is updated every `checkpoint_every` epochs."""
    if len(source_windows) < 2:
        raise ValueError("MLDG needs at least 2 source regions to hold one out as meta-test")

    regions = list(source_windows.keys())
    optimizer = torch.optim.Adam(encoder.parameters(), lr=outer_lr)
    mse_loss = nn.MSELoss()
    losses = []
    start_epoch = 0

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        encoder.load_state_dict(checkpoint["encoder_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        losses = checkpoint["losses"]
        start_epoch = checkpoint["epoch"]

    for epoch in range(start_epoch, epochs):
        meta_test_region = random.choice(regions)
        meta_train_regions = [r for r in regions if r != meta_test_region]

        x_meta_train = torch.cat([source_windows[r][0] for r in meta_train_regions], dim=0)
        x_meta_test, y_meta_test = source_windows[meta_test_region]

        params = dict(encoder.named_parameters())
        buffers = dict(encoder.named_buffers())
        trainable_names = [name for name, p in params.items() if p.requires_grad]

        domain_weights = {r: compute_domain_weight(source_windows[r][1]) for r in meta_train_regions}
        weight_sum = sum(domain_weights.values())

        meta_train_loss = torch.zeros(())
        for region in meta_train_regions:
            x_r, y_r = source_windows[region]
            pred_r, _ = functional_call(encoder, (params, buffers), (x_r,))
            region_loss = mse_loss(pred_r, y_r)
            meta_train_loss = meta_train_loss + (domain_weights[region] / weight_sum) * region_loss
        meta_train_loss = meta_train_loss + consistency_weight * consistency_loss(encoder, x_meta_train)

        grads = torch.autograd.grad(
            meta_train_loss, [params[name] for name in trainable_names], create_graph=True, allow_unused=True
        )
        grad_by_name = dict(zip(trainable_names, grads))
        updated_params = {
            name: p if grad_by_name.get(name) is None else p - inner_lr * grad_by_name[name]
            for name, p in params.items()
        }

        pred_meta_test, _ = functional_call(encoder, (updated_params, buffers), (x_meta_test,))
        meta_test_loss = mse_loss(pred_meta_test, y_meta_test)

        total_loss = meta_train_loss + meta_test_weight * meta_test_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())

        if checkpoint_path is not None and (epoch + 1) % checkpoint_every == 0:
            torch.save(
                {
                    "encoder_state": encoder.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "losses": losses,
                    "epoch": epoch + 1,
                },
                checkpoint_path,
            )

    return losses
