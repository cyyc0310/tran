"""More fundamental SA1 domain-adaptation techniques, following two recent-conference
directions surfaced by literature search: (D) gradual-unfreezing supervised fine-tuning
of the MLDG-pretrained encoder on the real target-region calibration split -- modeled on
IBM Research's AAAI 2024 workshop paper "Domain Adaptation for Time series Transformers
using One-step fine-tuning" (pretrain on a data-rich source, fine-tune on a data-scarce
target with progressive unfreezing to fight catastrophic forgetting), and validated as
high-impact specifically for cross-region carbon-intensity forecasting by a WWW'26 paper
whose ablation showed removing target-region fine-tuning costs 11.4% MAPE; and (E) Deep
CORAL (Sun & Saenko 2016) unsupervised feature-covariance alignment between source-region
and target-region pooled encoder features, requiring only the target's unlabeled inputs.

Both are implemented as additions alongside (not modifications to) `train_multi_source.py`'s
tested `train_multi_source_mldg`, matching this project's existing convention of adding a
parallel variant function for each SA1-mitigation ablation switch rather than growing the
production function's parameter surface for an experimental toggle."""

import os
import random

import torch
import torch.nn as nn
from torch.func import functional_call

from transcif.training.consistency import consistency_loss
from transcif.training.train_multi_source import compute_domain_weight


def coral_loss(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    """Deep CORAL: L_CORAL = (1 / 4d^2) * || Cov(source) - Cov(target) ||_F^2, over the
    pooled `fused` feature vector from DomainInvariantEncoder.forward_features. Unlike a
    supervised loss, this needs only the target domain's *inputs* -- no target labels --
    so it applies directly to SA1's calibration-split x_calib regardless of whether the
    real y_target_share labels are trusted for a given experiment."""
    source_centered = source_features - source_features.mean(dim=0, keepdim=True)
    target_centered = target_features - target_features.mean(dim=0, keepdim=True)
    n_source = max(source_features.shape[0] - 1, 1)
    n_target = max(target_features.shape[0] - 1, 1)
    cov_source = (source_centered.T @ source_centered) / n_source
    cov_target = (target_centered.T @ target_centered) / n_target
    feature_dim = source_features.shape[1]
    return ((cov_source - cov_target) ** 2).sum() / (4 * feature_dim * feature_dim)


def train_multi_source_mldg_coral(
    encoder: nn.Module,
    source_windows: dict,
    x_target_unlabeled: torch.Tensor,
    epochs: int = 150,
    outer_lr: float = 5e-3,
    inner_lr: float = 1e-2,
    meta_test_weight: float = 1.0,
    consistency_weight: float = 0.05,
    coral_weight: float = 0.1,
    checkpoint_path: str = None,
    checkpoint_every: int = 10,
) -> list:
    """Direction E: mirrors `train_multi_source_mldg`'s domain-weighted MLDG loop exactly,
    adding a Deep CORAL term that aligns the pooled meta-train regions' `fused` features
    against the target region's unlabeled `fused` features -- computed on the real encoder
    (like `consistency_loss`), not on the virtual meta-test-updated parameters.

    `checkpoint_path` is an optional resumability escape hatch for environments that may
    kill a long-running process before all `epochs` complete: if the file exists, training
    resumes from its saved epoch/optimizer/loss-history state instead of starting over,
    and the file is updated every `checkpoint_every` epochs. Purely an execution-robustness
    concern -- does not change what is being trained or how many total epochs it sees."""
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

        source_fused, _ = encoder.base_encoder.forward_features(x_meta_train)
        target_fused, _ = encoder.base_encoder.forward_features(x_target_unlabeled)
        meta_train_loss = meta_train_loss + coral_weight * coral_loss(source_fused, target_fused)

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


DEFAULT_UNFREEZE_GROUPS = (
    ("gate_logit", "volatility_gain_raw", "base_encoder.predict"),
    ("base_encoder.cv_dwcc",),
    ("base_encoder.lt_mwkc",),
)


def fine_tune_on_calibration(
    model: nn.Module,
    x_calib: torch.Tensor,
    y_calib: torch.Tensor,
    epochs_per_stage: int = 20,
    lr: float = 1e-3,
    unfreeze_groups: tuple = DEFAULT_UNFREEZE_GROUPS,
) -> list:
    """Direction D: supervised fine-tuning of an MLDG-pretrained encoder on the real
    target-region calibration split, with gradual unfreezing across `unfreeze_groups` --
    each stage keeps every previously-unfrozen parameter trainable and adds one more group,
    following IBM Research's AAAI 2024 workshop "one-step fine-tuning + gradual unfreezing"
    method to fight catastrophic forgetting of the MLDG-pretrained source-region dynamics.
    A parameter is matched to a group by exact name or dotted-prefix (e.g.
    "base_encoder.cv_dwcc" matches "base_encoder.cv_dwcc.weight")."""
    mse_loss = nn.MSELoss()
    losses = []

    for param in model.parameters():
        param.requires_grad_(False)

    unfrozen_prefixes = []
    for group in unfreeze_groups:
        unfrozen_prefixes.extend(group)
        for name, param in model.named_parameters():
            if any(name == prefix or name.startswith(prefix + ".") for prefix in unfrozen_prefixes):
                param.requires_grad_(True)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        for _ in range(epochs_per_stage):
            pred, _ = model(x_calib)
            loss = mse_loss(pred, y_calib)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return losses
