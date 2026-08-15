"""Phys-IRM: Physics-Informed Invariant Risk Minimization for Zero-Shot CIF.

Extends the base TransCIF zero-shot pipeline with an IRM-style penalty that
encourages the share-prediction head to be invariant across source regions,
combined with a physics-consistency loss anchoring predictions to the CIF
closed form.  See ``RESEARCH_DIRECTIONS.md`` §2.3 for motivation.
"""

import random

import numpy as np
import torch
import torch.nn as nn

from transcif.config import SEQ_LEN, HORIZON
from transcif.models.base import AdaptivePersistDLinear
from transcif.data.windows import build_windows
from transcif.physics.decompose import cif_from_shares
from transcif.physics.bounds import config_weight, unify_config_dim, pad_config
from transcif.training.schedulers import get_cosine_warmup_scheduler


def train_phys_irm(all_regions, target_name, seed=42, epochs=200, lr=1e-3,
                    gamma_irm=0.1, lambda_cif=0.5, device=None, pbar=None):
    """Train a Phys-IRM model: base model + IRM penalty + physics loss."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cfg_dim = unify_config_dim(all_regions)
    model = AdaptivePersistDLinear(seq_len=SEQ_LEN, horizon=HORIZON, config_dim=cfg_dim)
    if device:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_cosine_warmup_scheduler(optimizer, max(1, epochs // 10), epochs)

    target_mean_rs = all_regions[target_name]["mean_rs"]
    region_data = {}
    for name, data in all_regions.items():
        if name == target_name:
            continue
        x_win, y_win, y_cif_win = build_windows(data["rs"], data["cif"])
        if len(x_win) == 0:
            continue
        # Config-distance source weight (matches base TransCIF-ZS): regions
        # close to the target in mean_rs contribute more gradient.  Stored per
        # region and applied to the share + CIF losses so the IRM cross-region
        # structure is preserved while still biasing toward the target.
        w = config_weight(data["mean_rs"], target_mean_rs)
        region_data[name] = {
            "x": torch.tensor(x_win, dtype=torch.float32),
            "y_share": torch.tensor(y_win, dtype=torch.float32),
            "y_cif": torch.tensor(y_cif_win, dtype=torch.float32),
            "config": torch.tensor(
                np.tile(pad_config(data["config"], cfg_dim), (len(x_win), 1)),
                dtype=torch.float32),
            "ef_r": data["ef_r"], "ef_nr": data["ef_nr"],
            "w": float(w),
        }

    model.train()
    log = []
    names = list(region_data.keys())
    # Physics-consistency uses the TARGET's emission factors: at inference the
    # share head is converted to CIF with the target's ef_r / ef_nr, so training
    # the share->CIF reconstruction with the same factors keeps train/serve
    # consistent.  The CIF loss is NORMALISED by the target region's CIF scale
    # so that high-renewable-share targets (where |ef_r - ef_nr| is large) do
    # not produce gradient explosions that destroy share learning.
    ef_r_tgt = torch.tensor(all_regions[target_name]["ef_r"],
                             dtype=torch.float32, device=device)
    ef_nr_tgt = torch.tensor(all_regions[target_name]["ef_nr"],
                             dtype=torch.float32, device=device)
    # Target-region CIF scale (detached) for normalising the physics loss.
    y_cif_tgt_all = torch.cat([region_data[n]["y_cif"] for n in names])
    if device:
        y_cif_tgt_all = y_cif_tgt_all.to(device)
    cif_scale = y_cif_tgt_all.abs().mean().detach() + 1e-6
    prev_region_share_loss = None  # detached per-region share risks (last epoch)
    for epoch in range(epochs):
        total_irm_penalty = 0.0
        n_regions = 0
        total_loss = 0.0
        cur_share_losses = {}
        for name in names:
            rd = region_data[name]
            n = rd["x"].shape[0]
            idx = torch.randperm(n)[:min(256, n)]
            x_b, y_s_b, c_b = rd["x"][idx], rd["y_share"][idx], rd["config"][idx]
            y_cif_b = rd["y_cif"][idx]
            if device:
                x_b, y_s_b, c_b = x_b.to(device), y_s_b.to(device), c_b.to(device)
                y_cif_b = y_cif_b.to(device)

            s_pred = model(x_b, c_b)
            # base share loss, config-distance weighted (per-region scalar w)
            w_e = rd["w"]
            L_share = w_e * torch.abs(s_pred - y_s_b).mean()
            # physics-consistency: CIF predicted via physics vs true CIF, with
            # the TARGET's factors, normalised so gradients stay on scale.
            cif_pred = s_pred * ef_r_tgt + (1.0 - s_pred) * ef_nr_tgt
            L_cif = w_e * (torch.abs(cif_pred - y_cif_b) / cif_scale).mean()
            # IRM penalty: pull THIS region's share risk toward the mean share
            # risk of the OTHER regions (cross-environment invariance, per
            # RESEARCH_DIRECTIONS.md §2.3).  Other regions' risks are detached
            # from the previous epoch; L_share keeps the graph so the penalty
            # actually backpropagates into the shared head.
            if prev_region_share_loss is not None and len(names) > 1:
                others = [prev_region_share_loss[m]
                          for m in names if m != name]
                mean_other = torch.stack(others).mean()
                irm_penalty = (L_share - mean_other) ** 2
            else:
                irm_penalty = torch.zeros_like(L_share)
            loss = L_share + lambda_cif * L_cif + gamma_irm * irm_penalty
            cur_share_losses[name] = L_share.detach()
            total_loss += loss.item()
            total_irm_penalty += irm_penalty.item()
            n_regions += 1
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        prev_region_share_loss = cur_share_losses
        scheduler.step()
        if pbar is not None:
            pbar(epoch, epochs, total_loss / max(n_regions, 1),
                 extra=f"irm={total_irm_penalty/max(n_regions,1):.4f}")
        log.append({
            "epoch": epoch + 1,
            "loss": total_loss / max(n_regions, 1),
            "irm_penalty": total_irm_penalty / max(n_regions, 1),
        })

    model.eval()
    if pbar is not None:
        pbar.finish()
    return model, log


def train_phys_weighted_only(all_regions, target_name, seed=42, epochs=200,
                             lr=1e-3, lambda_cif=0.5, device=None, pbar=None):
    """Ablation: physics-weighted (no IRM penalty) to isolate IRM benefit.

    Identical to ``train_phys_irm`` except for the IRM penalty: same config-
    distance source weighting, same target-factor normalised CIF loss, same
    cosine scheduler.  The only difference between the two models is the IRM
    term, so ``irm_benefit`` cleanly isolates it.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cfg_dim = unify_config_dim(all_regions)
    model = AdaptivePersistDLinear(seq_len=SEQ_LEN, horizon=HORIZON, config_dim=cfg_dim)
    if device:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_cosine_warmup_scheduler(optimizer, max(1, epochs // 10), epochs)

    target_mean_rs = all_regions[target_name]["mean_rs"]
    xs, ys, y_cifs, cfgs, ws = [], [], [], [], []
    for name, data in all_regions.items():
        if name == target_name:
            continue
        x_win, y_win, y_cif_win = build_windows(data["rs"], data["cif"])
        if len(x_win) == 0:
            continue
        w = config_weight(data["mean_rs"], target_mean_rs)
        xs.append(x_win)
        ys.append(y_win)
        y_cifs.append(y_cif_win)
        cfgs.append(np.tile(pad_config(data["config"], cfg_dim), (len(x_win), 1)))
        ws.append(np.full(len(x_win), w, dtype=np.float32))

    x_all = torch.tensor(np.concatenate(xs), dtype=torch.float32)
    y_s_all = torch.tensor(np.concatenate(ys), dtype=torch.float32)
    y_cif_all = torch.tensor(np.concatenate(y_cifs), dtype=torch.float32)
    c_all = torch.tensor(np.concatenate(cfgs), dtype=torch.float32)
    w_all = torch.tensor(np.concatenate(ws), dtype=torch.float32)
    w_all = w_all / w_all.sum() * len(w_all)
    n = len(x_all)
    batch_size = min(512, n)

    model.train()
    ef_r_tgt = torch.tensor(all_regions[target_name]["ef_r"],
                             dtype=torch.float32, device=device)
    ef_nr_tgt = torch.tensor(all_regions[target_name]["ef_nr"],
                             dtype=torch.float32, device=device)
    cif_scale = (y_cif_all.abs().mean().detach() + 1e-6).to(device) \
        if device else (y_cif_all.abs().mean().detach() + 1e-6)
    for epoch in range(epochs):
        idx = torch.multinomial(w_all, batch_size, replacement=True)
        x_b, y_s_b, c_b = x_all[idx], y_s_all[idx], c_all[idx]
        y_cif_b = y_cif_all[idx]
        if device:
            x_b, y_s_b, c_b = x_b.to(device), y_s_b.to(device), c_b.to(device)
            y_cif_b = y_cif_b.to(device)
        s_pred = model(x_b, c_b)
        L_share = torch.abs(s_pred - y_s_b).mean()
        # tensor-space CIF with the TARGET's factors (keep gradients)
        cif_pred = (s_pred * ef_r_tgt
                    + (1.0 - s_pred) * ef_nr_tgt)
        L_cif = (torch.abs(cif_pred - y_cif_b) / cif_scale).mean()
        loss = L_share + lambda_cif * L_cif
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if pbar is not None:
            pbar(epoch, epochs, loss.item())

    model.eval()
    if pbar is not None:
        pbar.finish()
    return model, []


def predict_phys_irm(model, x_rs, config, ef_r, ef_nr):
    """Zero-shot inference: share prediction -> physics CIF conversion."""
    model.eval()
    dev = next(model.parameters()).device
    x_t = torch.tensor(x_rs, dtype=torch.float32).to(dev)
    config = pad_config(np.asarray(config), getattr(model, "config_dim", len(config))) \
             if not isinstance(config, torch.Tensor) else config
    c_t = torch.tensor(config).unsqueeze(0).expand(len(x_rs), -1).to(dev)
    with torch.no_grad():
        s_pred = model(x_t, c_t).cpu().numpy()
    return cif_from_shares(s_pred, ef_r, ef_nr)


BATCH_SIZE = 512
