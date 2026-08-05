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


def train_phys_irm(all_regions, target_name, seed=42, epochs=200, lr=1e-3,
                    gamma_irm=0.1, lambda_cif=0.5, device=None):
    """Train a Phys-IRM model: base model + IRM penalty + physics loss."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = AdaptivePersistDLinear(seq_len=SEQ_LEN, horizon=HORIZON)
    if device:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    region_data = {}
    for name, data in all_regions.items():
        if name == target_name:
            continue
        x_win, y_win, y_cif_win = build_windows(data["rs"], data["cif"])
        if len(x_win) == 0:
            continue
        region_data[name] = {
            "x": torch.tensor(x_win, dtype=torch.float32),
            "y_share": torch.tensor(y_win, dtype=torch.float32),
            "y_cif": torch.tensor(y_cif_win, dtype=torch.float32),
            "config": torch.tensor(
                np.tile(data["config"], (len(x_win), 1)), dtype=torch.float32),
            "ef_r": data["ef_r"], "ef_nr": data["ef_nr"],
        }

    model.train()
    log = []
    names = list(region_data.keys())
    # Physics-consistency must use the TARGET's emission factors, because at
    # inference time the share prediction is converted to CIF with the target's
    # ef_r / ef_nr.  Training with each source region's own factors would
    # create a train/serving mismatch.
    ef_r_tgt = all_regions[target_name]["ef_r"]
    ef_nr_tgt = all_regions[target_name]["ef_nr"]
    for epoch in range(epochs):
        total_irm_penalty = 0.0
        n_regions = 0
        total_loss = 0.0
        for name in names:
            rd = region_data[name]
            n = rd["x"].shape[0]
            idx = torch.randperm(n)[:min(256, n)]
            x_b, y_s_b, c_b = rd["x"][idx], rd["y_share"][idx], rd["config"][idx]
            if device:
                x_b, y_s_b, c_b = x_b.to(device), y_s_b.to(device), c_b.to(device)

            s_pred = model(x_b, c_b)
            # base share loss
            L_share = torch.abs(s_pred - y_s_b).mean()
            # physics-consistency: CIF predicted via physics vs true CIF,
            # expressed with the target region's emission factors.  Computed
            # in tensor space (NOT cif_from_shares, which returns numpy) so the
            # loss keeps gradients flowing back into the share head.
            cif_pred = s_pred * ef_r_tgt + (1.0 - s_pred) * ef_nr_tgt
            L_cif = torch.abs(cif_pred - rd["y_cif"][idx]).mean()
            # IRM penalty: variance of per-sample loss across the region batch
            sample_loss = torch.abs(s_pred - y_s_b).mean(dim=1)
            irm_penalty = sample_loss.var()
            loss = L_share + lambda_cif * L_cif + gamma_irm * irm_penalty
            total_loss += loss.item()
            total_irm_penalty += irm_penalty.item()
            n_regions += 1
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            log.append({
                "epoch": epoch + 1,
                "loss": total_loss / max(n_regions, 1),
                "irm_penalty": total_irm_penalty / max(n_regions, 1),
            })

    model.eval()
    return model, log


def train_phys_weighted_only(all_regions, target_name, seed=42, epochs=200,
                             lr=1e-3, lambda_cif=0.5, device=None):
    """Ablation: physics-weighted (no IRM penalty) to isolate IRM benefit."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = AdaptivePersistDLinear(seq_len=SEQ_LEN, horizon=HORIZON)
    if device:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    xs, ys, y_cifs, cfgs, ef_r_list, ef_nr_list = [], [], [], [], [], []
    for name, data in all_regions.items():
        if name == target_name:
            continue
        x_win, y_win, y_cif_win = build_windows(data["rs"], data["cif"])
        if len(x_win) == 0:
            continue
        xs.append(x_win)
        ys.append(y_win)
        y_cifs.append(y_cif_win)
        cfgs.append(np.tile(data["config"], (len(x_win), 1)))
        ef_r_list.append(np.full(len(x_win), data["ef_r"], dtype=np.float32))
        ef_nr_list.append(np.full(len(x_win), data["ef_nr"], dtype=np.float32))

    x_all = torch.tensor(np.concatenate(xs), dtype=torch.float32)
    y_s_all = torch.tensor(np.concatenate(ys), dtype=torch.float32)
    y_cif_all = torch.tensor(np.concatenate(y_cifs), dtype=torch.float32)
    c_all = torch.tensor(np.concatenate(cfgs), dtype=torch.float32)
    ef_r_all = torch.tensor(np.concatenate(ef_r_list), dtype=torch.float32)
    ef_nr_all = torch.tensor(np.concatenate(ef_nr_list), dtype=torch.float32)
    n = len(x_all)
    batch_size = min(512, n)

    model.train()
    # Ablation baseline for Phys-IRM: physics-weighted share training WITH the
    # CIF consistency loss but WITHOUT the IRM variance penalty.  This isolates
    # the contribution of the IRM term.  The CIF target is reconstructed from
    # each source region's own emission factors (consistent with its y_cif).
    for epoch in range(epochs):
        idx = torch.randperm(n)[:batch_size]
        x_b, y_s_b, c_b = x_all[idx], y_s_all[idx], c_all[idx]
        ef_r_b, ef_nr_b = ef_r_all[idx], ef_nr_all[idx]
        y_cif_b = y_cif_all[idx]
        if device:
            x_b, y_s_b, c_b = x_b.to(device), y_s_b.to(device), c_b.to(device)
            ef_r_b, ef_nr_b = ef_r_b.to(device), ef_nr_b.to(device)
            y_cif_b = y_cif_b.to(device)
        s_pred = model(x_b, c_b)
        L_share = torch.abs(s_pred - y_s_b).mean()
        # tensor-space CIF (keep gradients); ef_r_b/ef_nr_b are (B,) tensors
        cif_pred = (s_pred * ef_r_b.unsqueeze(1)
                    + (1.0 - s_pred) * ef_nr_b.unsqueeze(1))
        L_cif = torch.abs(cif_pred - y_cif_b).mean()
        loss = L_share + lambda_cif * L_cif
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    return model, []


def predict_phys_irm(model, x_rs, config, ef_r, ef_nr):
    """Zero-shot inference: share prediction -> physics CIF conversion."""
    model.eval()
    dev = next(model.parameters()).device
    x_t = torch.tensor(x_rs, dtype=torch.float32).to(dev)
    c_t = torch.tensor(config).unsqueeze(0).expand(len(x_rs), -1).to(dev)
    with torch.no_grad():
        s_pred = model(x_t, c_t).cpu().numpy()
    return cif_from_shares(s_pred, ef_r, ef_nr)


BATCH_SIZE = 512
