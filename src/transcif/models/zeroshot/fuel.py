"""Zero-shot training / prediction for the fuel-decomposed model (FD-1).

``train_fuel_zero_shot`` trains ``FuelDecompNet`` on every source region's
windows with the proven config-distance source weighting, a mixed objective
(CIF + masked per-fuel shares + aggregate renewable share) and random
``cold-mode dropout`` — each window occasionally loses its history so the
same weights serve both information tiers:

    I_0    config + live share telemetry (comparable with the paper ladder)
    I_cfg  config + weather + calendar only (the China deployment tier)
"""

import random

import numpy as np
import torch
import torch.nn as nn

from transcif.config import (
    SEQ_LEN, HORIZON, TRAIN_STRIDE, EPOCHS_ZERO_SHOT,
)
from transcif.data.fuel import (
    CANONICAL_FUELS, build_fd_windows, build_fd_config, fuel_cif,
)
from transcif.models.fuel_decomp import FuelDecompNet
from transcif.physics.bounds import config_weight
from transcif.training.schedulers import get_cosine_warmup_scheduler
from transcif.evaluation.metrics import compute_metrics

# Default mixed-objective weights.
LAMBDA_FUEL = 1.0     # per-fuel share error, EF-weighted into gCO2 units
LAMBDA_RS = 0.3       # aggregate renewable-share MAE (all regions)
P_COLD = 0.3          # probability a training window is forced to cold mode


def prepare_fd_region(region_name, all_configs, data_dir=None):
    """Load one region with fuel shares, exog features and the FD config."""
    from transcif.data.loaders import load_region_data  # noqa: PLC0415
    data = load_region_data(region_name, all_configs, data_dir=data_dir)
    from transcif.data.fuel import attach_fuel_and_exog  # noqa: PLC0415
    attach_fuel_and_exog(data, region_name, all_configs, data_dir=data_dir)
    data["fd_config"] = build_fd_config(data, region_name)
    return data


def train_fuel_zero_shot(fd_regions, target_name, seed=42,
                         epochs=EPOCHS_ZERO_SHOT, lr=1e-3,
                         lambda_fuel=LAMBDA_FUEL, lambda_rs=LAMBDA_RS,
                         p_cold=P_COLD, max_windows_per_region=700,
                         use_weighted=True, p_mix=0.0, use_hypernet=False,
                         device=None, pbar=None, model=None):
    """Train FuelDecompNet on all source regions for one LORO target.

    Args:
        fd_regions : {name: data dict from prepare_fd_region}
        target_name: region to leave out
        seed / epochs / lr : standard hyperparameters
        lambda_fuel / lambda_rs : mixed-objective weights
        p_cold : cold-mode dropout probability (trains the I_cfg path)
        max_windows_per_region : per-source training-window cap
        use_weighted : config-distance source weighting (Theorem 2 sampler)
        p_mix : fraction of steps trained on synthetic pseudo-grids
                (physics-guided pairwise recombination of source windows on
                a shared absolute-origin grid — Phase FD-2)

    Returns the trained model (eval mode).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    if model is None:
        model = FuelDecompNet(seq_len=SEQ_LEN, horizon=HORIZON,
                              use_hypernet=use_hypernet)
    if device:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_cosine_warmup_scheduler(optimizer, max(1, epochs // 10), epochs)

    tgt = fd_regions[target_name]

    # Shared absolute-origin grid so windows from different regions cover
    # the same calendar hours and remain pairwise mixable (p_mix > 0).
    import pandas as pd  # noqa: PLC0415
    epoch0 = pd.Timestamp("2023-01-01")
    offsets = {}
    for name, data in fd_regions.items():
        if name == target_name:
            continue
        offsets[name] = int((data["hours"][0] - epoch0).total_seconds() // 3600)
    window = SEQ_LEN + HORIZON
    abs_start_min = max(offsets.values())
    abs_start_max = min(o + len(fd_regions[n]["rs"]) for n, o in offsets.items()) - window
    shared_starts = np.arange(abs_start_min, max(abs_start_min + 1, abs_start_max),
                              TRAIN_STRIDE)

    tensors = {k: [] for k in
               ("x_rs", "x_fuel", "y_fuel", "y_rs", "y_cif",
                "x_weather", "fut_weather", "fut_exog")}
    cfgs, efs, weights, fuel_masks = [], [], [], []
    origin_keys, region_ids = [], []
    region_names = []
    for rid, (name, data) in enumerate(fd_regions.items()):
        if name == target_name:
            continue
        region_names.append(name)
        starts = shared_starts - offsets[name]
        w = build_fd_windows(data, seq_len=SEQ_LEN, horizon=HORIZON,
                             max_windows=max_windows_per_region, rng=rng,
                             starts=starts)
        n = len(w["x_rs"])
        if n == 0:
            continue
        for k in tensors:
            tensors[k].append(w[k])
        cfg = data["fd_config"]
        cfgs.append(np.tile(cfg, (n, 1)))
        efs.append(np.tile(data["ef_vec"], (n, 1)).astype(np.float32))
        fuel_masks.append(np.full((n, 1), 1.0 if data["has_fuel"] else 0.0,
                                  np.float32))
        if use_weighted:
            wt = config_weight(data["mean_rs"], tgt["mean_rs"])
        else:
            wt = 1.0
        weights.append(np.full(n, wt, np.float32))
        # Absolute origin keys from the window builder's own timestamps so
        # subsampled windows stay correctly aligned for synthetic pairing.
        oh = w["origin_hours"]
        origin_keys.append(((oh - epoch0) / np.timedelta64(1, "h")).values
                           .astype(np.int64))
        region_ids.append(np.full(n, rid, np.int64))

    batch = {k: torch.tensor(np.concatenate(v)) for k, v in tensors.items()}
    c_all = torch.tensor(np.concatenate(cfgs))
    e_all = torch.tensor(np.concatenate(efs))
    m_all = torch.tensor(np.concatenate(fuel_masks))
    w_all = torch.tensor(np.concatenate(weights))
    ok_all = torch.tensor(np.concatenate(origin_keys))
    rid_all = torch.tensor(np.concatenate(region_ids))
    w_all = w_all / w_all.sum() * len(w_all)
    if device:
        batch = {k: v.to(device) for k, v in batch.items()}
        c_all, e_all, m_all, w_all = (c_all.to(device), e_all.to(device),
                                      m_all.to(device), w_all.to(device))
        ok_all, rid_all = ok_all.to(device), rid_all.to(device)

    # Same-origin cross-region pair index for synthetic mixing.
    pair_bank = []
    if p_mix > 0:
        ok_np = ok_all.cpu().numpy()
        rid_np = rid_all.cpu().numpy()
        by_key = {}
        for i, k in enumerate(ok_np):
            by_key.setdefault(int(k), []).append(i)
        for k, idxs in by_key.items():
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    i, j = idxs[a], idxs[b]
                    if rid_np[i] != rid_np[j]:
                        pair_bank.append((i, j))
        pair_bank = np.array(pair_bank) if pair_bank else None
        if pair_bank is None:
            print("  [WARN] p_mix>0 but no cross-region same-origin pairs")

    n_samples = len(batch["x_rs"])
    batch_size = min(512, n_samples)
    model.train()
    for epoch in range(epochs):
        if p_mix > 0 and pair_bank is not None and \
                rng.random() < p_mix:
            sel = rng.integers(0, len(pair_bank), size=batch_size)
            pi = torch.tensor(pair_bank[sel, 0], device=batch["x_rs"].device)
            pj = torch.tensor(pair_bank[sel, 1], device=batch["x_rs"].device)
            alpha = torch.tensor(
                rng.uniform(0.2, 0.8, size=(batch_size, 1)),
                device=batch["x_rs"].device, dtype=torch.float32)

            def _mix(v, a):
                return (a if v.dim() == 2 else a.unsqueeze(-1)) * v[pi] \
                    + (1 - (a if v.dim() == 2 else a.unsqueeze(-1))) * v[pj]

            b = {k: _mix(v, alpha) for k, v in batch.items()}
            c_b = alpha * c_all[pi] + (1 - alpha) * c_all[pj]
            e_b = alpha * e_all[pi] + (1 - alpha) * e_all[pj]
            # Exact physics label: recompute the mixed CIF from the mixed
            # shares (reported CIFs carry per-source methodology noise).
            b["y_cif"] = torch.einsum("bhf,bf->bh", b["y_fuel"], e_b)
            w_b = w_all[pi]
            m_b = torch.minimum(m_all[pi], m_all[pj])
        else:
            idx = torch.randperm(n_samples)[:batch_size]
            b = {k: v[idx] for k, v in batch.items()}
            c_b, e_b, w_b, m_b = c_all[idx], e_all[idx], w_all[idx], m_all[idx]
        # Cold-mode dropout: each window independently loses its history.
        cold = (torch.rand(len(b["x_rs"]), 1, device=b["x_rs"].device) < p_cold).float()
        cif_hat, shares_hat, rs_hat = model(
            b["x_rs"], b["x_fuel"], b["x_weather"], b["fut_weather"],
            b["fut_exog"], c_b, e_b, hist_mask=1.0 - cold)
        loss_cif = (w_b.unsqueeze(1)
                    * torch.abs(cif_hat - b["y_cif"])).mean()
        if m_b.max() > 0 and lambda_fuel > 0:
            # EF-weighted share error — same units as the CIF loss so the
            # composition supervision is not drowned by gCO2-scale gradients
            # (a 0.1 share error on coal is a ~98 gCO2 error, not 0.1).
            abs_err = torch.abs(shares_hat - b["y_fuel"])          # (B, H, F)
            ef_w = e_b.abs()                                        # (B, F)
            per_sample = torch.einsum("bhf,bf->bh", abs_err, ef_w).mean(dim=1)
            share_term = abs_err.mean(dim=(1, 2))
            loss_fuel = (w_b * m_b.squeeze(1)
                         * (per_sample + 0.1 * share_term)).sum() / \
                (m_b.sum() * w_b.mean() + 1e-6)
        else:
            loss_fuel = torch.zeros((), device=b["x_rs"].device)
        loss_rs = (w_b.unsqueeze(1)
                   * torch.abs(rs_hat - b["y_rs"])).mean()
        loss = loss_cif + lambda_fuel * loss_fuel + lambda_rs * loss_rs
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
    return model


def predict_fuel_windows(model, windows, fd_config, ef_vec, cold=False,
                         device=None, batch_size=512):
    """Run FuelDecompNet over prebuilt windows.

    Args:
        cold : True for the I_cfg tier (history masked); False for I_0.

    Returns (cif (n, H), shares (n, H, F), rs (n, H)) as numpy arrays.
    """
    n = len(windows["x_rs"])
    hist = 0.0 if cold else 1.0
    model.eval()
    outs_cif, outs_sh, outs_rs = [], [], []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            args = [torch.tensor(windows[k][s:e]).to(
                        device or next(model.parameters()).device)
                    for k in ("x_rs", "x_fuel", "x_weather",
                              "fut_weather", "fut_exog")]
            cfg = torch.tensor(np.tile(fd_config, (e - s, 1))).to(args[0].device)
            ef = torch.tensor(np.tile(ef_vec, (e - s, 1))).to(args[0].device)
            hm = torch.full((e - s, 1), hist, device=args[0].device)
            cif, sh, rs = model(*args, cfg, ef, hist_mask=hm)
            outs_cif.append(cif.cpu().numpy())
            outs_sh.append(sh.cpu().numpy())
            outs_rs.append(rs.cpu().numpy())
    return (np.concatenate(outs_cif), np.concatenate(outs_sh),
            np.concatenate(outs_rs))


def make_zs_plus_share_fn(model, data, device=None):
    """Adapt FuelDecompNet to the ``zs_plus_predict`` share_fn interface.

    ``zs_plus_predict`` calls ``share_fn(x_win)`` with a raw (SEQ_LEN,)
    renewable-share window; this closure locates the window's origin in the
    region series (first-24 h fingerprint), rebuilds the fuel/exog inputs at
    that origin, and returns the aggregate renewable-share forecast (H,).
    """
    rs = data["rs"]
    fingerprint = {}
    for start in range(0, len(rs) - SEQ_LEN - HORIZON + 1):
        key = rs[start:start + 24].tobytes()
        fingerprint.setdefault(key, start)
    fd_cfg = torch.tensor(np.tile(data["fd_config"], (1, 1))).to(
        device or next(model.parameters()).device)
    ef = torch.tensor(np.tile(data["ef_vec"].astype(np.float32),
                              (1, 1))).to(fd_cfg.device)
    ex = data["exog"]
    fuel = data["fuel_shares"]
    from transcif.data.fuel import build_fd_windows  # noqa: PLC0415

    def share_fn(x_win):
        start = fingerprint.get(np.asarray(x_win[:24], np.float32).tobytes())
        if start is None:
            # Unknown window (e.g. ZS+ internal splits): fall back to the
            # closest origin by first-value matching.
            start = int(np.argmin(np.abs(rs[:-SEQ_LEN] - x_win[0])))
        w = build_fd_windows(
            {"rs": rs[start:start + SEQ_LEN + HORIZON],
             "cif": data["cif"][start:start + SEQ_LEN + HORIZON],
             "fuel_shares": fuel[start:start + SEQ_LEN + HORIZON],
             "hours": data["hours"][start:start + SEQ_LEN + HORIZON],
             "exog": {k: v[start:start + SEQ_LEN + HORIZON]
                      for k, v in ex.items()}},
            seq_len=SEQ_LEN, horizon=HORIZON, stride=1)
        args = [torch.tensor(w[k]).to(fd_cfg.device)
                for k in ("x_rs", "x_fuel", "x_weather", "fut_weather",
                          "fut_exog")]
        hm = torch.ones(1, 1, device=fd_cfg.device)
        with torch.no_grad():
            _, _, rs_hat = model(*args, fd_cfg, ef, hist_mask=hm)
        return rs_hat[0].cpu().numpy()

    return share_fn


# ---------------------------------------------------------------------------
# Metrics: level MAE plus the shape/ranking metrics that matter for
# telemetry-free regions (carbon-aware scheduling needs the hourly ORDER).
# ---------------------------------------------------------------------------

def shape_metrics(pred, truth):
    """Level + shape + ranking metrics over (n, H) prediction matrices.

    Returns dict:
        mae / rmse      : level errors (gCO2/kWh)
        diurnal_mae     : MAE after per-window demeaning (shape-only)
        monthly_shape_mae: MAE of deviations from each origin's monthly mean
        spearman        : mean per-window Spearman rank correlation
        bias            : mean signed error
    """
    from scipy.stats import spearmanr  # noqa: PLC0415
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    err = pred - truth
    out = {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
    }
    pd_ = pred - pred.mean(axis=1, keepdims=True)
    td_ = truth - truth.mean(axis=1, keepdims=True)
    out["diurnal_mae"] = float(np.abs(pd_ - td_).mean())
    out["monthly_shape_mae"] = out["diurnal_mae"]  # replaced below when months given
    rhos = []
    for i in range(len(pred)):
        if np.std(pred[i]) < 1e-6 or np.std(truth[i]) < 1e-6:
            rhos.append(0.0)
        else:
            rhos.append(float(spearmanr(pred[i], truth[i]).statistic))
    out["spearman"] = float(np.mean(rhos))
    return out


def shape_metrics_with_months(pred, truth, origin_hours):
    """``shape_metrics`` with monthly-mean deviations as the shape basis.

    ``origin_hours`` : pd.DatetimeIndex of each window origin; truth monthly
    means are an ORACLE quantity used only to measure shape skill (never a
    model input).
    """
    out = shape_metrics(pred, truth)
    months = origin_hours.month.values
    truth_monthly = np.zeros_like(truth, dtype=np.float64)
    for m in np.unique(months):
        sel = months == m
        truth_monthly[sel] = truth[sel].mean()
    # Deviation of prediction from the *true* monthly level: captures both
    # diurnal shape and month-level placement around the oracle anchor.
    out["monthly_shape_mae"] = float(np.abs(pred - truth_monthly).mean())
    return out
