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
LAMBDA_SHAPE = 0.5    # per-window DEmeaned CIF MAE — the diurnal-shape term
                      # (FD-6: level is near-oracle at I_cfg; shape is the
                      # frontier — this term optimises it directly without
                      # touching the level anchoring)
P_COLD = 0.3          # probability a training window is forced to cold mode


def fuel_config_weight(src, tgt):
    """Source weighting that upgrades the legacy mean_rs distance with the
    10-dim fuel-structure distance (MAE-floor analysis lever #2; Theorem 2:
    config-structure distance is the better transfer proxy).

    ``legacy / (0.5 * L1(fuel shares) + 0.1)`` when both regions have fuel
    telemetry; the legacy ``1/(|Δmean_rs|+0.05)`` weight otherwise (AU
    regions carry no fuel config — falls back to the proven sampler).
    """
    legacy = config_weight(src["mean_rs"], tgt["mean_rs"])
    if not (src.get("has_fuel") and tgt.get("has_fuel")):
        return legacy
    a = src["fd_config"][2:12]
    b = tgt["fd_config"][2:12]
    dist = float(np.abs(np.asarray(a) - np.asarray(b)).sum())
    return legacy / (0.5 * dist + 0.1)


def prepare_fd_region(region_name, all_configs, data_dir=None, multi_year=False,
                      monthly_history_only=False, use_au_state=False):
    """Load one region with fuel shares, exog features and the FD config."""
    from transcif.data.loaders import load_region_data  # noqa: PLC0415
    data = load_region_data(region_name, all_configs, data_dir=data_dir,
                            multi_year=multi_year)
    from transcif.data.fuel import (attach_fuel_and_exog,  # noqa: PLC0415
                                    build_fd_config,
                                    build_monthly_config_table)
    attach_fuel_and_exog(data, region_name, all_configs, data_dir=data_dir,
                         use_au_state=use_au_state)
    data["fd_config"] = build_fd_config(data, region_name)
    data["monthly_table"] = build_monthly_config_table(
        data, region_name, history_only=monthly_history_only)
    return data


def train_fuel_zero_shot(fd_regions, target_name, seed=42,
                         epochs=EPOCHS_ZERO_SHOT, lr=1e-3,
                         lambda_fuel=LAMBDA_FUEL, lambda_rs=LAMBDA_RS,
                         lambda_shape=LAMBDA_SHAPE,
                         p_cold=P_COLD, max_windows_per_region=700,
                         use_weighted=True, p_mix=0.0, use_hypernet=False,
                         weight_mode="legacy", ef_corr_bound=0.35,
                         use_monthly=False, lag_months=1,
                         evening_weight=1.0, solar_mod_bound=0.4,
                         wind_route_tau=1.1,
                         dynamic_residual=False,
                         dynamic_residual_bound=220.0,
                         same_jurisdiction=False,
                         domain_penalty=0.0, physics_target=False,
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
        domain_penalty : optional source-domain risk regularizer.  It adds
                the standard deviation of per-region batch risks to the
                ERM objective, which discourages a global model from winning
                by sacrificing one source regime.  It is disabled by
                default because route selection is the validated gain.
        physics_target : use ``future fuel shares × effective EF`` as the
                CIF target for source regions with fuel telemetry.  This
                removes source-specific reporting/accounting noise from the
                shared forecaster; telemetry-free sources keep their report
                CIF target.  Disabled by default for apples-to-apples
                comparison with earlier FD runs.

    Returns the trained model (eval mode).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    if model is None:
        model = FuelDecompNet(seq_len=SEQ_LEN, horizon=HORIZON,
                              use_hypernet=use_hypernet,
                              ef_corr_bound=ef_corr_bound,
                              solar_mod_bound=solar_mod_bound,
                              wind_route_tau=wind_route_tau,
                              dynamic_residual=dynamic_residual,
                              dynamic_residual_bound=dynamic_residual_bound)
    if device:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_cosine_warmup_scheduler(optimizer, max(1, epochs // 10), epochs)

    tgt = fd_regions[target_name]

    # Shared absolute-origin grid so windows from different regions cover
    # the same calendar hours and remain pairwise mixable (p_mix > 0).
    import pandas as pd  # noqa: PLC0415
    epoch0 = pd.Timestamp("2023-01-01")
    from transcif.data.fuel import jurisdiction_of
    offsets = {}
    for name, data in fd_regions.items():
        if name == target_name:
            continue
        if same_jurisdiction and jurisdiction_of(name) != jurisdiction_of(target_name):
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
        if same_jurisdiction and jurisdiction_of(name) != jurisdiction_of(target_name):
            continue
        region_names.append(name)
        starts = shared_starts - offsets[name]
        w = build_fd_windows(data, seq_len=SEQ_LEN, horizon=HORIZON,
                             max_windows=max_windows_per_region, rng=rng,
                             starts=starts,
                             monthly_table=(data.get("monthly_table")
                                            if use_monthly else None),
                             lag_months=lag_months)
        n = len(w["x_rs"])
        if n == 0:
            continue
        for k in tensors:
            tensors[k].append(w[k])
        cfg = data["fd_config"]
        if "config" in w:
            cfgs.append(w["config"])
        else:
            cfgs.append(np.tile(cfg, (n, 1)))
        efs.append(np.tile(data["ef_vec"], (n, 1)).astype(np.float32))
        fuel_masks.append(np.full((n, 1), 1.0 if data["has_fuel"] else 0.0,
                                  np.float32))
        if use_weighted:
            if weight_mode == "fuel":
                wt = fuel_config_weight(data, tgt)
            else:
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
    # Evening-peak loss weighting (roadmap C-class): recover the local hour
    # from the calendar sin/cos channels (indices 4/5 of fut_exog) and
    # weight the 17-21 h horizon targets up (the duck-curve ramp).
    hour_w = None
    if evening_weight != 1.0:
        sin_h = batch["fut_exog"][..., 4]
        cos_h = batch["fut_exog"][..., 5]
        frac = torch.remainder(torch.atan2(sin_h, cos_h) / (2 * torch.pi), 1.0)
        hour = (frac + 0.5 / 24.0) * 24.0
        hour_w = torch.where((hour >= 17.0) & (hour < 21.0),
                             evening_weight, 1.0).to(batch["x_rs"].device)
    model.train()
    for epoch in range(epochs):
        domain_b = None
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
            hw_b = hour_w[pi] if hour_w is not None else None
        else:
            idx = torch.randperm(n_samples)[:batch_size]
            b = {k: v[idx] for k, v in batch.items()}
            c_b, e_b, w_b, m_b = c_all[idx], e_all[idx], w_all[idx], m_all[idx]
            hw_b = hour_w[idx] if hour_w is not None else None
            domain_b = rid_all[idx]
        # Cold-mode dropout: each window independently loses its history.
        cold = (torch.rand(len(b["x_rs"]), 1, device=b["x_rs"].device) < p_cold).float()
        cif_hat, shares_hat, rs_hat = model(
            b["x_rs"], b["x_fuel"], b["x_weather"], b["fut_weather"],
            b["fut_exog"], c_b, e_b, hist_mask=1.0 - cold)
        y_cif_loss = b["y_cif"]
        if physics_target and m_b.max() > 0:
            y_phys = torch.einsum("bhf,bf->bh", b["y_fuel"], e_b)
            y_cif_loss = torch.where(m_b > 0.5, y_phys, y_cif_loss)
        per_el = torch.abs(cif_hat - y_cif_loss)               # (B, H)
        if hw_b is not None:
            # Evening-peak up-weighting, scale-normalised so the loss
            # magnitude stays comparable across evening_weight settings.
            per_el = per_el * hw_b / hw_b.mean().clamp_min(1e-6)
        loss_cif = (w_b.unsqueeze(1) * per_el).mean()
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
        if lambda_shape > 0:
            # Diurnal-shape term: per-window demeaned CIF MAE.  Level skill
            # comes from the config/anchor pathways; this term focuses the
            # gradient budget on the intra-day trajectory (what Spearman
            # and carbon-aware scheduling consume) without letting level
            # errors dominate the objective.
            pred_dm = cif_hat - cif_hat.mean(dim=1, keepdim=True)
            y_dm = y_cif_loss - y_cif_loss.mean(dim=1, keepdim=True)
            shape_el = torch.abs(pred_dm - y_dm)
            if hw_b is not None:
                shape_el = shape_el * hw_b / hw_b.mean().clamp_min(1e-6)
            loss_shape = (w_b.unsqueeze(1) * shape_el).mean()
            loss = loss + lambda_shape * loss_shape
        # Domain-generalisation regularizer (FOIL-inspired, but deliberately
        # lightweight): source region IDs are available during training, so
        # penalise dispersion of per-region risks within a minibatch.  This
        # preserves the physics architecture and avoids a learned domain
        # adapter, while reducing the chance that a few easy grids dominate
        # the shared objective.  Synthetic mixed batches have no single
        # domain label and therefore skip this term.
        if domain_penalty > 0 and domain_b is not None:
            group_risks = []
            for gid in torch.unique(domain_b):
                gm = domain_b == gid
                if gm.any():
                    group_risks.append(per_el[gm].mean())
            if len(group_risks) > 1:
                loss = loss + domain_penalty * torch.stack(group_risks).std(
                    unbiased=False)
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


def finetune_fuel_supervised(model, data, epochs=120, lr=3e-4,
                             lambda_fuel=1.0, lambda_rs=0.3,
                             lambda_shape=0.5, device=None, seed=0):
    """Same-architecture supervised upper bound (I_S tier).

    Fine-tunes a zero-shot-trained FuelDecompNet on the TARGET region's
    train split (first ``TRAIN_FRACTION`` of the year, the I_S protocol's
    80% local labels) with the same four-loss objective, no cold-mode
    dropout and no cross-region mixing — pure local adaptation.  Returns
    the model in eval mode; the caller predicts the untouched test split.
    """
    from transcif.data.fuel import build_fd_windows  # noqa: PLC0415
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device) if device else None
    model.train()
    split = int(len(data["rs"]) * 0.8)
    w = build_fd_windows(
        {**data, "rs": data["rs"][:split], "cif": data["cif"][:split],
         "fuel_shares": data["fuel_shares"][:split],
         "hours": data["hours"][:split],
         "exog": {k: v[:split] for k, v in data["exog"].items()}},
        seq_len=SEQ_LEN, horizon=HORIZON, stride=24,
        max_windows=700, rng=np.random.default_rng(seed))
    tensors = {k: torch.tensor(np.asarray(v), dtype=torch.float32,
                               device=device)
               for k, v in w.items() if isinstance(v, np.ndarray)}
    cfg = torch.tensor(data["fd_config"], dtype=torch.float32, device=device)
    ef = torch.tensor(data["ef_vec"], dtype=torch.float32, device=device)
    has_fuel = 1.0 if data.get("has_fuel") else 0.0
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(tensors["x_rs"])
    if n == 0:
        return model
    for _ in range(epochs):
        idx = torch.randperm(n)[:64]
        b = {k: v[idx] for k, v in tensors.items()}
        B = len(b["x_rs"])
        ef_b = ef.expand(B, -1)
        cif_hat, shares_hat, rs_hat = model(
            b["x_rs"], b["x_fuel"], b["x_weather"], b["fut_weather"],
            b["fut_exog"], cfg.expand(B, -1), ef_b)
        loss = torch.abs(cif_hat - b["y_cif"]).mean()
        if has_fuel:
            abs_err = torch.abs(shares_hat - b["y_fuel"])
            ef_w = ef_b.abs()
            per_s = torch.einsum("bhf,bf->bh", abs_err, ef_w).mean(dim=1)
            loss = loss + lambda_fuel * (
                per_s + 0.1 * abs_err.mean(dim=(1, 2))).mean()
        loss = loss + lambda_rs * torch.abs(rs_hat - b["y_rs"]).mean()
        pred_dm = cif_hat - cif_hat.mean(dim=1, keepdim=True)
        y_dm = b["y_cif"] - b["y_cif"].mean(dim=1, keepdim=True)
        loss = loss + lambda_shape * torch.abs(pred_dm - y_dm).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return model


def predict_fuel_windows(model, windows, fd_config, ef_vec, cold=False,
                         device=None, batch_size=512):
    """Run FuelDecompNet over prebuilt windows.

    Args:
        cold : True for the I_cfg tier (history masked); False for I_0.
        windows : may carry per-window ``config`` (n, D) from a monthly
                  table; otherwise the region-level ``fd_config`` is tiled.

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
            dev = args[0].device
            if "config" in windows:
                cfg = torch.tensor(windows["config"][s:e]).to(dev)
            else:
                cfg = torch.tensor(
                    np.tile(fd_config, (e - s, 1))).to(dev)
            ef = torch.tensor(np.tile(ef_vec, (e - s, 1))).to(dev)
            hm = torch.full((e - s, 1), hist, device=dev)
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


def apply_day_ahead_weather_error(windows, rng,
                                  sigma_temp=2.0, sigma_sw=0.25,
                                  sigma_wind=0.20):
    """Degrade FUTURE weather channels to day-ahead NWP forecast skill.

    Day-ahead evaluation legality: astronomy and calendar channels stay
    exact (deterministic for any future date) and past weather stays
    reanalysis (observable history); only the 24 h horizon weather is
    imperfect knowledge at deployment.  This helper perturbs it to typical
    24 h NWP error — temperature += N(0, 2 K), shortwave and wind speed
    multiplied by (1 + N(0, sigma)) — and recomputes the derived channels
    (wind capacity factor, clear-sky index) from the perturbed values so
    the model sees a self-consistent forecast.  Returns a perturbed
    shallow copy; the input dict is untouched.
    """
    from transcif.physics.astro import wind_capacity_factor  # noqa: PLC0415
    w = dict(windows)
    fw = windows["fut_weather"].copy()
    B, H = fw.shape[0], fw.shape[1]
    fw[..., 0] = (fw[..., 0]
                  + rng.normal(0.0, sigma_temp, (B, H))).astype(np.float32)
    fw[..., 1] = np.clip(
        fw[..., 1] * (1 + rng.normal(0.0, sigma_sw, (B, H))),
        0.0, None).astype(np.float32)
    fw[..., 2] = np.clip(
        fw[..., 2] * (1 + rng.normal(0.0, sigma_wind, (B, H))),
        0.0, None).astype(np.float32)
    fe = windows["fut_exog"].copy()
    fe[..., 2] = wind_capacity_factor(fw[..., 2]).astype(np.float32)
    clearsky = np.maximum(fe[..., 1], 1.0)  # astronomy column, untouched
    fe[..., 3] = np.clip(fw[..., 1] / clearsky, 0.0, 1.3).astype(np.float32)
    w["fut_exog"] = fe
    # Recompute the DERIVED fut_weather channels so the model sees a
    # self-consistent forecast everywhere (the wind head reads the wcf
    # channel directly; regime/tend ride on the same capacity factor).
    if fw.shape[2] > 8:
        fw[..., 3] = fe[..., 2]                      # wind CF
        fw[..., 4] = fe[..., 3]                      # clear-sky index
        past_wcf = windows["x_weather"][:, -24:, 3]  # observable reanalysis
        series = np.concatenate([past_wcf, fw[:, :, 3]], axis=1)  # (B, 24+H)
        cs = np.cumsum(np.concatenate(
            [np.zeros((B, 1), np.float32), series], axis=1), axis=1)
        k = np.arange(1, series.shape[1] + 1)
        reg = (cs[:, 1:] - cs[:, np.maximum(k - 24, 0)]) / np.minimum(k, 24)
        fw[:, :, 8] = reg[:, -H:].astype(np.float32)      # regime24
        tend = np.zeros_like(reg)
        tend[:, 6:] = reg[:, 6:] - reg[:, :-6]
        fw[:, :, 9] = tend[:, -H:]                          # 6 h tendency
    w["fut_weather"] = fw
    return w


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
