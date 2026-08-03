"""Phase 1.3: Unified Evaluation Protocol (LORO, 5-seed, 29 regions).

This is the DEFINITIVE experiment for the paper. All methods evaluated under
identical conditions with statistical rigor.

Protocol:
- LORO: Leave-One-Region-Out (target has NO training data)
- 5 random seeds for variance estimation
- Metrics: MAE, RMSE, sMAPE (point prediction)
- Time split: first 80% train, last 20% test
- All 29 regions: 4 AU + 17 UK + 8 US

Output: JSON results file + summary tables for paper

Usage: PYTHONPATH=src python scripts/run_unified_eval.py [--quick]
  --quick: 3 seeds, AU only (for debugging)
"""

import argparse
import glob
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data_2023"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEQ_LEN = 336
HORIZON = 24
TRAIN_STRIDE = 6
TEST_STRIDE = 24
TRAIN_FRACTION = 0.8
SEEDS_FULL = [0, 1, 2, 3, 4]
SEEDS_QUICK = [0, 1, 2]
EPOCHS_SUPERVISED = 300
EPOCHS_CARBONCAST = 300
EPOCHS_ZERO_SHOT = 150
BATCH_SIZE = 256

# All regions
AU_REGIONS = {
    "QLD1": {"file": "QLD1_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 841.59},
    "NSW1": {"file": "NSW1_2023_hourly.csv", "ef_r": 0.09, "ef_nr": 875.23},
    "VIC1": {"file": "VIC1_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 1160.12},
    "SA1":  {"file": "SA1_2023_hourly.csv",  "ef_r": 0.0, "ef_nr": 490.43},
}

US_REGIONS = {
    "US_CISO": {"file": "US_CISO_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 342.8},
    "US_PJM":  {"file": "US_PJM_2023_hourly.csv",  "ef_r": 0.0, "ef_nr": 347.6},
    "US_MISO": {"file": "US_MISO_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 534.1},
    "US_ERCO": {"file": "US_ERCO_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 470.3},
    "US_ISNE": {"file": "US_ISNE_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 299.1},
    "US_NYIS": {"file": "US_NYIS_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 287.3},
    "US_FPL":  {"file": "US_FPL_2023_hourly.csv",  "ef_r": 0.0, "ef_nr": 340.9},
    "US_BPAT": {"file": "US_BPAT_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 207.5},
}

UK_REGIONS = {}


# ---------------------------------------------------------------------------
# Data Loading (unchanged)
# ---------------------------------------------------------------------------

def discover_uk_regions():
    global UK_REGIONS
    for f in sorted(glob.glob(str(DATA_DIR / "UK_*_2023_hourly.csv"))):
        name = Path(f).stem.replace("_2023_hourly", "")
        df = pd.read_csv(f)
        rs = df["renew_share"].values
        cif = df["cif_real_gco2_per_kwh"].values
        mask = (rs < 0.95) & (rs > 0.05) & (cif > 0)
        if mask.sum() > 500:
            ef_nr_est = float(np.median(cif[mask] / (1 - rs[mask])))
            if 100 < ef_nr_est < 2000:
                UK_REGIONS[name] = {"file": Path(f).name, "ef_r": 0.0, "ef_nr": ef_nr_est}


def load_region_data(region_name: str, all_configs: dict) -> dict:
    info = all_configs[region_name]
    path = DATA_DIR / info["file"]
    ef_r, ef_nr = info["ef_r"], info["ef_nr"]
    df = pd.read_csv(path, parse_dates=["hour"])
    df = df.sort_values("hour").reset_index(drop=True)
    rs = df["renew_share"].values.astype(np.float32)
    cif = df["cif_real_gco2_per_kwh"].values.astype(np.float32)
    valid = np.isfinite(rs) & np.isfinite(cif) & (cif >= 0)
    rs, cif = rs[valid], cif[valid]
    return {
        "rs": rs, "cif": cif,
        "mean_rs": float(rs.mean()),
        "ef_r": ef_r, "ef_nr": ef_nr,
        "config": np.array([rs.mean(), ef_nr / 1000.0], dtype=np.float32),
    }


def build_windows(rs, cif, seq_len, horizon, stride):
    window = seq_len + horizon
    x_rs, y_rs, y_cif = [], [], []
    for start in range(0, len(rs) - window + 1, stride):
        x_rs.append(rs[start:start + seq_len])
        y_rs.append(rs[start + seq_len:start + window])
        y_cif.append(cif[start + seq_len:start + window])
    if not x_rs:
        return np.empty((0, seq_len)), np.empty((0, horizon)), np.empty((0, horizon))
    return np.stack(x_rs), np.stack(y_rs), np.stack(y_cif)


def cif_from_shares(rs, ef_r, ef_nr):
    return rs * ef_r + (1 - rs) * ef_nr


# ---------------------------------------------------------------------------
# TransCIF-ZS+ : test-time calibration (zero target-domain training)
# ---------------------------------------------------------------------------

K_BACKTEST = 7    # recent observed 24h blocks used for self-validation
ANCHOR_WIN = 24   # hours of observed rs used for level anchoring
RESID_WIN = 48    # hours used to estimate the physics residual delta_t
BLEND_GAMMA = 2.0  # sharpness of the per-lead precision-weighted self-validation blend
WEEKLY_LAG = 168  # lag of the weekly-persistence fusion branch
SELECT_DAYS = 56  # rolling window of observed days used to self-select the fusion config
SELECT_MARGIN = 0.015  # relative improvement required to leave the default config
SELECT_METRIC = "dual"  # aggregation of replayed daily MAEs: dual/mean/sqrt/log/ratio/median
SELECT_TOL = None  # dual gate: max relative loss allowed on the other metric (None=margin)

# Menu of fusion configurations the regional self-selection chooses from.
# Branch indices: 0=calibrated model, 1=lag-24h persistence, 2=lag-168h weekly
# persistence, 3=7-day same-hour climatology, 4=4-week same-weekday
# climatology, 5=raw model. The first entry is the default blend and the
# fallback whenever the observed history is too short to run the selection
# backtest; the second adds the weekly-climatology branch for grids with
# strong weekly seasonality (e.g. demand-driven markets); the third is the
# conservative two-branch legacy blend that dominates in highly renewable
# grids where the daily cycle is weak and climatology averages mislead.
FUSION_MENU = (
    dict(branches=(0, 1, 3), gamma=2.5, k_backtest=28),     # default blend
    dict(branches=(0, 1, 3, 4), gamma=2.5, k_backtest=28),  # + weekly climatology
    dict(branches=(0, 1), gamma=2.0, k_backtest=7),         # conservative 2-branch
)


def zs_plus_predict(model, config, rs, cif, ef_r, ef_nr, origins, horizon=HORIZON,
                    fusion=None):
    r"""Test-time calibrated zero-shot prediction (TransCIF-ZS+).

    Three calibration steps, all using only the target's observable streams
    up to each forecast origin (same information persistence uses); the model
    is never trained on target data:
      1. Level anchor: shift predicted shares so their mean matches the
         last ANCHOR_WIN hours of observed rs.
      2. Residual correction: add the mean physics residual
         delta = cif_obs - CIF(rs_obs) over the last RESID_WIN hours.
      3. Self-selected per-lead precision-weighted fusion: backtest the
         candidate branches -- (M) the calibrated model, (P24) lag-24h CIF
         persistence, (P168) lag-168h weekly persistence, (C7) the 7-day
         same-hour climatology, (WC4) the 4-week same-weekday climatology --
         on the k most recent observed 24h blocks, accumulate each branch's
         mean absolute error at every lead time h, and blend the branches
         lead-by-lead with inverse-power weights
         w_b,h \propto e_b,h^{-gamma}
         (for two branches and gamma=2 this reduces exactly to
         alpha_h = p_err_h^2 / (m_err_h^2 + p_err_h^2), i.e. inverse-variance
         weighting). The fusion configuration (default blend, + weekly
         climatology, or the conservative two-branch blend) is re-selected at
         every forecast origin by replaying all menu configs on the
         SELECT_DAYS most recent observed days strictly before that origin;
         replayed daily MAEs are compared under two aggregations at once
         (arithmetic mean, which is dominated by hard days, and log-mean,
         which captures a systematic everyday edge), and a candidate replaces
         the default only if it wins by SELECT_MARGIN on one aggregation
         without losing more than the margin on the other -- pure past-only
         information, the same stream persistence uses, so each grid recruits
         the configuration its own recent history supports.

    fusion : optional dict(branches=, gamma=, k_backtest=) to force one
             configuration (used by probes/ablations); None self-selects.
    """
    cfg1 = torch.tensor(config).unsqueeze(0)
    branch_cache = {}

    def branch_preds(t0):
        """(6, horizon): calibrated model, lag-24 persistence, lag-168 weekly,
        7d climatology, 4-week same-weekday climatology, raw model."""
        if t0 not in branch_cache:
            x = torch.tensor(rs[t0 - SEQ_LEN:t0], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                s_raw = model(x, cfg1).numpy()[0]
            s = np.clip(s_raw - s_raw.mean() + rs[t0 - ANCHOR_WIN:t0].mean(), 0.0, 1.0)
            delta = (cif[t0 - RESID_WIN:t0]
                     - cif_from_shares(rs[t0 - RESID_WIN:t0], ef_r, ef_nr)).mean()
            weekly_lags = [j * WEEKLY_LAG for j in range(1, 5) if t0 - j * WEEKLY_LAG >= 0]
            branch_cache[t0] = np.stack([
                cif_from_shares(s, ef_r, ef_nr) + delta,
                cif[t0 - 24:t0 - 24 + horizon],
                cif[t0 - WEEKLY_LAG:t0 - WEEKLY_LAG + horizon],
                np.mean([cif[t0 - j * 24:t0 - j * 24 + horizon]
                         for j in range(1, 8)], axis=0),
                np.mean([cif[t0 - lag:t0 - lag + horizon]
                         for lag in weekly_lags], axis=0),
                cif_from_shares(np.clip(s_raw, 0.0, 1.0), ef_r, ef_nr),
            ])
        return branch_cache[t0]

    def fuse_at(t0, branches, gamma, k_backtest):
        idx = list(branches)
        bp, yt = [], []
        for k in range(1, k_backtest + 1):
            o = t0 - k * 24
            if o - SEQ_LEN < 0 or o + 24 > min(t0, len(cif)):
                break
            bp.append(branch_preds(o)[idx, :24])
            yt.append(cif[o:o + 24])
        live = branch_preds(t0)[idx]
        if not bp:
            return 0.5 * live[0] + 0.5 * live[1]
        mean_err = np.abs(np.stack(bp, axis=1) - np.stack(yt)[None]).mean(axis=1)
        # normalize by the per-lead best branch before the inverse power for
        # numerical stability at large gamma (near winner-take-all)
        ratio = mean_err / (mean_err.min(axis=0, keepdims=True) + 1e-8)
        with np.errstate(divide="ignore"):
            w = ratio ** (-gamma)
        # a branch with zero backtest error (e.g. flat cif segments in fully
        # renewable hours) gives ratio 0 -> inf weight; resolve such leads to
        # winner-take-all among the zero-error branches instead of nan
        bad = ~np.isfinite(w)
        if bad.any():
            cols = bad.any(axis=0)
            w[:, cols] = bad[:, cols].astype(float)
        w /= w.sum(axis=0, keepdims=True)
        return (w * live).sum(axis=0)

    if fusion is not None:
        return np.stack([fuse_at(t0, **fusion) for t0 in origins])

    # Rolling per-origin self-selection: at each forecast origin, replay every
    # menu config on the SELECT_DAYS most recent observed days (past-only, the
    # same information persistence uses) and keep the winner; the default
    # config is retained unless a candidate beats it by its margin (menu
    # entries may carry their own "margin" so riskier configs need stronger
    # replay evidence to be recruited).
    sim_cache = {}

    def cfg_args(cfg):
        return {k: v for k, v in cfg.items() if k != "margin"}

    def sim_mae(o, ci):
        if (o, ci) not in sim_cache:
            sim_cache[(o, ci)] = np.abs(
                fuse_at(o, **cfg_args(FUSION_MENU[ci]))[:24] - cif[o:o + 24]).mean()
        return sim_cache[(o, ci)]

    preds = []
    for t0 in origins:
        errs = []  # (n_days, n_cfg) replayed daily MAEs
        for j in range(1, SELECT_DAYS + 1):
            o_s = t0 - j * 24
            if o_s - SEQ_LEN - 24 < 0:
                break
            errs.append([sim_mae(o_s, i) for i in range(len(FUSION_MENU))])
        chosen = FUSION_MENU[0]
        if errs:
            e = np.array(errs)
            if SELECT_METRIC == "dual":
                # a candidate replaces the default when it beats it by its
                # margin on either aggregation (mean captures the big-error
                # days, log the systematic everyday edge) without being more
                # than the tolerance worse on the other
                sm, sl = e.sum(axis=0), np.log1p(e).sum(axis=0)
                best, best_key = 0, 2.0
                for i in range(1, len(FUSION_MENU)):
                    mi = FUSION_MENU[i].get("margin", SELECT_MARGIN)
                    tol = mi if SELECT_TOL is None else SELECT_TOL
                    rm, rl = sm[i] / sm[0], sl[i] / sl[0]
                    wins = ((rm < 1.0 - mi and rl <= 1.0 + tol)
                            or (rl < 1.0 - mi and rm <= 1.0 + tol))
                    if wins and rm + rl < best_key:
                        best, best_key = i, rm + rl
                chosen = FUSION_MENU[best]
            else:
                if SELECT_METRIC == "log":
                    scores = np.log1p(e).sum(axis=0)
                elif SELECT_METRIC == "sqrt":
                    scores = np.sqrt(e).sum(axis=0)
                elif SELECT_METRIC == "ratio":
                    scores = (e / (e.min(axis=1, keepdims=True) + 1e-8)).sum(axis=0)
                elif SELECT_METRIC == "median":
                    scores = np.median(e, axis=0)
                else:
                    scores = e.sum(axis=0)
                best = int(np.argmin(scores))
                if scores[best] < scores[0] * (1.0 - SELECT_MARGIN):
                    chosen = FUSION_MENU[best]
        preds.append(fuse_at(t0, **cfg_args(chosen)))
    return np.stack(preds)


# ---------------------------------------------------------------------------
# Models (same as phase1_complete)
# ---------------------------------------------------------------------------

class RevIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True) + self.eps
            return (x - self.mean) / self.std
        else:
            return x * self.std + self.mean


class PatchTSTFixed(nn.Module):
    def __init__(self, seq_len=336, horizon=24, patch_len=24, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.patch_len = patch_len
        n_patches = seq_len // patch_len
        self.revin = RevIN()
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128,
            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(n_patches * d_model), nn.Linear(n_patches * d_model, horizon))

    def forward(self, x):
        x_norm = self.revin(x, 'norm')
        B = x_norm.shape[0]
        patches = x_norm.unfold(1, self.patch_len, self.patch_len)
        x_emb = self.patch_embed(patches) + self.pos_embed
        x_enc = self.transformer(x_emb)
        out_norm = self.head(x_enc.reshape(B, -1))
        return self.revin(out_norm, 'denorm')


class AdaptivePersistDLinear(nn.Module):
    def __init__(self, seq_len=336, horizon=24, config_dim=2):
        super().__init__()
        self.horizon = horizon
        self.avg_pool = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.linear_trend = nn.Linear(seq_len, horizon)
        self.linear_seasonal = nn.Linear(seq_len, horizon)
        self.config_bias = nn.Sequential(nn.Linear(config_dim, 16), nn.ReLU(), nn.Linear(16, horizon))
        self.gate_net = nn.Sequential(nn.Linear(config_dim + 2, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, config):
        x3 = x.unsqueeze(1)
        trend = self.avg_pool(x3).squeeze(1)
        seasonal = x - trend
        dlinear_out = torch.sigmoid(
            self.linear_trend(trend) + self.linear_seasonal(seasonal) + self.config_bias(config))
        persist = x[:, -self.horizon:]
        recent_mean = x[:, -48:].mean(dim=1, keepdim=True)
        recent_std = x[:, -48:].std(dim=1, keepdim=True)
        gate_input = torch.cat([config, recent_mean, recent_std], dim=1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        return gate * persist + (1 - gate) * dlinear_out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_patchtst(x_train, y_train, epochs=300, lr=3e-4):
    model = PatchTSTFixed(seq_len=SEQ_LEN, horizon=HORIZON)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_cosine_warmup_scheduler(optimizer, 30, epochs)
    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    n = len(x_t)
    model.train()
    for epoch in range(epochs):
        idx = torch.randperm(n)[:min(BATCH_SIZE, n)]
        pred = model(x_t[idx])
        loss = nn.functional.l1_loss(pred, y_t[idx])
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    model.eval()
    return model


def train_zero_shot(all_regions, target_name, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = AdaptivePersistDLinear(seq_len=SEQ_LEN, horizon=HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = get_cosine_warmup_scheduler(optimizer, 15, EPOCHS_ZERO_SHOT)
    xs, ys, cfgs, weights = [], [], [], []
    for name, data in all_regions.items():
        if name == target_name:
            continue
        x_win, y_win, _ = build_windows(data["rs"], data["cif"], SEQ_LEN, HORIZON, TRAIN_STRIDE)
        if len(x_win) == 0:
            continue
        xs.append(x_win)
        ys.append(y_win)
        cfgs.append(np.tile(data["config"], (len(x_win), 1)))
        dist = abs(data["mean_rs"] - all_regions[target_name]["mean_rs"])
        w = 1.0 / (dist + 0.05)
        weights.append(np.full(len(x_win), w, dtype=np.float32))
    x_all = torch.tensor(np.concatenate(xs))
    y_all = torch.tensor(np.concatenate(ys))
    c_all = torch.tensor(np.concatenate(cfgs))
    w_all = torch.tensor(np.concatenate(weights))
    w_all = w_all / w_all.sum() * len(w_all)
    n_samples = len(x_all)
    batch_size = min(512, n_samples)
    model.train()
    for epoch in range(EPOCHS_ZERO_SHOT):
        idx = torch.randperm(n_samples)[:batch_size]
        pred = model(x_all[idx], c_all[idx])
        loss = (w_all[idx].unsqueeze(1) * torch.abs(pred - y_all[idx])).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred, true):
    """Compute MAE, RMSE, sMAPE."""
    mae = float(np.abs(pred - true).mean())
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    denom = (np.abs(pred) + np.abs(true)) / 2 + 1e-8
    smape = float(np.mean(np.abs(pred - true) / denom) * 100)
    return {"mae": mae, "rmse": rmse, "smape": smape}


# ---------------------------------------------------------------------------
# Evaluation per target
# ---------------------------------------------------------------------------

def evaluate_target(target_name, all_regions, seed=42):
    """Evaluate on one target region: persistence + PatchTST-sup + TransCIF-ZS."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    data = all_regions[target_name]
    rs, cif = data["rs"], data["cif"]
    ef_r, ef_nr = data["ef_r"], data["ef_nr"]

    n_hours = len(rs)
    split_hour = int(n_hours * TRAIN_FRACTION)

    # Build windows
    x_rs_test, _, y_cif_test = build_windows(
        rs[split_hour - SEQ_LEN:], cif[split_hour - SEQ_LEN:], SEQ_LEN, HORIZON, TEST_STRIDE)
    cif_offset = cif[split_hour - SEQ_LEN:]
    x_cif_test = []
    for start in range(0, len(cif_offset) - SEQ_LEN - HORIZON + 1, TEST_STRIDE):
        x_cif_test.append(cif_offset[start:start + SEQ_LEN])
    if not x_cif_test:
        return None
    x_cif_test = np.stack(x_cif_test)

    # Training windows for supervised
    x_cif_train, y_cif_train = [], []
    for start in range(0, split_hour - SEQ_LEN - HORIZON + 1, TRAIN_STRIDE):
        x_cif_train.append(cif[start:start + SEQ_LEN])
        y_cif_train.append(cif[start + SEQ_LEN:start + SEQ_LEN + HORIZON])
    x_cif_train = np.stack(x_cif_train)
    y_cif_train = np.stack(y_cif_train)

    results = {"target": target_name, "seed": seed, "mean_rs": data["mean_rs"],
               "ef_nr": data["ef_nr"], "n_test": len(x_rs_test)}

    # --- 1. Persistence ---
    persist_pred = x_cif_test[:, -HORIZON:]
    results["persistence"] = compute_metrics(persist_pred, y_cif_test)

    # --- 2. PatchTST supervised ---
    ptst = train_patchtst(x_cif_train, y_cif_train)
    with torch.no_grad():
        ptst_pred = ptst(torch.tensor(x_cif_test, dtype=torch.float32)).numpy()
    results["patchtst_sup"] = compute_metrics(ptst_pred, y_cif_test)

    # --- 3. TransCIF zero-shot ---
    zs_model = train_zero_shot(all_regions, target_name, seed=seed)
    target_cfg = torch.tensor(data["config"]).unsqueeze(0).expand(len(x_rs_test), -1)
    with torch.no_grad():
        zs_rs_pred = zs_model(torch.tensor(x_rs_test, dtype=torch.float32), target_cfg).numpy()
    zs_cif_pred = cif_from_shares(zs_rs_pred, ef_r, ef_nr)
    results["transcif_zs"] = compute_metrics(zs_cif_pred, y_cif_test)

    # --- 4. TransCIF-ZS+ (test-time calibration, zero target training) ---
    origins = [split_hour + st
               for st in range(0, len(cif_offset) - SEQ_LEN - HORIZON + 1, TEST_STRIDE)]
    zsp_pred = zs_plus_predict(zs_model, data["config"], rs, cif, ef_r, ef_nr, origins)
    results["transcif_zs_plus"] = compute_metrics(zsp_pred, y_cif_test)

    # --- Ratios ---
    results["ratio_vs_patchtst"] = results["transcif_zs"]["mae"] / results["patchtst_sup"]["mae"]
    results["ratio_vs_persist"] = results["transcif_zs"]["mae"] / results["persistence"]["mae"]
    results["ratio_plus_vs_patchtst"] = (
        results["transcif_zs_plus"]["mae"] / results["patchtst_sup"]["mae"])
    results["ratio_plus_vs_persist"] = (
        results["transcif_zs_plus"]["mae"] / results["persistence"]["mae"])

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick mode: 3 seeds, AU only")
    args = parser.parse_args()

    seeds = SEEDS_QUICK if args.quick else SEEDS_FULL
    mode_str = "QUICK (AU only, 3 seeds)" if args.quick else "FULL (29 regions, 5 seeds)"

    print("=" * 80)
    print(f"Phase 1.3: Unified Evaluation Protocol — {mode_str}")
    print("=" * 80)

    # Load all regions
    discover_uk_regions()
    all_configs = {**AU_REGIONS, **UK_REGIONS, **US_REGIONS}
    all_regions = {}
    for name in all_configs:
        try:
            all_regions[name] = load_region_data(name, all_configs)
        except Exception as e:
            print(f"  [WARN] Skip {name}: {e}")

    n_total = len(all_regions)
    print(f"Loaded: {n_total} regions")

    if args.quick:
        targets = ["QLD1", "NSW1", "VIC1", "SA1"]
    else:
        targets = sorted(all_regions.keys(), key=lambda x: all_regions[x]["mean_rs"])

    print(f"Targets: {len(targets)} regions, Seeds: {seeds}")
    print(f"Total evaluations: {len(targets) * len(seeds)}")
    t0 = time.time()

    # Resume support: reload partial results if a previous run was interrupted
    tag = "quick" if args.quick else "full"
    partial_file = RESULTS_DIR / f"unified_eval_{tag}.partial.json"
    all_results = []
    done_targets = set()
    if partial_file.exists():
        with open(partial_file) as f:
            all_results = json.load(f)
        # Only skip targets with ALL seeds completed
        for t in {r["target"] for r in all_results}:
            t_seeds = {r["seed"] for r in all_results if r["target"] == t}
            if set(seeds) <= t_seeds:
                done_targets.add(t)
        print(f"Resuming: {len(done_targets)} targets already complete in {partial_file}")
        all_results = [r for r in all_results if r["target"] in done_targets]

    for i, target in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}] {target} (rs={all_regions[target]['mean_rs']:.3f})", flush=True)
        if target in done_targets:
            print("  (cached from partial results)", flush=True)
            continue
        for seed in seeds:
            r = evaluate_target(target, all_regions, seed=seed)
            if r is None:
                continue
            all_results.append(r)
            print(f"  s{seed}: Persist={r['persistence']['mae']:.1f} "
                  f"PatchTST={r['patchtst_sup']['mae']:.1f} "
                  f"TransCIF={r['transcif_zs']['mae']:.1f} "
                  f"ZS+={r['transcif_zs_plus']['mae']:.1f} "
                  f"ratio={r['ratio_vs_patchtst']:.3f} "
                  f"ratio+={r['ratio_plus_vs_patchtst']:.3f}", flush=True)
        # Checkpoint after each region so an interrupted run can resume
        with open(partial_file, "w") as f:
            json.dump(all_results, f)

    elapsed = time.time() - t0
    print(f"\n\nTotal time: {elapsed/60:.1f} min")

    # Save raw results
    results_file = RESULTS_DIR / f"unified_eval_{tag}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved: {results_file}")
    if partial_file.exists():
        partial_file.unlink()

    # Summary table
    print("\n\n" + "=" * 100)
    print("SUMMARY TABLE (Mean ± Std across seeds)")
    print("=" * 100)
    print(f"{'Region':<25} {'mean_rs':<8} {'Persist':<12} {'PatchTST-S':<12} "
          f"{'TransCIF-ZS':<12} {'Ratio':<8} {'ZS/P':<6}")
    print("-" * 100)

    for target in targets:
        tr = [r for r in all_results if r["target"] == target]
        if not tr:
            continue
        persist_vals = [r["persistence"]["mae"] for r in tr]
        ptst_vals = [r["patchtst_sup"]["mae"] for r in tr]
        zs_vals = [r["transcif_zs"]["mae"] for r in tr]
        ratio_vals = [r["ratio_vs_patchtst"] for r in tr]
        zsp_vals = [r["ratio_vs_persist"] for r in tr]

        print(f"{target:<25} {tr[0]['mean_rs']:<8.3f} "
              f"{np.mean(persist_vals):.1f}±{np.std(persist_vals):.1f}  "
              f"{np.mean(ptst_vals):.1f}±{np.std(ptst_vals):.1f}  "
              f"{np.mean(zs_vals):.1f}±{np.std(zs_vals):.1f}  "
              f"{np.mean(ratio_vals):<8.3f} {np.mean(zsp_vals):<6.3f}")

    # Overall statistics
    all_ratios = [r["ratio_vs_patchtst"] for r in all_results]
    all_zsp = [r["ratio_vs_persist"] for r in all_results]
    all_ratios_p = [r["ratio_plus_vs_patchtst"] for r in all_results]
    all_zspp = [r["ratio_plus_vs_persist"] for r in all_results]
    print(f"\n{'OVERALL':<25} {'':8} {'':12} {'':12} {'':12} "
          f"{np.mean(all_ratios):<8.3f} {np.mean(all_zsp):<6.3f}")
    print(f"\nMedian Ratio vs PatchTST: ZS={np.median(all_ratios):.3f}  "
          f"ZS+={np.median(all_ratios_p):.3f}")
    print(f"Mean Ratio vs PatchTST:   ZS={np.mean(all_ratios):.3f}  "
          f"ZS+={np.mean(all_ratios_p):.3f}")
    print(f"Regions where ZS  < Persist: {sum(1 for r in all_zsp if r < 1)}/{len(all_zsp)}")
    print(f"Regions where ZS+ < Persist: {sum(1 for r in all_zspp if r < 1)}/{len(all_zspp)}")


if __name__ == "__main__":
    main()
