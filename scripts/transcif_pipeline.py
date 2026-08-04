"""Unified TransCIF pipeline: training, inference, ZS+ calibration, evaluation.

Extracted from run_unified_eval.py for single-point import across all scripts.

Exports:
    Data:       discover_uk_regions, load_region_data, build_windows,
                cif_from_shares, get_cosine_warmup_scheduler
    Training:   train_patchtst, train_zero_shot
    Evaluation: compute_metrics, evaluate_target
    ZS-plus:    zs_plus_predict
    Config:     DATA_DIR, SEQ_LEN, HORIZON, region configs, defaults
"""

import glob
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from transcif_model import (
    AdaptivePersistDLinear, RichConfigAdaptivePersist, PatchTSTFixed,
)
from transcif_data import ramp_aware_loss, MissingMaskAugmentor

# ---------------------------------------------------------------------------
# Paths & global constants
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data_2023"

SEQ_LEN = 336
HORIZON = 24
TRAIN_STRIDE = 6
TEST_STRIDE = 24
TRAIN_FRACTION = 0.8
EPOCHS_SUPERVISED = 300
EPOCHS_CARBONCAST = 300
EPOCHS_ZERO_SHOT = 150
assert EPOCHS_CARBONCAST == 300  # keep linter happy, used by external imports
BATCH_SIZE = 256

# Seed configurations (shared across all experiment scripts)
SEEDS_FULL = [0, 1, 2, 3, 4]
SEEDS_QUICK = [0, 1, 2]

# Region configurations
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
# Data loading
# ---------------------------------------------------------------------------

def discover_uk_regions(data_dir=None):
    """Populate UK_REGIONS glob by scanning the data directory."""
    global UK_REGIONS
    if data_dir is None:
        data_dir = _DATA_DIR
    for f in sorted(glob.glob(str(data_dir / "UK_*_2023_hourly.csv"))):
        name = Path(f).stem.replace("_2023_hourly", "")
        df = pd.read_csv(f)
        rs = df["renew_share"].values
        cif = df["cif_real_gco2_per_kwh"].values
        mask = (rs < 0.95) & (rs > 0.05) & (cif > 0)
        if mask.sum() > 500:
            ef_nr_est = float(np.median(cif[mask] / (1 - rs[mask])))
            if 100 < ef_nr_est < 2000:
                UK_REGIONS[name] = {
                    "file": Path(f).name, "ef_r": 0.0, "ef_nr": ef_nr_est}


def load_region_data(region_name: str, all_configs: dict,
                     data_dir=None) -> dict:
    """Load a single region's rs / cif timeseries."""
    if data_dir is None:
        data_dir = _DATA_DIR
    info = all_configs[region_name]
    path = data_dir / info["file"]
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


def build_windows(rs, cif, seq_len=SEQ_LEN, horizon=HORIZON, stride=TRAIN_STRIDE):
    """Build (x_rs, y_rs, y_cif) sliding windows."""
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
    """Physics layer: CIF = share * ef_ren + (1-share) * ef_nonren."""
    return rs * ef_r + (1 - rs) * ef_nr


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def get_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_patchtst(x_train, y_train, epochs=300, lr=3e-4, device=None):
    """Train a supervised PatchTST baseline."""
    model = PatchTSTFixed(seq_len=SEQ_LEN, horizon=HORIZON)
    if device:
        model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_cosine_warmup_scheduler(optimizer, 30, epochs)
    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    if device:
        x_t, y_t = x_t.to(device), y_t.to(device)
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


def train_zero_shot(all_regions, target_name, seed=42,
                    model_class=None, use_weighted=True,
                    use_ramp_loss=False, mask_augment_prob=0.0,
                    epochs=EPOCHS_ZERO_SHOT, lr=1e-3, device=None):
    """Train the zero-shot model on all source regions for one LORO target.

    Args:
        all_regions : dict  {name: {"rs":..., "cif":..., "config":...}}
        target_name : str   region to leave out
        seed        : int
        model_class : nn.Module class (default AdaptivePersistDLinear)
        use_weighted: bool  config-distance source weighting
        use_ramp_loss : bool  use ramp-weighted L1 loss instead of plain L1
        mask_augment_prob : float  probability of input masking (0=off)
        epochs, lr  : training hyperparameters
    """
    if model_class is None:
        model_class = AdaptivePersistDLinear
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = model_class(seq_len=SEQ_LEN, horizon=HORIZON)
    if device:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_cosine_warmup_scheduler(optimizer, max(1, epochs // 10), epochs)
    mask_aug = MissingMaskAugmentor(prob=mask_augment_prob) if mask_augment_prob > 0 else None
    xs, ys, cfgs, weights = [], [], [], []
    for name, data in all_regions.items():
        if name == target_name:
            continue
        x_win, y_win, _ = build_windows(data["rs"], data["cif"])
        if len(x_win) == 0:
            continue
        xs.append(x_win)
        ys.append(y_win)
        cfgs.append(np.tile(data["config"], (len(x_win), 1)))
        if use_weighted:
            dist = abs(data["mean_rs"] - all_regions[target_name]["mean_rs"])
            w = 1.0 / (dist + 0.05)
        else:
            w = 1.0
        weights.append(np.full(len(x_win), w, dtype=np.float32))
    x_all = torch.tensor(np.concatenate(xs))
    y_all = torch.tensor(np.concatenate(ys))
    c_all = torch.tensor(np.concatenate(cfgs))
    w_all = torch.tensor(np.concatenate(weights))
    w_all = w_all / w_all.sum() * len(w_all)
    if device:
        x_all, y_all, c_all, w_all = x_all.to(device), y_all.to(device), c_all.to(device), w_all.to(device)
    n_samples = len(x_all)
    batch_size = min(512, n_samples)
    model.train()
    for epoch in range(epochs):
        idx = torch.randperm(n_samples)[:batch_size]
        x_batch = x_all[idx]
        y_batch = y_all[idx]
        c_batch = c_all[idx]
        w_batch = w_all[idx]
        if mask_aug is not None:
            x_batch, _ = mask_aug(x_batch)
        pred = model(x_batch, c_batch)
        if use_ramp_loss:
            per_element = ramp_aware_loss(pred, y_batch, reduction='none')
            loss = (w_batch.unsqueeze(1) * per_element).mean()
        else:
            loss = (w_batch.unsqueeze(1) * torch.abs(pred - y_batch)).mean()
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
    """MAE, RMSE, sMAPE."""
    mae = float(np.abs(pred - true).mean())
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    denom = (np.abs(pred) + np.abs(true)) / 2 + 1e-8
    smape = float(np.mean(np.abs(pred - true) / denom) * 100)
    return {"mae": mae, "rmse": rmse, "smape": smape}


# ---------------------------------------------------------------------------
# ZS+ test-time calibration
# ---------------------------------------------------------------------------

K_BACKTEST = 7
ANCHOR_WIN = 24
RESID_WIN = 48
BLEND_GAMMA = 2.0
WEEKLY_LAG = 168
SELECT_DAYS = 56
SELECT_MARGIN = 0.015
SELECT_METRIC = "dual"
SELECT_TOL = None

FUSION_MENU = (
    dict(branches=(0, 1, 3), gamma=2.5, k_backtest=28),
    dict(branches=(0, 1, 3, 4), gamma=2.5, k_backtest=28),
    dict(branches=(0, 1), gamma=2.0, k_backtest=7),
)


def zs_plus_predict(model, config, rs, cif, ef_r, ef_nr, origins,
                    horizon=HORIZON, fusion=None):
    """Test-time calibrated zero-shot prediction (TransCIF-ZS+).

    For details see the docstring in run_unified_eval.py.
    """
    cfg1 = torch.tensor(config).unsqueeze(0)
    branch_cache = {}

    def branch_preds(t0):
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
        ratio = mean_err / (mean_err.min(axis=0, keepdims=True) + 1e-8)
        with np.errstate(divide="ignore"):
            w = ratio ** (-gamma)
        bad = ~np.isfinite(w)
        if bad.any():
            cols = bad.any(axis=0)
            w[:, cols] = bad[:, cols].astype(float)
        w /= w.sum(axis=0, keepdims=True)
        return (w * live).sum(axis=0)

    if fusion is not None:
        return np.stack([fuse_at(t0, **fusion) for t0 in origins])

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
        errs = []
        for j in range(1, SELECT_DAYS + 1):
            o_s = t0 - j * 24
            if o_s - SEQ_LEN - 24 < 0:
                break
            errs.append([sim_mae(o_s, i) for i in range(len(FUSION_MENU))])
        chosen = FUSION_MENU[0]
        if errs:
            e = np.array(errs)
            if SELECT_METRIC == "dual":
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
# Single-target evaluation
# ---------------------------------------------------------------------------

def evaluate_target(target_name, all_regions, seed=42,
                    model_class=None, use_ramp_loss=False,
                    use_rag=False, use_phys_irm=False,
                    use_causal=False, use_icl=False, use_hier=False):
    """Full evaluation on one target: persistence, PatchTST-sup, ZS, ZS+, [RAG], [Phys-IRM], [Causal], [ICL], [Hier].

    Args:
        model_class : override model architecture (None → AdaptivePersistDLinear)
        use_ramp_loss : use ramp-weighted L1 during training
        use_rag : if True, also train & evaluate RagDLinear
        use_phys_irm : if True, also train & evaluate Phys-IRM model
        use_causal : if True, also train & evaluate CausalDomainVAE
        use_icl : if True, also train & evaluate ICTransformer
        use_hier : if True, also train & evaluate HierDLinear
    """
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
        rs[split_hour - SEQ_LEN:], cif[split_hour - SEQ_LEN:],
        seq_len=SEQ_LEN, horizon=HORIZON, stride=TEST_STRIDE)
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

    # 1. Persistence
    persist_pred = x_cif_test[:, -HORIZON:]
    results["persistence"] = compute_metrics(persist_pred, y_cif_test)

    # 2. PatchTST supervised
    ptst = train_patchtst(x_cif_train, y_cif_train, epochs=EPOCHS_SUPERVISED)
    with torch.no_grad():
        ptst_pred = ptst(torch.tensor(x_cif_test, dtype=torch.float32)).numpy()
    results["patchtst_sup"] = compute_metrics(ptst_pred, y_cif_test)

    # 3. TransCIF zero-shot
    zs_model = train_zero_shot(all_regions, target_name, seed=seed,
                               model_class=model_class,
                               use_ramp_loss=use_ramp_loss)
    target_cfg = torch.tensor(data["config"]).unsqueeze(0).expand(len(x_rs_test), -1)
    with torch.no_grad():
        zs_rs_pred = zs_model(torch.tensor(x_rs_test, dtype=torch.float32), target_cfg).numpy()
    zs_cif_pred = cif_from_shares(zs_rs_pred, ef_r, ef_nr)
    results["transcif_zs"] = compute_metrics(zs_cif_pred, y_cif_test)

    # 4. TransCIF-ZS+
    origins = [split_hour + st
               for st in range(0, len(cif_offset) - SEQ_LEN - HORIZON + 1, TEST_STRIDE)]
    zsp_pred = zs_plus_predict(zs_model, data["config"], rs, cif, ef_r, ef_nr, origins)
    results["transcif_zs_plus"] = compute_metrics(zsp_pred, y_cif_test)

    # 5. TransCIF-RAG (optional, retrieval-augmented)
    if use_rag:
        try:
            print(f"    [RAG] training...", end="", flush=True)
            from transcif_rag import RagMemoryBank, RagDLinear, train_rag_zero_shot, predict_rag_zs
            rag_model, bank = train_rag_zero_shot(all_regions, target_name, seed=seed)
            cif_rag = predict_rag_zs(rag_model, bank, x_rs_test.astype(np.float32),
                                     data["config"].astype(np.float32), ef_r, ef_nr)
            results["transcif_rag"] = compute_metrics(cif_rag, y_cif_test)
            results["ratio_rag_vs_zs"] = results["transcif_rag"]["mae"] / max(
                results["transcif_zs"]["mae"], 1e-6)
            print(" done", flush=True)
        except Exception as e:
            results["transcif_rag"] = None
            results["ratio_rag_vs_zs"] = None
            print(f"  [WARN] RAG failed for {target_name}: {e}")

    # 6. TransCIF-PhysIRM (optional, physics-informed IRM)
    if use_phys_irm:
        try:
            print(f"    [Phys-IRM] training...", end="", flush=True)
            from transcif_phys_irm import train_phys_irm, predict_phys_irm
            phys_model, phys_log = train_phys_irm(
                all_regions, target_name, seed=seed, gamma_irm=0.1, lambda_cif=0.5)
            cif_phys = predict_phys_irm(phys_model, x_rs_test.astype(np.float32),
                                        data["config"].astype(np.float32), ef_r, ef_nr)
            results["transcif_phys_irm"] = compute_metrics(cif_phys, y_cif_test)
            results["ratio_phys_irm_vs_zs"] = results["transcif_phys_irm"]["mae"] / max(
                results["transcif_zs"]["mae"], 1e-6)
            # Also run weighted-only ablation
            from transcif_phys_irm import train_phys_weighted_only
            pw_model, _ = train_phys_weighted_only(
                all_regions, target_name, seed=seed, lambda_cif=0.5)
            cif_pw = predict_phys_irm(pw_model, x_rs_test.astype(np.float32),
                                      data["config"].astype(np.float32), ef_r, ef_nr)
            results["transcif_phys_weighted"] = compute_metrics(cif_pw, y_cif_test)
            results["ratio_phys_weighted_vs_zs"] = results["transcif_phys_weighted"]["mae"] / max(
                results["transcif_zs"]["mae"], 1e-6)
            results["irm_benefit"] = results["transcif_phys_irm"]["mae"] / max(
                results["transcif_phys_weighted"]["mae"], 1e-6)
        except Exception as e:
            results["transcif_phys_irm"] = None
            results["ratio_phys_irm_vs_zs"] = None
            print(f"\n  [WARN] Phys-IRM failed for {target_name}: {e}")

    # 7. TransCIF-Causal (optional, domain disentanglement)
    if use_causal:
        try:
            print(f"    [Causal] training...", end="", flush=True)
            from transcif_causal import train_causal_zero_shot, predict_causal_zs
            causal_model, _ = train_causal_zero_shot(
                all_regions, target_name, seed=seed)
            cif_causal = predict_causal_zs(
                causal_model, x_rs_test.astype(np.float32),
                data["config"].astype(np.float32), ef_r, ef_nr)
            results["transcif_causal"] = compute_metrics(cif_causal, y_cif_test)
            results["ratio_causal_vs_zs"] = results["transcif_causal"]["mae"] / max(
                results["transcif_zs"]["mae"], 1e-6)
        except Exception as e:
            results["transcif_causal"] = None
            results["ratio_causal_vs_zs"] = None
            print(f"\n  [WARN] Causal failed for {target_name}: {e}")

    # 8. TransCIF-ICL (optional, in-context learning)
    if use_icl:
        try:
            print(f"    [ICL] training...", end="", flush=True)
            from transcif_icl import ICTransformer, train_icl, predict_icl_zs
            icl_model = train_icl(all_regions, target_name, seed=seed)
            cif_icl = predict_icl_zs(
                icl_model, all_regions, target_name,
                x_rs_test.astype(np.float32), ef_r, ef_nr)
            results["transcif_icl"] = compute_metrics(cif_icl, y_cif_test)
            results["ratio_icl_vs_zs"] = results["transcif_icl"]["mae"] / max(
                results["transcif_zs"]["mae"], 1e-6)
        except Exception as e:
            results["transcif_icl"] = None
            results["ratio_icl_vs_zs"] = None
            print(f"\n  [WARN] ICL failed for {target_name}: {e}")

    # 9. TransCIF-Hier (optional, hierarchical debiased)
    if use_hier:
        try:
            print(f"    [Hier] training...", end="", flush=True)
            from transcif_hier import train_hier, predict_hier_zs
            hier_model = train_hier(all_regions, target_name, seed=seed)
            cif_hier = predict_hier_zs(
                hier_model, x_rs_test.astype(np.float32),
                data["config"].astype(np.float32), ef_r, ef_nr)
            results["transcif_hier"] = compute_metrics(cif_hier, y_cif_test)
            results["ratio_hier_vs_zs"] = results["transcif_hier"]["mae"] / max(
                results["transcif_zs"]["mae"], 1e-6)
        except Exception as e:
            results["transcif_hier"] = None
            results["ratio_hier_vs_zs"] = None
            print(f"  [WARN] Hier failed for {target_name}: {e}")

    # Ratios
    results["ratio_vs_patchtst"] = results["transcif_zs"]["mae"] / results["patchtst_sup"]["mae"]
    results["ratio_vs_persist"] = results["transcif_zs"]["mae"] / results["persistence"]["mae"]
    results["ratio_plus_vs_patchtst"] = (
        results["transcif_zs_plus"]["mae"] / results["patchtst_sup"]["mae"])
    results["ratio_plus_vs_persist"] = (
        results["transcif_zs_plus"]["mae"] / results["persistence"]["mae"])

    return results
