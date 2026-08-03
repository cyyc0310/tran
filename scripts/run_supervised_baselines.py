"""Phase 1.1: Supervised baselines for Q1 journal paper.

Runs DLinear-supervised, XGBoost-supervised, and persistence on each AU target region
using the target region's OWN data (80% train, 20% test). Compares against zero-shot
AdaptivePersistDLinear (LORO protocol: all other regions as sources, zero target data).

Establishes the "transfer efficiency ratio" = zero-shot MAE / supervised-best MAE.
Goal: ratio <= 1.25 (zero-shot within 25% of supervised performance).

Usage: PYTHONPATH=src python scripts/run_supervised_baselines.py
"""

import glob
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data_2023"
SEQ_LEN = 336  # 14 days (matching AdaptivePersistDLinear)
HORIZON = 24   # 24-hour ahead forecast
STRIDE = 24    # non-overlapping evaluation windows
TRAIN_FRACTION = 0.8
SEEDS = [0, 1, 2, 3, 4]

# AU regions with known emission factors
AU_REGIONS = {
    "QLD1": {"file": "QLD1_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 841.59},
    "NSW1": {"file": "NSW1_2023_hourly.csv", "ef_r": 0.09, "ef_nr": 875.23},
    "VIC1": {"file": "VIC1_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 1160.12},
    "SA1":  {"file": "SA1_2023_hourly.csv",  "ef_r": 0.0, "ef_nr": 490.43},
}

# UK regions (emission factors estimated from data)
UK_REGIONS = {}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_region_data(region_name: str) -> dict:
    """Load hourly CSV and compute config vector."""
    if region_name in AU_REGIONS:
        info = AU_REGIONS[region_name]
        path = DATA_DIR / info["file"]
        ef_r, ef_nr = info["ef_r"], info["ef_nr"]
    elif region_name in UK_REGIONS:
        info = UK_REGIONS[region_name]
        path = DATA_DIR / info["file"]
        ef_r, ef_nr = info["ef_r"], info["ef_nr"]
    else:
        raise ValueError(f"Unknown region: {region_name}")

    df = pd.read_csv(path, parse_dates=["hour"])
    df = df.sort_values("hour").reset_index(drop=True)
    rs = df["renew_share"].values.astype(np.float32)
    cif = df["cif_real_gco2_per_kwh"].values.astype(np.float32)

    return {
        "rs": rs,
        "cif": cif,
        "mean_rs": float(rs.mean()),
        "ef_r": ef_r,
        "ef_nr": ef_nr,
        "config": np.array([rs.mean(), ef_nr / 1000.0], dtype=np.float32),
    }


def discover_uk_regions():
    """Discover UK regions from data_2023/ and estimate emission factors."""
    global UK_REGIONS
    for f in sorted(glob.glob(str(DATA_DIR / "UK_*_2023_hourly.csv"))):
        name = Path(f).stem.replace("_2023_hourly", "")
        df = pd.read_csv(f)
        rs = df["renew_share"].values
        cif = df["cif_real_gco2_per_kwh"].values

        # Estimate ef_nr from data: CIF ≈ (1-rs) * ef_nr when ef_r ≈ 0
        mask = (rs < 0.95) & (rs > 0.05) & (cif > 0)
        if mask.sum() > 500:
            ef_nr_est = float(np.median(cif[mask] / (1 - rs[mask])))
            if 100 < ef_nr_est < 2000:  # sanity check
                UK_REGIONS[name] = {
                    "file": Path(f).name,
                    "ef_r": 0.0,
                    "ef_nr": ef_nr_est,
                }


def build_windows(rs: np.ndarray, cif: np.ndarray, seq_len: int, horizon: int, stride: int):
    """Build sliding windows for renew_share input and CIF target."""
    window = seq_len + horizon
    x_windows, y_rs_windows, y_cif_windows = [], [], []

    for start in range(0, len(rs) - window + 1, stride):
        x_windows.append(rs[start:start + seq_len])
        y_rs_windows.append(rs[start + seq_len:start + window])
        y_cif_windows.append(cif[start + seq_len:start + window])

    return (
        np.stack(x_windows).astype(np.float32),
        np.stack(y_rs_windows).astype(np.float32),
        np.stack(y_cif_windows).astype(np.float32),
    )


def cif_from_shares(rs: np.ndarray, ef_r: float, ef_nr: float) -> np.ndarray:
    """CIF = rs * ef_r + (1-rs) * ef_nr."""
    return rs * ef_r + (1 - rs) * ef_nr


# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------

class DLinearSupervised(nn.Module):
    """DLinear (Zeng et al., AAAI'23) trained on target domain to predict renew_share."""

    def __init__(self, seq_len: int = 336, horizon: int = 24):
        super().__init__()
        self.seq_len = seq_len
        self.avg_pool = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.linear_trend = nn.Linear(seq_len, horizon)
        self.linear_seasonal = nn.Linear(seq_len, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        x3 = x.unsqueeze(1)  # (batch, 1, seq_len)
        trend = self.avg_pool(x3).squeeze(1)  # (batch, seq_len)
        seasonal = x - trend
        out = torch.sigmoid(self.linear_trend(trend) + self.linear_seasonal(seasonal))
        return out


class AdaptivePersistDLinear(nn.Module):
    """Zero-shot model: DLinear + adaptive persistence gate + config conditioning.
    This is the model that achieved 4/4 clean sweep in LORO experiments."""

    def __init__(self, seq_len: int = 336, horizon: int = 24, config_dim: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.avg_pool = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.linear_trend = nn.Linear(seq_len, horizon)
        self.linear_seasonal = nn.Linear(seq_len, horizon)
        self.config_bias = nn.Sequential(
            nn.Linear(config_dim, 16), nn.ReLU(), nn.Linear(16, horizon)
        )
        self.gate_net = nn.Sequential(
            nn.Linear(config_dim + 2, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor, config: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len), config: (batch, config_dim)
        x3 = x.unsqueeze(1)
        trend = self.avg_pool(x3).squeeze(1)
        seasonal = x - trend
        dlinear_out = torch.sigmoid(
            self.linear_trend(trend) + self.linear_seasonal(seasonal) + self.config_bias(config)
        )
        # Persistence branch: repeat last 24h
        persist = x[:, -self.horizon:]
        # Adaptive gate based on config + recent statistics
        recent_mean = x[:, -48:].mean(dim=1, keepdim=True)
        recent_std = x[:, -48:].std(dim=1, keepdim=True)
        gate_input = torch.cat([config, recent_mean, recent_std], dim=1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        return gate * persist + (1 - gate) * dlinear_out


# ---------------------------------------------------------------------------
# Training Functions
# ---------------------------------------------------------------------------

def train_dlinear_supervised(x_train, y_train, seed=42, epochs=100, lr=1e-3):
    """Train DLinear on target domain data."""
    torch.manual_seed(seed)
    model = DLinearSupervised(seq_len=SEQ_LEN, horizon=HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_t = torch.tensor(x_train)
    y_t = torch.tensor(y_train)

    for _ in range(epochs):
        pred = model(x_t)
        loss = nn.functional.l1_loss(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def train_xgboost_supervised(x_train, y_train_cif, seed=42):
    """Train gradient boosting on target domain with lag features to predict CIF directly."""
    from sklearn.ensemble import GradientBoostingRegressor

    # Feature engineering: statistical features from the input window
    features = []
    for i in range(len(x_train)):
        rs_window = x_train[i]
        feat = [
            rs_window[-1],    # last value
            rs_window[-24:].mean(),  # last day mean
            rs_window[-24:].std(),   # last day std
            rs_window[-168:].mean(), # last week mean
            rs_window.mean(),        # full window mean
            rs_window.std(),         # full window std
            rs_window[-1] - rs_window[-24],  # 24h change
            rs_window[-1] - rs_window[-168], # 7d change
            np.percentile(rs_window, 25),
            np.percentile(rs_window, 75),
        ]
        features.append(feat)

    X = np.array(features)
    # Predict mean CIF over horizon
    Y = y_train_cif.mean(axis=1)

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=seed,
    )
    model.fit(X, Y)
    return model


def predict_xgboost(model, x_test):
    """Predict with XGBoost model."""
    features = []
    for i in range(len(x_test)):
        rs_window = x_test[i]
        feat = [
            rs_window[-1],
            rs_window[-24:].mean(),
            rs_window[-24:].std(),
            rs_window[-168:].mean(),
            rs_window.mean(),
            rs_window.std(),
            rs_window[-1] - rs_window[-24],
            rs_window[-1] - rs_window[-168],
            np.percentile(rs_window, 25),
            np.percentile(rs_window, 75),
        ]
        features.append(feat)
    X = np.array(features)
    # XGBoost predicts mean CIF; expand to horizon
    mean_cif = model.predict(X)
    return np.repeat(mean_cif[:, None], HORIZON, axis=1)


def train_zero_shot(sources: dict, target_name: str, seed=42, epochs=100, lr=1e-3):
    """Train AdaptivePersistDLinear on source domains (LORO protocol)."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = AdaptivePersistDLinear(seq_len=SEQ_LEN, horizon=HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Build source data with config-distance weighted sampling
    target_config = sources[target_name]["config"]
    xs, ys, cfgs, weights = [], [], [], []

    for name, data in sources.items():
        if name == target_name:
            continue
        x_win, y_win, _ = build_windows(data["rs"], data["cif"], SEQ_LEN, HORIZON, STRIDE)
        if len(x_win) == 0:
            continue
        xs.append(x_win)
        ys.append(y_win)
        cfg = np.tile(data["config"], (len(x_win), 1))
        cfgs.append(cfg)
        # Config distance weighting
        dist = abs(data["mean_rs"] - sources[target_name]["mean_rs"])
        w = 1.0 / (dist + 0.05)
        weights.append(np.full(len(x_win), w))

    x_all = torch.tensor(np.concatenate(xs))
    y_all = torch.tensor(np.concatenate(ys))
    c_all = torch.tensor(np.concatenate(cfgs))
    w_all = torch.tensor(np.concatenate(weights))
    w_all = w_all / w_all.sum() * len(w_all)

    for _ in range(epochs):
        pred = model(x_all, c_all)
        # Weighted L1 loss on renew_share prediction
        loss = (w_all.unsqueeze(1) * torch.abs(pred - y_all)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_all(target_name: str, all_regions: dict, seed: int = 42) -> dict:
    """Run all baselines on one target region."""
    data = all_regions[target_name]
    rs, cif = data["rs"], data["cif"]
    ef_r, ef_nr = data["ef_r"], data["ef_nr"]

    # Build windows
    x_win, y_rs_win, y_cif_win = build_windows(rs, cif, SEQ_LEN, HORIZON, STRIDE)
    n = len(x_win)
    split = int(n * TRAIN_FRACTION)

    x_train, x_test = x_win[:split], x_win[split:]
    y_rs_train, y_rs_test = y_rs_win[:split], y_rs_win[split:]
    y_cif_train, y_cif_test = y_cif_win[:split], y_cif_win[split:]

    results = {"target": target_name, "seed": seed, "n_test": len(x_test)}

    # --- Persistence baseline (lag-24h) ---
    persist_rs = x_test[:, -HORIZON:]
    persist_cif = cif_from_shares(persist_rs, ef_r, ef_nr)
    persist_mae = float(np.abs(persist_cif - y_cif_test).mean())
    results["persistence_mae"] = persist_mae

    # --- DLinear-supervised ---
    dlinear = train_dlinear_supervised(x_train, y_rs_train, seed=seed)
    with torch.no_grad():
        rs_pred = dlinear(torch.tensor(x_test)).numpy()
    dlinear_cif = cif_from_shares(rs_pred, ef_r, ef_nr)
    dlinear_mae = float(np.abs(dlinear_cif - y_cif_test).mean())
    results["dlinear_supervised_mae"] = dlinear_mae

    # --- XGBoost-supervised (direct CIF prediction) ---
    try:
        xgb = train_xgboost_supervised(x_train, y_cif_train, seed=seed)
        if xgb is not None:
            xgb_cif = predict_xgboost(xgb, x_test)
            xgb_mae = float(np.abs(xgb_cif - y_cif_test).mean())
            results["xgboost_supervised_mae"] = xgb_mae
        else:
            results["xgboost_supervised_mae"] = None
    except Exception as e:
        print(f"  [WARN] XGBoost failed: {e}")
        results["xgboost_supervised_mae"] = None

    # --- Zero-shot AdaptivePersistDLinear (LORO) ---
    zs_model = train_zero_shot(all_regions, target_name, seed=seed)
    target_config = torch.tensor(data["config"]).unsqueeze(0).expand(len(x_test), -1)
    with torch.no_grad():
        zs_rs_pred = zs_model(torch.tensor(x_test), target_config).numpy()
    zs_cif = cif_from_shares(zs_rs_pred, ef_r, ef_nr)
    zs_mae = float(np.abs(zs_cif - y_cif_test).mean())
    results["zero_shot_mae"] = zs_mae

    # --- Transfer efficiency ratio ---
    best_supervised = min(
        r for r in [results["dlinear_supervised_mae"], results.get("xgboost_supervised_mae")]
        if r is not None
    )
    results["best_supervised_mae"] = best_supervised
    results["transfer_efficiency_ratio"] = zs_mae / best_supervised if best_supervised > 0 else float("inf")
    results["zs_vs_persistence"] = zs_mae / persist_mae

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Phase 1.1: Supervised Baselines vs Zero-Shot (LORO Protocol)")
    print("=" * 70)

    # Discover all regions
    discover_uk_regions()
    print(f"\nDiscovered {len(UK_REGIONS)} UK regions with valid emission factors")

    # Load all region data
    all_regions = {}
    for name in AU_REGIONS:
        all_regions[name] = load_region_data(name)
    for name in UK_REGIONS:
        all_regions[name] = load_region_data(name)

    print(f"Total regions available: {len(all_regions)}")
    print(f"  AU: {list(AU_REGIONS.keys())}")
    print(f"  UK: {list(UK_REGIONS.keys())}")
    print(f"\nConfig vectors:")
    for name, data in all_regions.items():
        if name in AU_REGIONS:
            print(f"  {name:20s} mean_rs={data['mean_rs']:.3f}  ef_nr={data['ef_nr']:.1f}")

    print(f"\n{'='*70}")
    print(f"Running LORO evaluation on 4 AU target regions")
    print(f"  SEQ_LEN={SEQ_LEN}, HORIZON={HORIZON}, STRIDE={STRIDE}")
    print(f"  TRAIN_FRACTION={TRAIN_FRACTION}, SEEDS={SEEDS}")
    print(f"{'='*70}\n")

    # Run evaluations
    all_results = []
    for target in ["QLD1", "NSW1", "VIC1", "SA1"]:
        print(f"\n--- Target: {target} ---")
        seed_results = []
        for seed in SEEDS:
            print(f"  seed={seed} ...", end=" ", flush=True)
            result = evaluate_all(target, all_regions, seed=seed)
            seed_results.append(result)
            print(f"ZS={result['zero_shot_mae']:.1f} DL={result['dlinear_supervised_mae']:.1f} "
                  f"XGB={result.get('xgboost_supervised_mae', 'N/A')} "
                  f"Persist={result['persistence_mae']:.1f} "
                  f"Ratio={result['transfer_efficiency_ratio']:.3f}")

        all_results.append(seed_results)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Mean ± Std across 5 seeds (MAE in gCO₂/kWh)")
    print("=" * 70)
    print(f"{'Region':<8} {'Persistence':<14} {'DLinear-sup':<14} {'XGBoost-sup':<14} "
          f"{'Zero-shot':<14} {'Best-sup':<12} {'Ratio':<10} {'ZS/Persist':<10}")
    print("-" * 100)

    for target_idx, target in enumerate(["QLD1", "NSW1", "VIC1", "SA1"]):
        sr = all_results[target_idx]
        persist = np.array([r["persistence_mae"] for r in sr])
        dl = np.array([r["dlinear_supervised_mae"] for r in sr])
        xgb_vals = [r["xgboost_supervised_mae"] for r in sr if r.get("xgboost_supervised_mae") is not None]
        xgb = np.array(xgb_vals) if xgb_vals else None
        zs = np.array([r["zero_shot_mae"] for r in sr])
        best_sup = np.array([r["best_supervised_mae"] for r in sr])
        ratio = np.array([r["transfer_efficiency_ratio"] for r in sr])
        zs_p = np.array([r["zs_vs_persistence"] for r in sr])

        xgb_str = f"{xgb.mean():.1f}±{xgb.std():.1f}" if xgb is not None else "N/A"
        print(f"{target:<8} {persist.mean():.1f}±{persist.std():.1f}{'':4s} "
              f"{dl.mean():.1f}±{dl.std():.1f}{'':4s} "
              f"{xgb_str:<14} "
              f"{zs.mean():.1f}±{zs.std():.1f}{'':4s} "
              f"{best_sup.mean():.1f}{'':6s} "
              f"{ratio.mean():.3f}{'':5s} "
              f"{zs_p.mean():.3f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("  Ratio = zero_shot_MAE / best_supervised_MAE")
    print("  Ratio ~1.0 = zero-shot matches supervised (ideal)")
    print("  Ratio <1.25 = zero-shot within 25% of supervised (publishable)")
    print("  ZS/Persist <1.0 = zero-shot beats persistence (minimum bar)")
    print("=" * 70)


if __name__ == "__main__":
    main()
