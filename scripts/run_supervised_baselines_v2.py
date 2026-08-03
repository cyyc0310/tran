"""Phase 1.1 v2: Improved supervised baselines with direct CIF prediction.

Key improvements over v1:
- DLinear-direct: predicts CIF directly (not rs->CIF), stronger supervised baseline
- More training epochs (200) for better convergence
- Overlapping test windows (stride=6) for more stable evaluation
- PatchTST-lite: a multi-head attention model for stronger comparison

Usage: PYTHONPATH=src python scripts/run_supervised_baselines_v2.py
"""

import glob
import random
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
SEQ_LEN = 336   # 14 days
HORIZON = 24    # 24-hour ahead
TRAIN_STRIDE = 6   # overlapping windows for training
TEST_STRIDE = 24   # non-overlapping for test (clean evaluation)
TRAIN_FRACTION = 0.8
SEEDS = [0, 1, 2]  # 3 seeds for speed, expand to 5 for paper
EPOCHS_SUPERVISED = 200
EPOCHS_ZERO_SHOT = 150

# AU regions with measured emission factors (from NEMED 2023)
AU_REGIONS = {
    "QLD1": {"file": "QLD1_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 841.59},
    "NSW1": {"file": "NSW1_2023_hourly.csv", "ef_r": 0.09, "ef_nr": 875.23},
    "VIC1": {"file": "VIC1_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 1160.12},
    "SA1":  {"file": "SA1_2023_hourly.csv",  "ef_r": 0.0, "ef_nr": 490.43},
}

UK_REGIONS = {}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def discover_uk_regions():
    """Discover UK regions and estimate emission factors from data."""
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


def load_region_data(region_name: str) -> dict:
    """Load hourly CSV and compute config vector."""
    all_regions = {**AU_REGIONS, **UK_REGIONS}
    info = all_regions[region_name]
    path = DATA_DIR / info["file"]
    ef_r, ef_nr = info["ef_r"], info["ef_nr"]

    df = pd.read_csv(path, parse_dates=["hour"])
    df = df.sort_values("hour").reset_index(drop=True)
    rs = df["renew_share"].values.astype(np.float32)
    cif = df["cif_real_gco2_per_kwh"].values.astype(np.float32)

    return {
        "rs": rs, "cif": cif,
        "mean_rs": float(rs.mean()),
        "ef_r": ef_r, "ef_nr": ef_nr,
        "config": np.array([rs.mean(), ef_nr / 1000.0], dtype=np.float32),
    }


def build_windows(rs, cif, seq_len, horizon, stride):
    """Build sliding windows."""
    window = seq_len + horizon
    x_rs, y_rs, y_cif = [], [], []
    for start in range(0, len(rs) - window + 1, stride):
        x_rs.append(rs[start:start + seq_len])
        y_rs.append(rs[start + seq_len:start + window])
        y_cif.append(cif[start + seq_len:start + window])
    return np.stack(x_rs), np.stack(y_rs), np.stack(y_cif)


def cif_from_shares(rs, ef_r, ef_nr):
    return rs * ef_r + (1 - rs) * ef_nr


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class DLinearDirect(nn.Module):
    """DLinear that directly predicts CIF (no physics constraint). Stronger supervised baseline."""
    def __init__(self, seq_len=336, horizon=24):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.linear_trend = nn.Linear(seq_len, horizon)
        self.linear_seasonal = nn.Linear(seq_len, horizon)

    def forward(self, x):
        # x: (batch, seq_len) - uses CIF history directly
        x3 = x.unsqueeze(1)
        trend = self.avg_pool(x3).squeeze(1)
        seasonal = x - trend
        return self.linear_trend(trend) + self.linear_seasonal(seasonal)


class DLinearRS(nn.Module):
    """DLinear that predicts renew_share (comparable to our approach)."""
    def __init__(self, seq_len=336, horizon=24):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.linear_trend = nn.Linear(seq_len, horizon)
        self.linear_seasonal = nn.Linear(seq_len, horizon)

    def forward(self, x):
        x3 = x.unsqueeze(1)
        trend = self.avg_pool(x3).squeeze(1)
        seasonal = x - trend
        return torch.sigmoid(self.linear_trend(trend) + self.linear_seasonal(seasonal))


class PatchTSTLite(nn.Module):
    """Simplified PatchTST-style model for supervised baseline."""
    def __init__(self, seq_len=336, horizon=24, patch_len=24, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.patch_len = patch_len
        n_patches = seq_len // patch_len
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(n_patches * d_model, horizon)

    def forward(self, x):
        # x: (batch, seq_len)
        B = x.shape[0]
        patches = x.unfold(1, self.patch_len, self.patch_len)  # (B, n_patches, patch_len)
        x_emb = self.patch_embed(patches) + self.pos_embed
        x_enc = self.transformer(x_emb)
        return self.head(x_enc.reshape(B, -1))


class AdaptivePersistDLinear(nn.Module):
    """Zero-shot model (our method)."""
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
            self.linear_trend(trend) + self.linear_seasonal(seasonal) + self.config_bias(config)
        )
        persist = x[:, -self.horizon:]
        recent_mean = x[:, -48:].mean(dim=1, keepdim=True)
        recent_std = x[:, -48:].std(dim=1, keepdim=True)
        gate_input = torch.cat([config, recent_mean, recent_std], dim=1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        return gate * persist + (1 - gate) * dlinear_out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, x_train, y_train, epochs=200, lr=1e-3, loss_fn="l1"):
    """Generic supervised training loop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    x_t, y_t = torch.tensor(x_train), torch.tensor(y_train)

    for _ in range(epochs):
        pred = model(x_t)
        if loss_fn == "l1":
            loss = nn.functional.l1_loss(pred, y_t)
        else:
            loss = nn.functional.mse_loss(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    return model


def train_gbrt(x_train_rs, y_train_cif, seed=42):
    """GradientBoosting for direct CIF prediction with rich features."""
    features = []
    for rs in x_train_rs:
        feat = [
            rs[-1], rs[-24:].mean(), rs[-24:].std(),
            rs[-48:].mean(), rs[-48:].std(),
            rs[-168:].mean(), rs[-168:].std(),
            rs.mean(), rs.std(),
            rs[-1] - rs[-24], rs[-1] - rs[-168],
            np.percentile(rs, 10), np.percentile(rs, 25),
            np.percentile(rs, 75), np.percentile(rs, 90),
            # Hour-of-day proxy (cyclic patterns)
            float(np.argmax(rs[-24:])) / 24.0,
        ]
        features.append(feat)
    X = np.array(features)
    Y = y_train_cif.mean(axis=1)  # mean over horizon

    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=seed,
    )
    model.fit(X, Y)
    return model, X


def predict_gbrt(model, x_test_rs, horizon):
    features = []
    for rs in x_test_rs:
        feat = [
            rs[-1], rs[-24:].mean(), rs[-24:].std(),
            rs[-48:].mean(), rs[-48:].std(),
            rs[-168:].mean(), rs[-168:].std(),
            rs.mean(), rs.std(),
            rs[-1] - rs[-24], rs[-1] - rs[-168],
            np.percentile(rs, 10), np.percentile(rs, 25),
            np.percentile(rs, 75), np.percentile(rs, 90),
            float(np.argmax(rs[-24:])) / 24.0,
        ]
        features.append(feat)
    X = np.array(features)
    mean_cif = model.predict(X)
    return np.repeat(mean_cif[:, None], horizon, axis=1)


def train_zero_shot(all_regions, target_name, seed=42):
    """Train AdaptivePersistDLinear on ALL other regions (LORO)."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = AdaptivePersistDLinear(seq_len=SEQ_LEN, horizon=HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_ZERO_SHOT)

    target_config = all_regions[target_name]["config"]
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

    # Mini-batch training for large datasets
    n_samples = len(x_all)
    batch_size = min(512, n_samples)

    for epoch in range(EPOCHS_ZERO_SHOT):
        idx = torch.randperm(n_samples)[:batch_size]
        pred = model(x_all[idx], c_all[idx])
        loss = (w_all[idx].unsqueeze(1) * torch.abs(pred - y_all[idx])).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_all(target_name: str, all_regions: dict, seed: int = 42) -> dict:
    """Run all methods on one target region."""
    torch.manual_seed(seed)
    data = all_regions[target_name]
    rs, cif = data["rs"], data["cif"]
    ef_r, ef_nr = data["ef_r"], data["ef_nr"]

    # Split time series: first 80% for train, last 20% for test
    n_hours = len(rs)
    split_hour = int(n_hours * TRAIN_FRACTION)

    # Build train windows (overlapping for more data)
    x_rs_train, y_rs_train, y_cif_train = build_windows(
        rs[:split_hour], cif[:split_hour], SEQ_LEN, HORIZON, TRAIN_STRIDE)

    # Also build CIF-history windows for direct prediction
    x_cif_train = []
    for start in range(0, split_hour - SEQ_LEN - HORIZON + 1, TRAIN_STRIDE):
        x_cif_train.append(cif[start:start + SEQ_LEN])
    x_cif_train = np.stack(x_cif_train)

    # Build test windows (non-overlapping for clean eval)
    x_rs_test, y_rs_test, y_cif_test = build_windows(
        rs[split_hour - SEQ_LEN:], cif[split_hour - SEQ_LEN:], SEQ_LEN, HORIZON, TEST_STRIDE)
    x_cif_test = []
    rs_test_offset = rs[split_hour - SEQ_LEN:]
    cif_test_offset = cif[split_hour - SEQ_LEN:]
    for start in range(0, len(rs_test_offset) - SEQ_LEN - HORIZON + 1, TEST_STRIDE):
        x_cif_test.append(cif_test_offset[start:start + SEQ_LEN])
    x_cif_test = np.stack(x_cif_test)

    results = {"target": target_name, "seed": seed, "n_train": len(x_rs_train), "n_test": len(x_rs_test)}

    # --- 1. Persistence (lag-24h on CIF) ---
    persist_cif = x_cif_test[:, -HORIZON:]
    results["persistence_mae"] = float(np.abs(persist_cif - y_cif_test).mean())

    # --- 2. DLinear-RS supervised (predict rs, then physics) ---
    dlinear_rs = DLinearRS(seq_len=SEQ_LEN, horizon=HORIZON)
    train_model(dlinear_rs, x_rs_train, y_rs_train, epochs=EPOCHS_SUPERVISED, lr=5e-4)
    with torch.no_grad():
        rs_pred = dlinear_rs(torch.tensor(x_rs_test)).numpy()
    results["dlinear_rs_mae"] = float(np.abs(cif_from_shares(rs_pred, ef_r, ef_nr) - y_cif_test).mean())

    # --- 3. DLinear-Direct supervised (predict CIF directly from CIF history) ---
    dlinear_direct = DLinearDirect(seq_len=SEQ_LEN, horizon=HORIZON)
    train_model(dlinear_direct, x_cif_train, y_cif_train, epochs=EPOCHS_SUPERVISED, lr=5e-4)
    with torch.no_grad():
        cif_pred = dlinear_direct(torch.tensor(x_cif_test)).numpy()
    results["dlinear_direct_mae"] = float(np.abs(cif_pred - y_cif_test).mean())

    # --- 4. PatchTST-lite supervised (direct CIF prediction) ---
    patchtst = PatchTSTLite(seq_len=SEQ_LEN, horizon=HORIZON)
    train_model(patchtst, x_cif_train, y_cif_train, epochs=EPOCHS_SUPERVISED, lr=1e-3, loss_fn="mse")
    with torch.no_grad():
        cif_pred_ptst = patchtst(torch.tensor(x_cif_test)).numpy()
    results["patchtst_mae"] = float(np.abs(cif_pred_ptst - y_cif_test).mean())

    # --- 5. GBRT supervised (feature-engineered) ---
    try:
        gbrt, _ = train_gbrt(x_rs_train, y_cif_train, seed=seed)
        gbrt_pred = predict_gbrt(gbrt, x_rs_test, HORIZON)
        results["gbrt_mae"] = float(np.abs(gbrt_pred - y_cif_test).mean())
    except Exception as e:
        results["gbrt_mae"] = None
        print(f"  [WARN] GBRT failed: {e}")

    # --- 6. Zero-shot AdaptivePersistDLinear (LORO, ALL other regions) ---
    zs_model = train_zero_shot(all_regions, target_name, seed=seed)
    target_cfg = torch.tensor(data["config"]).unsqueeze(0).expand(len(x_rs_test), -1)
    with torch.no_grad():
        zs_rs_pred = zs_model(torch.tensor(x_rs_test), target_cfg).numpy()
    zs_cif = cif_from_shares(zs_rs_pred, ef_r, ef_nr)
    results["zero_shot_mae"] = float(np.abs(zs_cif - y_cif_test).mean())

    # --- Compute ratios ---
    supervised_maes = [v for k, v in results.items()
                       if k.endswith("_mae") and k not in ("persistence_mae", "zero_shot_mae") and v is not None]
    results["best_supervised_mae"] = min(supervised_maes) if supervised_maes else None
    if results["best_supervised_mae"]:
        results["transfer_ratio"] = results["zero_shot_mae"] / results["best_supervised_mae"]
    results["zs_vs_persist"] = results["zero_shot_mae"] / results["persistence_mae"]

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Phase 1.1 v2: Comprehensive Supervised Baselines vs Zero-Shot")
    print("=" * 80)

    discover_uk_regions()
    print(f"Regions: {len(AU_REGIONS)} AU + {len(UK_REGIONS)} UK = {len(AU_REGIONS) + len(UK_REGIONS)} total")

    all_regions = {}
    for name in AU_REGIONS:
        all_regions[name] = load_region_data(name)
    for name in UK_REGIONS:
        all_regions[name] = load_region_data(name)

    print(f"\nAU Config: " + " | ".join(f"{n}: rs={d['mean_rs']:.3f}" for n, d in all_regions.items() if n in AU_REGIONS))
    print(f"\nSettings: SEQ={SEQ_LEN} HOR={HORIZON} epochs_sup={EPOCHS_SUPERVISED} epochs_zs={EPOCHS_ZERO_SHOT}")
    print(f"Seeds: {SEEDS}\n")

    all_results = {}
    for target in ["QLD1", "NSW1", "VIC1", "SA1"]:
        print(f"\n{'='*60}")
        print(f"  Target: {target} (mean_rs={all_regions[target]['mean_rs']:.3f}, ef_nr={all_regions[target]['ef_nr']:.0f})")
        print(f"{'='*60}")
        seed_results = []
        for seed in SEEDS:
            print(f"  seed={seed}:", flush=True)
            r = evaluate_all(target, all_regions, seed=seed)
            seed_results.append(r)
            gbrt_str = f"{r['gbrt_mae']:.1f}" if isinstance(r.get('gbrt_mae'), (int, float)) else "N/A"
            print(f"    Persist={r['persistence_mae']:.1f} | DL-RS={r['dlinear_rs_mae']:.1f} "
                  f"DL-Direct={r['dlinear_direct_mae']:.1f} PatchTST={r['patchtst_mae']:.1f} "
                  f"GBRT={gbrt_str} "
                  f"| ZS={r['zero_shot_mae']:.1f} | Ratio={r.get('transfer_ratio', 0):.3f}")
        all_results[target] = seed_results

    # Final summary table
    print("\n\n" + "=" * 100)
    print("FINAL SUMMARY (Mean across seeds, MAE in gCO₂/kWh)")
    print("=" * 100)
    header = f"{'Region':<6} {'Persist':<9} {'DL-RS':<9} {'DL-Direct':<11} {'PatchTST':<10} {'GBRT':<9} {'Zero-Shot':<11} {'Best-Sup':<10} {'Ratio':<7} {'ZS/P':<6}"
    print(header)
    print("-" * len(header))

    for target in ["QLD1", "NSW1", "VIC1", "SA1"]:
        sr = all_results[target]
        persist = np.mean([r["persistence_mae"] for r in sr])
        dl_rs = np.mean([r["dlinear_rs_mae"] for r in sr])
        dl_dir = np.mean([r["dlinear_direct_mae"] for r in sr])
        ptst = np.mean([r["patchtst_mae"] for r in sr])
        gbrt_vals = [r["gbrt_mae"] for r in sr if r.get("gbrt_mae") is not None]
        gbrt = np.mean(gbrt_vals) if gbrt_vals else float("nan")
        zs = np.mean([r["zero_shot_mae"] for r in sr])
        best_sup = np.mean([r["best_supervised_mae"] for r in sr if r.get("best_supervised_mae")])
        ratio = np.mean([r["transfer_ratio"] for r in sr if r.get("transfer_ratio")])
        zs_p = np.mean([r["zs_vs_persist"] for r in sr])

        print(f"{target:<6} {persist:<9.1f} {dl_rs:<9.1f} {dl_dir:<11.1f} {ptst:<10.1f} "
              f"{gbrt:<9.1f} {zs:<11.1f} {best_sup:<10.1f} {ratio:<7.3f} {zs_p:<6.3f}")

    print("\n" + "=" * 100)
    print("Ratio = ZeroShot / BestSupervised (lower=better, <1.25 is publishable)")
    print("ZS/P  = ZeroShot / Persistence (lower=better, <1.0 means beating persistence)")
    print("=" * 100)


if __name__ == "__main__":
    main()
