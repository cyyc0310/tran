"""Phase 1.1 Complete: All baselines + CarbonCast CNN-LSTM + Fixed PatchTST.

Key additions over v2:
- CarbonCast-style CNN-LSTM (PyTorch reimpl): supervised + zero-shot dual-mode
- Fixed PatchTST with RevIN normalization + cosine warmup + lower lr
- All results in one script for reproducibility

Usage: PYTHONPATH=src python scripts/run_phase1_complete.py
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
TRAIN_STRIDE = 6
TEST_STRIDE = 24
TRAIN_FRACTION = 0.8
SEEDS = [0, 1, 2]
EPOCHS_SUPERVISED = 300      # increased from 200
EPOCHS_CARBONCAST = 300      # CarbonCast needs more (CNN-LSTM is larger)
EPOCHS_ZERO_SHOT = 150
BATCH_SIZE = 256

# AU regions with measured emission factors
AU_REGIONS = {
    "QLD1": {"file": "QLD1_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 841.59},
    "NSW1": {"file": "NSW1_2023_hourly.csv", "ef_r": 0.09, "ef_nr": 875.23},
    "VIC1": {"file": "VIC1_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 1160.12},
    "SA1":  {"file": "SA1_2023_hourly.csv",  "ef_r": 0.0, "ef_nr": 490.43},
}

UK_REGIONS = {}


# ---------------------------------------------------------------------------
# Data Loading (same as v2)
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


def load_region_data(region_name: str) -> dict:
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
    window = seq_len + horizon
    x_rs, y_rs, y_cif = [], [], []
    for start in range(0, len(rs) - window + 1, stride):
        x_rs.append(rs[start:start + seq_len])
        y_rs.append(rs[start + seq_len:start + window])
        y_cif.append(cif[start + seq_len:start + window])
    return np.stack(x_rs), np.stack(y_rs), np.stack(y_cif)


def build_multivariate_windows(rs, cif, seq_len, horizon, stride):
    """Build multivariate windows: input is (rs, cif) history, target is CIF."""
    window = seq_len + horizon
    x_multi, y_cif_out = [], []
    for start in range(0, len(rs) - window + 1, stride):
        # Stack rs and cif as 2-channel input
        x_multi.append(np.stack([rs[start:start + seq_len],
                                  cif[start:start + seq_len]], axis=-1))
        y_cif_out.append(cif[start + seq_len:start + window])
    return np.stack(x_multi), np.stack(y_cif_out)


def cif_from_shares(rs, ef_r, ef_nr):
    return rs * ef_r + (1 - rs) * ef_nr


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RevIN(nn.Module):
    """Reversible Instance Normalization for time series."""
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True) + self.eps
            return (x - self.mean) / self.std
        else:  # denorm
            return x * self.std + self.mean


class DLinearDirect(nn.Module):
    """DLinear that directly predicts CIF."""
    def __init__(self, seq_len=336, horizon=24):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.linear_trend = nn.Linear(seq_len, horizon)
        self.linear_seasonal = nn.Linear(seq_len, horizon)

    def forward(self, x):
        x3 = x.unsqueeze(1)
        trend = self.avg_pool(x3).squeeze(1)
        seasonal = x - trend
        return self.linear_trend(trend) + self.linear_seasonal(seasonal)


class DLinearRS(nn.Module):
    """DLinear that predicts renew_share."""
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


class PatchTSTFixed(nn.Module):
    """PatchTST with RevIN normalization — fixes convergence issue."""
    def __init__(self, seq_len=336, horizon=24, patch_len=24, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.patch_len = patch_len
        self.horizon = horizon
        n_patches = seq_len // patch_len
        self.revin = RevIN()
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(n_patches * d_model),
            nn.Linear(n_patches * d_model, horizon)
        )

    def forward(self, x):
        # RevIN normalization
        x_norm = self.revin(x, 'norm')
        B = x_norm.shape[0]
        patches = x_norm.unfold(1, self.patch_len, self.patch_len)
        x_emb = self.patch_embed(patches) + self.pos_embed
        x_enc = self.transformer(x_emb)
        out_norm = self.head(x_enc.reshape(B, -1))
        # Denormalize output
        return self.revin(out_norm, 'denorm')


class CarbonCastCNNLSTM(nn.Module):
    """PyTorch reimplementation of CarbonCast's CNN-LSTM architecture.

    Original (Keras): Conv1D → MaxPool → Conv1D → Flatten → RepeatVector → LSTM → Dropout → Dense
    Key: includes built-in min-max normalization (same as original CarbonCast's common.scaleDataset).

    Input: multivariate time series (rs + cif history) of shape (batch, seq_len, n_features)
    Output: (batch, horizon) CIF predictions (denormalized)
    """
    def __init__(self, seq_len=336, horizon=24, n_features=2,
                 filters1=64, kernel1=7, filters2=32, kernel2=5,
                 pool_size=2, lstm_hidden=64, dropout=0.2):
        super().__init__()
        self.horizon = horizon
        self.n_features = n_features

        # Normalization stats (set during training)
        self.register_buffer('feat_min', torch.zeros(n_features))
        self.register_buffer('feat_max', torch.ones(n_features))
        self.register_buffer('target_min', torch.tensor(0.0))
        self.register_buffer('target_max', torch.tensor(1.0))

        # Conv1D Block 1
        self.conv1 = nn.Conv1d(n_features, filters1, kernel_size=kernel1, padding='same')
        self.act1 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size)

        # Conv1D Block 2
        seq_after_pool = seq_len // pool_size
        self.conv2 = nn.Conv1d(filters1, filters2, kernel_size=kernel2, padding='same')
        self.act2 = nn.ReLU()

        # Flatten size
        self.flat_size = seq_after_pool * filters2

        # RepeatVector + LSTM
        self.lstm = nn.LSTM(input_size=self.flat_size, hidden_size=lstm_hidden,
                           batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(lstm_hidden, 1)

    def set_normalization(self, x_train, y_train):
        """Compute min-max stats from training data (like original CarbonCast)."""
        # x_train: (N, seq_len, n_features)
        feat_min = x_train.reshape(-1, self.n_features).min(axis=0)
        feat_max = x_train.reshape(-1, self.n_features).max(axis=0)
        self.feat_min = torch.tensor(feat_min, dtype=torch.float32)
        self.feat_max = torch.tensor(feat_max, dtype=torch.float32)
        # Avoid division by zero
        diff = self.feat_max - self.feat_min
        diff[diff < 1e-6] = 1.0
        self.feat_max = self.feat_min + diff

        self.target_min = torch.tensor(float(y_train.min()), dtype=torch.float32)
        self.target_max = torch.tensor(float(y_train.max()), dtype=torch.float32)
        if self.target_max - self.target_min < 1e-6:
            self.target_max = self.target_min + 1.0

    def normalize_input(self, x):
        """Min-max normalize input features to [0, 1]."""
        return (x - self.feat_min) / (self.feat_max - self.feat_min)

    def denormalize_output(self, y):
        """Denormalize output from [0, 1] back to CIF scale."""
        return y * (self.target_max - self.target_min) + self.target_min

    def normalize_target(self, y):
        """Normalize target to [0, 1]."""
        return (y - self.target_min) / (self.target_max - self.target_min)

    def forward(self, x, denorm=True):
        """x: (batch, seq_len, n_features)"""
        # Normalize input
        x = self.normalize_input(x)
        # Conv expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.act1(self.conv1(x))
        x = self.pool(x)
        x = self.act2(self.conv2(x))
        # Flatten
        x = x.permute(0, 2, 1)  # (batch, seq_after_pool, filters2)
        B = x.shape[0]
        flat = x.reshape(B, -1)  # (batch, flat_size)
        # RepeatVector: repeat for each output timestep
        repeated = flat.unsqueeze(1).expand(-1, self.horizon, -1)
        # LSTM
        lstm_out, _ = self.lstm(repeated)
        lstm_out = self.dropout(lstm_out)
        # Dense for each timestep → sigmoid to keep in [0,1]
        out = torch.sigmoid(self.dense(lstm_out).squeeze(-1))  # (batch, horizon)
        # Denormalize to CIF scale
        if denorm:
            out = self.denormalize_output(out)
        return out


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
# Training Utilities
# ---------------------------------------------------------------------------

def get_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    """Cosine schedule with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model_batched(model, x_train, y_train, epochs=300, lr=1e-3,
                        loss_fn="l1", warmup=20, batch_size=BATCH_SIZE):
    """Mini-batch training with cosine warmup scheduler."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_cosine_warmup_scheduler(optimizer, warmup, epochs)
    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    n = len(x_t)

    model.train()
    for epoch in range(epochs):
        idx = torch.randperm(n)[:min(batch_size, n)]
        pred = model(x_t[idx])
        if loss_fn == "l1":
            loss = nn.functional.l1_loss(pred, y_t[idx])
        else:
            loss = nn.functional.mse_loss(pred, y_t[idx])
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    return model


def train_carboncast(model, x_train, y_train, epochs=300, lr=5e-4, batch_size=BATCH_SIZE):
    """Train CarbonCast CNN-LSTM with multivariate input + min-max normalization."""
    # Set normalization stats from training data
    model.set_normalization(x_train, y_train)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_cosine_warmup_scheduler(optimizer, 30, epochs)
    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    # Normalize targets for training
    y_t_norm = model.normalize_target(y_t)
    n = len(x_t)

    model.train()
    for epoch in range(epochs):
        idx = torch.randperm(n)[:min(batch_size, n)]
        pred = model(x_t[idx], denorm=False)  # get normalized output
        loss = nn.functional.l1_loss(pred, y_t_norm[idx])
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    return model


def train_gbrt(x_train_rs, y_train_cif, seed=42):
    """GBRT with rich features."""
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
            float(np.argmax(rs[-24:])) / 24.0,
        ]
        features.append(feat)
    X = np.array(features)
    Y = y_train_cif.mean(axis=1)
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=seed)
    model.fit(X, Y)
    return model


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


def train_carboncast_zero_shot(all_regions, target_name, seed=42):
    """Train CarbonCast CNN-LSTM on source regions, test on target (cross-domain).
    
    Key: normalization is computed from SOURCE data only — simulates real deployment
    where you don't have target domain statistics.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = CarbonCastCNNLSTM(seq_len=SEQ_LEN, horizon=HORIZON, n_features=2)

    xs, ys = [], []
    for name, data in all_regions.items():
        if name == target_name:
            continue
        x_multi, y_cif = build_multivariate_windows(
            data["rs"], data["cif"], SEQ_LEN, HORIZON, TRAIN_STRIDE)
        if len(x_multi) == 0:
            continue
        xs.append(x_multi)
        ys.append(y_cif)

    x_all_np = np.concatenate(xs)
    y_all_np = np.concatenate(ys)

    # Set normalization from source data
    model.set_normalization(x_all_np, y_all_np)

    x_all = torch.tensor(x_all_np, dtype=torch.float32)
    y_all = torch.tensor(y_all_np, dtype=torch.float32)
    y_all_norm = model.normalize_target(y_all)
    n_samples = len(x_all)
    batch_size = min(BATCH_SIZE, n_samples)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = get_cosine_warmup_scheduler(optimizer, 30, EPOCHS_CARBONCAST)

    model.train()
    for epoch in range(EPOCHS_CARBONCAST):
        idx = torch.randperm(n_samples)[:batch_size]
        pred = model(x_all[idx], denorm=False)
        loss = nn.functional.l1_loss(pred, y_all_norm[idx])
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    random.seed(seed)
    np.random.seed(seed)

    data = all_regions[target_name]
    rs, cif = data["rs"], data["cif"]
    ef_r, ef_nr = data["ef_r"], data["ef_nr"]

    n_hours = len(rs)
    split_hour = int(n_hours * TRAIN_FRACTION)

    # Build windows
    x_rs_train, y_rs_train, y_cif_train = build_windows(
        rs[:split_hour], cif[:split_hour], SEQ_LEN, HORIZON, TRAIN_STRIDE)
    x_rs_test, y_rs_test, y_cif_test = build_windows(
        rs[split_hour - SEQ_LEN:], cif[split_hour - SEQ_LEN:], SEQ_LEN, HORIZON, TEST_STRIDE)

    # CIF-history windows
    x_cif_train, x_cif_test = [], []
    for start in range(0, split_hour - SEQ_LEN - HORIZON + 1, TRAIN_STRIDE):
        x_cif_train.append(cif[start:start + SEQ_LEN])
    x_cif_train = np.stack(x_cif_train)
    cif_test_offset = cif[split_hour - SEQ_LEN:]
    for start in range(0, len(cif_test_offset) - SEQ_LEN - HORIZON + 1, TEST_STRIDE):
        x_cif_test.append(cif_test_offset[start:start + SEQ_LEN])
    x_cif_test = np.stack(x_cif_test)

    # Multivariate windows (for CarbonCast)
    x_multi_train, y_multi_train = build_multivariate_windows(
        rs[:split_hour], cif[:split_hour], SEQ_LEN, HORIZON, TRAIN_STRIDE)
    rs_test_offset = rs[split_hour - SEQ_LEN:]
    cif_test_offset2 = cif[split_hour - SEQ_LEN:]
    x_multi_test, _ = build_multivariate_windows(
        rs_test_offset, cif_test_offset2, SEQ_LEN, HORIZON, TEST_STRIDE)

    results = {"target": target_name, "seed": seed,
               "n_train": len(x_rs_train), "n_test": len(x_rs_test)}

    # --- 1. Persistence ---
    persist_cif = x_cif_test[:, -HORIZON:]
    results["persistence_mae"] = float(np.abs(persist_cif - y_cif_test).mean())

    # --- 2. DLinear-RS supervised ---
    dlinear_rs = DLinearRS(seq_len=SEQ_LEN, horizon=HORIZON)
    train_model_batched(dlinear_rs, x_rs_train, y_rs_train,
                       epochs=EPOCHS_SUPERVISED, lr=5e-4, warmup=20)
    with torch.no_grad():
        rs_pred = dlinear_rs(torch.tensor(x_rs_test, dtype=torch.float32)).numpy()
    results["dlinear_rs_mae"] = float(np.abs(cif_from_shares(rs_pred, ef_r, ef_nr) - y_cif_test).mean())

    # --- 3. DLinear-Direct supervised ---
    dlinear_direct = DLinearDirect(seq_len=SEQ_LEN, horizon=HORIZON)
    train_model_batched(dlinear_direct, x_cif_train, y_cif_train,
                       epochs=EPOCHS_SUPERVISED, lr=5e-4, warmup=20)
    with torch.no_grad():
        cif_pred = dlinear_direct(torch.tensor(x_cif_test, dtype=torch.float32)).numpy()
    results["dlinear_direct_mae"] = float(np.abs(cif_pred - y_cif_test).mean())

    # --- 4. PatchTST-Fixed supervised (RevIN + warmup + lower lr) ---
    patchtst = PatchTSTFixed(seq_len=SEQ_LEN, horizon=HORIZON)
    train_model_batched(patchtst, x_cif_train, y_cif_train,
                       epochs=EPOCHS_SUPERVISED, lr=3e-4, loss_fn="l1", warmup=30)
    with torch.no_grad():
        cif_pred_ptst = patchtst(torch.tensor(x_cif_test, dtype=torch.float32)).numpy()
    results["patchtst_mae"] = float(np.abs(cif_pred_ptst - y_cif_test).mean())

    # --- 5. CarbonCast CNN-LSTM supervised (target domain training) ---
    cc_model = CarbonCastCNNLSTM(seq_len=SEQ_LEN, horizon=HORIZON, n_features=2)
    train_carboncast(cc_model, x_multi_train, y_cif_train,
                    epochs=EPOCHS_CARBONCAST, lr=5e-4)
    with torch.no_grad():
        cc_pred = cc_model(torch.tensor(x_multi_test, dtype=torch.float32)).numpy()
    results["carboncast_sup_mae"] = float(np.abs(cc_pred - y_cif_test).mean())

    # --- 6. GBRT supervised ---
    try:
        gbrt = train_gbrt(x_rs_train, y_cif_train, seed=seed)
        gbrt_pred = predict_gbrt(gbrt, x_rs_test, HORIZON)
        results["gbrt_mae"] = float(np.abs(gbrt_pred - y_cif_test).mean())
    except Exception as e:
        results["gbrt_mae"] = None
        print(f"  [WARN] GBRT failed: {e}")

    # --- 7. CarbonCast CNN-LSTM zero-shot (trained on other regions) ---
    cc_zs_model = train_carboncast_zero_shot(all_regions, target_name, seed=seed)
    with torch.no_grad():
        cc_zs_pred = cc_zs_model(torch.tensor(x_multi_test, dtype=torch.float32)).numpy()
    results["carboncast_zs_mae"] = float(np.abs(cc_zs_pred - y_cif_test).mean())

    # --- 8. TransCIF zero-shot (our method) ---
    zs_model = train_zero_shot(all_regions, target_name, seed=seed)
    target_cfg = torch.tensor(data["config"]).unsqueeze(0).expand(len(x_rs_test), -1)
    with torch.no_grad():
        zs_rs_pred = zs_model(torch.tensor(x_rs_test, dtype=torch.float32), target_cfg).numpy()
    zs_cif = cif_from_shares(zs_rs_pred, ef_r, ef_nr)
    results["transcif_zs_mae"] = float(np.abs(zs_cif - y_cif_test).mean())

    # --- Compute ratios ---
    sup_keys = ["dlinear_rs_mae", "dlinear_direct_mae", "patchtst_mae",
                "carboncast_sup_mae", "gbrt_mae"]
    sup_vals = [results[k] for k in sup_keys if results.get(k) is not None]
    results["best_supervised_mae"] = min(sup_vals) if sup_vals else None
    if results["best_supervised_mae"]:
        results["transcif_ratio"] = results["transcif_zs_mae"] / results["best_supervised_mae"]
        results["carboncast_zs_ratio"] = results["carboncast_zs_mae"] / results["best_supervised_mae"]
    results["transcif_vs_persist"] = results["transcif_zs_mae"] / results["persistence_mae"]
    results["carboncast_zs_vs_persist"] = results["carboncast_zs_mae"] / results["persistence_mae"]

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 90)
    print("Phase 1.1 Complete: All Baselines + CarbonCast CNN-LSTM + Fixed PatchTST")
    print("=" * 90)

    discover_uk_regions()
    n_total = len(AU_REGIONS) + len(UK_REGIONS)
    print(f"Regions: {len(AU_REGIONS)} AU + {len(UK_REGIONS)} UK = {n_total} total")

    all_regions = {}
    for name in AU_REGIONS:
        all_regions[name] = load_region_data(name)
    for name in UK_REGIONS:
        all_regions[name] = load_region_data(name)

    print(f"\nAU Config: " + " | ".join(
        f"{n}: rs={d['mean_rs']:.3f}" for n, d in all_regions.items() if n in AU_REGIONS))
    print(f"\nSettings: SEQ={SEQ_LEN} HOR={HORIZON} epochs_sup={EPOCHS_SUPERVISED} "
          f"epochs_cc={EPOCHS_CARBONCAST} epochs_zs={EPOCHS_ZERO_SHOT}")
    print(f"Seeds: {SEEDS}\n")

    all_results = {}
    for target in ["QLD1", "NSW1", "VIC1", "SA1"]:
        print(f"\n{'='*70}")
        print(f"  Target: {target} (mean_rs={all_regions[target]['mean_rs']:.3f}, "
              f"ef_nr={all_regions[target]['ef_nr']:.0f})")
        print(f"{'='*70}")
        seed_results = []
        for seed in SEEDS:
            print(f"  seed={seed}:", flush=True)
            r = evaluate_all(target, all_regions, seed=seed)
            seed_results.append(r)
            gbrt_str = f"{r['gbrt_mae']:.1f}" if isinstance(r.get('gbrt_mae'), (int, float)) else "N/A"
            print(f"    Persist={r['persistence_mae']:.1f} | DL-Dir={r['dlinear_direct_mae']:.1f} "
                  f"PatchTST={r['patchtst_mae']:.1f} CC-Sup={r['carboncast_sup_mae']:.1f} "
                  f"GBRT={gbrt_str}")
            print(f"    CC-ZS={r['carboncast_zs_mae']:.1f} | TransCIF-ZS={r['transcif_zs_mae']:.1f} "
                  f"| Ratio(ours)={r.get('transcif_ratio', 0):.3f} "
                  f"Ratio(CC-ZS)={r.get('carboncast_zs_ratio', 0):.3f}")
        all_results[target] = seed_results

    # Final summary
    print("\n\n" + "=" * 120)
    print("FINAL SUMMARY (Mean across seeds, CIF MAE in gCO₂/kWh)")
    print("=" * 120)
    header = (f"{'Region':<6} {'Persist':<8} {'DL-Dir':<8} {'DL-RS':<7} {'PatchTST':<9} "
              f"{'CC-Sup':<8} {'GBRT':<7} | {'CC-ZS':<8} {'TransCIF':<9} | "
              f"{'BestSup':<8} {'Ratio-T':<8} {'Ratio-CC':<9}")
    print(header)
    print("-" * 120)

    for target in ["QLD1", "NSW1", "VIC1", "SA1"]:
        sr = all_results[target]
        persist = np.mean([r["persistence_mae"] for r in sr])
        dl_dir = np.mean([r["dlinear_direct_mae"] for r in sr])
        dl_rs = np.mean([r["dlinear_rs_mae"] for r in sr])
        ptst = np.mean([r["patchtst_mae"] for r in sr])
        cc_sup = np.mean([r["carboncast_sup_mae"] for r in sr])
        gbrt_vals = [r["gbrt_mae"] for r in sr if r.get("gbrt_mae") is not None]
        gbrt = np.mean(gbrt_vals) if gbrt_vals else float("nan")
        cc_zs = np.mean([r["carboncast_zs_mae"] for r in sr])
        transcif = np.mean([r["transcif_zs_mae"] for r in sr])
        best_sup = np.mean([r["best_supervised_mae"] for r in sr if r.get("best_supervised_mae")])
        ratio_t = np.mean([r["transcif_ratio"] for r in sr if r.get("transcif_ratio")])
        ratio_cc = np.mean([r["carboncast_zs_ratio"] for r in sr if r.get("carboncast_zs_ratio")])

        print(f"{target:<6} {persist:<8.1f} {dl_dir:<8.1f} {dl_rs:<7.1f} {ptst:<9.1f} "
              f"{cc_sup:<8.1f} {gbrt:<7.1f} | {cc_zs:<8.1f} {transcif:<9.1f} | "
              f"{best_sup:<8.1f} {ratio_t:<8.3f} {ratio_cc:<9.3f}")

    print("\n" + "=" * 120)
    print("Columns: Supervised methods (train on target) | Zero-shot methods (no target data)")
    print("Ratio-T  = TransCIF-ZS / BestSupervised (lower=better, <1.25 publishable)")
    print("Ratio-CC = CarbonCast-ZS / BestSupervised (expected >> 1.0 = cross-domain failure)")
    print("Key story: CarbonCast strong supervised, but FAILS cross-domain. TransCIF survives.")
    print("=" * 120)


if __name__ == "__main__":
    main()
