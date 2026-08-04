"""Retrieval-Augmented Generation for Time-Series Forecasting (RAG-TS).

Top-priority research direction from RESEARCH_DIRECTIONS.md.

Core idea: instead of training one model on all source regions and then applying
it zero-shot to a target, we build a memory bank of source-region patterns and
retrieve the most relevant windows at inference time to condition the prediction.

Components:
    RagMemoryBank  – two-stage retrieval (config distance → L2 pattern similarity)
    RagDLinear     – DLinear backbone with cross-attention over retrieved windows
    train_rag_zero_shot – LORO training loop with retrieval augmentation

Usage:
    from transcif_rag import RagMemoryBank, RagDLinear, train_rag_zero_shot
"""

import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transcif_model import RevIN
from transcif_pipeline import (
    SEQ_LEN, HORIZON, TRAIN_STRIDE, EPOCHS_ZERO_SHOT,
    build_windows, get_cosine_warmup_scheduler,
)


# ============================================================================
# Memory Bank
# ============================================================================

class RagMemoryBank:
    """Two-stage memory bank for source-region patterns.

    Stage 1 (coarse): filter by config distance → keep top candidate regions
    Stage 2 (fine):   rank by L2 distance on input windows → keep top k
    """

    def __init__(self, all_regions, target_name, seq_len=SEQ_LEN, horizon=HORIZON,
                 stride=TRAIN_STRIDE, n_coarse=3, k_retrieve=5):
        self.seq_len = seq_len
        self.horizon = horizon
        self.k = k_retrieve
        self.target_config = all_regions[target_name]["config"]

        # Build region-level index: region_name → config vector
        self.region_configs = {}
        for name, data in all_regions.items():
            if name == target_name:
                continue
            self.region_configs[name] = data["config"]

        # Pre-build window bank grouped by region
        self._bank = defaultdict(list)
        for name, data in all_regions.items():
            if name == target_name:
                continue
            x_win, y_win, _ = build_windows(data["rs"], data["cif"],
                                            seq_len, horizon, stride)
            if len(x_win) < 2:
                continue
            for i in range(len(x_win)):
                self._bank[name].append({
                    "x": x_win[i].astype(np.float32),
                    "y": y_win[i].astype(np.float32),
                })

        # Pre-compute config ranking
        self._config_rank = sorted(
            self.region_configs.keys(),
            key=lambda n: np.linalg.norm(self.region_configs[n] - self.target_config),
        )
        self._n_coarse = min(n_coarse, len(self._config_rank))

    def query(self, x_target, rs_target=None):
        """Retrieve top-k source windows for a target input.

        Args:
            x_target   : (seq_len,) numpy array
            rs_target  : optional mean_rs of target (used for config look-up)

        Returns:
            x_retrieved : (k, seq_len)  top-k input windows
            y_retrieved : (k, horizon)  corresponding output windows
            scores      : (k,)          similarity scores (higher = more similar)
        """
        # Stage 1: pick top regions by config distance
        candidate_regions = self._config_rank[:self._n_coarse]

        # Stage 2: rank individual windows by L2 distance
        candidates = []
        for region in candidate_regions:
            for entry in self._bank[region]:
                score = -np.linalg.norm(x_target - entry["x"])  # negative L2
                candidates.append((score, entry["x"], entry["y"]))

        if not candidates:
            return (np.zeros((0, self.seq_len), dtype=np.float32),
                    np.zeros((0, self.horizon), dtype=np.float32),
                    np.array([], dtype=np.float32))

        candidates.sort(key=lambda c: c[0], reverse=True)  # higher score first
        k = min(self.k, len(candidates))
        top = candidates[:k]

        x_ret = np.stack([t[1] for t in top])
        y_ret = np.stack([t[2] for t in top])
        scores = np.array([t[0] for t in top], dtype=np.float32)

        # Normalize scores to [0, 1] for confidence gating
        if len(scores) > 1 and scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        else:
            scores = np.ones_like(scores) * 0.5

        return x_ret, y_ret, scores

    def query_batch(self, x_batch, rs_batch=None):
        """Batch retrieval for training.

        Args:
            x_batch  : (B, seq_len) tensor or numpy
            rs_batch : (B,) optional mean_rs values

        Returns:
            x_ret_batch : (B, k, seq_len)
            y_ret_batch : (B, k, horizon)
            sim_scores  : (B, k)
        """
        if isinstance(x_batch, torch.Tensor):
            x_batch = x_batch.numpy()
        B = x_batch.shape[0]
        xr, yr, ss = [], [], []
        for i in range(B):
            xi, yi, si = self.query(x_batch[i])
            if len(xi) < self.k:
                # Pad with zeros if not enough results
                pad_x = np.zeros((self.k - len(xi), self.seq_len), dtype=np.float32)
                pad_y = np.zeros((self.k - len(yi), self.horizon), dtype=np.float32)
                pad_s = np.zeros(self.k - len(si), dtype=np.float32)
                xi = np.concatenate([xi, pad_x], axis=0)
                yi = np.concatenate([yi, pad_y], axis=0)
                si = np.concatenate([si, pad_s], axis=0)
            xr.append(xi[:self.k])
            yr.append(yi[:self.k])
            ss.append(si[:self.k])
        return np.stack(xr), np.stack(yr), np.stack(ss)


# ============================================================================
# RagDLinear – Retrieval-Augmented DLinear
# ============================================================================

class RagDLinear(nn.Module):
    """DLinear augmented with retrieved source-region patterns.

    Architecture:
        Target input → DLinear (trend/seasonal + config bias) → ŝ_target
        Retrieved windows → lightweight encoder → ŝ_retrieval
        Fusion gate (confidence-aware) → ŝ_final
        Persistence gate → final output ∈ [0, 1]

    The fusion gate is conditioned on:
        - config similarity between target and retrieved regions
        - input pattern similarity (retrieval scores)
        - recent volatility of the target
    """

    def __init__(self, seq_len=336, horizon=24, config_dim=2, k_retrieve=5):
        super().__init__()
        self.horizon = horizon
        self.k_retrieve = k_retrieve

        # ---- DLinear backbone (shares structure with AdaptivePersistDLinear) ----
        self.avg_pool = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.linear_trend = nn.Linear(seq_len, horizon)
        self.linear_seasonal = nn.Linear(seq_len, horizon)
        self.config_bias = nn.Sequential(
            nn.Linear(config_dim, 16), nn.ReLU(), nn.Linear(16, horizon))

        # ---- Retrieval encoder ----
        # Encode each retrieved window into a horizon-length prediction
        self.retrieve_encoder = nn.Sequential(
            nn.Linear(seq_len, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, horizon),
        )
        # Encode retrieval similarity scores → confidence features
        self.score_encoder = nn.Sequential(
            nn.Linear(k_retrieve, 16), nn.ReLU(), nn.Linear(16, 8))

        # ---- Fusion gate ----
        # Concept: gate ∈ [0, 1] controls how much to trust retrieval vs model
        fusion_dim = config_dim + 2 + 8  # config + recent_mean/std + score features
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_dim, 16), nn.ReLU(), nn.Linear(16, horizon),
            nn.Sigmoid())

        # ---- Persistence gate (same as AdaptivePersistDLinear) ----
        self.persist_gate = nn.Sequential(
            nn.Linear(config_dim + 2, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, config, x_retrieved=None, sim_scores=None):
        """Forward pass.

        Args:
            x           : (B, seq_len)       target input
            config      : (B, config_dim)    target config
            x_retrieved : (B, k, seq_len)    retrieved windows (optional)
            sim_scores  : (B, k)             retrieval similarity ∈ [0,1]
        """
        # ---- DLinear prediction ----
        x3 = x.unsqueeze(1)
        trend = self.avg_pool(x3).squeeze(1)
        seasonal = x - trend
        s_target = torch.sigmoid(
            self.linear_trend(trend) +
            self.linear_seasonal(seasonal) +
            self.config_bias(config))          # (B, horizon)

        # ---- Retrieval-augmented prediction ----
        if x_retrieved is not None and x_retrieved.shape[1] > 0:
            B, k, L = x_retrieved.shape
            # Average over retrieved windows → shared encoder is efficient
            x_ret_flat = x_retrieved.reshape(B * k, L)
            s_ret_flat = torch.sigmoid(self.retrieve_encoder(x_ret_flat))
            s_ret = s_ret_flat.reshape(B, k, -1)         # (B, k, horizon)
            ret_score = s_ret.mean(dim=1)                 # (B, horizon)  simple avg

            # ---- Fusion gate ----
            score_feat = self.score_encoder(sim_scores) if sim_scores is not None else \
                         torch.zeros(B, 8, device=x.device)
            recent_mean = x[:, -48:].mean(dim=1, keepdim=True)  # (B, 1)
            recent_std = x[:, -48:].std(dim=1, keepdim=True)    # (B, 1)
            gate_input = torch.cat([config, recent_mean, recent_std, score_feat], dim=1)
            gate = self.fusion_gate(gate_input)          # (B, horizon)

            s_dlinear = gate * s_target + (1.0 - gate) * ret_score
        else:
            s_dlinear = s_target

        # ---- Persistence gate ----
        persist = x[:, -self.horizon:]
        recent_mean2 = x[:, -48:].mean(dim=1, keepdim=True)
        recent_std2 = x[:, -48:].std(dim=1, keepdim=True)
        pgate_input = torch.cat([config, recent_mean2, recent_std2], dim=1)
        pgate = torch.sigmoid(self.persist_gate(pgate_input))

        return pgate * persist + (1.0 - pgate) * s_dlinear


# ============================================================================
# Training
# ============================================================================

def train_rag_zero_shot(all_regions, target_name, seed=42,
                        use_weighted=True, k_retrieve=5, n_coarse=3,
                        epochs=EPOCHS_ZERO_SHOT, lr=1e-3, device=None):
    """Train the RAG-DLinear model on all source regions for one LORO target.

    Key difference from standard zero-shot training: during training, the memory
    bank is built from source regions and used to retrieve patterns for each
    training batch. The model learns to leverage retrieval during training so
    it knows how to use retrieved patterns at inference time.

    Args:
        all_regions : dict  {name: {"rs":..., "cif":..., "config":...}}
        target_name : str   region to leave out
        use_weighted: bool  config-distance source weighting
        k_retrieve  : int   number of windows to retrieve per query
        n_coarse    : int   number of candidate regions for coarse filtering
        epochs, lr  : training hyperparams
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Build memory bank from source regions
    bank = RagMemoryBank(all_regions, target_name,
                         k_retrieve=k_retrieve, n_coarse=n_coarse)

    model = RagDLinear(seq_len=SEQ_LEN, horizon=HORIZON,
                       k_retrieve=k_retrieve)
    if device:
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_cosine_warmup_scheduler(optimizer, max(1, epochs // 10), epochs)

    # Collect training windows from all source regions
    xs, ys, cfgs, weights = [], [], [], []
    target_data = all_regions[target_name]
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
            dist = abs(data["mean_rs"] - target_data["mean_rs"])
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
    batch_size = min(256, n_samples)  # smaller batch because retrieval is expensive

    model.train()
    for epoch in range(epochs):
        idx = torch.randperm(n_samples)[:batch_size]
        x_batch = x_all[idx]
        y_batch = y_all[idx]
        c_batch = c_all[idx]
        w_batch = w_all[idx]

        # Retrieve patterns for this batch
        x_ret, y_ret, sim = bank.query_batch(x_batch.cpu().numpy())
        x_ret_t = torch.tensor(x_ret).to(x_batch.device) if len(x_ret) > 0 else None
        sim_t = torch.tensor(sim).to(x_batch.device) if len(sim) > 0 else None

        pred = model(x_batch, c_batch, x_ret_t, sim_t)
        loss = (w_batch.unsqueeze(1) * torch.abs(pred - y_batch)).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    return model, bank


# ============================================================================
# Inference (for evaluate_target in run_unified_eval)
# ============================================================================

def predict_rag_zs(model, bank, x_test, config_target, ef_r, ef_nr):
    """Zero-shot prediction with retrieval augmentation.

    Args:
        model          : trained RagDLinear
        bank           : RagMemoryBank (built from source regions)
        x_test         : (N, seq_len)  test input windows
        config_target  : (config_dim,) target config vector
        ef_r, ef_nr    : emission factors

    Returns:
        cif_pred : (N, horizon)  predicted CIF
    """
    model.eval()
    N = len(x_test)
    x_ret_all, sim_all = [], []
    for i in range(N):
        xr, _, si = bank.query(x_test[i])
        x_ret_all.append(xr)
        sim_all.append(np.pad(si, (0, bank.k - len(si))) if len(si) > 0
                       else np.zeros(bank.k, dtype=np.float32))

    x_ret_t = torch.tensor(np.stack(x_ret_all), dtype=torch.float32)
    sim_t = torch.tensor(np.array(sim_all), dtype=torch.float32)
    x_t = torch.tensor(x_test, dtype=torch.float32)
    cfg_t = torch.tensor(config_target).unsqueeze(0).expand(N, -1)

    with torch.no_grad():
        share_pred = model(x_t, cfg_t, x_ret_t, sim_t).numpy()

    # Physics layer
    return share_pred * ef_r + (1.0 - share_pred) * ef_nr
