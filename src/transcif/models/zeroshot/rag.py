"""RAG-ZS: Retrieval-Augmented Generation for Zero-Shot CIF Forecasting.

This module extends the base TransCIF zero-shot pipeline with a KNN-based
retrieval memory bank: at inference, historical (context -> target CIF) pairs
from *source* regions are retrieved by nearest-neighbour similarity of the
input window, and the target CIF is reconstructed by applying the target's
physics emission factors to the retrieved share trajectories.
"""

import random

import numpy as np
import torch
import torch.nn as nn

from transcif.config import SEQ_LEN, HORIZON
from transcif.models.base import AdaptivePersistDLinear
from transcif.physics.decompose import cif_from_shares
from transcif.training.schedulers import get_cosine_warmup_scheduler


# ---------------------------------------------------------------------------
# Memory bank
# ---------------------------------------------------------------------------

class RagMemoryBank:
    """Stores (context_window, target_share) pairs from source regions.

    NOTE: the retrieval target is the *renewable share* (in [0, 1]), NOT the
    CIF.  The RAG bias is added on top of the model's share prediction, so the
    two must share the same units; converting to CIF only happens once at the
    very end via ``cif_from_shares``.
    """

    def __init__(self):
        self.contexts = []   # list of np.float32 arrays (SEQ_LEN,)
        self.targets = []    # list of np.float32 arrays (HORIZON,) — share

    def add(self, context, target):
        self.contexts.append(np.asarray(context, np.float32).ravel())
        self.targets.append(np.asarray(target, np.float32).ravel())

    def build(self):
        """Stack into arrays for fast NN lookup."""
        if not self.contexts:
            self.X = np.zeros((0, SEQ_LEN), np.float32)
            self.Y = np.zeros((0, HORIZON), np.float32)
            return
        self.X = np.stack(self.contexts)
        self.Y = np.stack(self.targets)

    def retrieve(self, query, k=5):
        """Return top-k (context, target, dist) by L2 distance to query."""
        if len(self.contexts) == 0:
            return [], [], np.zeros(0, np.float32)
        q = np.asarray(query, np.float32).ravel()
        d = np.linalg.norm(self.X - q, axis=1)
        idx = np.argpartition(d, min(k, len(d) - 1))[:k]
        idx = idx[np.argsort(d[idx])]
        return [self.X[i] for i in idx], [self.Y[i] for i in idx], d[idx]

    def retrieve_batch(self, queries, k=5):
        """Vectorised top-k retrieval for a batch of queries.

        Returns (targets_mean [B,H], dists_mean [B]) — the mean of the top-k
        retrieved share trajectories and the mean retrieval distance, which is
        all the training/predict loops actually use.  Much faster than calling
        ``retrieve`` per sample (one matmul vs B loops over the full bank).
        """
        n = len(self.contexts)
        if n == 0:
            B = len(queries)
            return (np.zeros((B, HORIZON), np.float32),
                    np.zeros(B, np.float32))
        Q = np.asarray(queries, np.float32)  # (B, L)
        # squared L2 via (a-b)^2 = a^2 - 2ab + b^2
        q2 = (Q ** 2).sum(axis=1, keepdims=True)        # (B,1)
        x2 = (self.X ** 2).sum(axis=1, keepdims=True).T  # (1,N)
        d2 = q2 - 2 * Q @ self.X.T + x2                 # (B,N)
        d2 = np.maximum(d2, 0)
        kk = min(k, n)
        idx = np.argpartition(d2, kk - 1, axis=1)[:, :kk]  # (B,kk)
        row = np.arange(Q.shape[0])[:, None]
        dists_sel = np.sqrt(d2[row, idx])                   # (B,kk)
        tgt_sel = self.Y[idx]                               # (B,kk,H)
        return tgt_sel.mean(axis=1), dists_sel.mean(axis=1)


# ---------------------------------------------------------------------------
# RAG-augmented DLinear
# ---------------------------------------------------------------------------

class RagDLinear(nn.Module):
    """AdaptivePersistDLinear with an extra RAG-retrieved bias channel.

    The retrieved share trajectory is fused with the DLinear prediction through
    a dedicated sigmoid-gated branch (keeping the output in [0,1], unlike an
    additive bias).  A second gate, conditioned on the retrieval distance,
    controls how much the RAG branch contributes — when retrieval is a poor
    match the gate closes and the model falls back to the plain DLinear path.
    """

    def __init__(self, seq_len=SEQ_LEN, horizon=HORIZON, config_dim=2):
        super().__init__()
        self.horizon = horizon
        self.avg_pool = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.linear_trend = nn.Linear(seq_len, horizon)
        self.linear_seasonal = nn.Linear(seq_len, horizon)
        self.config_bias = nn.Sequential(
            nn.Linear(config_dim, 16), nn.ReLU(), nn.Linear(16, horizon))
        # RAG branch: project retrieved share -> sigmoid so it stays in [0,1]
        self.rag_proj = nn.Sequential(
            nn.Linear(horizon, horizon), nn.ReLU(), nn.Linear(horizon, horizon))
        # RAG gate: conditioned on config + recent stats + retrieval distance
        # (a scalar: mean L2 distance of the retrieved neighbours).
        self.rag_gate = nn.Sequential(
            nn.Linear(config_dim + 3, 16), nn.ReLU(), nn.Linear(16, 1))
        self.gate_net = nn.Sequential(
            nn.Linear(config_dim + 2, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, config, rag_target=None, rag_dist=None):
        x3 = x.unsqueeze(1)
        trend = self.avg_pool(x3).squeeze(1)
        seasonal = x - trend
        dlinear_out = torch.sigmoid(
            self.linear_trend(trend) +
            self.linear_seasonal(seasonal) +
            self.config_bias(config))
        # RAG branch: sigmoid-projected retrieved share, gated by retrieval quality
        if rag_target is not None:
            rag_out = torch.sigmoid(self.rag_proj(rag_target))
            recent_mean = x[:, -48:].mean(dim=1, keepdim=True)
            recent_std = x[:, -48:].std(dim=1, keepdim=True)
            d_feat = rag_dist if rag_dist is not None else \
                torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
            rg_input = torch.cat([config, recent_mean, recent_std, d_feat], dim=1)
            rag_gate = torch.sigmoid(self.rag_gate(rg_input))
            dlinear_out = rag_gate * rag_out + (1 - rag_gate) * dlinear_out
        persist = x[:, -self.horizon:]
        recent_mean = x[:, -48:].mean(dim=1, keepdim=True)
        recent_std = x[:, -48:].std(dim=1, keepdim=True)
        gate_input = torch.cat([config, recent_mean, recent_std], dim=1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        return gate * persist + (1 - gate) * dlinear_out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_rag_zero_shot(all_regions, target_name, seed=42,
                        epochs=200, lr=1e-3, k=5, device=None, pbar=None):
    """Train RAG-augmented model + build memory bank from source windows."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = RagDLinear(seq_len=SEQ_LEN, horizon=HORIZON)
    if device:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_cosine_warmup_scheduler(optimizer, max(1, epochs // 10), epochs)

    target_mean_rs = all_regions[target_name]["mean_rs"]
    xs, ys, cfgs, ws = [], [], [], []
    bank = RagMemoryBank()
    for name, data in all_regions.items():
        if name == target_name:
            continue
        arr = data["rs"].astype(np.float32)
        n = len(arr) - SEQ_LEN - HORIZON
        # Config-distance source weighting (matches the base TransCIF-ZS
        # sampler): windows from regions close to the target in mean_rs are
        # sampled more often, which is what makes zero-shot transfer work.
        dist = abs(data["mean_rs"] - target_mean_rs)
        w = 1.0 / (dist + 0.05)
        for t in range(0, n, 6):
            ctx = arr[t:t + SEQ_LEN]
            # Store the renewable SHARE as the retrieval target (not CIF) so the
            # RAG bias stays in the same units as the model's share prediction.
            tgt = arr[t + SEQ_LEN:t + SEQ_LEN + HORIZON].astype(np.float32)
            bank.add(ctx, tgt)
            xs.append(ctx)
            ys.append(arr[t + SEQ_LEN:t + SEQ_LEN + HORIZON])
            cfgs.append(data["config"])
            ws.append(w)
    bank.build()

    x_all = torch.tensor(np.stack(xs), dtype=torch.float32)
    y_all = torch.tensor(np.stack(ys), dtype=torch.float32)
    c_all = torch.tensor(np.stack(cfgs), dtype=torch.float32)
    w_all = torch.tensor(np.array(ws, dtype=np.float32))
    w_all = w_all / w_all.sum() * len(w_all)
    n_samples = len(x_all)
    batch_size = min(256, n_samples)

    model.train()
    for epoch in range(epochs):
        idx = torch.multinomial(w_all, batch_size, replacement=True)
        x_b, y_b, c_b = x_all[idx], y_all[idx], c_all[idx]
        rag_np, dist_np = bank.retrieve_batch(x_b.numpy(), k=k)
        rag_t = torch.tensor(rag_np, dtype=torch.float32)
        rag_d = torch.tensor(dist_np, dtype=torch.float32).unsqueeze(1)
        if device:
            x_b, y_b, c_b, rag_t = x_b.to(device), y_b.to(device), c_b.to(device), rag_t.to(device)
            rag_d = rag_d.to(device)
        pred = model(x_b, c_b, rag_target=rag_t, rag_dist=rag_d)
        loss = torch.abs(pred - y_b).mean()
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
    return model, bank


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_rag_zs(model, bank, x_rs, config, ef_r, ef_nr, k=5):
    """Zero-shot prediction with RAG retrieval + physics conversion."""
    model.eval()
    dev = next(model.parameters()).device
    x_t = torch.tensor(x_rs, dtype=torch.float32).to(dev)
    c_t = torch.tensor(config).unsqueeze(0).expand(len(x_rs), -1).to(dev)
    rag_np, dist_np = bank.retrieve_batch(x_rs, k=k)
    rag_t = torch.tensor(rag_np, dtype=torch.float32).to(dev)
    rag_d = torch.tensor(dist_np, dtype=torch.float32).unsqueeze(1).to(dev)
    with torch.no_grad():
        s_pred = model(x_t, c_t, rag_target=rag_t, rag_dist=rag_d).cpu().numpy()
    return cif_from_shares(s_pred, ef_r, ef_nr)
