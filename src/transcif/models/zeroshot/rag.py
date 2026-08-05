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
        """Return top-k (context, target) by L2 distance to query."""
        if len(self.contexts) == 0:
            return [], []
        q = np.asarray(query, np.float32).ravel()
        d = np.linalg.norm(self.X - q, axis=1)
        idx = np.argsort(d)[:k]
        return [self.X[i] for i in idx], [self.Y[i] for i in idx]


# ---------------------------------------------------------------------------
# RAG-augmented DLinear
# ---------------------------------------------------------------------------

class RagDLinear(nn.Module):
    """AdaptivePersistDLinear with an extra RAG-retrieved bias channel."""

    def __init__(self, seq_len=SEQ_LEN, horizon=HORIZON, config_dim=2):
        super().__init__()
        self.horizon = horizon
        self.avg_pool = nn.AvgPool1d(kernel_size=25, stride=1, padding=12)
        self.linear_trend = nn.Linear(seq_len, horizon)
        self.linear_seasonal = nn.Linear(seq_len, horizon)
        self.config_bias = nn.Sequential(
            nn.Linear(config_dim, 16), nn.ReLU(), nn.Linear(16, horizon))
        self.rag_bias = nn.Linear(horizon, horizon)
        self.gate_net = nn.Sequential(
            nn.Linear(config_dim + 2, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, config, rag_target=None):
        x3 = x.unsqueeze(1)
        trend = self.avg_pool(x3).squeeze(1)
        seasonal = x - trend
        dlinear_out = torch.sigmoid(
            self.linear_trend(trend) +
            self.linear_seasonal(seasonal) +
            self.config_bias(config))
        if rag_target is not None:
            dlinear_out = dlinear_out + self.rag_bias(rag_target)
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
                        epochs=200, lr=1e-3, k=5, device=None):
    """Train RAG-augmented model + build memory bank from source windows."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = RagDLinear(seq_len=SEQ_LEN, horizon=HORIZON)
    if device:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    xs, ys, cfgs = [], [], []
    bank = RagMemoryBank()
    for name, data in all_regions.items():
        if name == target_name:
            continue
        arr = data["rs"].astype(np.float32)
        n = len(arr) - SEQ_LEN - HORIZON
        for t in range(0, n, 6):
            ctx = arr[t:t + SEQ_LEN]
            # Store the renewable SHARE as the retrieval target (not CIF) so the
            # RAG bias stays in the same units as the model's share prediction.
            tgt = arr[t + SEQ_LEN:t + SEQ_LEN + HORIZON].astype(np.float32)
            bank.add(ctx, tgt)
            xs.append(ctx)
            ys.append(arr[t + SEQ_LEN:t + SEQ_LEN + HORIZON])
            cfgs.append(data["config"])
    bank.build()

    x_all = torch.tensor(np.stack(xs), dtype=torch.float32)
    y_all = torch.tensor(np.stack(ys), dtype=torch.float32)
    c_all = torch.tensor(np.stack(cfgs), dtype=torch.float32)
    n_samples = len(x_all)
    batch_size = min(256, n_samples)

    model.train()
    for epoch in range(epochs):
        idx = torch.randperm(n_samples)[:batch_size]
        x_b, y_b, c_b = x_all[idx], y_all[idx], c_all[idx]
        rag_b = []
        for i in range(x_b.shape[0]):
            _, tgts = bank.retrieve(x_b[i].numpy(), k=k)
            rag_b.append(np.mean(tgts, axis=0) if tgts else np.zeros(HORIZON, np.float32))
        rag_t = torch.tensor(np.stack(rag_b), dtype=torch.float32)
        if device:
            x_b, y_b, c_b, rag_t = x_b.to(device), y_b.to(device), c_b.to(device), rag_t.to(device)
        pred = model(x_b, c_b, rag_target=rag_t)
        loss = torch.abs(pred - y_b).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
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
    rag_list = []
    for i in range(len(x_rs)):
        _, tgts = bank.retrieve(x_rs[i], k=k)
        rag_list.append(np.mean(tgts, axis=0) if tgts else np.zeros(HORIZON, np.float32))
    rag_t = torch.tensor(np.stack(rag_list), dtype=torch.float32).to(dev)
    with torch.no_grad():
        s_pred = model(x_t, c_t, rag_target=rag_t).cpu().numpy()
    return cif_from_shares(s_pred, ef_r, ef_nr)
