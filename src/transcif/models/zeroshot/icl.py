"""IC-TSF: In-Context Time Series Forecaster for Zero-Shot CIF Prediction.

Core idea (from RESEARCH_DIRECTIONS.md §5):
    Borrow the In-Context Learning (ICL) paradigm from LLMs:
    - Construct context as (m example pairs, 1 query) sequence
    - Causal Transformer learns to predict query output by "analogy" with examples
    - Examples selected by config-distance + input similarity
    - Physics layer ensures CIF consistency

Key difference from ICTSP (ICLR 2025):
    ICTSP uses cross-attention across time dimension.
    IC-TSF uses causal Transformer in a GPT-style autoregressive ICL setup,
    where the model "sees" example input-output pairs and then predicts
    the query output in one forward pass.

Architecture:
    Context Window = [e1_in, e1_out, e2_in, e2_out, ..., em_in, em_out, q_in]
    Position encoding: role-aware (example_input, example_output, query_input)
    Causal Transformer → last H tokens → share prediction
    Physics Layer → CIF


Exports:
    ICTransformer        — causal Transformer for ICL
    build_context         — construct (examples + query) input tensor
    train_icl            — ICL training on source regions
    predict_icl_zs       — zero-shot inference via ICL
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transcif.data.windows import build_windows
from transcif.physics.decompose import cif_from_shares
from transcif.physics.bounds import config_weight
from transcif.training.schedulers import get_cosine_warmup_scheduler


# ---------------------------------------------------------------------------
# Role-aware position encoding
# ---------------------------------------------------------------------------

class RoleAwarePositionalEncoding(nn.Module):
    """Distinguishes example_input / example_output / query_input roles.

    Each token in the context has a role:
        0 = example_input
        1 = example_output
        2 = query_input
    """

    def __init__(self, d_model, max_len=2000):
        super().__init__()
        self.role_embed = nn.Embedding(3, d_model)
        # Standard sinusoidal PE
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, roles):
        """Add positional + role embedding.

        Args:
            x      : (B, T, d_model)
            roles  : (B, T) int tensor [0,1,2]
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len] + self.role_embed(roles)


# ---------------------------------------------------------------------------
# IC Transformer
# ---------------------------------------------------------------------------

class ICTransformer(nn.Module):
    """Causal Transformer for In-Context Time Series learning.

    Input:  (B, T, 2) — each token is [value, is_query_mask]
    Output: share prediction for the query positions

    T = horizon * (2*m + 1)
        m examples × 2 (input + output) + 1 query input
    """

    def __init__(self, horizon=24, n_examples=3, d_model=64, n_layers=3, n_heads=4):
        super().__init__()
        self.horizon = horizon
        self.n_examples = n_examples
        self.total_tokens = horizon * (2 * n_examples + 1)

        self.input_proj = nn.Linear(2, d_model)  # [value, is_query_mask]
        self.pos_encoder = RoleAwarePositionalEncoding(d_model, max_len=self.total_tokens + 100)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
            norm_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model), enable_nested_tensor=False)

        # Predict share values for query positions only
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, values, roles, causal_mask=None):
        """Forward pass.

        Args:
            values      : (B, T) scalar values (share or special value for query)
            roles       : (B, T) int role codes
            causal_mask : (T, T) attention mask (causal)

        Returns:
            pred : (B, horizon) share prediction for query positions
        """
        B, T = values.shape
        # Build is_query flag
        is_query = (roles == 2).float()  # (B, T)
        x = torch.stack([values, is_query], dim=-1)  # (B, T, 2)
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_encoder(x, roles)

        if causal_mask is None:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=values.device)

        h = self.transformer(x, mask=causal_mask)  # (B, T, d_model)

        # Select only query input positions (the last horizon tokens)
        query_mask = (roles == 2)  # (B, T)
        # Gather query-position features
        q_feats = h[query_mask].view(B, self.horizon, -1)  # (B, horizon, d_model)
        pred = torch.sigmoid(self.pred_head(q_feats).squeeze(-1))  # (B, horizon)
        return pred


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------

def build_context(target_window, example_windows, example_outputs,
                  horizon=24, fill_value=0.0):
    """Build ICL context tensor from examples + query.

    Layout (T = horizon * (2*m + 1)):
        Position  0:h-1      : example_1 input
        Position  h:2h-1     : example_1 output
        Position  2h:3h-1    : example_2 input
        Position  3h:4h-1    : example_2 output
        ...
        Position  2m*h:2m*h+h-1 : query input (last horizon tokens)

    Roles:
        0 = example_input
        1 = example_output
        2 = query_input

    Args:
        target_window   : (horizon,) — last H values of target input
        example_windows : list of (horizon,) — example input segments
        example_outputs : list of (horizon,) — example output segments
        horizon         : H
        fill_value      : value for query_output positions (masked)

    Returns:
        values : (B, T) context values
        roles  : (B, T) role codes
    """
    m = len(example_windows)
    T = horizon * (2 * m + 1)
    values = np.full((1, T), fill_value, dtype=np.float32)
    roles = np.full((1, T), 0, dtype=np.int64)

    pos = 0
    for i in range(m):
        # Example input
        values[0, pos:pos + horizon] = example_windows[i]
        roles[0, pos:pos + horizon] = 0  # example_input
        pos += horizon
        # Example output
        values[0, pos:pos + horizon] = example_outputs[i]
        roles[0, pos:pos + horizon] = 1  # example_output
        pos += horizon

    # Query input
    values[0, pos:pos + horizon] = target_window
    roles[0, pos:pos + horizon] = 2  # query_input

    return values, roles


def select_examples(all_regions, target_name, target_window, n_examples=3,
                     horizon=24, n_coarse=5):
    """Select best-matching example (input, output) pairs from source regions.

    Strategy:
        1. Coarse filtering: top-n_coarse regions by config distance
        2. Fine ranking: L2 similarity of input windows
        3. Pick top-n_examples

    Returns: (example_windows, example_outputs) lists
    """
    target_cfg = all_regions[target_name]["config"]
    # Config distance only over the comparable leading dims (min of the two
    # configs) so mixed-dim pools (2-D AU vs 12-D US/UK) don't broadcast-fail.
    scores = []

    for name, data in all_regions.items():
        if name == target_name:
            continue
        # Config distance
        n_cmp = min(len(data["config"]), len(target_cfg))
        config_dist = np.linalg.norm(data["config"][:n_cmp] - target_cfg[:n_cmp])
        # Find best-matching window in this region
        x_win, y_win, _ = build_windows(data["rs"], data["cif"])
        if len(x_win) == 0:
            continue
        # Input similarity: compare the FULL target window to the full source
        # window (not just the last horizon) — a longer context yields more
        # stable neighbour matching.  Source windows are SEQ_LEN wide; the
        # target window passed in is also SEQ_LEN wide.
        target_full = target_window.reshape(1, -1)
        src_full = x_win[:, :target_full.shape[1]] if x_win.shape[1] >= target_full.shape[1] \
            else x_win
        sims = -np.linalg.norm(src_full - target_full[:, :src_full.shape[1]], axis=1)
        best_idx = int(np.argmax(sims))
        score = sims[best_idx] - 0.1 * config_dist  # combined
        scores.append((name, score, best_idx, x_win, y_win))

    # Sort by score descending
    scores.sort(key=lambda t: -t[1])

    example_windows = []
    example_outputs = []
    for i in range(min(n_examples, len(scores))):
        _, _, idx, xw, yw = scores[i]
        example_windows.append(xw[idx, -horizon:])
        example_outputs.append(yw[idx])

    return example_windows, example_outputs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_icl(all_regions, target_name, seed=42, n_examples=3,
               epochs=200, lr=1e-3, device=None, pbar=None):
    """Train ICTransformer with ICL format on source regions.

    Training: for each batch, randomly pick m examples from source regions,
    construct context, predict query, supervise with ground-truth share values.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = ICTransformer(horizon=24, n_examples=n_examples)
    if device:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_cosine_warmup_scheduler(optimizer, max(1, epochs // 10), epochs)

    # Gather source data with config-distance weights (matches base TransCIF-ZS)
    target_mean_rs = all_regions[target_name]["mean_rs"]
    region_windows = {}
    region_weights = {}
    for name, data in all_regions.items():
        if name == target_name:
            continue
        x_win, y_win, _ = build_windows(data["rs"], data["cif"])
        if len(x_win) > 0:
            region_windows[name] = (x_win, y_win, data["config"])
            region_weights[name] = config_weight(data["mean_rs"], target_mean_rs)

    if not region_windows:
        print(f"  [WARN] No source data for {target_name}")
        return model, []

    region_names = list(region_windows.keys())
    # Normalised sampling weights over regions (config-distance biased)
    w_vals = np.array([region_weights[n] for n in region_names], dtype=np.float64)
    region_probs = w_vals / w_vals.sum()
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_iters = 0

        for _ in range(32):  # 32 batches per epoch (was 4 — undertrained)
            # Pick a config-weighted "query" region from sources
            query_name = np.random.choice(region_names, p=region_probs)
            q_x_win, q_y_win, q_cfg = region_windows[query_name]
            q_idx = random.randrange(len(q_x_win))
            query_x = q_x_win[q_idx, -24:]  # last H as query input
            query_y = q_y_win[q_idx]

            # Select examples from OTHER regions (also config-weighted)
            ex_windows, ex_outputs = [], []
            candidates = [n for n in region_names if n != query_name]
            if len(candidates) < n_examples:
                candidates = region_names  # fallback (allow self)
            cand_w = np.array([region_weights[n] for n in candidates], dtype=np.float64)
            cand_probs = cand_w / cand_w.sum()

            for i in range(n_examples):
                ex_name = np.random.choice(candidates, p=cand_probs)
                ex_x_win, ex_y_win, _ = region_windows[ex_name]
                ex_idx = random.randrange(len(ex_x_win))
                ex_windows.append(ex_x_win[ex_idx, -24:])
                ex_outputs.append(ex_y_win[ex_idx])

            # Build context
            values, roles = build_context(
                query_x, ex_windows, ex_outputs, horizon=24)
            v_t = torch.tensor(values)
            r_t = torch.tensor(roles)
            y_t = torch.tensor(query_y.reshape(1, -1))

            if device:
                v_t, r_t, y_t = v_t.to(device), r_t.to(device), y_t.to(device)

            pred = model(v_t.squeeze(0).unsqueeze(0), r_t.squeeze(0).unsqueeze(0))
            loss = F.l1_loss(pred, y_t)
            total_loss += loss.item()
            n_iters += 1

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        if pbar is not None:
            pbar(epoch, epochs, total_loss / max(n_iters, 1))

    model.eval()
    if pbar is not None:
        pbar.finish()
    return model


def predict_icl_zs(model, all_regions, target_name, x_rs_test, ef_r, ef_nr,
                    n_examples=3):
    """Zero-shot inference: select examples, build context, predict.

    For each test window, select top-n_examples matching examples from source
    regions, construct context, and predict share via ICL transformer.
    """
    model.eval()
    dev = next(model.parameters()).device
    cif_preds = []

    for i in range(len(x_rs_test)):
        target_window = x_rs_test[i]
        ex_w, ex_o = select_examples(
            all_regions, target_name, target_window,
            n_examples=n_examples, horizon=24)

        # Pad to n_examples if not enough found
        while len(ex_w) < n_examples:
            ex_w.append(np.zeros(24, dtype=np.float32))
            ex_o.append(np.zeros(24, dtype=np.float32))

        values, roles = build_context(
            target_window[-24:], ex_w, ex_o, horizon=24)

        v_t = torch.tensor(values).to(dev)
        r_t = torch.tensor(roles, dtype=torch.long).to(dev)

        with torch.no_grad():
            pred = model(v_t.squeeze(0).unsqueeze(0), r_t.squeeze(0).unsqueeze(0))
            share = pred.cpu().numpy().squeeze(0)

        cif_preds.append(cif_from_shares(share, ef_r, ef_nr))

    return np.stack(cif_preds)
