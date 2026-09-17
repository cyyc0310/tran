#!/usr/bin/env python
"""Gap-4: supervised-baseline (PatchTST + CarbonCast CNN-LSTM) vs the FD-41
fuel-decomposition stack, paired on the official protocol.

The FD stack numbers come from results/fuel_decomp_eval_full_fd41.json
(29 regions x 5 seeds, official).  The supervised baselines are re-trained
here on each target's own train split (80%) and scored on the same test
windows (last 20%, stride 24) — identical window protocol to
run_fuel_decomp_eval.build_target_test_windows.

CarbonCast CNN-LSTM is ported from scripts/experiments/run_phase1_complete.py
(PyTorch reimpl with built-in min-max normalization, the paper-faithful
variant).  CC-ZS (cross-domain zero-shot) is NOT re-run: its collapse
(2.4x degradation) is architecture-level, established on the same loader in
Phase 1.1; re-training the FD stack does not change CC-ZS.  We re-run the
supervised arm so the paired comparison (FD-stack vs supervised) uses the
same test windows and current data assets.

Usage:
    PYTHONPATH=src .venv-nemed/bin/python scripts/experiments/gap4_supervised_baselines.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from transcif.config import (
    SEQ_LEN, HORIZON, TEST_STRIDE, TRAIN_FRACTION, TRAIN_STRIDE,
    RESULTS_DIR,
)
from transcif.data.loaders import all_region_configs
from transcif.data.fuel import build_fd_windows
from transcif.models.zeroshot.fuel import prepare_fd_region

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PatchTST(nn.Module):
    """Minimal PatchTST: patching + RevIN + channel-independent Transformer.

    Matches the supervised upper bound used in the paper (Phase 1.1: RevIN +
    cosine warmup + 300 epochs, strongest supervised baseline).
    """

    def __init__(self, seq_len=SEQ_LEN, horizon=HORIZON,
                 patch_len=16, stride=8, d_model=64, nhead=4,
                 nlayers=3, dropout=0.1):
        super().__init__()
        self.revin = RevIN(num_features=1)
        n_patches = (seq_len - patch_len) // stride + 1
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, nlayers)
        self.head = nn.Linear(d_model * n_patches, horizon)
        self.horizon = horizon

    def forward(self, x):
        # x: (B, L) univariate CIF history
        z = self.revin(x.unsqueeze(-1), mode="norm")      # (B, L, 1)
        z = z.squeeze(-1)
        p = z.unfold(dimension=-1, size=16, step=8)       # (B, n_patches, 16)
        h = self.encoder(self.patch_embed(p) + self.pos)  # (B, P, D)
        out = self.head(h.flatten(1))                     # (B, H)
        return self.revin(out.unsqueeze(-1), mode="denorm").squeeze(-1)


class RevIN(nn.Module):
    def __init__(self, num_features=1, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self.mu = x.mean(1, keepdim=True)
            self.sigma = (x.std(1, keepdim=True) + self.eps)
            x = (x - self.mu) / self.sigma * self.gamma + self.beta
        else:
            x = (x - self.beta) / (self.gamma + 1e-10) * self.sigma + self.mu
        return x


class CarbonCastCNNLSTM(nn.Module):
    """PyTorch reimpl of CarbonCast's CNN-LSTM (from run_phase1_complete.py).

    Input (rs, CIF) 2-channel history; built-in min-max normalization from
    training stats (mirrors original CarbonCast common.scaleDataset).
    """

    def __init__(self, seq_len=SEQ_LEN, horizon=HORIZON, n_features=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        conv_len = seq_len // 4
        self.lstm = nn.LSTM(128, 100, batch_first=True)
        self.head = nn.Linear(100, horizon)
        self.register_buffer("x_min", torch.zeros(n_features))
        self.register_buffer("x_max", torch.ones(n_features))
        self.register_buffer("y_min", torch.zeros(horizon))
        self.register_buffer("y_max", torch.ones(horizon))

    def set_normalization(self, x_train, y_train):
        self.x_min = torch.from_numpy(
            np.asarray(x_train.reshape(-1, x_train.shape[-1]).min(0))).float()
        self.x_max = torch.from_numpy(
            np.asarray(x_train.reshape(-1, x_train.shape[-1]).max(0))).float()
        self.y_min = torch.tensor(float(np.min(y_train))).float()
        self.y_max = torch.tensor(float(np.max(y_train))).float()

    def normalize_input(self, x):
        return (x - self.x_min) / (self.x_max - self.x_min + 1e-8)

    def denormalize_output(self, y):
        return y * (self.y_max - self.y_min + 1e-8) + self.y_min

    def forward(self, x, denorm=True):
        h = self.normalize_input(x)
        h = self.conv(h.transpose(1, 2))
        h, _ = self.lstm(h.transpose(1, 2))
        y = self.head(h[:, -1])
        return self.denormalize_output(y) if denorm else y


def train_torch(model, x, y, epochs, lr, device, tag=""):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(1, epochs // 5))
    xb_all = torch.tensor(x, dtype=torch.float32)      # stays on CPU
    yb_all = torch.tensor(y, dtype=torch.float32)      # stays on CPU
    n = len(xb_all)
    bs = 64
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            xb = xb_all[idx].to(device)
            yb = yb_all[idx].to(device)
            opt.zero_grad()
            loss = nn.functional.smooth_l1_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", default=str(RESULTS_DIR / "fuel_decomp_eval_full_fd41.json"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "gap4_supervised_vs_fd41.json"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    with open(args.fd) as f:
        fd_doc = json.load(f)
    fd_rows = fd_doc["rows"]
    fd_meta = fd_doc["meta"]
    print(f"[gap4] FD stack: {len(fd_rows)} pairs from {args.fd}")
    print(f"[gap4] FD meta epochs={fd_meta.get('epochs')} monthly={fd_meta.get('use_monthly')}")

    cfgs = all_region_configs()

    out_rows = []
    targets = sorted({r["target"] for r in fd_rows})
    for target in targets:
        data = prepare_fd_region(target, cfgs)
        rs, cif = data["rs"], data["cif"]
        split = int(len(rs) * TRAIN_FRACTION)

        # test windows (same protocol as fuel eval)
        w = None
        sl = slice(split - SEQ_LEN, None)
        sliced = {**data, "rs": rs[sl], "cif": cif[sl],
                  "fuel_shares": data["fuel_shares"][sl],
                  "hours": data["hours"][sl],
                  "exog": {k: v[sl] for k, v in data["exog"].items()}}
        w = build_fd_windows(sliced, seq_len=SEQ_LEN, horizon=HORIZON,
                             stride=TEST_STRIDE)
        y = w["y_cif"]
        n = len(w["x_rs"])

        # supervised training windows: CIF history only, own region
        x_cif_train, y_cif_train = [], []
        for start in range(0, split - SEQ_LEN - HORIZON + 1, TRAIN_STRIDE):
            x_cif_train.append(cif[start:start + SEQ_LEN])
            y_cif_train.append(cif[start + SEQ_LEN:start + SEQ_LEN + HORIZON])
        x_cif_train = np.stack(x_cif_train)
        y_cif_train = np.stack(y_cif_train)

        # CarbonCast input: (rs, CIF) 2-channel
        x2_train = []
        for start in range(0, split - SEQ_LEN - HORIZON + 1, TRAIN_STRIDE):
            x2_train.append(np.stack([rs[start:start + SEQ_LEN],
                                      cif[start:start + SEQ_LEN]], -1))
        x2_train = np.stack(x2_train)
        x2_test = []
        for s0 in range(0, n):
            origin = (split - SEQ_LEN) + s0 * TEST_STRIDE
            x2_test.append(np.stack([rs[origin:origin + SEQ_LEN],
                                     cif[origin:origin + SEQ_LEN]], -1))
        x2_test = np.stack(x2_test)

        row = {"target": target, "n_test": n}
        # persistence reference on identical windows
        cif_off = cif[split - SEQ_LEN:]
        persist = np.stack([
            cif_off[s + SEQ_LEN - HORIZON:s + SEQ_LEN]
            for s in range(0, len(cif_off) - SEQ_LEN - HORIZON + 1,
                           TEST_STRIDE)][:n])
        row["persistence_mae"] = float(np.abs(persist - y).mean())

        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            t0 = time.time()
            # PatchTST
            ptst = train_torch(PatchTST(), x_cif_train, y_cif_train,
                               args.epochs, 1e-3, device)
            with torch.no_grad():
                pred = ptst(torch.tensor(x_cif_test := np.stack([
                    cif[(split - SEQ_LEN) + s * TEST_STRIDE:
                        (split - SEQ_LEN) + s * TEST_STRIDE + SEQ_LEN]
                    for s in range(n)]), dtype=torch.float32,
                    device=device)).cpu().numpy()
            row[f"patchtst_mae_s{seed}"] = float(np.abs(pred - y).mean())

            # CarbonCast supervised
            cc = CarbonCastCNNLSTM()
            cc.set_normalization(x2_train, y_cif_train)
            train_torch(cc, x2_train, y_cif_train, args.epochs, 5e-4, device)
            with torch.no_grad():
                pred_cc = cc(torch.tensor(x2_test, dtype=torch.float32,
                                          device=device)).cpu().numpy()
            row[f"carboncast_mae_s{seed}"] = float(np.abs(pred_cc - y).mean())
            row[f"train_s_s{seed}"] = round(time.time() - t0, 1)

        # FD-stack numbers for this target (median over seeds)
        fd_t = [r for r in fd_rows if r["target"] == target]
        row["fd_i0_median"] = float(np.median(
            [r["fuel_i0"]["mae"] for r in fd_t]))
        row["fd_icfg_median"] = float(np.median(
            [r["fuel_i_cfg"]["mae"] for r in fd_t]))
        row["fd_iplus_median"] = float(np.median(
            [r["fuel_i_plus"]["mae"] for r in fd_t if r.get("fuel_i_plus")]))

        # paired deltas
        pt = np.mean([row[f"patchtst_mae_s{s}"] for s in args.seeds])
        cc = np.mean([row[f"carboncast_mae_s{s}"] for s in args.seeds])
        row["iplus_vs_patchtst"] = row["fd_iplus_median"] - pt
        row["iplus_vs_carboncast"] = row["fd_iplus_median"] - cc
        row["i0_vs_patchtst"] = row["fd_i0_median"] - pt

        out_rows.append(row)
        print(f"[gap4] {target:28s} persist {row['persistence_mae']:6.1f} | "
              f"PatchTST {pt:6.1f} | CC {cc:6.1f} | FD I_0 {row['fd_i0_median']:6.1f} "
              f"I_cfg {row['fd_icfg_median']:6.1f} I_+ {row['fd_iplus_median']:6.1f} "
              f"({row[f'train_s_s{args.seeds[0]}']}s)", flush=True)
        Path(args.out).write_text(json.dumps(
            {"rows": out_rows,
             "meta": {"epochs": args.epochs, "seeds": args.seeds,
                      "fd_source": args.fd,
                      "protocol": "supervised own-region 80/20, same test windows as FD stack"}},
            indent=1))

    # summary
    for key, label in (
            ("iplus_vs_patchtst", "I_+ vs PatchTST"),
            ("iplus_vs_carboncast", "I_+ vs CarbonCast"),
            ("i0_vs_patchtst", "I_0 vs PatchTST")):
        vals = [r[key] for r in out_rows]
        wins = sum(v < 0 for v in vals)
        print(f"[gap4] {label}: median delta {np.median(vals):+.2f} | "
              f"win {wins}/{len(vals)}")

    print(f"[gap4] wrote {args.out}")


if __name__ == "__main__":
    main()
