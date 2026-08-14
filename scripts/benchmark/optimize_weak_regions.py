"""Diagnose & fix weak-region zero-shot performance.

Variants tested per target (seed 0 quick pass):
  V0: current AdaptivePersistDLinear -> physics          (reproduces paper)
  V1: V0 + online residual correction delta48            (Term-2 estimation)
  V2: V0 + residual correction delta168
  V3: gate-floor OOD fallback + delta48
Also logs mean gate value for diagnosis.

Usage: PYTHONPATH=scripts python scripts/optimize_weak_regions.py
"""

import json
from pathlib import Path

import numpy as np
import torch

from transcif.config import (
    SEQ_LEN, HORIZON, TEST_STRIDE, TRAIN_FRACTION,
    AU_REGIONS, US_REGIONS, UK_REGIONS,
)
from transcif.data.loaders import discover_uk_regions, load_region_data
from transcif.physics.decompose import cif_from_shares
from transcif.evaluation.metrics import compute_metrics
from transcif.models.zeroshot.base_zs import train_zero_shot

RESULTS = Path(__file__).resolve().parent.parent / "results"
# per-region patchtst MAE (seed 0) from the existing full eval, for ratio calc
_full = json.load(open(RESULTS / "unified_eval_full.json"))
PTST0 = {r["target"]: r["patchtst_sup"]["mae"] for r in _full if r["seed"] == 0}

TARGETS = None  # None => all regions


def residual_estimate(x_rs, x_cif, ef_r, ef_nr, win):
    """Online estimate of physics residual delta_t from observed stream."""
    phys_hist = cif_from_shares(x_rs[:, -win:], ef_r, ef_nr)
    return (x_cif[:, -win:] - phys_hist).mean(axis=1, keepdims=True)


def main():
    discover_uk_regions()
    all_configs = {**AU_REGIONS, **UK_REGIONS, **US_REGIONS}
    all_regions = {name: load_region_data(name, all_configs) for name in all_configs}

    seed = 0
    targets = TARGETS or sorted(all_regions.keys(), key=lambda x: all_regions[x]["mean_rs"])
    header = (f"{'target':<20}{'persist':>8}{'V0_cur':>8}{'V5_a+d':>8}"
              f"{'V6_sv':>8}{'r0':>6}{'r6':>6}")
    print(header)
    print("-" * len(header))
    agg = {"r0": [], "r6": []}

    for target in targets:
        data = all_regions[target]
        rs, cif = data["rs"], data["cif"]
        ef_r, ef_nr = data["ef_r"], data["ef_nr"]
        split = int(len(rs) * TRAIN_FRACTION)

        rs_o, cif_o = rs[split - SEQ_LEN:], cif[split - SEQ_LEN:]
        x_rs, x_cif, y_cif = [], [], []
        for st in range(0, len(rs_o) - SEQ_LEN - HORIZON + 1, TEST_STRIDE):
            x_rs.append(rs_o[st:st + SEQ_LEN])
            x_cif.append(cif_o[st:st + SEQ_LEN])
            y_cif.append(cif_o[st + SEQ_LEN:st + SEQ_LEN + HORIZON])
        x_rs, x_cif, y_cif = np.stack(x_rs), np.stack(x_cif), np.stack(y_cif)

        persist_mae = compute_metrics(x_cif[:, -HORIZON:], y_cif)["mae"]

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = train_zero_shot(all_regions, target, seed=seed)

        cfg = torch.tensor(data["config"]).unsqueeze(0).expand(len(x_rs), -1)
        x_t = torch.tensor(x_rs, dtype=torch.float32)
        with torch.no_grad():
            s_hat = model(x_t, cfg).numpy()
            # gate diagnosis
            recent_mean = x_t[:, -48:].mean(dim=1, keepdim=True)
            recent_std = x_t[:, -48:].std(dim=1, keepdim=True)
            gate = torch.sigmoid(model.gate_net(
                torch.cat([cfg, recent_mean, recent_std], dim=1))).mean().item()

        base = cif_from_shares(s_hat, ef_r, ef_nr)
        v0 = compute_metrics(base, y_cif)["mae"]

        d48 = residual_estimate(x_rs, x_cif, ef_r, ef_nr, 48)
        v1 = compute_metrics(base + d48, y_cif)["mae"]

        # V4: level anchor — shift s_hat so its mean matches last-24h observed rs
        anchor = x_rs[:, -24:].mean(axis=1, keepdims=True)
        s_anchor = np.clip(s_hat - s_hat.mean(axis=1, keepdims=True) + anchor, 0.0, 1.0)
        v4 = compute_metrics(cif_from_shares(s_anchor, ef_r, ef_nr), y_cif)["mae"]
        v5 = compute_metrics(cif_from_shares(s_anchor, ef_r, ef_nr) + d48, y_cif)["mae"]

        # V6: self-validation selection — backtest model vs CIF-persistence on
        # K recent observed 24h blocks (all inside available history), then
        # weight the two branches by inverse backtest MAE.
        K = 7
        rs_hist = rs_o  # full observable stream
        v6_preds = np.zeros_like(y_cif)
        alphas = []
        for i, st in enumerate(range(0, len(rs_o) - SEQ_LEN - HORIZON + 1, TEST_STRIDE)):
            t_end = st + SEQ_LEN  # forecast origin
            m_err, p_err = [], []
            for k in range(1, K + 1):
                o = t_end - k * 24  # past origin with observed outcome
                if o - SEQ_LEN < 0:
                    break
                xb = torch.tensor(rs_hist[o - SEQ_LEN:o], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    sb = model(xb, torch.tensor(data["config"]).unsqueeze(0)).numpy()[0]
                sb = np.clip(sb - sb.mean() + rs_hist[o - 24:o].mean(), 0, 1)
                cif_b = cif_from_shares(sb, ef_r, ef_nr) + (
                    cif_o[o - 48:o] - cif_from_shares(rs_hist[o - 48:o], ef_r, ef_nr)).mean()
                y_b = cif_o[o:o + 24]
                m_err.append(np.abs(cif_b - y_b).mean())
                p_err.append(np.abs(cif_o[o - 24:o] - y_b).mean())
            if m_err:
                me, pe = np.mean(m_err), np.mean(p_err)
                alpha = pe / (me + pe)  # weight on model branch
            else:
                alpha = 0.5
            alphas.append(alpha)
            model_branch = cif_from_shares(s_anchor[i], ef_r, ef_nr) + d48[i]
            persist_branch = x_cif[i, -HORIZON:]
            v6_preds[i] = alpha * model_branch + (1 - alpha) * persist_branch
        v6 = compute_metrics(v6_preds, y_cif)["mae"]

        pt = PTST0.get(target, float("nan"))
        r0, r6 = v0 / pt, v6 / pt
        agg["r0"].append(r0)
        agg["r6"].append(r6)
        print(f"{target:<20}{persist_mae:>8.1f}{v0:>8.1f}{v5:>8.1f}"
              f"{v6:>8.1f}{r0:>6.2f}{r6:>6.2f}")

    print("-" * len(header))
    r0a, r6a = np.array(agg["r0"]), np.array(agg["r6"])
    print(f"{'MEDIAN ratio':<20}{'':8}{'':8}{'':8}{'':8}"
          f"{np.median(r0a):>6.2f}{np.median(r6a):>6.2f}")
    print(f"{'MEAN ratio':<20}{'':8}{'':8}{'':8}{'':8}"
          f"{r0a.mean():>6.2f}{r6a.mean():>6.2f}")
    print(f"within 1.25x:  V0={int((r0a<=1.25).sum())}/{len(r0a)}  "
          f"V6={int((r6a<=1.25).sum())}/{len(r6a)}")
    print(f"within 1.5x:   V0={int((r0a<=1.5).sum())}/{len(r0a)}  "
          f"V6={int((r6a<=1.5).sum())}/{len(r6a)}")


if __name__ == "__main__":
    main()
