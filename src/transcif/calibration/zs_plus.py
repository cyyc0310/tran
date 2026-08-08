"""Test-time calibration for TransCIF-ZS+ (adaptive branch fusion).

Implements the ``zs_plus_predict`` routine: given a trained zero-shot model
and a target region's history, it blends several forecast "branches" using
backtested per-day error to produce a calibrated prediction.  See
``run_unified_eval.py`` for the conceptual description.
"""

import numpy as np
import torch

from transcif.config import SEQ_LEN, HORIZON
from transcif.physics.decompose import cif_from_shares

# --- ZS+ hyperparameters ----------------------------------------------------
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
                    horizon=HORIZON, fusion=None, share_fn=None):
    """Test-time calibrated zero-shot prediction (TransCIF-ZS+).

    Args:
        model    : a zero-shot model exposing ``model(x, config) -> share``.
                   Ignored when ``share_fn`` is provided.
        share_fn : optional callable ``share_fn(x_window_np) -> share_np``
                   (shape ``(horizon,)``).  When given, it replaces the
                   ``model(x, cfg1)`` call inside branch 0/5, letting models
                   with a different forward signature (RAG/Causal/ICL) reuse
                   the same calibration pipeline.
    """
    cfg1 = torch.tensor(config).unsqueeze(0)
    branch_cache = {}

    def branch_preds(t0):
        if t0 not in branch_cache:
            dev = next(model.parameters()).device if share_fn is None else None
            x_win = rs[t0 - SEQ_LEN:t0]
            if share_fn is not None:
                s_raw = np.asarray(share_fn(x_win), dtype=np.float64).ravel()
            else:
                x = torch.tensor(x_win, dtype=torch.float32).unsqueeze(0).to(dev)
                with torch.no_grad():
                    s_raw = model(x, cfg1.to(dev)).cpu().numpy()[0]
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
