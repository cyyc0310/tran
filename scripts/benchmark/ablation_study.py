"""Phase 2.3: Complete Ablation Study.

Systematically removes each component to quantify its contribution.

Configurations:
| Config              | Removed                    |
|---------------------|----------------------------|
| Full model          | -                          |
| w/o adaptive gate   | Fixed gate=0.5             |
| w/o config bias     | Remove config_bias MLP     |
| w/o weighted sample | Uniform source sampling    |
| w/o physics layer   | Direct CIF prediction      |
| w/o decomposition   | No trend/seasonal split    |
| Persistence only    | Upper baseline (no model)  |

Usage: python scripts/ablation_study.py [--quick]
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from transcif.config import (
    DATA_DIR, SEQ_LEN, HORIZON, TRAIN_STRIDE, TEST_STRIDE, TRAIN_FRACTION,
    AU_REGIONS, US_REGIONS, UK_REGIONS, EPOCHS_ZERO_SHOT,
)
from transcif.calibration.zs_plus import (
    ANCHOR_WIN, RESID_WIN, WEEKLY_LAG, SELECT_DAYS, SELECT_MARGIN, FUSION_MENU,
)
from transcif.data.loaders import discover_uk_regions, load_region_data
from transcif.data.windows import build_windows
from transcif.physics.decompose import cif_from_shares
from transcif.models.zeroshot.base_zs import (
    compute_metrics, get_cosine_warmup_scheduler,
    train_zero_shot,
)
from transcif.models.base import (
    AdaptivePersistDLinear,
    NoAdaptiveGate, NoConfigBias, NoDecomposition,
    DirectCIF, NoPhysicsConversion,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# Training variant for ablation
# --------------------------------------------------------------------------

def train_ablation(all_regions, target_name, model_class, use_weighted=True,
                   use_physics=True, seed=42):
    """Train an ablation variant.

    Args:
        model_class : The model architecture to use
        use_weighted: Whether to use config-distance weighting
        use_physics  : True=predict rs, False/rs_to_cif=predict CIF directly
        seed         : Random seed
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = model_class(seq_len=SEQ_LEN, horizon=HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = get_cosine_warmup_scheduler(optimizer, 15, EPOCHS_ZERO_SHOT)

    xs, ys, cfgs, weights = [], [], [], []
    target_data = all_regions[target_name]

    for name, data in all_regions.items():
        if name == target_name:
            continue
        rs = data["rs"]
        cif = data["cif"]
        split = int(len(rs) * TRAIN_FRACTION)

        if use_physics == True:
            x_win, y_win, _ = build_windows(
                rs[:split], cif[:split], SEQ_LEN, HORIZON, TRAIN_STRIDE)
        elif use_physics == "rs_to_cif":
            x_win, _, y_cif_win = build_windows(
                rs[:split], cif[:split], SEQ_LEN, HORIZON, TRAIN_STRIDE)
            y_win = y_cif_win
        else:
            _, _, y_cif_win = build_windows(
                rs[:split], cif[:split], SEQ_LEN, HORIZON, TRAIN_STRIDE)
            x_cif = []
            for start in range(0, split - SEQ_LEN - HORIZON + 1, TRAIN_STRIDE):
                x_cif.append(cif[start:start + SEQ_LEN])
            if not x_cif:
                continue
            x_win = np.stack(x_cif)
            y_win = y_cif_win

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


ABLATION_CONFIGS = {
    "Full model": {
        "model_class": AdaptivePersistDLinear,
        "use_weighted": True,
        "use_physics": True,
    },
    "w/o adaptive gate": {
        "model_class": NoAdaptiveGate,
        "use_weighted": True,
        "use_physics": True,
    },
    "w/o config bias": {
        "model_class": NoConfigBias,
        "use_weighted": True,
        "use_physics": True,
    },
    "w/o decomposition": {
        "model_class": NoDecomposition,
        "use_weighted": True,
        "use_physics": True,
    },
    "w/o weighted sampling": {
        "model_class": AdaptivePersistDLinear,
        "use_weighted": False,
        "use_physics": True,
    },
    "Direct CIF (oracle)": {
        "model_class": DirectCIF,
        "use_weighted": True,
        "use_physics": False,
    },
}


def evaluate_ablation(target_name, all_regions, config_name, config, seed=42):
    """Evaluate one ablation configuration on one target."""
    data = all_regions[target_name]
    rs, cif = data["rs"], data["cif"]
    ef_r, ef_nr = data["ef_r"], data["ef_nr"]

    split = int(len(rs) * TRAIN_FRACTION)
    use_physics = config["use_physics"]

    if use_physics == True or use_physics == "rs_to_cif":
        x_rs_test, _, y_cif_test = build_windows(
            rs[split - SEQ_LEN:], cif[split - SEQ_LEN:],
            SEQ_LEN, HORIZON, TEST_STRIDE)
        x_test = x_rs_test
    else:
        _, _, y_cif_test = build_windows(
            rs[split - SEQ_LEN:], cif[split - SEQ_LEN:],
            SEQ_LEN, HORIZON, TEST_STRIDE)
        x_cif_test = []
        cif_offset = cif[split - SEQ_LEN:]
        for start in range(0, len(cif_offset) - SEQ_LEN - HORIZON + 1, TEST_STRIDE):
            x_cif_test.append(cif_offset[start:start + SEQ_LEN])
        if not x_cif_test:
            return None
        x_test = np.stack(x_cif_test)

    if len(x_test) == 0:
        return None

    model = train_ablation(
        all_regions, target_name,
        config["model_class"], config["use_weighted"], use_physics, seed)

    target_cfg = torch.tensor(data["config"]).unsqueeze(0).expand(len(x_test), -1)
    with torch.no_grad():
        pred = model(torch.tensor(x_test, dtype=torch.float32), target_cfg).numpy()

    if use_physics == True:
        cif_pred = cif_from_shares(pred, ef_r, ef_nr)
    else:
        cif_pred = pred

    return compute_metrics(cif_pred, y_cif_test)


# --------------------------------------------------------------------------
# ZS+ test-time calibration component ablation
# --------------------------------------------------------------------------

def zs_plus_variant(model, config, rs, cif, ef_r, ef_nr, origins,
                    use_anchor=True, use_residual=True, use_selfval=True,
                    fusion=None, horizon=HORIZON):
    """zs_plus_predict with each calibration step individually switchable."""
    from transcif.config import SEQ_LEN as L
    cfg1 = torch.tensor(config).unsqueeze(0)
    branch_cache = {}

    def branch_preds(t0):
        if t0 not in branch_cache:
            x = torch.tensor(rs[t0 - L:t0], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                s_raw = model(x, cfg1).numpy()[0]
            s = s_raw
            if use_anchor:
                s = np.clip(s_raw - s_raw.mean() + rs[t0 - ANCHOR_WIN:t0].mean(), 0.0, 1.0)
            out0 = cif_from_shares(s, ef_r, ef_nr)
            if use_residual:
                delta = (cif[t0 - RESID_WIN:t0]
                         - cif_from_shares(rs[t0 - RESID_WIN:t0], ef_r, ef_nr)).mean()
                out0 = out0 + delta
            weekly_lags = [j * WEEKLY_LAG for j in range(1, 5)
                           if t0 - j * WEEKLY_LAG >= 0]
            branch_cache[t0] = np.stack([
                out0,
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
            if o - L < 0 or o + 24 > min(t0, len(cif)):
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

    if not use_selfval:
        return np.stack([branch_preds(t0)[0] for t0 in origins])

    if fusion is not None:
        return np.stack([fuse_at(t0, **fusion) for t0 in origins])

    sim_cache = {}

    def sim_mae(o, ci):
        if (o, ci) not in sim_cache:
            sim_cache[(o, ci)] = np.abs(
                fuse_at(o, FUSION_MENU[ci]["branches"],
                        FUSION_MENU[ci]["gamma"],
                        FUSION_MENU[ci]["k_backtest"])[:24] - cif[o:o + 24]).mean()
        return sim_cache[(o, ci)]

    preds = []
    for t0 in origins:
        errs = []
        for j in range(1, SELECT_DAYS + 1):
            o_s = t0 - j * 24
            if o_s - L - 24 < 0:
                break
            errs.append([sim_mae(o_s, i) for i in range(len(FUSION_MENU))])
        chosen = FUSION_MENU[0]
        if errs:
            e = np.array(errs)
            sm, sl = e.sum(axis=0), np.log1p(e).sum(axis=0)
            best, best_key = 0, 2.0
            for i in range(1, len(FUSION_MENU)):
                rm, rl = sm[i] / sm[0], sl[i] / sl[0]
                wins = ((rm < 1.0 - SELECT_MARGIN and rl <= 1.0 + SELECT_MARGIN)
                        or (rl < 1.0 - SELECT_MARGIN and rm <= 1.0 + SELECT_MARGIN))
                if wins and rm + rl < best_key:
                    best, best_key = i, rm + rl
            chosen = FUSION_MENU[best]
        preds.append(fuse_at(t0, chosen["branches"], chosen["gamma"],
                            chosen["k_backtest"]))
    return np.stack(preds)


ZSP_VARIANTS = {
    "ZS+ (full calib)": dict(use_anchor=True, use_residual=True, use_selfval=True),
    "ZS+ w/o anchor": dict(use_anchor=False, use_residual=True, use_selfval=True),
    "ZS+ w/o residual": dict(use_anchor=True, use_residual=False, use_selfval=True),
    "ZS+ w/o self-val": dict(use_anchor=True, use_residual=True, use_selfval=False),
    "ZS+ w/o rolling selection": dict(fusion=FUSION_MENU[0]),
    "ZS+ legacy 2-branch fusion": dict(
        fusion=dict(branches=(0, 1), gamma=2.0, k_backtest=7)),
    "Raw ZS (no calib)": dict(use_anchor=False, use_residual=False, use_selfval=False),
}


def evaluate_zsplus_ablation(target_name, all_regions, seed=42):
    """Train the full ZS model once, then ablate the ZS+ calibration steps."""
    data = all_regions[target_name]
    rs, cif = data["rs"], data["cif"]
    ef_r, ef_nr = data["ef_r"], data["ef_nr"]
    split = int(len(rs) * TRAIN_FRACTION)

    _, _, y_cif_test = build_windows(
        rs[split - SEQ_LEN:], cif[split - SEQ_LEN:],
        SEQ_LEN, HORIZON, TEST_STRIDE)
    if len(y_cif_test) == 0:
        return None

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = train_zero_shot(all_regions, target_name, seed=seed)

    n_off = len(rs) - (split - SEQ_LEN)
    origins = [split + st
               for st in range(0, n_off - SEQ_LEN - HORIZON + 1, TEST_STRIDE)]

    out = {}
    for name, flags in ZSP_VARIANTS.items():
        pred = zs_plus_variant(model, data["config"], rs, cif, ef_r, ef_nr,
                               origins, **flags)
        out[name] = compute_metrics(pred, y_cif_test)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick: AU only, 3 seeds")
    args = parser.parse_args()

    seeds = [0, 1, 2] if args.quick else [0, 1, 2, 3, 4]

    print("=" * 80)
    print(f"Phase 2.3: Complete Ablation Study ({'QUICK' if args.quick else 'FULL'})")
    print("=" * 80)

    discover_uk_regions()
    all_configs_dict = {**AU_REGIONS, **UK_REGIONS, **US_REGIONS}
    all_regions = {}
    for name in all_configs_dict:
        try:
            all_regions[name] = load_region_data(name, all_configs_dict)
        except Exception:
            pass

    print(f"Loaded: {len(all_regions)} regions")

    if args.quick:
        targets = ["QLD1", "NSW1", "VIC1", "SA1"]
    else:
        targets = ["US_FPL", "US_MISO", "QLD1", "NSW1", "VIC1", "SA1",
                   "UK_07_South_Wales", "UK_01_North_Scotland", "US_BPAT"]
        targets = [t for t in targets if t in all_regions]

    print(f"Targets: {targets}")
    print(f"Seeds: {seeds}")
    print(f"Configs: {list(ABLATION_CONFIGS.keys())}")
    total = len(targets) * len(seeds) * len(ABLATION_CONFIGS)
    print(f"Total evaluations: {total}")

    t0 = time.time()
    all_results = []

    for i, target in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}] {target} (rs={all_regions[target]['mean_rs']:.3f})")
        for config_name, config in ABLATION_CONFIGS.items():
            maes = []
            for seed in seeds:
                metrics = evaluate_ablation(target, all_regions, config_name, config, seed)
                if metrics:
                    maes.append(metrics["mae"])
                    all_results.append({
                        "target": target, "config": config_name,
                        "seed": seed, **metrics
                    })
            if maes:
                print(f"  {config_name:<25} MAE={np.mean(maes):.1f} ± {np.std(maes):.1f}")

    # ZS+ calibration component ablation
    print("\n" + "=" * 80)
    print("ZS+ CALIBRATION COMPONENT ABLATION")
    print("=" * 80)
    zsp_seeds = seeds[:3]
    for i, target in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}] {target}")
        per_variant = {name: [] for name in ZSP_VARIANTS}
        for seed in zsp_seeds:
            out = evaluate_zsplus_ablation(target, all_regions, seed=seed)
            if out is None:
                continue
            for name, metrics in out.items():
                per_variant[name].append(metrics["mae"])
                all_results.append({
                    "target": target, "config": name,
                    "seed": seed, **metrics
                })
        for name, maes in per_variant.items():
            if maes:
                print(f"  {name:<25} MAE={np.mean(maes):.1f} ± {np.std(maes):.1f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")

    tag = "quick" if args.quick else "full"
    results_file = RESULTS_DIR / f"ablation_{tag}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved: {results_file}")

    # Ablation summary
    print("\n" + "=" * 90)
    print("ABLATION SUMMARY (Mean MAE across all targets and seeds)")
    print("=" * 90)
    print(f"\n{'Config':<25}", end="")
    for t in targets:
        print(f" {t:<10}", end="")
    print(f" {'MEAN':<8}")
    print("-" * (25 + 10 * len(targets) + 8))

    for config_name in ABLATION_CONFIGS:
        print(f"{config_name:<25}", end="")
        config_maes = []
        for t in targets:
            t_results = [r["mae"] for r in all_results
                        if r["target"] == t and r["config"] == config_name]
            if t_results:
                mean_mae = np.mean(t_results)
                config_maes.append(mean_mae)
                print(f" {mean_mae:<10.1f}", end="")
            else:
                print(f" {'N/A':<10}", end="")
        overall = np.mean(config_maes) if config_maes else 0
        print(f" {overall:<8.1f}")

    print(f"\n{'Relative to Full (%)':<25}", end="")
    for t in targets:
        print(f" {t:<10}", end="")
    print(f" {'MEAN':<8}")
    print("-" * (25 + 10 * len(targets) + 8))

    full_maes = {}
    for t in targets:
        t_results = [r["mae"] for r in all_results
                    if r["target"] == t and r["config"] == "Full model"]
        if t_results:
            full_maes[t] = np.mean(t_results)

    for config_name in ABLATION_CONFIGS:
        if config_name == "Full model":
            continue
        print(f"{config_name:<25}", end="")
        deltas = []
        for t in targets:
            t_results = [r["mae"] for r in all_results
                        if r["target"] == t and r["config"] == config_name]
            if t_results and t in full_maes:
                delta = (np.mean(t_results) - full_maes[t]) / full_maes[t] * 100
                deltas.append(delta)
                marker = chr(8593) if delta > 0 else chr(8595)
                print(f" {marker}{abs(delta):<8.1f}%", end="")
            else:
                print(f" {'N/A':<10}", end="")
        overall_delta = np.mean(deltas) if deltas else 0
        marker = chr(8593) if overall_delta > 0 else chr(8595)
        print(f" {marker}{abs(overall_delta):<7.1f}%")

    # ZS+ calibration ablation summary
    print("\n" + "=" * 90)
    print("ZS+ CALIBRATION ABLATION SUMMARY (Mean MAE across targets and seeds)")
    print("=" * 90)
    zsp_full = {}
    for t in targets:
        vals = [r["mae"] for r in all_results
                if r["target"] == t and r["config"] == "ZS+ (full calib)"]
        if vals:
            zsp_full[t] = np.mean(vals)
    for name in ZSP_VARIANTS:
        print(f"{name:<25}", end="")
        maes, deltas = [], []
        for t in targets:
            vals = [r["mae"] for r in all_results
                    if r["target"] == t and r["config"] == name]
            if vals:
                m = np.mean(vals)
                maes.append(m)
                if t in zsp_full and zsp_full[t] > 0:
                    deltas.append((m - zsp_full[t]) / zsp_full[t] * 100)
        overall = np.mean(maes) if maes else 0
        d = np.mean(deltas) if deltas else 0
        print(f" mean MAE={overall:<8.1f} vs full: {d:+.1f}%")

    print("\n✓ Ablation study complete!")


if __name__ == "__main__":
    main()
