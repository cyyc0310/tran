"""Phase 3.3: Temporal OOD Experiment.

Tests whether the model generalizes across time:
- Train on first 9 months, test on last 3 months
- Compare with the standard 80/20 split
- Shows model isn't just memorizing seasonal patterns

Motivation: A reviewer might argue that the model memorizes year-specific patterns.
By testing on temporally out-of-distribution data, we prove it learns
generalizable dynamics.

Usage: PYTHONPATH=src python scripts/temporal_ood.py
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_unified_eval import (
    DATA_DIR, SEQ_LEN, HORIZON, TRAIN_STRIDE, TEST_STRIDE,
    AU_REGIONS, US_REGIONS, UK_REGIONS,
    discover_uk_regions, load_region_data, build_windows,
    cif_from_shares, train_zero_shot, compute_metrics, zs_plus_predict,
    AdaptivePersistDLinear, get_cosine_warmup_scheduler,
    EPOCHS_ZERO_SHOT, BATCH_SIZE, TRAIN_FRACTION,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def evaluate_temporal_split(all_regions, target_name, train_frac, seed=42):
    """Evaluate TransCIF with a specific temporal train/test split.
    
    Args:
        train_frac: Fraction of data used for source training (e.g., 0.75 = 9/12 months)
    """
    data = all_regions[target_name]
    rs, cif = data["rs"], data["cif"]
    ef_r, ef_nr = data["ef_r"], data["ef_nr"]
    
    split = int(len(rs) * train_frac)
    
    # Build test windows from temporal OOD portion
    x_rs_test, y_rs_test, y_cif_test = build_windows(
        rs[split - SEQ_LEN:], cif[split - SEQ_LEN:], SEQ_LEN, HORIZON, TEST_STRIDE)
    
    if len(x_rs_test) == 0:
        return None
    
    # Train zero-shot model (sources use same temporal fraction)
    # Create modified region data with the new split
    modified_regions = {}
    for name, rdata in all_regions.items():
        mod_split = int(len(rdata["rs"]) * train_frac)
        modified_regions[name] = {
            **rdata,
            "_train_end": mod_split,  # Marker for custom split
        }
    
    # Train using standard function but with modified TRAIN_FRACTION logic
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    import random
    random.seed(seed)
    
    model = AdaptivePersistDLinear(seq_len=SEQ_LEN, horizon=HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = get_cosine_warmup_scheduler(optimizer, 15, EPOCHS_ZERO_SHOT)
    
    xs, ys, cfgs, wts = [], [], [], []
    target_data = all_regions[target_name]
    
    for name, rdata in all_regions.items():
        if name == target_name:
            continue
        src_rs = rdata["rs"]
        src_cif = rdata["cif"]
        src_split = int(len(src_rs) * train_frac)
        
        x_win, y_win, _ = build_windows(src_rs[:src_split], src_cif[:src_split],
                                         SEQ_LEN, HORIZON, TRAIN_STRIDE)
        if len(x_win) == 0:
            continue
        
        xs.append(x_win)
        ys.append(y_win)
        cfgs.append(np.tile(rdata["config"], (len(x_win), 1)))
        
        dist = abs(rdata["mean_rs"] - target_data["mean_rs"])
        w = 1.0 / (dist + 0.05)
        wts.append(np.full(len(x_win), w, dtype=np.float32))
    
    x_all = torch.tensor(np.concatenate(xs))
    y_all = torch.tensor(np.concatenate(ys))
    c_all = torch.tensor(np.concatenate(cfgs))
    w_all = torch.tensor(np.concatenate(wts))
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    model.eval()
    
    # Predict on test
    target_cfg = torch.tensor(data["config"]).unsqueeze(0).expand(len(x_rs_test), -1)
    with torch.no_grad():
        rs_pred = model(torch.tensor(x_rs_test, dtype=torch.float32), target_cfg).numpy()
    
    cif_pred = cif_from_shares(rs_pred, ef_r, ef_nr)
    
    # ZS+ branch: test-time calibration on the same model
    n_off = len(rs) - (split - SEQ_LEN)
    origins = [split + st
               for st in range(0, n_off - SEQ_LEN - HORIZON + 1, TEST_STRIDE)]
    zsp_pred = zs_plus_predict(model, data["config"], rs, cif, ef_r, ef_nr, origins)
    
    # Persistence baseline
    persist_pred = cif_from_shares(x_rs_test[:, -HORIZON:], ef_r, ef_nr)
    
    metrics_zs = compute_metrics(cif_pred, y_cif_test)
    metrics_zsp = compute_metrics(zsp_pred, y_cif_test)
    metrics_persist = compute_metrics(persist_pred, y_cif_test)
    
    return {
        "transcif_zs": metrics_zs,
        "transcif_zs_plus": metrics_zsp,
        "persistence": metrics_persist,
        "n_test": len(x_rs_test),
        "ratio_vs_persist": metrics_zs["mae"] / (metrics_persist["mae"] + 1e-10),
        "ratio_plus_vs_persist": metrics_zsp["mae"] / (metrics_persist["mae"] + 1e-10),
    }


def main():
    print("=" * 80)
    print("Phase 3.3: Temporal OOD Experiment")
    print("=" * 80)
    print("\nTest: Does the model generalize to unseen TIME periods?")
    print("Compare: 80/20 (standard) vs 75/25 (last quarter) vs 50/50 (last half)")
    
    # Load regions
    discover_uk_regions()
    all_configs = {**AU_REGIONS, **UK_REGIONS, **US_REGIONS}
    all_regions = {}
    for name in all_configs:
        try:
            all_regions[name] = load_region_data(name, all_configs)
        except:
            pass
    
    print(f"\nLoaded: {len(all_regions)} regions")
    
    # Representative targets
    targets = ["US_FPL", "US_MISO", "QLD1", "NSW1", "VIC1", "SA1",
               "UK_07_South_Wales", "UK_01_North_Scotland", "US_BPAT"]
    targets = [t for t in targets if t in all_regions]
    
    # Temporal splits to test
    splits = {
        "Standard (80/20)": 0.80,
        "9-month (75/25)": 0.75,
        "6-month (50/50)": 0.50,
    }
    
    seeds = [0, 1, 2]
    
    print(f"Targets: {targets}")
    print(f"Splits: {list(splits.keys())}")
    print(f"Seeds: {seeds}")
    
    t0 = time.time()
    all_results = []
    
    # Run evaluations
    print(f"\n{'Region':<15}", end="")
    for split_name in splits:
        print(f" {split_name[:12]:<14}", end="")
    print()
    print("-" * 60)
    
    for target in targets:
        print(f"{target:<15}", end="")
        target_results = {"region": target, "mean_rs": all_regions[target]["mean_rs"]}
        
        for split_name, train_frac in splits.items():
            maes = []
            ratios = []
            maes_plus = []
            ratios_plus = []
            for seed in seeds:
                result = evaluate_temporal_split(all_regions, target, train_frac, seed)
                if result:
                    maes.append(result["transcif_zs"]["mae"])
                    ratios.append(result["ratio_vs_persist"])
                    maes_plus.append(result["transcif_zs_plus"]["mae"])
                    ratios_plus.append(result["ratio_plus_vs_persist"])
            
            if maes:
                mean_mae = np.mean(maes)
                mean_ratio = np.mean(ratios)
                target_results[f"mae_{split_name}"] = mean_mae
                target_results[f"ratio_{split_name}"] = mean_ratio
                target_results[f"mae_plus_{split_name}"] = np.mean(maes_plus)
                target_results[f"ratio_plus_{split_name}"] = np.mean(ratios_plus)
                print(f" {np.mean(maes_plus):<6.1f}±{np.std(maes_plus):<5.1f}", end="")
            else:
                print(f" {'N/A':<14}", end="")
        
        print()
        all_results.append(target_results)
    
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")
    
    # Analysis
    print("\n" + "=" * 80)
    print("TEMPORAL OOD ANALYSIS")
    print("=" * 80)
    
    # Compute degradation from standard to OOD splits (ZS+ variant, primary)
    print(f"\n{'Region':<15} {'Std MAE':<10} {'75% MAE':<10} {'50% MAE':<10} "
          f"{'75%/Std':<8} {'50%/Std':<8}")
    print("-" * 65)
    
    degradations_75 = []
    degradations_50 = []
    
    for r in all_results:
        std_key = [k for k in r if "mae_plus_Standard" in k]
        m75_key = [k for k in r if "mae_plus_9-month" in k]
        m50_key = [k for k in r if "mae_plus_6-month" in k]
        
        if std_key and m75_key and m50_key:
            std_mae = r[std_key[0]]
            m75_mae = r[m75_key[0]]
            m50_mae = r[m50_key[0]]
            
            deg_75 = m75_mae / std_mae
            deg_50 = m50_mae / std_mae
            degradations_75.append(deg_75)
            degradations_50.append(deg_50)
            
            print(f"{r['region']:<15} {std_mae:<10.1f} {m75_mae:<10.1f} {m50_mae:<10.1f} "
                  f"{deg_75:<8.3f} {deg_50:<8.3f}")
    
    print(f"\n{'MEAN':<15} {'—':<10} {'—':<10} {'—':<10} "
          f"{np.mean(degradations_75):<8.3f} {np.mean(degradations_50):<8.3f}")
    
    print(f"\nKey finding:")
    print(f"  75/25 split (last quarter): MAE ratio = {np.mean(degradations_75):.3f} vs standard")
    print(f"  50/50 split (last half): MAE ratio = {np.mean(degradations_50):.3f} vs standard")
    if np.mean(degradations_75) < 1.15:
        print(f"  → GOOD: Model generalizes well to unseen time periods (<15% degradation)")
    elif np.mean(degradations_75) < 1.3:
        print(f"  → MODERATE: Some temporal degradation, but model still transfers")
    else:
        print(f"  → CONCERN: Significant temporal degradation, may overfit to seasons")
    
    # Save results
    output_file = RESULTS_DIR / "temporal_ood.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved: {output_file}")
    
    # --- Generate Figure ---
    print("\n--- GENERATING FIGURE ---")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: MAE comparison across splits
    ax = axes[0]
    x_pos = np.arange(len(all_results))
    width = 0.25
    
    std_maes = [r.get(f"mae_plus_Standard (80/20)", 0) for r in all_results]
    m75_maes = [r.get(f"mae_plus_9-month (75/25)", 0) for r in all_results]
    m50_maes = [r.get(f"mae_plus_6-month (50/50)", 0) for r in all_results]
    
    ax.bar(x_pos - width, std_maes, width, label='Standard (80/20)', alpha=0.8)
    ax.bar(x_pos, m75_maes, width, label='Last quarter (75/25)', alpha=0.8)
    ax.bar(x_pos + width, m50_maes, width, label='Last half (50/50)', alpha=0.8)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r["region"][:8] for r in all_results], rotation=45, ha='right')
    ax.set_ylabel("CIF MAE (gCO₂/kWh)", fontsize=11)
    ax.set_title("(a) Temporal OOD: ZS+ MAE by Split", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Degradation ratio
    ax = axes[1]
    ax.bar(x_pos - 0.15, degradations_75, 0.3, label='75/25 vs Standard', alpha=0.8)
    ax.bar(x_pos + 0.15, degradations_50, 0.3, label='50/50 vs Standard', alpha=0.8)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='No degradation')
    ax.axhline(1.15, color='r', linestyle=':', alpha=0.5, label='15% threshold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r["region"][:8] for r in all_results], rotation=45, ha='right')
    ax.set_ylabel("MAE ratio (OOD / Standard)", fontsize=11)
    ax.set_title("(b) Temporal Degradation", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "temporal_ood.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {fig_path}")
    
    print("\n✓ Phase 3.3 Temporal OOD complete!")


if __name__ == "__main__":
    main()
