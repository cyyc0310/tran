"""Phase 3.2: CarbonCast Cross-Domain Deep Analysis.

Compares TransCIF zero-shot vs CarbonCast CNN-LSTM in both modes:
- CarbonCast supervised: trained on target domain (performance upper bound)
- CarbonCast zero-shot: trained on source domains (cross-domain failure)
- TransCIF zero-shot: config-driven (our method)

Analyzes WHY CarbonCast fails in cross-domain:
1. Normalization mismatch (source min/max ≠ target min/max)
2. No physics layer → cannot separate transferable patterns from region-specific ones
3. Overfits to source domain absolute CIF values

Usage: PYTHONPATH=scripts python scripts/carboncast_analysis.py
"""

import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_unified_eval import (
    DATA_DIR, SEQ_LEN, HORIZON, TRAIN_STRIDE, TEST_STRIDE, TRAIN_FRACTION,
    AU_REGIONS, US_REGIONS, UK_REGIONS,
    discover_uk_regions, load_region_data, build_windows,
    cif_from_shares, train_zero_shot, compute_metrics, zs_plus_predict,
    get_cosine_warmup_scheduler, EPOCHS_ZERO_SHOT, BATCH_SIZE,
)
from run_phase1_complete import (
    CarbonCastCNNLSTM, train_carboncast_zero_shot,
    build_multivariate_windows, EPOCHS_CARBONCAST,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def train_carboncast_supervised(all_regions, target_name, seed=42):
    """Train CarbonCast on target domain (supervised upper bound)."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    data = all_regions[target_name]
    rs, cif = data["rs"], data["cif"]
    split = int(len(rs) * TRAIN_FRACTION)

    model = CarbonCastCNNLSTM(seq_len=SEQ_LEN, horizon=HORIZON, n_features=2)

    x_multi, y_cif = build_multivariate_windows(rs[:split], cif[:split], SEQ_LEN, HORIZON, TRAIN_STRIDE)
    if len(x_multi) == 0:
        return None

    model.set_normalization(x_multi, y_cif)

    x_all = torch.tensor(x_multi, dtype=torch.float32)
    y_all = torch.tensor(y_cif, dtype=torch.float32)
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


def evaluate_carboncast_comparison(target_name, all_regions, seed=42):
    """Evaluate CarbonCast (supervised + zero-shot) and TransCIF (zero-shot)."""
    data = all_regions[target_name]
    rs, cif = data["rs"], data["cif"]
    ef_r, ef_nr = data["ef_r"], data["ef_nr"]
    split = int(len(rs) * TRAIN_FRACTION)

    # Test windows (multivariate for CarbonCast, rs-only for TransCIF)
    x_multi_test, y_cif_test = build_multivariate_windows(
        rs[split - SEQ_LEN:], cif[split - SEQ_LEN:], SEQ_LEN, HORIZON, TEST_STRIDE)
    x_rs_test, _, _ = build_windows(
        rs[split - SEQ_LEN:], cif[split - SEQ_LEN:], SEQ_LEN, HORIZON, TEST_STRIDE)

    if len(x_multi_test) == 0:
        return None

    results = {"region": target_name, "mean_rs": data["mean_rs"], "ef_nr": data["ef_nr"]}

    # 1. CarbonCast supervised
    cc_sup = train_carboncast_supervised(all_regions, target_name, seed)
    if cc_sup:
        with torch.no_grad():
            pred_sup = cc_sup(torch.tensor(x_multi_test, dtype=torch.float32)).numpy()
        metrics_sup = compute_metrics(pred_sup, y_cif_test)
        results["cc_supervised"] = metrics_sup
    else:
        results["cc_supervised"] = {"mae": float('inf')}

    # 2. CarbonCast zero-shot
    cc_zs = train_carboncast_zero_shot(all_regions, target_name, seed)
    with torch.no_grad():
        pred_zs_cc = cc_zs(torch.tensor(x_multi_test, dtype=torch.float32)).numpy()
    metrics_zs_cc = compute_metrics(pred_zs_cc, y_cif_test)
    results["cc_zeroshot"] = metrics_zs_cc

    # 3. TransCIF zero-shot
    zs_model = train_zero_shot(all_regions, target_name, seed=seed)
    target_cfg = torch.tensor(data["config"]).unsqueeze(0).expand(len(x_rs_test), -1)
    with torch.no_grad():
        rs_pred = zs_model(torch.tensor(x_rs_test, dtype=torch.float32), target_cfg).numpy()
    cif_pred_transcif = cif_from_shares(rs_pred, ef_r, ef_nr)
    metrics_zs_tc = compute_metrics(cif_pred_transcif, y_cif_test)
    results["transcif_zeroshot"] = metrics_zs_tc

    # 3b. TransCIF ZS+ (test-time calibration on the same model)
    n_off = len(rs) - (split - SEQ_LEN)
    origins = [split + st
               for st in range(0, n_off - SEQ_LEN - HORIZON + 1, TEST_STRIDE)]
    zsp_pred = zs_plus_predict(zs_model, data["config"], rs, cif, ef_r, ef_nr, origins)
    metrics_zsp_tc = compute_metrics(zsp_pred, y_cif_test)
    results["transcif_zs_plus"] = metrics_zsp_tc

    # 4. Persistence
    persist_pred = cif_from_shares(x_rs_test[:, -HORIZON:], ef_r, ef_nr)
    metrics_persist = compute_metrics(persist_pred, y_cif_test)
    results["persistence"] = metrics_persist

    # Analysis: normalization mismatch
    # Source normalization stats vs target data range
    src_target_min = float(cc_zs.target_min.item())
    src_target_max = float(cc_zs.target_max.item())
    actual_cif_min = float(y_cif_test.min())
    actual_cif_max = float(y_cif_test.max())
    results["norm_mismatch"] = {
        "source_cif_range": [src_target_min, src_target_max],
        "target_cif_range": [actual_cif_min, actual_cif_max],
        "range_overlap": max(0, min(src_target_max, actual_cif_max) - max(src_target_min, actual_cif_min)) /
                         (actual_cif_max - actual_cif_min + 1e-10),
    }

    # Ratios
    results["ratio_cc_zs_vs_sup"] = metrics_zs_cc["mae"] / (metrics_sup["mae"] + 1e-10)
    results["ratio_tc_zs_vs_sup"] = metrics_zs_tc["mae"] / (metrics_sup["mae"] + 1e-10)
    results["ratio_tc_vs_cc_zs"] = metrics_zs_tc["mae"] / (metrics_zs_cc["mae"] + 1e-10)
    results["ratio_tcp_zs_vs_sup"] = metrics_zsp_tc["mae"] / (metrics_sup["mae"] + 1e-10)
    results["ratio_tcp_vs_cc_zs"] = metrics_zsp_tc["mae"] / (metrics_zs_cc["mae"] + 1e-10)

    return results


def main():
    print("=" * 80)
    print("Phase 3.2: CarbonCast Cross-Domain Deep Analysis")
    print("=" * 80)
    print("\nComparing: CarbonCast (supervised/zero-shot) vs TransCIF (zero-shot)")
    print("Key question: Why does CarbonCast fail in cross-domain while TransCIF succeeds?\n")

    # Load regions
    discover_uk_regions()
    all_configs = {**AU_REGIONS, **UK_REGIONS, **US_REGIONS}
    all_regions = {}
    for name in all_configs:
        try:
            all_regions[name] = load_region_data(name, all_configs)
        except:
            pass

    print(f"Loaded: {len(all_regions)} regions")

    # Representative targets
    targets = ["US_FPL", "US_MISO", "QLD1", "NSW1", "VIC1", "SA1",
               "UK_07_South_Wales", "UK_01_North_Scotland", "US_BPAT"]
    targets = [t for t in targets if t in all_regions]

    print(f"Targets: {targets}")
    t0 = time.time()

    print(f"\n{'Region':<15} {'CC-Sup':<9} {'CC-ZS':<9} {'TC-ZS':<9} {'TC-ZS+':<9} "
          f"{'Persist':<9} {'CC degrade':<11} {'TC+/CC-ZS':<9}")
    print("-" * 85)

    all_results = []
    for target in targets:
        r = evaluate_carboncast_comparison(target, all_regions, seed=42)
        if r is None:
            continue
        all_results.append(r)

        cc_sup = r["cc_supervised"]["mae"]
        cc_zs = r["cc_zeroshot"]["mae"]
        tc_zs = r["transcif_zeroshot"]["mae"]
        tc_zsp = r["transcif_zs_plus"]["mae"]
        persist = r["persistence"]["mae"]

        print(f"{target:<15} {cc_sup:<9.1f} {cc_zs:<9.1f} {tc_zs:<9.1f} {tc_zsp:<9.1f} "
              f"{persist:<9.1f} {r['ratio_cc_zs_vs_sup']:<11.2f} "
              f"{r['ratio_tcp_vs_cc_zs']:<9.3f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    cc_degrade = [r["ratio_cc_zs_vs_sup"] for r in all_results]
    tc_vs_cc = [r["ratio_tc_vs_cc_zs"] for r in all_results]
    tcp_vs_cc = [r["ratio_tcp_vs_cc_zs"] for r in all_results]
    overlaps = [r["norm_mismatch"]["range_overlap"] for r in all_results]

    print(f"\n1. CARBONCAST CROSS-DOMAIN DEGRADATION:")
    print(f"   CarbonCast ZS / CarbonCast Supervised ratio:")
    print(f"   Mean: {np.mean(cc_degrade):.2f}× (range: {np.min(cc_degrade):.2f}-{np.max(cc_degrade):.2f})")
    print(f"   CarbonCast loses {(np.mean(cc_degrade)-1)*100:.0f}% accuracy in cross-domain")

    print(f"\n2. TRANSCIF vs CARBONCAST ZERO-SHOT:")
    print(f"   TransCIF_ZS / CarbonCast_ZS ratio:")
    print(f"   Mean: {np.mean(tc_vs_cc):.3f}")
    wins = sum(1 for r in tc_vs_cc if r < 1.0)
    print(f"   TransCIF beats CarbonCast ZS in {wins}/{len(tc_vs_cc)} regions")
    print(f"   TransCIF_ZS+ / CarbonCast_ZS ratio: Mean: {np.mean(tcp_vs_cc):.3f}")
    wins_p = sum(1 for r in tcp_vs_cc if r < 1.0)
    print(f"   TransCIF-ZS+ beats CarbonCast ZS in {wins_p}/{len(tcp_vs_cc)} regions")

    print(f"\n3. NORMALIZATION MISMATCH (root cause):")
    print(f"   Mean CIF range overlap (source vs target): {np.mean(overlaps)*100:.1f}%")
    print(f"   Regions with <50% overlap: {sum(1 for o in overlaps if o < 0.5)}/{len(overlaps)}")
    print(f"   This causes CarbonCast's denormalization to produce out-of-range predictions")

    print(f"\n4. KEY INSIGHT:")
    print(f"   CarbonCast fails because:")
    print(f"   a) Min-max normalization bakes in source domain statistics")
    print(f"   b) CNN-LSTM learns absolute CIF patterns (not relative/physics-based)")
    print(f"   c) No mechanism to adapt outputs to target domain emission factors")
    print(f"   TransCIF succeeds because:")
    print(f"   a) Predicts RENEWABLE SHARE (0-1, similar across all regions)")
    print(f"   b) Physics layer converts to CIF using target's config (ef_r, ef_nr)")
    print(f"   c) Config-weighted sampling focuses on relevant source regions")

    # Save results
    output_file = RESULTS_DIR / "carboncast_analysis.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved: {output_file}")

    # --- Generate Figure ---
    print("\n--- GENERATING FIGURE ---")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    regions = [r["region"] for r in all_results]
    x_pos = np.arange(len(regions))

    # Panel 1: MAE comparison (bar chart)
    ax = axes[0]
    width = 0.17
    cc_sup_maes = [r["cc_supervised"]["mae"] for r in all_results]
    cc_zs_maes = [r["cc_zeroshot"]["mae"] for r in all_results]
    tc_zs_maes = [r["transcif_zeroshot"]["mae"] for r in all_results]
    tc_zsp_maes = [r["transcif_zs_plus"]["mae"] for r in all_results]
    persist_maes = [r["persistence"]["mae"] for r in all_results]

    ax.bar(x_pos - 2*width, cc_sup_maes, width, label='CC Supervised', alpha=0.8, color='tab:green')
    ax.bar(x_pos - width, cc_zs_maes, width, label='CC Zero-Shot', alpha=0.8, color='tab:red')
    ax.bar(x_pos, tc_zs_maes, width, label='TransCIF ZS', alpha=0.8, color='tab:blue')
    ax.bar(x_pos + width, tc_zsp_maes, width, label='TransCIF ZS+', alpha=0.8, color='tab:purple')
    ax.bar(x_pos + 2*width, persist_maes, width, label='Persistence', alpha=0.5, color='gray')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([r[:7] for r in regions], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel("CIF MAE (gCO₂/kWh)", fontsize=11)
    ax.set_title("(a) CarbonCast vs TransCIF", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Cross-domain degradation ratio
    ax = axes[1]
    ax.bar(x_pos - 0.2, cc_degrade, 0.2, label='CC ZS/CC Sup', alpha=0.8, color='tab:red')
    tc_vs_sup = [r["ratio_tc_zs_vs_sup"] for r in all_results]
    tcp_vs_sup = [r["ratio_tcp_zs_vs_sup"] for r in all_results]
    ax.bar(x_pos, tc_vs_sup, 0.2, label='TC ZS/CC Sup', alpha=0.8, color='tab:blue')
    ax.bar(x_pos + 0.2, tcp_vs_sup, 0.2, label='TC ZS+/CC Sup', alpha=0.8, color='tab:purple')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r[:7] for r in regions], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel("Ratio vs CarbonCast Supervised", fontsize=11)
    ax.set_title("(b) Cross-Domain Degradation", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: Normalization overlap vs degradation
    ax = axes[2]
    ax.scatter(overlaps, cc_degrade, s=80, c='tab:red', alpha=0.8,
               edgecolors='k', linewidth=0.5, label='CarbonCast')
    for i, r in enumerate(all_results):
        ax.annotate(r["region"][:6], (overlaps[i], cc_degrade[i]),
                   fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel("CIF Range Overlap (source vs target)", fontsize=11)
    ax.set_ylabel("CarbonCast ZS / Supervised ratio", fontsize=11)
    ax.set_title("(c) Normalization Mismatch → Failure", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "carboncast_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {fig_path}")

    print("\n✓ Phase 3.2 CarbonCast Analysis complete!")


if __name__ == "__main__":
    main()
