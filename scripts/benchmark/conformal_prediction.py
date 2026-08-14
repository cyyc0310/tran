"""Phase 3.1: Probabilistic Extension via Conformal Prediction.

Adds calibrated prediction intervals to TransCIF zero-shot predictions.
Key guarantee: coverage >= (1 - alpha) with finite-sample validity.

Reports:
- CRPS (Continuous Ranked Probability Score)
- Coverage at 90% and 95% nominal levels
- Mean interval width (narrower = more informative)
- Reliability diagram (coverage vs nominal level)

Approach:
1. Train zero-shot model (same as Phase 1.3)
2. Split test set: first 50% = calibration, last 50% = evaluation
3. Compute nonconformity scores on calibration set
4. Apply split-conformal intervals to evaluation set
5. Verify coverage guarantee holds

Additionally supports:
- Per-horizon conformal (tighter intervals)
- Adaptive Online Conformal (AOC) – P2 from IMPROVEMENT_PLAN.md
- State-Conditioned Conformal – per-regime calibration

Usage: python scripts/conformal_prediction.py [--quick]
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from transcif.config import (
    DATA_DIR, RESULTS_DIR, SEQ_LEN, HORIZON, TRAIN_STRIDE, TEST_STRIDE, TRAIN_FRACTION,
    AU_REGIONS, US_REGIONS, UK_REGIONS,
)
from transcif.data.loaders import discover_uk_regions, load_region_data
from transcif.data.windows import build_windows
from transcif.physics.decompose import cif_from_shares
from transcif.models.zeroshot.base_zs import (
    train_zero_shot, compute_metrics, zs_plus_predict,
)
from transcif.calibration.conformal import (
    split_conformal_calibrate,
    split_conformal_calibrate_per_horizon,
    AdaptiveOnlineConformal,
    StateConditionedConformal,
    classify_state,
    compute_crps, compute_coverage, compute_interval_width,
)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# Main evaluation
# --------------------------------------------------------------------------

def evaluate_conformal(target_name, all_regions, seed=42, coverages=None,
                       use_aoc=True):
    """Run conformal prediction for one target region.

    Args:
        target_name : region to evaluate
        all_regions : dict of loaded region data
        seed        : random seed
        coverages   : list of coverage levels (default [0.90, 0.95])
        use_aoc     : if True, also run Adaptive Online Conformal (default True)
    """
    if coverages is None:
        coverages = [0.90, 0.95]

    data = all_regions[target_name]
    rs, cif = data["rs"], data["cif"]
    ef_r, ef_nr = data["ef_r"], data["ef_nr"]

    split = int(len(rs) * TRAIN_FRACTION)

    # Build test windows
    x_rs_test, y_rs_test, y_cif_test = build_windows(
        rs[split - SEQ_LEN:], cif[split - SEQ_LEN:],
        SEQ_LEN, HORIZON, TEST_STRIDE)

    if len(x_rs_test) < 20:
        return None

    # Train zero-shot model
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = train_zero_shot(all_regions, target_name, seed=seed)

    # Raw ZS prediction
    target_cfg = torch.tensor(data["config"]).unsqueeze(0).expand(len(x_rs_test), -1)
    with torch.no_grad():
        rs_pred = model(
            torch.tensor(x_rs_test, dtype=torch.float32), target_cfg).numpy()
    cif_pred_raw = cif_from_shares(rs_pred, ef_r, ef_nr)

    # ZS+ point prediction
    n_off = len(rs) - (split - SEQ_LEN)
    origins = [split + st
               for st in range(0, n_off - SEQ_LEN - HORIZON + 1, TEST_STRIDE)]
    assert len(origins) == len(x_rs_test)
    cif_pred = zs_plus_predict(model, data["config"], rs, cif, ef_r, ef_nr, origins)

    # Split into calibration / evaluation
    n_total = len(cif_pred)
    n_cal = n_total // 2

    cif_pred_cal = cif_pred[:n_cal]
    cif_true_cal = y_cif_test[:n_cal]
    cif_pred_eval = cif_pred[n_cal:]
    cif_true_eval = y_cif_test[n_cal:]

    mae = float(np.abs(cif_pred_eval - cif_true_eval).mean())
    mae_raw = float(np.abs(cif_pred_raw[n_cal:] - cif_true_eval).mean())

    results = {
        "region": target_name,
        "mean_rs": data["mean_rs"],
        "n_calibration": n_cal,
        "n_evaluation": n_total - n_cal,
        "point_mae": mae,
        "point_mae_raw_zs": mae_raw,
    }

    # --- Split conformal (global & per-horizon) ---
    for cov in coverages:
        hw_global, cal_scores = split_conformal_calibrate(
            cif_true_cal, cif_pred_cal, coverage=cov)
        hw_per_h = split_conformal_calibrate_per_horizon(
            cif_true_cal, cif_pred_cal, coverage=cov)

        coverage_global = compute_coverage(cif_true_eval, cif_pred_eval, hw_global)
        coverage_per_h = compute_coverage(cif_true_eval, cif_pred_eval, hw_per_h)
        width_global = compute_interval_width(hw_global)
        width_per_h = compute_interval_width(hw_per_h)
        crps_global = compute_crps(cif_true_eval, cif_pred_eval, hw_global)

        cv = f"{int(cov*100)}"
        results[f"coverage_{cv}_global"] = coverage_global
        results[f"coverage_{cv}_per_h"] = coverage_per_h
        results[f"width_{cv}_global"] = width_global
        results[f"width_{cv}_per_h"] = width_per_h
        results[f"halfwidth_{cv}_global"] = hw_global
        results[f"halfwidth_{cv}_per_h"] = hw_per_h.tolist() if isinstance(hw_per_h, np.ndarray) else hw_per_h
        results[f"crps_{cv}"] = crps_global

    # --- Adaptive Online Conformal (AOC) ---
    if use_aoc:
        aoc = AdaptiveOnlineConformal(horizon=HORIZON, coverage=0.90, gamma=0.005)
        aoc_covs, aoc_widths = [], []
        for i in range(len(cif_pred_eval)):
            aoc.update(cif_pred_eval[i], cif_true_eval[i])
            if i >= 20:  # warm-up
                aoc_cov = compute_coverage(
                    cif_true_eval[max(0, i-48):i+1],
                    cif_pred_eval[max(0, i-48):i+1],
                    aoc.get_halfwidths())
                aoc_covs.append(aoc_cov)
                aoc_widths.append(compute_interval_width(aoc.get_halfwidths()))
        if aoc_covs:
            results["aoc_final_coverage"] = aoc_covs[-1]
            results["aoc_final_width"] = aoc_widths[-1]
            results["aoc_mean_coverage"] = float(np.mean(aoc_covs))
            results["aoc_mean_width"] = float(np.mean(aoc_widths))

    # --- State-Conditioned Conformal ---
    state_labels_cal = []
    for i in range(n_cal):
        # Use the rs input window to classify state
        assert i + n_cal <= len(x_rs_test)  # Use full rs for cal
    # Use the corresponding input windows for state classification
    for i in range(len(x_rs_test)):
        if i < n_cal:
            state_labels_cal.append(
                classify_state(data["config"], x_rs_test[i]))

    if len(set(state_labels_cal)) >= 2:
        scc = StateConditionedConformal(horizon=HORIZON, coverage=0.90)
        scc.calibrate(cif_pred_cal, cif_true_cal,
                      np.array(state_labels_cal))

        state_labels_eval = []
        for i in range(n_cal, min(n_cal + len(x_rs_test), len(x_rs_test))):
            j = i - n_cal
            if j < len(cif_pred_eval):
                state_labels_eval.append(
                    classify_state(data["config"],
                                  x_rs_test[min(i, len(x_rs_test)-1)]))
        state_labels_eval = state_labels_eval[:len(cif_pred_eval)]

        scc_coverage = 0.0
        scc_width = 0.0
        n_scc = 0
        for i, sl in enumerate(state_labels_eval):
            lower, upper = scc.predict_interval(cif_pred_eval[i], sl)
            covered = np.mean((cif_true_eval[i] >= lower) &
                            (cif_true_eval[i] <= upper))
            scc_coverage += covered
            scc_width += np.mean(upper - lower)
            n_scc += 1
        if n_scc > 0:
            results["scc_coverage"] = scc_coverage / n_scc
            results["scc_width"] = scc_width / n_scc

    # Multi-level coverage for reliability diagram
    levels = np.arange(0.05, 1.0, 0.05)
    empirical_coverages = []
    for level in levels:
        hw, _ = split_conformal_calibrate(cif_true_cal, cif_pred_cal, coverage=level)
        ec = compute_coverage(cif_true_eval, cif_pred_eval, hw)
        empirical_coverages.append(ec)
    results["reliability_levels"] = levels.tolist()
    results["reliability_coverages"] = empirical_coverages

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick: AU only, 1 seed")
    parser.add_argument("--aoc", action="store_true", help="Enable Adaptive Online Conformal")
    args = parser.parse_args()

    print("=" * 80)
    tag = "QUICK" if args.quick else "FULL"
    print(f"Phase 3.1: Conformal Prediction Extension ({tag})")
    if args.aoc:
        print("  AOC (Adaptive Online Conformal) enabled")
    print("=" * 80)

    discover_uk_regions()
    all_configs = {**AU_REGIONS, **UK_REGIONS, **US_REGIONS}
    all_regions = {}
    for name in all_configs:
        try:
            all_regions[name] = load_region_data(name, all_configs)
        except Exception:
            pass

    print(f"Loaded: {len(all_regions)} regions")

    if args.quick:
        targets = ["QLD1", "NSW1", "VIC1", "SA1"]
        seeds = [42]
    else:
        targets = sorted(all_regions.keys())
        seeds = [42]

    print(f"Targets: {len(targets)} regions")

    t0 = time.time()
    all_results = []

    print(f"\n{'Region':<15} {'MAE':<8} {'Cov90%':<8} {'Cov95%':<8} "
          f"{'Width90':<9} {'Width95':<9} {'CRPS90':<8}")
    print("-" * 70)

    for target in targets:
        if target not in all_regions:
            continue
        result = evaluate_conformal(target, all_regions, seed=seeds[0],
                                   use_aoc=args.aoc)
        if result is None:
            continue
        all_results.append(result)
        print(f"{target:<15} {result['point_mae']:<8.1f} "
              f"{result['coverage_90_per_h']:<8.3f} "
              f"{result['coverage_95_per_h']:<8.3f} "
              f"{result['width_90_per_h']:<9.1f} "
              f"{result['width_95_per_h']:<9.1f} "
              f"{result['crps_90']:<8.1f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    cov90_vals = [r["coverage_90_per_h"] for r in all_results]
    cov95_vals = [r["coverage_95_per_h"] for r in all_results]
    width90_vals = [r["width_90_per_h"] for r in all_results]
    width95_vals = [r["width_95_per_h"] for r in all_results]
    crps90_vals = [r["crps_90"] for r in all_results]
    mae_vals = [r["point_mae"] for r in all_results]

    print(f"\n90% Conformal Intervals:")
    print(f"  Mean empirical coverage: {np.mean(cov90_vals):.3f} (target: 0.900)")
    print(f"  Min coverage: {np.min(cov90_vals):.3f}")
    print(f"  Regions with coverage ≥ 0.90: "
          f"{sum(c >= 0.90 for c in cov90_vals)}/{len(cov90_vals)}")
    print(f"  Mean interval width: {np.mean(width90_vals):.1f} gCO₂/kWh")
    print(f"  Mean CRPS: {np.mean(crps90_vals):.1f} gCO₂/kWh")

    print(f"\n95% Conformal Intervals:")
    print(f"  Mean empirical coverage: {np.mean(cov95_vals):.3f} (target: 0.950)")
    print(f"  Min coverage: {np.min(cov95_vals):.3f}")
    print(f"  Regions with coverage ≥ 0.95: "
          f"{sum(c >= 0.95 for c in cov95_vals)}/{len(cov95_vals)}")
    print(f"  Mean interval width: {np.mean(width95_vals):.1f} gCO₂/kWh")

    print(f"\nSharpness ratio (width / MAE):")
    sharpness = [w / m for w, m in zip(width90_vals, mae_vals)]
    print(f"  Mean: {np.mean(sharpness):.2f} (lower = more informative)")

    if args.aoc:
        aoc_covs = [r.get("aoc_mean_coverage", 0) for r in all_results
                    if "aoc_mean_coverage" in r]
        aoc_widths = [r.get("aoc_mean_width", 0) for r in all_results
                     if "aoc_mean_width" in r]
        if aoc_covs:
            print(f"\nAOC (Adaptive Online Conformal):")
            print(f"  Mean coverage: {np.mean(aoc_covs):.3f}")
            print(f"  Mean width: {np.mean(aoc_widths):.1f}")

    # Save results
    output_file = RESULTS_DIR / "conformal_prediction.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved: {output_file}")

    # Figures
    print("\n--- GENERATING FIGURES ---")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Reliability diagram
    ax = axes[0, 0]
    all_levels = np.array(all_results[0]["reliability_levels"])
    all_empirical = np.array([r["reliability_coverages"] for r in all_results])
    mean_empirical = all_empirical.mean(axis=0)
    std_empirical = all_empirical.std(axis=0)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.plot(all_levels, mean_empirical, 'b-o', markersize=4, label='TransCIF (mean +/- std)')
    ax.fill_between(all_levels, mean_empirical - std_empirical,
                    mean_empirical + std_empirical, alpha=0.2)
    ax.set_xlabel("Nominal coverage level", fontsize=11)
    ax.set_ylabel("Empirical coverage", fontsize=11)
    ax.set_title("(a) Reliability Diagram", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Panel 2: Coverage by region
    ax = axes[0, 1]
    regions = [r["region"] for r in all_results]
    x_pos = np.arange(len(regions))
    ax.bar(x_pos - 0.15, cov90_vals, 0.3, label='90% intervals', alpha=0.7)
    ax.bar(x_pos + 0.15, cov95_vals, 0.3, label='95% intervals', alpha=0.7)
    ax.axhline(0.90, color='tab:blue', linestyle=':', alpha=0.7)
    ax.axhline(0.95, color='tab:orange', linestyle=':', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r[:6] for r in regions], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel("Empirical coverage", fontsize=11)
    ax.set_title("(b) Coverage by Region", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.7, 1.05)

    # Panel 3: Width vs MAE
    ax = axes[1, 0]
    colors = []
    for r in all_results:
        if r["region"].startswith("US_"):
            colors.append("tab:blue")
        elif r["region"].startswith("UK_"):
            colors.append("tab:green")
        else:
            colors.append("tab:red")
    ax.scatter(mae_vals, width90_vals, c=colors, s=60, alpha=0.8,
               edgecolors='k', linewidth=0.5)
    max_mae = max(mae_vals) * 1.1
    ax.plot([0, max_mae], [0, 2*max_mae], 'k--', alpha=0.4, label='width = 2x MAE')
    ax.plot([0, max_mae], [0, 3*max_mae], 'k:', alpha=0.3, label='width = 3x MAE')
    for i, r in enumerate(all_results):
        if r["region"] in ["VIC1", "QLD1", "US_FPL", "US_BPAT"]:
            ax.annotate(r["region"], (mae_vals[i], width90_vals[i]),
                       fontsize=7, xytext=(5, 3), textcoords='offset points')
    ax.set_xlabel("Point forecast MAE (gCO2/kWh)", fontsize=11)
    ax.set_ylabel("90% interval width (gCO2/kWh)", fontsize=11)
    ax.set_title("(c) Sharpness: Width vs MAE", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Per-horizon intervals
    ax = axes[1, 1]
    example = next((r for r in all_results if r["region"] == "NSW1"), all_results[0])
    hw_90 = np.array(example["halfwidth_90_per_h"])
    hw_95 = np.array(example["halfwidth_95_per_h"])
    hours = np.arange(1, HORIZON + 1)
    ax.fill_between(hours, -hw_95, hw_95, alpha=0.15, color='tab:orange', label='95% interval')
    ax.fill_between(hours, -hw_90, hw_90, alpha=0.3, color='tab:blue', label='90% interval')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel("Forecast horizon (hours)", fontsize=11)
    ax.set_ylabel("Interval half-width (gCO2/kWh)", fontsize=11)
    ax.set_title(f"(d) Per-Horizon Intervals ({example['region']})", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='tab:red', label='AU'),
        Patch(facecolor='tab:green', label='UK'),
        Patch(facecolor='tab:blue', label='US'),
    ]
    axes[1, 0].legend(handles=legend_elements, fontsize=9, loc='upper left')

    plt.tight_layout()
    fig_path = FIGURES_DIR / "conformal_prediction.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {fig_path}")

    print("\nPhase 3.1 Conformal Prediction complete!")


if __name__ == "__main__":
    main()
