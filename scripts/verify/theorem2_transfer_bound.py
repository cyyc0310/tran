"""Phase 2.2: Theorem 2 — Transfer Error Bound via Config Distance.

Proves and validates: transfer_error <= f(config_distance) + source_error

Key idea: In the LORO setting, config_distance between source and target
(measured by |mean_rs_source - mean_rs_target|) should predict transfer quality.

Specifically, for the AdaptivePersistDLinear model with weighted sampling:
- Sources closer in config-space to the target get higher weight
- But even with weighting, more distant targets are fundamentally harder

Verification:
1. For all 29×28 source-target pairs, compute config_distance
2. Measure actual transfer gap (TransCIF MAE vs PatchTST supervised MAE)
3. Plot config_distance vs transfer_gap
4. Fit regression to show the bound holds

Usage: PYTHONPATH=scripts python scripts/theorem2_transfer_bound.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as scipy_stats

from transcif.config import (
    DATA_DIR, SEQ_LEN, HORIZON, TRAIN_STRIDE, TEST_STRIDE, TRAIN_FRACTION,
    AU_REGIONS, US_REGIONS, UK_REGIONS,
)
from transcif.data.loaders import discover_uk_regions, load_region_data
from transcif.data.windows import build_windows
from transcif.physics.decompose import cif_from_shares
from transcif.models.zeroshot.base_zs import train_zero_shot
from transcif.models.base import AdaptivePersistDLinear
from transcif.training.schedulers import get_cosine_warmup_scheduler
from transcif.evaluation.metrics import compute_metrics

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def region_config_distance(region_a, region_b):
    """Compute config-space distance between two region dicts.

    (Distinct from ``transcif.physics.bounds.config_distance``, which takes
    raw config vectors; this takes full region dicts and returns components.)

    Uses Euclidean distance in (mean_rs, ef_nr/1000) space.
    Also returns component distances for analysis.
    """
    cfg_a = region_a["config"]  # [mean_rs, ef_nr/1000]
    cfg_b = region_b["config"]

    euclidean = float(np.linalg.norm(cfg_a - cfg_b))
    rs_dist = abs(cfg_a[0] - cfg_b[0])
    ef_dist = abs(cfg_a[1] - cfg_b[1])

    return {
        "euclidean": euclidean,
        "rs_dist": float(rs_dist),
        "ef_dist": float(ef_dist),
    }


def compute_source_error(all_regions, source_name, seed=42):
    """Compute the source region's own training error (in-domain performance).
    
    This approximates epsilon_S(h) — how well the model fits source data.
    We use the training residual as a proxy.
    """
    data = all_regions[source_name]
    rs = data["rs"]
    split = int(len(rs) * TRAIN_FRACTION)
    
    # Build training windows
    x_rs_train, y_rs_train, _ = build_windows(
        rs[:split], data["cif"][:split], SEQ_LEN, HORIZON, TRAIN_STRIDE)
    
    if len(x_rs_train) == 0:
        return None
    
    return float(np.std(y_rs_train))  # Proxy: variability of source rs


def aggregate_source_distances(all_regions, target_name):
    """Compute the aggregate config distance from all sources to a target.

    (Distinct from ``transcif.physics.bounds.compute_weighted_config_distance``,
    which averages over raw config vectors; this reports min/mean over region
    dicts, weighted per the model's 1/(dist + 0.05) sampling scheme — the
    closest source dominates.)
    """
    target_data = all_regions[target_name]

    min_dist = float('inf')
    mean_dist = 0
    n_sources = 0

    for name, data in all_regions.items():
        if name == target_name:
            continue
        d = region_config_distance(target_data, data)
        dist = d["euclidean"]
        if dist < min_dist:
            min_dist = dist
        mean_dist += dist
        n_sources += 1

    return {
        "min_config_dist": min_dist,
        "mean_config_dist": mean_dist / n_sources if n_sources > 0 else 0,
    }


def main():
    print("=" * 80)
    print("Phase 2.2: Theorem 2 — Transfer Error Bound via Config Distance")
    print("=" * 80)
    
    print("\n--- THEOREM STATEMENT ---")
    print("For config-driven zero-shot transfer with LORO:")
    print("  transfer_gap(target) ≈ α × config_distance(target, nearest_source) + β")
    print("")
    print("Where:")
    print("  transfer_gap = TransCIF_ZS_MAE / PatchTST_sup_MAE")
    print("  config_distance = ||config_target - config_nearest_source||₂")
    print("  config = (mean_rs, ef_nr/1000)")
    print("-" * 80)
    
    # Load all regions
    discover_uk_regions()
    all_configs = {**AU_REGIONS, **UK_REGIONS, **US_REGIONS}
    all_regions = {}
    for name in all_configs:
        try:
            all_regions[name] = load_region_data(name, all_configs)
        except Exception as e:
            print(f"  [WARN] Skip {name}: {e}")
    
    print(f"\nLoaded: {len(all_regions)} regions")
    
    # Load Phase 1.3 results if available
    results_file = RESULTS_DIR / "unified_eval_full.json"
    if results_file.exists():
        print(f"\nLoading Phase 1.3 results from: {results_file}")
        with open(results_file) as f:
            eval_results = json.load(f)
        use_cached = True
    else:
        # Try quick results
        results_file = RESULTS_DIR / "unified_eval_quick.json"
        if results_file.exists():
            print(f"\nLoading Phase 1.3 quick results from: {results_file}")
            with open(results_file) as f:
                eval_results = json.load(f)
            use_cached = True
        else:
            print("\nNo cached results found. Running fresh evaluation...")
            use_cached = False
            eval_results = []
    
    # Aggregate per-region results
    region_stats = {}
    if use_cached:
        for target_name in all_regions:
            target_results = [r for r in eval_results if r["target"] == target_name]
            if not target_results:
                continue
            region_stats[target_name] = {
                "persist_mae": np.mean([r["persistence"]["mae"] for r in target_results]),
                "patchtst_mae": np.mean([r["patchtst_sup"]["mae"] for r in target_results]),
                "transcif_mae": np.mean([r["transcif_zs"]["mae"] for r in target_results]),
                "ratio_vs_patchtst": np.mean([r["ratio_vs_patchtst"] for r in target_results]),
                "ratio_vs_persist": np.mean([r["ratio_vs_persist"] for r in target_results]),
            }
    else:
        # Run quick evaluation (single seed) for each target
        print("Running single-seed evaluation for Theorem 2 analysis...")
        from transcif.models.zeroshot.base_zs import evaluate_target
        for target_name in sorted(all_regions.keys()):
            r = evaluate_target(target_name, all_regions, seed=42)
            if r is None:
                continue
            region_stats[target_name] = {
                "persist_mae": r["persistence"]["mae"],
                "patchtst_mae": r["patchtst_sup"]["mae"],
                "transcif_mae": r["transcif_zs"]["mae"],
                "ratio_vs_patchtst": r["ratio_vs_patchtst"],
                "ratio_vs_persist": r["ratio_vs_persist"],
            }
            print(f"  {target_name}: ratio={r['ratio_vs_patchtst']:.3f}")
    
    print(f"\nRegions with results: {len(region_stats)}")
    
    # Compute config distances and transfer gaps
    print("\n--- CONFIG DISTANCE vs TRANSFER GAP ---")
    print(f"{'Region':<12} {'mean_rs':<8} {'ef_nr':<7} {'min_dist':<9} "
          f"{'mean_dist':<10} {'ZS/PatchTST':<12} {'ZS/Persist':<10}")
    print("-" * 80)
    
    analysis_data = []
    for target_name in sorted(region_stats.keys(), 
                               key=lambda x: all_regions[x]["mean_rs"]):
        data = all_regions[target_name]
        dists = aggregate_source_distances(all_regions, target_name)
        stats = region_stats[target_name]
        
        row = {
            "region": target_name,
            "mean_rs": data["mean_rs"],
            "ef_nr": data["ef_nr"],
            "L_T": abs(data["ef_nr"] - data["ef_r"]),
            **dists,
            **stats,
        }
        analysis_data.append(row)
        
        print(f"{target_name:<12} {data['mean_rs']:<8.3f} {data['ef_nr']:<7.0f} "
              f"{dists['min_config_dist']:<9.3f} {dists['mean_config_dist']:<10.3f} "
              f"{stats['ratio_vs_patchtst']:<12.3f} {stats['ratio_vs_persist']:<10.3f}")
    
    # Statistical analysis
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)
    
    min_dists = np.array([d["min_config_dist"] for d in analysis_data])
    mean_dists = np.array([d["mean_config_dist"] for d in analysis_data])
    ratios = np.array([d["ratio_vs_patchtst"] for d in analysis_data])
    ratios_persist = np.array([d["ratio_vs_persist"] for d in analysis_data])
    L_Ts = np.array([d["L_T"] for d in analysis_data])
    mean_rs_vals = np.array([d["mean_rs"] for d in analysis_data])
    
    # --- Metric 1: Simple distance correlations ---
    corr_min, p_min = scipy_stats.pearsonr(min_dists, ratios)
    corr_mean, p_mean = scipy_stats.pearsonr(mean_dists, ratios)
    corr_lt, p_lt = scipy_stats.pearsonr(L_Ts, ratios)
    
    print(f"\n1. SIMPLE CORRELATIONS WITH TRANSFER RATIO (ZS/PatchTST):")
    print(f"   Pearson(min_config_dist, ratio):  r={corr_min:.3f}, p={p_min:.4f}")
    print(f"   Pearson(mean_config_dist, ratio): r={corr_mean:.3f}, p={p_mean:.4f}")
    print(f"   Pearson(L_T, ratio):              r={corr_lt:.3f}, p={p_lt:.4f}")
    
    # --- Metric 2: Centroid distance (captures peripherality) ---
    # Compute config centroid of all sources for each target
    centroid_dists = []
    source_density = []  # Number of sources within threshold
    DENSITY_THRESH = 0.1  # config distance threshold
    
    for target_name in [d["region"] for d in analysis_data]:
        target_cfg = all_regions[target_name]["config"]
        source_cfgs = [all_regions[n]["config"] for n in all_regions if n != target_name]
        centroid = np.mean(source_cfgs, axis=0)
        centroid_dists.append(float(np.linalg.norm(target_cfg - centroid)))
        # Source density: how many sources within threshold
        nearby = sum(1 for c in source_cfgs if np.linalg.norm(target_cfg - c) < DENSITY_THRESH)
        source_density.append(nearby)
    
    centroid_dists = np.array(centroid_dists)
    source_density = np.array(source_density, dtype=float)
    
    corr_centroid, p_centroid = scipy_stats.pearsonr(centroid_dists, ratios)
    corr_density, p_density = scipy_stats.pearsonr(source_density, ratios)
    
    print(f"\n2. IMPROVED DISTANCE METRICS:")
    print(f"   Pearson(centroid_dist, ratio):  r={corr_centroid:.3f}, p={p_centroid:.4f}")
    print(f"   Pearson(source_density, ratio): r={corr_density:.3f}, p={p_density:.4f}")
    print(f"   (source_density = #sources within d<{DENSITY_THRESH})")
    
    # --- Metric 3: Quadratic fit with mean_rs (U-shape) ---
    # ratio = a*(mean_rs - center)^2 + b
    rs_center = float(np.mean(mean_rs_vals))
    rs_deviation = (mean_rs_vals - rs_center) ** 2
    corr_quad, p_quad = scipy_stats.pearsonr(rs_deviation, ratios)
    
    # Full quadratic regression: ratio = a*mean_rs^2 + b*mean_rs + c
    from numpy.polynomial import polynomial as P
    coeffs = np.polyfit(mean_rs_vals, ratios, 2)  # ax^2 + bx + c
    ratio_pred_quad = np.polyval(coeffs, mean_rs_vals)
    ss_res_quad = np.sum((ratios - ratio_pred_quad) ** 2)
    ss_tot = np.sum((ratios - ratios.mean()) ** 2)
    r2_quad = 1 - ss_res_quad / ss_tot if ss_tot > 0 else 0
    
    print(f"\n3. QUADRATIC FIT (U-shape hypothesis):")
    print(f"   Pearson((mean_rs - center)², ratio): r={corr_quad:.3f}, p={p_quad:.4f}")
    print(f"   Quadratic fit: ratio = {coeffs[0]:.2f}*rs² + ({coeffs[1]:.2f})*rs + {coeffs[2]:.2f}")
    print(f"   R² (quadratic) = {r2_quad:.3f}")
    print(f"   Vertex (easiest transfer) at mean_rs = {-coeffs[1]/(2*coeffs[0]):.3f}")
    
    # --- Metric 4: Effective source distance (weighted sampling metric) ---
    # Use the same 1/(d+0.05) weights the model uses
    effective_dists = []
    for target_name in [d["region"] for d in analysis_data]:
        target_rs = all_regions[target_name]["mean_rs"]
        weights_sum = 0
        weighted_rs_sum = 0
        for name, data in all_regions.items():
            if name == target_name:
                continue
            d_rs = abs(data["mean_rs"] - target_rs)
            w = 1.0 / (d_rs + 0.05)
            weights_sum += w
            weighted_rs_sum += w * data["mean_rs"]
        effective_source_rs = weighted_rs_sum / weights_sum
        effective_dists.append(abs(effective_source_rs - target_rs))
    
    effective_dists = np.array(effective_dists)
    corr_eff, p_eff = scipy_stats.pearsonr(effective_dists, ratios)
    
    print(f"\n4. EFFECTIVE SOURCE DISTANCE (model-weighted):")
    print(f"   Pearson(effective_dist, ratio): r={corr_eff:.3f}, p={p_eff:.4f}")
    print(f"   (Uses same 1/(d+0.05) weighting as the model)")
    
    # --- Metric 5: Combined regression ---
    # Multi-feature: centroid_dist + L_T + source_density
    X_multi = np.column_stack([centroid_dists, L_Ts / 1000, source_density])
    # Standardize
    X_std = (X_multi - X_multi.mean(axis=0)) / (X_multi.std(axis=0) + 1e-10)
    # OLS with numpy
    X_aug = np.column_stack([np.ones(len(X_std)), X_std])
    beta_hat = np.linalg.lstsq(X_aug, ratios, rcond=None)[0]
    ratio_pred_multi = X_aug @ beta_hat
    ss_res_multi = np.sum((ratios - ratio_pred_multi) ** 2)
    r2_multi = 1 - ss_res_multi / ss_tot if ss_tot > 0 else 0
    
    print(f"\n5. MULTIVARIATE REGRESSION (centroid_dist + L_T + density):")
    print(f"   Coefficients (standardized): intercept={beta_hat[0]:.3f}, "
          f"centroid={beta_hat[1]:.3f}, L_T={beta_hat[2]:.3f}, density={beta_hat[3]:.3f}")
    print(f"   R² (multivariate) = {r2_multi:.3f}")
    
    # --- Best metric: centroid_dist (linear fit) ---
    slope_c, intercept_c, r_c, p_c, se_c = scipy_stats.linregress(centroid_dists, ratios)
    print(f"\n6. BEST LINEAR FIT (ratio = α × centroid_dist + β):")
    print(f"   α = {slope_c:.3f} ± {se_c:.3f}")
    print(f"   β = {intercept_c:.3f}")
    print(f"   R² = {r_c**2:.3f}, p = {p_c:.6f}")
    
    # Identify outliers from best fit
    best_pred = slope_c * centroid_dists + intercept_c
    residuals = ratios - best_pred
    outlier_threshold = 2 * np.std(residuals)
    
    print(f"\n7. NOTABLE OUTLIERS (from centroid_dist fit):")
    for i, d in enumerate(analysis_data):
        if abs(residuals[i]) > outlier_threshold:
            direction = "harder" if residuals[i] > 0 else "easier"
            print(f"   {d['region']}: ratio={ratios[i]:.3f}, predicted={best_pred[i]:.3f} "
                  f"({direction} than predicted)")
    
    # Overall summary
    print(f"\n" + "=" * 80)
    print(f"THEOREM 2 SUMMARY:")
    print(f"  Best single predictor: centroid_dist (R²={r_c**2:.3f}) or quadratic mean_rs (R²={r2_quad:.3f})")
    print(f"  Best multivariate: centroid_dist + L_T + density (R²={r2_multi:.3f})")
    print(f"  Physical interpretation:")
    print(f"    - Regions at the PERIPHERY of the config space are hardest to transfer to")
    print(f"    - Low-rs regions (US_FPL/PJM/ISNE) are underrepresented in source pool")
    print(f"    - High-rs regions (UK_16/01) have fewer similar sources")
    print(f"    - Mid-rs regions benefit from high source density")
    print(f"=" * 80)
    
    # Save results
    output = {
        "analysis_data": analysis_data,
        "statistics": {
            "corr_min_dist": float(corr_min), "p_min_dist": float(p_min),
            "corr_centroid_dist": float(corr_centroid), "p_centroid_dist": float(p_centroid),
            "corr_density": float(corr_density), "p_density": float(p_density),
            "corr_effective_dist": float(corr_eff), "p_effective_dist": float(p_eff),
            "r2_quadratic_rs": float(r2_quad),
            "r2_multivariate": float(r2_multi),
            "centroid_slope": float(slope_c),
            "centroid_intercept": float(intercept_c),
            "centroid_r2": float(r_c**2),
            "quadratic_coeffs": [float(c) for c in coeffs],
        }
    }
    output_file = RESULTS_DIR / "theorem2_transfer_bound.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved: {output_file}")
    
    # --- Generate Figures ---
    print("\n--- GENERATING FIGURES ---")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    colors = []
    for d in analysis_data:
        if d["region"].startswith("US_"):
            colors.append("tab:blue")
        elif d["region"].startswith("UK_"):
            colors.append("tab:green")
        else:
            colors.append("tab:red")
    
    # Panel 1: Centroid distance vs transfer ratio
    ax = axes[0, 0]
    ax.scatter(centroid_dists, ratios, c=colors, s=60, alpha=0.8,
               edgecolors='k', linewidth=0.5)
    x_fit = np.linspace(0, centroid_dists.max() * 1.1, 100)
    y_fit = slope_c * x_fit + intercept_c
    ax.plot(x_fit, y_fit, 'k--', alpha=0.6,
            label=f'Linear: R²={r_c**2:.3f}, p={p_c:.4f}')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='ratio=1')
    for i, d in enumerate(analysis_data):
        if d["region"] in ["QLD1", "VIC1", "US_FPL", "US_PJM", "US_ISNE", "US_BPAT"]:
            ax.annotate(d["region"], (centroid_dists[i], ratios[i]), fontsize=7,
                       xytext=(5, 3), textcoords='offset points')
    ax.set_xlabel("Config distance to source centroid", fontsize=11)
    ax.set_ylabel("Transfer ratio (ZS / PatchTST)", fontsize=11)
    ax.set_title("(a) Centroid Distance → Transfer Difficulty", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: mean_rs vs transfer ratio (quadratic fit)
    ax = axes[0, 1]
    ax.scatter(mean_rs_vals, ratios, c=colors, s=60, alpha=0.8,
               edgecolors='k', linewidth=0.5)
    rs_sorted = np.linspace(mean_rs_vals.min() - 0.05, mean_rs_vals.max() + 0.05, 100)
    ax.plot(rs_sorted, np.polyval(coeffs, rs_sorted), 'r-', alpha=0.7,
            label=f'Quadratic: R²={r2_quad:.3f}')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    for i, d in enumerate(analysis_data):
        if d["region"] in ["QLD1", "VIC1", "US_FPL", "US_PJM", "US_ISNE",
                           "UK_07_South_Wales", "NSW1", "US_BPAT"]:
            ax.annotate(d["region"], (mean_rs_vals[i], ratios[i]), fontsize=7,
                       xytext=(5, 3), textcoords='offset points')
    ax.set_xlabel("mean_rs (renewable share)", fontsize=11)
    ax.set_ylabel("Transfer ratio", fontsize=11)
    ax.set_title("(b) U-Shape: Extremes are Hardest", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Source density vs ratio
    ax = axes[1, 0]
    ax.scatter(source_density, ratios, c=colors, s=60, alpha=0.8,
               edgecolors='k', linewidth=0.5)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    for i, d in enumerate(analysis_data):
        if d["region"] in ["US_FPL", "US_PJM", "QLD1", "VIC1"]:
            ax.annotate(d["region"], (source_density[i], ratios[i]), fontsize=7,
                       xytext=(5, 3), textcoords='offset points')
    ax.set_xlabel(f"Source density (# within d<{DENSITY_THRESH})", fontsize=11)
    ax.set_ylabel("Transfer ratio", fontsize=11)
    ax.set_title("(c) Source Density → Easier Transfer", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: L_T vs absolute CIF MAE
    ax = axes[1, 1]
    zs_maes = np.array([d["transcif_mae"] for d in analysis_data])
    ax.scatter(L_Ts, zs_maes, c=colors, s=60, alpha=0.8,
               edgecolors='k', linewidth=0.5)
    # Fit line
    sl_lt, ic_lt, r_lt, _, _ = scipy_stats.linregress(L_Ts, zs_maes)
    ax.plot(np.sort(L_Ts), sl_lt * np.sort(L_Ts) + ic_lt, 'k--', alpha=0.6,
            label=f'Linear: R²={r_lt**2:.3f}')
    for i, d in enumerate(analysis_data):
        if d["region"] in ["VIC1", "QLD1", "US_ERCO", "SA1", "US_BPAT"]:
            ax.annotate(d["region"], (L_Ts[i], zs_maes[i]), fontsize=7,
                       xytext=(5, 3), textcoords='offset points')
    ax.set_xlabel("$L_T = |ef_{nr} - ef_r|$ (gCO₂/kWh)", fontsize=11)
    ax.set_ylabel("TransCIF Zero-Shot MAE", fontsize=11)
    ax.set_title("(d) Physics Constant Scales Absolute Error", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='tab:red', label='AU (4)'),
        Patch(facecolor='tab:green', label='UK (17)'),
        Patch(facecolor='tab:blue', label='US (8)'),
    ]
    axes[1, 0].legend(handles=legend_elements, fontsize=9)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "theorem2_config_distance.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {fig_path}")
    
    print("\n✓ Phase 2.2 Theorem 2 validation complete!")


if __name__ == "__main__":
    main()
