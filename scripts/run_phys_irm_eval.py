"""Phys-IRM Experiment: Physics-Informed Invariant Risk Minimization.

Standalone evaluation and ablation script. Compares:
    1. TransCIF-ZS (ERM baseline)
    2. Phys-IRM (1/L_T weighting + IRM penalty)
    3. Phys-IRM (1/L_T weighting only, no IRM penalty) — ablation
    4. ERM with config-distance weighting (current best baseline)
    5. TransCIF-ZS+ (calibrated baseline)

Ablation studies:
    - γ_irm sweep: {0.0, 0.01, 0.05, 0.1, 0.5, 1.0}
    - λ_cif sweep: {0.0, 0.1, 0.5, 1.0}
    - L_T analysis: low-L_T vs high-L_T region performance

Usage:
    python scripts/run_phys_irm_eval.py --quick          # 4 AU, 1 seed
    python scripts/run_phys_irm_eval.py                  # all regions, 3 seeds
    python scripts/run_phys_irm_eval.py --ablation-irm   # γ sweep
    python scripts/run_phys_irm_eval.py --ablation-cif   # λ_cif sweep
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_unified_eval import (
    DATA_DIR, RESULTS_DIR, SEQ_LEN, HORIZON, TRAIN_STRIDE, TEST_STRIDE,
    TRAIN_FRACTION, AU_REGIONS, US_REGIONS, UK_REGIONS,
    ANCHOR_WIN, RESID_WIN, WEEKLY_LAG, SELECT_DAYS, SELECT_MARGIN, FUSION_MENU,
)
from transcif_pipeline import (
    discover_uk_regions, load_region_data, build_windows,
    cif_from_shares, train_zero_shot, compute_metrics, zs_plus_predict,
)
from transcif_model import AdaptivePersistDLinear
from transcif_phys_irm import (
    PhysIRMDLinear, irm_penalty, train_phys_irm, predict_phys_irm,
    compute_L_T, irm_penalty_batched,
)


# ---------------------------------------------------------------------------
# Single-target evaluation
# ---------------------------------------------------------------------------

def evaluate_phys_irm_target(target_name, all_regions, seed=42,
                              gamma_irm=0.1, lambda_cif=0.5):
    """Compare ZS (ERM), Phys-IRM, and Phys-IRM (no IRM penalty)."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    data = all_regions[target_name]
    rs, cif = data["rs"], data["cif"]
    ef_r, ef_nr = data["ef_r"], data["ef_nr"]
    L_T = abs(ef_nr - ef_r)

    split = int(len(rs) * TRAIN_FRACTION)
    x_rs_test, _, y_cif_test = build_windows(
        rs[split - SEQ_LEN:], cif[split - SEQ_LEN:],
        SEQ_LEN, HORIZON, TEST_STRIDE)

    if len(x_rs_test) == 0:
        return None

    result = {
        "target": target_name, "seed": seed, "L_T": L_T,
        "mean_rs": data["mean_rs"], "ef_nr": ef_nr,
    }

    # 1. ERM baseline (standard train_zero_shot)
    zs_model = train_zero_shot(all_regions, target_name, seed=seed)
    cfg_t = torch.tensor(data["config"]).unsqueeze(0).expand(len(x_rs_test), -1)
    with torch.no_grad():
        zs_share = zs_model(torch.tensor(x_rs_test, dtype=torch.float32), cfg_t).numpy()
    zs_cif = cif_from_shares(zs_share, ef_r, ef_nr)
    result["zs_erm"] = compute_metrics(zs_cif, y_cif_test)

    # 2. Phys-IRM (1/L_T weighting + IRM penalty)
    t0 = time.time()
    phys_model, phys_log = train_phys_irm(
        all_regions, target_name, seed=seed,
        gamma_irm=gamma_irm, lambda_cif=lambda_cif)
    phys_cif = predict_phys_irm(phys_model, x_rs_test, data["config"], ef_r, ef_nr)
    result["phys_irm"] = compute_metrics(phys_cif, y_cif_test)
    result["phys_irm_time"] = time.time() - t0

    # 3. Phys-IRM without IRM penalty (only 1/L_T weighting)
    from transcif_phys_irm import train_phys_weighted_only
    pw_model, _ = train_phys_weighted_only(
        all_regions, target_name, seed=seed, lambda_cif=lambda_cif)
    pw_cif = predict_phys_irm(pw_model, x_rs_test, data["config"], ef_r, ef_nr)
    result["phys_weighted"] = compute_metrics(pw_cif, y_cif_test)

    # 4. ZS+ baseline
    origins = [split + st
               for st in range(0, len(cif[split - SEQ_LEN:]) - SEQ_LEN - HORIZON + 1,
                              TEST_STRIDE)]
    zsp_pred = zs_plus_predict(zs_model, data["config"], rs, cif, ef_r, ef_nr, origins)
    result["zs_plus"] = compute_metrics(zsp_pred, y_cif_test)

    # Ratios
    result["ratio_phys_irm_vs_zs"] = (
        result["phys_irm"]["mae"] / max(result["zs_erm"]["mae"], 1e-6))
    result["ratio_phys_weighted_vs_zs"] = (
        result["phys_weighted"]["mae"] / max(result["zs_erm"]["mae"], 1e-6))
    result["irm_benefit"] = (
        result["phys_irm"]["mae"] / max(result["phys_weighted"]["mae"], 1e-6))

    return result


# ---------------------------------------------------------------------------
# Main evaluation loops
# ---------------------------------------------------------------------------

def run_full_eval(all_regions, targets, seeds, gamma_irm=0.1, lambda_cif=0.5):
    """Run Phys-IRM evaluation across all targets and seeds."""
    results = []
    n_total = len(targets) * len(seeds)
    t_start = time.time()

    for i, target in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}] {target} "
              f"(rs={all_regions[target]['mean_rs']:.3f}, "
              f"L_T={abs(all_regions[target]['ef_nr']-all_regions[target]['ef_r']):.0f})",
              flush=True)
        for seed in seeds:
            r = evaluate_phys_irm_target(
                target, all_regions, seed=seed,
                gamma_irm=gamma_irm, lambda_cif=lambda_cif)
            if r is None:
                continue
            results.append(r)
            print(f"  s{seed}: ZS-ERM={r['zs_erm']['mae']:.1f} "
                  f"Phys-IRM={r['phys_irm']['mae']:.1f} "
                  f"(×{r['ratio_phys_irm_vs_zs']:.3f}) "
                  f"Weighted={r['phys_weighted']['mae']:.1f} "
                  f"(×{r['ratio_phys_weighted_vs_zs']:.3f}) "
                  f"IRM+delta={r['irm_benefit']:.3f} "
                  f"ZS+={r['zs_plus']['mae']:.1f}", flush=True)

    elapsed = time.time() - t_start
    _print_summary(results, elapsed, gamma_irm, lambda_cif)
    return results


def _print_summary(results, elapsed, gamma_irm, lambda_cif):
    """Print aggregate statistics."""
    if not results:
        print("\nNo results to summarize.")
        return

    zs_maes = [r["zs_erm"]["mae"] for r in results]
    phys_maes = [r["phys_irm"]["mae"] for r in results]
    pw_maes = [r["phys_weighted"]["mae"] for r in results]
    ratios = [r["ratio_phys_irm_vs_zs"] for r in results]
    pw_ratios = [r["ratio_phys_weighted_vs_zs"] for r in results]

    better_than_zs = sum(1 for r in ratios if r < 1.0)
    better_than_weighted = sum(1 for r in results if r["irm_benefit"] < 1.0)

    print("\n" + "=" * 70)
    print(f"Phys-IRM Summary (γ={gamma_irm}, λ_cif={lambda_cif}, "
          f"{len(set(r['target'] for r in results))} regions, "
          f"{len(results)} evals, {elapsed:.0f}s)")
    print(f"  ZS-ERM median MAE:       {np.median(zs_maes):.1f}")
    print(f"  Phys-IRM median MAE:      {np.median(phys_maes):.1f}  "
          f"(×{np.median(ratios):.3f} vs ZS)")
    print(f"  1/L_T weighted median:    {np.median(pw_maes):.1f}  "
          f"(×{np.median(pw_ratios):.3f} vs ZS)")
    print(f"  Phys-IRM beats ZS-ERM:    {better_than_zs}/{len(results)} "
          f"({100*better_than_zs/max(len(results),1):.0f}%)")
    print(f"  IRM benefit over weighted: {better_than_weighted}/{len(results)} "
          f"({100*better_than_weighted/max(len(results),1):.0f}%)")

    # L_T-stratified analysis
    print("\n  --- L_T-stratified ---")
    high_lt = [r for r in results if r["L_T"] > 500]
    low_lt = [r for r in results if r["L_T"] <= 500]
    for label, group in [("high-L_T (>500)", high_lt), ("low-L_T (≤500)", low_lt)]:
        if group:
            g_ratios = [r["ratio_phys_irm_vs_zs"] for r in group]
            print(f"  {label} (n={len(group)}): "
                  f"ZS={np.median([r['zs_erm']['mae'] for r in group]):.1f}, "
                  f"Phys-IRM={np.median([r['phys_irm']['mae'] for r in group]):.1f}, "
                  f"ratio={np.median(g_ratios):.3f}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Ablation: γ_irm sweep
# ---------------------------------------------------------------------------

def run_ablation_irm(all_regions, target, seed=42, gammas=None):
    """Sweep IRM penalty weight γ on a single target."""
    if gammas is None:
        gammas = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

    data = all_regions[target]
    split = int(len(data["rs"]) * TRAIN_FRACTION)
    x_rs_test, _, y_cif_test = build_windows(
        data["rs"][split - SEQ_LEN:],
        data["cif"][split - SEQ_LEN:],
        SEQ_LEN, HORIZON, TEST_STRIDE)

    print(f"\nγ_irm ablation for {target} (L_T={abs(data['ef_nr']-data['ef_r']):.0f})")
    print("-" * 70)

    results = []
    # ERM baseline
    zs_model = train_zero_shot(all_regions, target, seed=seed)
    cfg_t = torch.tensor(data["config"]).unsqueeze(0).expand(len(x_rs_test), -1)
    with torch.no_grad():
        zs_share = zs_model(torch.tensor(x_rs_test, dtype=torch.float32), cfg_t).numpy()
    zs_cif = cif_from_shares(zs_share, data["ef_r"], data["ef_nr"])
    zs_mae = compute_metrics(zs_cif, y_cif_test)["mae"]

    for g in gammas:
        model, log = train_phys_irm(
            all_regions, target, seed=seed, gamma_irm=g)
        cif_pred = predict_phys_irm(model, x_rs_test, data["config"],
                                    data["ef_r"], data["ef_nr"])
        m = compute_metrics(cif_pred, y_cif_test)
        ratio = m["mae"] / max(zs_mae, 1e-6)
        final_irm = log[-1]["L_irm"] if log else 0
        print(f"  γ={g:<6} MAE={m['mae']:.1f}  ×{ratio:.3f}  "
              f"L_irm_final={final_irm:.6f}")
        results.append({
            "gamma_irm": g, "mae": m["mae"], "ratio_vs_zs": ratio,
            "L_irm_final": final_irm,
        })

    return results


# ---------------------------------------------------------------------------
# Ablation: λ_cif sweep
# ---------------------------------------------------------------------------

def run_ablation_cif(all_regions, target, seed=42, lambdas=None):
    """Sweep CIF supervision weight λ_cif on a single target."""
    if lambdas is None:
        lambdas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

    data = all_regions[target]
    split = int(len(data["rs"]) * TRAIN_FRACTION)
    x_rs_test, _, y_cif_test = build_windows(
        data["rs"][split - SEQ_LEN:],
        data["cif"][split - SEQ_LEN:],
        SEQ_LEN, HORIZON, TEST_STRIDE)

    zs_model = train_zero_shot(all_regions, target, seed=seed)
    cfg_t = torch.tensor(data["config"]).unsqueeze(0).expand(len(x_rs_test), -1)
    with torch.no_grad():
        zs_share = zs_model(torch.tensor(x_rs_test, dtype=torch.float32), cfg_t).numpy()
    zs_cif = cif_from_shares(zs_share, data["ef_r"], data["ef_nr"])
    zs_mae = compute_metrics(zs_cif, y_cif_test)["mae"]

    print(f"\nλ_cif ablation for {target} (L_T={abs(data['ef_nr']-data['ef_r']):.0f})")
    print("-" * 70)

    results = []
    for lam in lambdas:
        model, log = train_phys_irm(
            all_regions, target, seed=seed, lambda_cif=lam)
        cif_pred = predict_phys_irm(model, x_rs_test, data["config"],
                                    data["ef_r"], data["ef_nr"])
        m = compute_metrics(cif_pred, y_cif_test)
        ratio = m["mae"] / max(zs_mae, 1e-6)
        print(f"  λ_cif={lam:<6} MAE={m['mae']:.1f}  ×{ratio:.3f}")
        results.append({
            "lambda_cif": lam, "mae": m["mae"], "ratio_vs_zs": ratio,
        })

    return results


# ---------------------------------------------------------------------------
# Spectral analysis: per-group L_irm
# ---------------------------------------------------------------------------

def run_spectral_analysis(all_regions, target, seed=42):
    """Diagnostic: compute IRM penalty per source region.

    Reveals which source regions contribute most to the IRM term,
    i.e., which regions have the largest gradient norm.
    """
    data = all_regions[target]
    split = int(len(data["rs"]) * TRAIN_FRACTION)
    x_rs_test, _, y_cif_test = build_windows(
        data["rs"][split - SEQ_LEN:],
        data["cif"][split - SEQ_LEN:],
        SEQ_LEN, HORIZON, TEST_STRIDE)

    # Train a single model
    model, _ = train_phys_irm(all_regions, target, seed=seed)
    model.eval()

    print(f"\nPer-source IRM penalty spectrum for {target}")
    print(f"{'source':>20}  {'L_T':>8}  {'L_share':>10}  {'IRM_penalty':>12}")
    print("-" * 60)

    per_source = []
    for name, src in all_regions.items():
        if name == target:
            continue
        x_win, y_win, _ = build_windows(src["rs"], src["cif"])
        if len(x_win) == 0:
            continue
        L_T = abs(src["ef_nr"] - src["ef_r"])
        cfg_t = torch.tensor(np.tile(src["config"], (len(x_win), 1)),
                              dtype=torch.float32)
        x_t = torch.tensor(x_win, dtype=torch.float32)
        y_t = torch.tensor(y_win, dtype=torch.float32)

        # Compute share error and IRM penalty
        with torch.no_grad():
            share_pred, feat = model(x_t, cfg_t)
            L_share_val = F.l1_loss(share_pred, y_t).item()
        L_irm_val = irm_penalty(feat, y_t).item()

        print(f"  {name:>18}  {L_T:>8.0f}  {L_share_val:>10.4f}  {L_irm_val:>12.6f}")
        per_source.append({
            "source": name, "L_T": L_T,
            "L_share": L_share_val, "L_irm": L_irm_val,
        })

    return per_source


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

import torch.nn.functional as F  # noqa: E402 (used in spectral analysis)


def main():
    parser = argparse.ArgumentParser(description="Phys-IRM Experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Quick: AU only, seed=0")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds (full mode)")
    parser.add_argument("--gamma-irm", type=float, default=0.1,
                        help="IRM penalty weight")
    parser.add_argument("--lambda-cif", type=float, default=0.5,
                        help="CIF supervision weight")
    parser.add_argument("--ablation-irm", action="store_true",
                        help="Sweep γ_irm on VIC1 + SA1")
    parser.add_argument("--ablation-cif", action="store_true",
                        help="Sweep λ_cif on VIC1")
    parser.add_argument("--spectral", action="store_true",
                        help="Per-source IRM penalty diagnostic")
    parser.add_argument("--target", type=str, default=None,
                        help="Single target for ablation/spectral")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    # Load regions
    discover_uk_regions()
    all_configs = {**AU_REGIONS, **UK_REGIONS, **US_REGIONS}
    all_regions = {}
    for name in all_configs:
        try:
            all_regions[name] = load_region_data(name, all_configs)
        except Exception as e:
            print(f"  [WARN] Skip {name}: {e}")

    # Compute L_T stats
    lt = compute_L_T(all_regions)
    print(f"\nLoaded {len(all_regions)} regions.")
    print(f"L_T range: {min(lt.values()):.0f} – {max(lt.values()):.0f}")
    print(f"Top-5 high-L_T: {list(lt.items())[:5]}")
    print(f"Bottom-5 low-L_T: {list(lt.items())[-5:]}")

    # --- Ablation branches ---
    if args.ablation_irm:
        targets = args.target.split(",") if args.target else ["VIC1", "SA1"]
        all_irm_results = []
        for tgt in targets:
            r = run_ablation_irm(all_regions, tgt, gammas=[
                0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
            all_irm_results.extend([{"target": tgt, **x} for x in r])
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            json.dump(all_irm_results, open(args.out, "w"), indent=2)
            print(f"\nSaved to {args.out}")
        return

    if args.ablation_cif:
        target = args.target or "VIC1"
        results = run_ablation_cif(all_regions, target, lambdas=[
            0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
        if args.out:
            json.dump(results, open(args.out, "w"), indent=2)
        return

    if args.spectral:
        target = args.target or "VIC1"
        run_spectral_analysis(all_regions, target)
        return

    # --- Full evaluation ---
    seeds_q = [0] if args.quick else list(range(min(args.seeds, 5)))
    targets_q = ["QLD1", "NSW1", "VIC1", "SA1"] if args.quick else sorted(
        all_regions.keys(), key=lambda x: all_regions[x]["mean_rs"])

    print(f"\nPhys-IRM Evaluation: {len(targets_q)} regions × {len(seeds_q)} seeds")
    print(f"  γ_irm={args.gamma_irm}  λ_cif={args.lambda_cif}")

    results = run_full_eval(all_regions, targets_q, seeds_q,
                             gamma_irm=args.gamma_irm,
                             lambda_cif=args.lambda_cif)

    # Save
    out_path = args.out or str(
        RESULTS_DIR / "phys_irm_eval.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
