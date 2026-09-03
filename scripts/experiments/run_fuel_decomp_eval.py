#!/usr/bin/env python
"""FuelDecompNet LORO evaluation (TransCIF-FD, Phase FD-1).

Evaluates the fuel-decomposed physics-structured model on two information
tiers simultaneously (one set of weights, cold-mode dropout at training):

    I_0    config + live share telemetry        (paper-comparable)
    I_cfg  config + weather + calendar only     (China deployment tier)

Baselines:
    persistence        lag-24 CIF (forecast reference; needs CIF history)
    config-constant    mean_rs*ef_r + (1-mean_rs)*ef_nr — the "official
                       annual factor" analog available to any region
    monthly-constant   per-month true mean (oracle level anchor)

Metrics: MAE/RMSE + shape metrics (diurnal MAE, monthly-shape MAE,
Spearman hourly ranking) — the shape/ranking metrics are what carbon-aware
scheduling in telemetry-free regions actually consumes.

Usage:
    python scripts/experiments/run_fuel_decomp_eval.py                 # 8 regions x seeds 0-1
    python scripts/experiments/run_fuel_decomp_eval.py --full          # 29 regions x seeds 0-4
    python scripts/experiments/run_fuel_decomp_eval.py --regions US_CISO --seeds 0 --epochs 60
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import torch

from transcif.config import (
    SEQ_LEN, HORIZON, TEST_STRIDE, TRAIN_FRACTION, SEEDS_FULL, RESULTS_DIR,
)
from transcif.data.loaders import all_region_configs
from transcif.data.fuel import build_fd_windows
from transcif.models.zeroshot.fuel import (
    prepare_fd_region, train_fuel_zero_shot, predict_fuel_windows,
    shape_metrics_with_months, apply_day_ahead_weather_error,
)
from transcif.evaluation.metrics import compute_metrics

QUICK_REGIONS = [
    "QLD1", "NSW1", "VIC1", "SA1",
    "US_BPAT", "US_PJM",
    "UK_02_South_Scotland", "UK_08_West_Midlands",
]

# FD-22 deployment route table (official PASS, fd22/fd24): wind-seasonal
# regions with REAL per-fuel telemetry run the fuel path (tau 1.1 never
# triggers the aggregate router); everything else keeps the default router.
# Eval-time only — training stays at the global tau, so the trained models
# are bit-identical to a uniform-tau run.  SA1 must stay aggregate: its
# fuel path carries a +56 cold bias (FD-22, classifier-synthetic shares).
ROUTE_TABLE = {"UK_01_North_Scotland": 1.1, "UK_02_South_Scotland": 1.1,
               "UK_16_Scotland": 1.1}


def build_target_test_windows(data, use_monthly=False):
    """Test-split windows (last 20%, stride 24) matching the paper protocol."""
    split = int(len(data["rs"]) * TRAIN_FRACTION)
    sl = slice(split - SEQ_LEN, None)
    sliced = {
        **data,
        "rs": data["rs"][sl], "cif": data["cif"][sl],
        "fuel_shares": data["fuel_shares"][sl], "hours": data["hours"][sl],
        "exog": {k: v[sl] for k, v in data["exog"].items()},
    }
    return build_fd_windows(sliced, seq_len=SEQ_LEN, horizon=HORIZON,
                            stride=TEST_STRIDE,
                            monthly_table=(data.get("monthly_table_target",
                                                     data.get("monthly_table"))
                                           if use_monthly else None),
                            lag_months=1)


def select_source_route(model, fd_regions, target, device,
                        candidates=(0.0, 0.45, 1.1)):
    """Select the route using source-only validation, never target labels.

    The model is already trained with the target region held out.  For each
    candidate router, score the source regions' final 20% cold windows and
    transfer that evidence to the target using config similarity.  This is a
    deployment-safe alternative to selecting a route on target history.
    """
    from transcif.physics.bounds import config_weight

    tgt = fd_regions[target]
    scores = {float(tau): [] for tau in candidates}
    for name, data in fd_regions.items():
        if name == target:
            continue
        split = int(len(data["rs"]) * TRAIN_FRACTION)
        sl = slice(split - SEQ_LEN, None)
        sliced = {**data,
                  "rs": data["rs"][sl], "cif": data["cif"][sl],
                  "fuel_shares": data["fuel_shares"][sl],
                  "hours": data["hours"][sl],
                  "exog": {k: v[sl] for k, v in data["exog"].items()}}
        w = build_fd_windows(sliced, seq_len=SEQ_LEN, horizon=HORIZON,
                             stride=TEST_STRIDE, max_windows=80)
        if len(w["x_rs"]) == 0:
            continue
        weight = float(config_weight(
            float(data["mean_rs"]), float(tgt["mean_rs"])))
        for tau in candidates:
            old = model.wind_route_tau
            model.wind_route_tau = float(tau)
            pred, _, _ = predict_fuel_windows(
                model, w, data["fd_config"],
                data["ef_vec"].astype(np.float32), cold=True, device=device)
            model.wind_route_tau = old
            scores[float(tau)].append((weight, float(np.abs(
                pred - w["y_cif"]).mean())))
    weighted = {
        tau: (sum(w * s for w, s in vals) / max(sum(w for w, _ in vals), 1e-6))
        for tau, vals in scores.items() if vals
    }
    if not weighted:
        return float(candidates[0]), weighted
    return min(weighted, key=weighted.get), weighted


def estimate_source_bias(model, fd_regions, target, device, cold=True,
                         max_windows=60):
    """Estimate a deployment-safe additive bias from source holdouts.

    The target CIF labels are never touched.  Each source contributes its
    final validation-window mean(prediction - label), weighted by target
    configuration similarity.  This corrects systematic regional accounting
    offsets that a shared physics head cannot identify from weather alone.
    """
    from transcif.physics.bounds import config_weight
    tgt = fd_regions[target]
    num = den = 0.0
    details = {}
    for name, data in fd_regions.items():
        if name == target:
            continue
        split = int(len(data["rs"]) * TRAIN_FRACTION)
        sl = slice(split - SEQ_LEN, None)
        sliced = {**data, "rs": data["rs"][sl], "cif": data["cif"][sl],
                  "fuel_shares": data["fuel_shares"][sl],
                  "hours": data["hours"][sl],
                  "exog": {k: v[sl] for k, v in data["exog"].items()}}
        w = build_fd_windows(sliced, seq_len=SEQ_LEN, horizon=HORIZON,
                             stride=TEST_STRIDE, max_windows=max_windows)
        if len(w["x_rs"]) == 0:
            continue
        pred, _, _ = predict_fuel_windows(
            model, w, data["fd_config"], data["ef_vec"].astype(np.float32),
            cold=cold, device=device)
        bias = float(np.mean(pred - w["y_cif"]))
        weight = float(config_weight(float(data["mean_rs"]),
                                     float(tgt["mean_rs"])))
        num += weight * bias
        den += weight
        details[name] = bias
    return (num / den if den > 0 else 0.0), details


def smooth_hourly_windows(pred, hours):
    """Smooth each forecast horizon with a centered moving average."""
    if hours <= 1:
        return pred
    k = int(hours)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    out = np.pad(pred, ((0, 0), (pad, pad)), mode="edge")
    cs = np.cumsum(out, axis=1)
    cs = np.concatenate([np.zeros((len(out), 1), dtype=out.dtype), cs], axis=1)
    return (cs[:, k:] - cs[:, :-k]) / float(k)


def evaluate_one(target, fd_regions, seed, epochs, device, p_cold=0.3,
                 p_mix=0.0, use_hypernet=False, weight_mode="legacy",
                 ef_corr_bound=0.35, lambda_shape=0.5,
                 use_monthly=False, lag_months=1, weather_noise=False,
                 evening_weight=1.0, solar_mod_bound=0.4,
                 wind_route_tau=1.1, domain_penalty=0.0,
                 dynamic_residual=False,
                 dynamic_residual_bound=220.0,
                 same_jurisdiction=False,
                 physics_target=False, source_route_select=False,
                 route_candidates=(0.0, 0.45, 1.1),
                 source_bias_calibrate=False,
                 smooth_hours=1,
                 lambda_fuel=1.0, lambda_rs=0.3,
                 deployment_route_table=False):
    t0 = time.time()
    model = train_fuel_zero_shot(
        fd_regions, target, seed=seed, epochs=epochs, device=device,
        p_cold=p_cold, p_mix=p_mix, use_hypernet=use_hypernet,
        weight_mode=weight_mode, ef_corr_bound=ef_corr_bound,
        lambda_shape=lambda_shape, use_monthly=use_monthly,
        lag_months=lag_months,
        evening_weight=evening_weight, solar_mod_bound=solar_mod_bound,
        wind_route_tau=wind_route_tau, dynamic_residual=dynamic_residual,
        dynamic_residual_bound=dynamic_residual_bound,
        same_jurisdiction=same_jurisdiction,
        domain_penalty=domain_penalty,
        physics_target=physics_target, lambda_fuel=lambda_fuel,
        lambda_rs=lambda_rs)
    route_scores = {}
    if source_route_select:
        wind_route_tau, route_scores = select_source_route(
            model, fd_regions, target, device, candidates=route_candidates)
        model.wind_route_tau = wind_route_tau
    data = fd_regions[target]
    if deployment_route_table and not source_route_select:
        wind_route_tau = ROUTE_TABLE.get(target, wind_route_tau)
        model.wind_route_tau = float(wind_route_tau)
    w = build_target_test_windows(data, use_monthly=use_monthly)
    n = len(w["x_rs"])
    if n == 0:
        return None
    y = w["y_cif"]
    ef_vec = data["ef_vec"].astype(np.float32)
    fd_cfg = data["fd_config"]

    res = {"target": target, "seed": seed, "n_test": n,
           "mean_rs": data["mean_rs"], "ef_nr": data["ef_nr"],
           "has_fuel": bool(data["has_fuel"]),
           "train_s": round(time.time() - t0, 1),
           "selected_route_tau": float(wind_route_tau),
           "source_route_scores": route_scores}
    source_bias = 0.0
    source_bias_details = {}
    if source_bias_calibrate:
        source_bias, source_bias_details = estimate_source_bias(
            model, fd_regions, target, device, cold=True)
    res["source_bias"] = float(source_bias)
    res["source_bias_details"] = source_bias_details

    # --- FuelDecompNet, both tiers
    for tier, cold in (("i0", False), ("i_cfg", True)):
        cif, _, _ = predict_fuel_windows(model, w, fd_cfg, ef_vec,
                                         cold=cold, device=device)
        if source_bias_calibrate and cold:
            cif = cif - source_bias
        cif = smooth_hourly_windows(cif, smooth_hours)
        res[f"fuel_{tier}"] = shape_metrics_with_months(cif, y, w["origin_hours"])
        res[f"fuel_{tier}"]["std_metrics"] = compute_metrics(cif, y)
        # Dual-track labels (roadmap #1): for fuel-telemetry regions also
        # score against the PHYSICS-reconstructed truth (shares x EF).
        # Import-accounting regions (UK South-East) carry 10-37% label
        # noise from the API methodology — this track isolates it.
        if data["has_fuel"]:
            from transcif.data.fuel import fuel_cif
            y_phys = fuel_cif(w["y_fuel"], data["ef_vec"]).astype(np.float32)
            res[f"fuel_{tier}_phys"] = shape_metrics_with_months(
                cif, y_phys, w["origin_hours"])

    # --- Day-ahead weather-forecast track: same model, same origins, but
    #     the 24 h horizon weather is degraded to NWP skill (astronomy and
    #     calendar stay exact — they are deterministic; past weather stays
    #     reanalysis).  Paired within one run so the sensitivity is
    #     directly readable as fuel_{tier}_wn vs fuel_{tier}.
    if weather_noise:
        wn_rng = np.random.default_rng(1000 + seed)
        w_wn = apply_day_ahead_weather_error(w, wn_rng)
        for tier, cold in (("i0", False), ("i_cfg", True)):
            cif, _, _ = predict_fuel_windows(model, w_wn, fd_cfg, ef_vec,
                                             cold=cold, device=device)
            res[f"fuel_{tier}_wn"] = shape_metrics_with_months(
                cif, y, w["origin_hours"])
            res[f"fuel_{tier}_wn"]["std_metrics"] = compute_metrics(cif, y)

    # --- FuelDecompNet + ZS+ calibration (I_+ tier): the workhorse from the
    #     TransCIF ladder (drop-one verdict: ZS_PLUS_WRAPPER), with the fuel
    #     model as the anchor branch via the share_fn hook.
    split = int(len(data["rs"]) * TRAIN_FRACTION)
    try:
        from transcif.calibration.zs_plus import zs_plus_predict
        from transcif.models.zeroshot.fuel import make_zs_plus_share_fn
        share_fn = make_zs_plus_share_fn(model, data, device=device)
        origins = [split + st for st in range(
            0, len(data["cif"][split - SEQ_LEN:]) - SEQ_LEN - HORIZON + 1,
            TEST_STRIDE)][:n]
        zsp = zs_plus_predict(model, data["fd_config"], data["rs"], data["cif"],
                              data["ef_r"], data["ef_nr"], origins,
                              share_fn=share_fn)
        if len(zsp) == n:
            zsp = smooth_hourly_windows(zsp, smooth_hours)
            res["fuel_i_plus"] = shape_metrics_with_months(
                zsp, y, w["origin_hours"])
            res["fuel_i_plus"]["std_metrics"] = compute_metrics(zsp, y)
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] ZS+ tier failed for {target}: {e}")

    # --- persistence (lag-24 CIF; forecast reference, needs CIF history)
    cif_off = data["cif"][split - SEQ_LEN:]
    persist = np.stack([
        cif_off[s + SEQ_LEN - HORIZON:s + SEQ_LEN]
        for s in range(0, len(cif_off) - SEQ_LEN - HORIZON + 1, TEST_STRIDE)
    ])[:n]
    res["persistence"] = shape_metrics_with_months(persist, y, w["origin_hours"])

    # --- config-constant (official-annual-factor analog; deployment-legal)
    cfg_const = np.full_like(y, data["mean_rs"] * data["ef_r"]
                             + (1 - data["mean_rs"]) * data["ef_nr"])
    res["config_constant"] = shape_metrics_with_months(cfg_const, y, w["origin_hours"])

    # --- monthly-constant (oracle level anchor)
    months = w["origin_hours"].month.values
    monthly = np.zeros_like(y, dtype=np.float64)
    for m in np.unique(months):
        monthly[months == m] = y[months == m].mean()
    res["monthly_constant"] = shape_metrics_with_months(monthly, y, w["origin_hours"])

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=QUICK_REGIONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--max-windows", type=int, default=700)
    ap.add_argument("--p-cold", type=float, default=0.3)
    ap.add_argument("--p-mix", type=float, default=0.0,
                    help="fraction of steps on synthetic mixed pseudo-grids (FD-2)")
    ap.add_argument("--use-hypernet", action="store_true",
                    help="config hypernet generates dynamic head weights (FD-2)")
    ap.add_argument("--weight-mode", choices=["legacy", "fuel"], default="legacy",
                    help="source weighting: mean_rs distance or +fuel-structure distance")
    ap.add_argument("--ef-corr-bound", type=float, default=0.35,
                    help="bound on the learned EF correction (hypernet: try 0.15)")
    ap.add_argument("--lambda-shape", type=float, default=0.5,
                    help="diurnal-shape loss weight (0 = FD-5 objective)")
    ap.add_argument("--lambda-fuel", type=float, default=1.0,
                    help="EF-weighted per-fuel share loss weight")
    ap.add_argument("--lambda-rs", type=float, default=0.3,
                    help="aggregate renewable-share auxiliary loss weight")
    ap.add_argument("--monthly-config", action="store_true",
                    help="per-window monthly fuel-mix configs (publication-lagged)")
    ap.add_argument("--lag-months", type=int, default=1,
                    help="monthly statistics publication lag")
    ap.add_argument("--weather-noise", action="store_true",
                    help="add the day-ahead NWP-error track (paired per run)")
    ap.add_argument("--evening-weight", type=float, default=1.0,
                    help="evening-peak (17-21h) loss up-weighting (C-class: 2.0)")
    ap.add_argument("--solar-mod-bound", type=float, default=0.4,
                    help="solar weather-modulation bound (C-class: 0.6)")
    ap.add_argument("--wind-route-tau", type=float, default=1.1,
                    help="fuel-vs-aggregate router threshold; 1.1 is the "
                         "validated fuel-first policy, 0.0 aggregate")
    ap.add_argument("--domain-penalty", type=float, default=0.0,
                    help="optional std. dev. penalty on per-source risks")
    ap.add_argument("--dynamic-residual", action="store_true",
                    help="enable bounded source-trained dynamic CIF residual")
    ap.add_argument("--dynamic-residual-bound", type=float, default=220.0)
    ap.add_argument("--physics-target", action="store_true",
                    help="train source CIF loss on fuel shares x effective EF")
    ap.add_argument("--source-route-select", action="store_true",                    help="select route from source-only validation, no target CIF")
    ap.add_argument("--deployment-route-table", action="store_true",
                    help="restore the FD-22 official route table at eval "
                         "(UK_01/02/16 -> fuel path); training unchanged")
    ap.add_argument("--route-candidates", nargs="+", type=float,
                    default=[0.0, 0.45, 1.1])
    ap.add_argument("--source-bias-calibrate", action="store_true",
                    help="transfer source-only cold-mode mean bias")
    ap.add_argument("--smooth-hours", type=int, default=1,
                    help="centered forecast smoothing width; 1 disables")
    ap.add_argument("--same-jurisdiction-sources", action="store_true",
                    help="train only on other regions in target jurisdiction")
    ap.add_argument("--full", action="store_true",
                    help="29-region protocol, seeds 0-4")
    ap.add_argument("--multi-year", action="store_true",
                    help="explicitly concatenate available 2022/2023/2024 files")
    ap.add_argument("--au-state", action="store_true",
                    help="use optional NEM regional state channel")
    ap.add_argument("--monthly-history-only", action="store_true",
                    help="build monthly config only from pre-test history")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.full:
        args.regions = None
        args.seeds = SEEDS_FULL

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[fuel-eval] device={device} epochs={args.epochs} "
          f"p_cold={args.p_cold} p_mix={args.p_mix} "
          f"max_windows={args.max_windows}")

    cfgs = all_region_configs()
    # Match the legacy 29-region protocol exactly: UK_18_GB (national
    # aggregate) IS included, UK_04 is absent (its ef_nr estimate falls
    # outside the discovery bounds) — same target set as unified_eval_full.json.
    pool_names = list(cfgs)
    targets = args.regions or pool_names

    print("[fuel-eval] preparing FD data for all regions ...")
    fd_regions = {}
    for name in pool_names:
        fd_regions[name] = prepare_fd_region(
            name, cfgs, multi_year=args.multi_year,
            monthly_history_only=args.monthly_history_only,
            use_au_state=args.au_state)
    print(f"[fuel-eval] {len(fd_regions)} regions ready")

    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / ("fuel_decomp_eval_full.json" if args.full
                       else "fuel_decomp_eval_quick.json"))
    # Resume support: skip (target, seed) pairs already recorded.
    rows = []
    expected_meta = {
        "epochs": args.epochs, "p_cold": args.p_cold,
        "p_mix": args.p_mix, "use_hypernet": bool(args.use_hypernet),
        "weight_mode": args.weight_mode, "ef_corr_bound": args.ef_corr_bound,
        "lambda_shape": args.lambda_shape,
        "lambda_fuel": args.lambda_fuel, "lambda_rs": args.lambda_rs,
        "use_monthly": bool(args.monthly_config), "lag_months": args.lag_months,
        "multi_year": bool(args.multi_year),
        "au_state": bool(args.au_state),
        "dynamic_residual": bool(args.dynamic_residual),
        "same_jurisdiction_sources": bool(args.same_jurisdiction_sources),
        "route_candidates": args.route_candidates,
        "source_bias_calibrate": bool(args.source_bias_calibrate),
        "smooth_hours": args.smooth_hours,
        "monthly_history_only": bool(args.monthly_history_only),
        "weather_noise": bool(args.weather_noise),
        "evening_weight": args.evening_weight,
        "solar_mod_bound": args.solar_mod_bound,
        "wind_route_tau": args.wind_route_tau,
        "domain_penalty": args.domain_penalty,
        "physics_target": bool(args.physics_target),
        "source_route_select": bool(args.source_route_select),
        "deployment_route_table": bool(args.deployment_route_table),
        "max_windows": args.max_windows,
    }
    if out_path.exists():
        with open(out_path) as f:
            doc = json.load(f)
        old_meta = doc.get("meta", {})
        mismatch = {k: (old_meta.get(k), v) for k, v in expected_meta.items()
                    if k in old_meta and old_meta.get(k) != v}
        if mismatch:
            print(f"[fuel-eval] metadata mismatch; starting fresh: {mismatch}")
            rows = []
        else:
            rows = doc.get("rows", [])
        done = {(r["target"], r["seed"]) for r in rows}
        print(f"[fuel-eval] resuming: {len(done)} pairs already done")
    else:
        done = set()

    for target in targets:
        for seed in args.seeds:
            if (target, seed) in done:
                continue
            t0 = time.time()
            try:
                row = evaluate_one(target, fd_regions, seed, args.epochs,
                                   device, p_cold=args.p_cold,
                                   p_mix=args.p_mix,
                                   use_hypernet=args.use_hypernet,
                                   weight_mode=args.weight_mode,
                                   ef_corr_bound=args.ef_corr_bound,
                                   lambda_shape=args.lambda_shape,
                                   use_monthly=args.monthly_config,
                                   lag_months=args.lag_months,
                                   weather_noise=args.weather_noise,
                                   evening_weight=args.evening_weight,
                                   solar_mod_bound=args.solar_mod_bound,
                                   wind_route_tau=args.wind_route_tau,
                                   dynamic_residual=args.dynamic_residual,
                                   dynamic_residual_bound=args.dynamic_residual_bound,
                                   same_jurisdiction=args.same_jurisdiction_sources,
                                   domain_penalty=args.domain_penalty,
                                   route_candidates=tuple(args.route_candidates),
                                   source_bias_calibrate=args.source_bias_calibrate,
                                   smooth_hours=args.smooth_hours,
                                   physics_target=args.physics_target,
                                   source_route_select=args.source_route_select,
                                   lambda_fuel=args.lambda_fuel,
                                   lambda_rs=args.lambda_rs,
                                   deployment_route_table=args.deployment_route_table)
            except Exception as e:  # noqa: BLE001
                print(f"  [WARN] {target} seed {seed} failed: {e}")
                continue
            if row is None:
                continue
            rows.append(row)
            r0, rc = row["fuel_i0"]["mae"], row["fuel_i_cfg"]["mae"]
            rp = row.get("fuel_i_plus", {}).get("mae", float("nan"))
            print(f"  {target:28s} seed {seed}: I_0 MAE {r0:6.1f} | "
                  f"I_cfg MAE {rc:6.1f} | I_+ MAE {rp:6.1f} "
                  f"(persist {row['persistence']['mae']:6.1f}, "
                  f"cfg-const {row['config_constant']['mae']:6.1f}) "
                  f"[{time.time() - t0:.0f}s]")
            with open(out_path, "w") as f:
                json.dump({"rows": rows, "meta": {
                    "epochs": args.epochs, "p_cold": args.p_cold,
                    "p_mix": args.p_mix,
                    "use_hypernet": bool(args.use_hypernet),
                    "weight_mode": args.weight_mode,
                    "ef_corr_bound": args.ef_corr_bound,
                    "lambda_shape": args.lambda_shape,
                    "lambda_fuel": args.lambda_fuel,
                    "lambda_rs": args.lambda_rs,
                    "use_monthly": bool(args.monthly_config),
                    "lag_months": args.lag_months,
                    "weather_noise": bool(args.weather_noise),
                    "evening_weight": args.evening_weight,
                    "solar_mod_bound": args.solar_mod_bound,
                    "wind_route_tau": args.wind_route_tau,
                    "domain_penalty": args.domain_penalty,
                    "physics_target": bool(args.physics_target),
                    "source_route_select": bool(args.source_route_select),
                    "max_windows": args.max_windows}}, f, indent=1)

    # --- summary
    if rows:
        for col in ("fuel_i0", "fuel_i_cfg", "fuel_i_plus",
                    "fuel_i0_wn", "fuel_i_cfg_wn", "fuel_i_plus_wn",
                    "persistence", "config_constant", "monthly_constant"):
            maes = [r[col]["mae"] for r in rows if r.get(col)]
            print(f"{col:18s} median MAE {np.median(maes):7.2f}  "
                  f"mean {np.mean(maes):7.2f}  n={len(maes)}")
        for col in ("fuel_i0", "fuel_i_cfg", "fuel_i_plus",
                    "fuel_i0_wn", "fuel_i_cfg_wn"):
            sp = [r[col]["spearman"] for r in rows if r.get(col)]
            dm = [r[col]["diurnal_mae"] for r in rows if r.get(col)]
            if not sp:
                continue
            print(f"{col:18s} median Spearman {np.median(sp):.3f}  "
                  f"diurnal MAE {np.median(dm):.2f}")
        print(f"[fuel-eval] wrote {out_path} ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
