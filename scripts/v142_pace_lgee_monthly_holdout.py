"""V142: targeted PACE/LGEE development with an expanded source budget."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import v139_physical_reserve_calibration as v139
import v140_monthly_conditional_risk as v140
from v141_source_diversity_ablation import original_budget_frame

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V142_PACE_LGEE_MONTHLY_HOLDOUT_PROTOCOL.md"
V140_DIR = ROOT / "results/v140-monthly-conditional-risk-r1"
PREPARED = ROOT / "results/v141-source-expansion-input-r1"
OUT = ROOT / "results/v142-pace-lgee-monthly-holdout-r1"
METHODS = ("physical", "anchor_disjoint12", "anchor_all_sources", "anchor_nearest12_sources",
           "direct_cqr_all", "direct_cqr_active", "occupancy_cqr")


def record(row, seed: int, method: str, alpha: float, lo: int, hi: int) -> dict:
    result = {"region": row.region, "month": row.month, "tau": row.tau, "seed": seed, "method": method,
              "alpha": alpha, "hours": row.hours, "lower": lo, "upper": hi, "width": hi - lo,
              "covered": lo <= row.factor_box_K_lower and hi >= row.factor_box_K_upper,
              "empty_fallback": False, "capacity_unavailable": False,
              "factor_box_K_lower": row.factor_box_K_lower, "factor_box_K_upper": row.factor_box_K_upper,
              "robust_interval_score90": (hi - lo + 20 * max(lo - row.factor_box_K_lower, 0)
                                           + 20 * max(row.factor_box_K_upper - hi, 0))}
    for rate in v139.RATES:
        required = math.ceil(rate * row.hours - 1e-10)
        pos, neg = lo >= required, hi < required
        result[f"positive_{rate}"] = pos
        result[f"negative_{rate}"] = neg
        result[f"abstain_{rate}"] = not (pos or neg)
        result[f"error_{rate}"] = ((pos and row.factor_box_K_lower < required)
                                  or (neg and row.factor_box_K_upper >= required))
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    protocol_hash = v139.sha(PROTOCOL)
    original = original_budget_frame(pd.read_csv(V140_DIR / "monthly_feature_panel.csv"))
    additions = pd.read_csv(PREPARED / "additional_monthly_panel.csv")
    additions = additions.loc[additions.eligible_donor]
    panel = pd.concat([original, additions], ignore_index=True, sort=False)
    v140.FEATURES = json.loads((V140_DIR / "summary.json").read_text())["common_features"]
    fuels = [f"fuelshare_{f}" for f in v140.v90.FUELS]
    if not np.isfinite(panel[v140.FEATURES].to_numpy(float)).all():
        raise ValueError("invalid common monthly features")
    outputs, roles, calibration_membership, failures = [], [], [], []
    for target_region in ("PACE", "LGEE"):
        for seed in v140.SEEDS:
            ordered = sorted(original.region.unique(), key=lambda region: hashlib.sha256(
                f"{seed}/{target_region}/{region}".encode()).hexdigest())
            calibration_regions = ordered[:12]
            fit_regions = ordered[12:] + sorted(set(additions.region) - {target_region})
            if (target_region in fit_regions or target_region in calibration_regions
                    or set(fit_regions) & set(calibration_regions)
                    or len(fit_regions) != 30 or len(calibration_regions) != 12):
                raise ValueError("held-out target leakage or wrong role count")
            for role, roster in (("fit", fit_regions), ("calibration", calibration_regions)):
                for region in roster:
                    roles.append({"target_region": target_region, "seed": seed, "role": role, "region": region})
            for tau in (600, 900):
                frame = panel.loc[panel.tau.eq(tau)]
                source = frame.loc[frame.region.ne(target_region)]
                fit = frame.loc[frame.region.isin(fit_regions)]
                calibration = frame.loc[frame.region.isin(calibration_regions)]
                target = frame.loc[frame.region.eq(target_region)]
                if source.region.nunique() != 42 or not target.physical_lower.le(target.factor_box_K_lower).all() or not target.physical_upper.ge(target.factor_box_K_upper).all():
                    raise ValueError("invalid source count or target physical containment")
                sa, sb = v140.projected_predictions(source, "anchor_absolute")
                source_scores = pd.DataFrame({"region": source.region.to_numpy(),
                                               "score": v140.score(source, sa, sb, False)}).groupby("region").score.max()
                source_centers = source.groupby("region")[fuels].mean()
                for method in METHODS:
                    normalized = method == "occupancy_cqr"
                    if method == "physical":
                        a, b = target.physical_lower.to_numpy(float), target.physical_upper.to_numpy(float)
                        score_pool = pd.Series(0.0, index=calibration_regions)
                    elif method.startswith("anchor"):
                        a, b = v140.projected_predictions(target, "anchor_absolute")
                        score_pool = (source_scores.loc[calibration_regions] if method == "anchor_disjoint12"
                                      else source_scores)
                    else:
                        models = v140.fitted_models(fit, method)
                        ca, cb = v140.projected_predictions(calibration, method, models)
                        score_pool = pd.DataFrame({"region": calibration.region.to_numpy(),
                            "score": v140.score(calibration, ca, cb, normalized)}).groupby("region").score.max()
                        a, b = v140.projected_predictions(target, method, models)
                    for j, row in enumerate(target.itertuples(index=False)):
                        pool = score_pool
                        if method == "anchor_nearest12_sources":
                            x = np.array([getattr(row, f) for f in fuels], dtype=float)
                            distances = (source_centers - x).abs().sum(axis=1)
                            nearest = sorted(distances.index, key=lambda region: (distances.loc[region], region))[:12]
                            pool = source_scores.loc[nearest]
                        if target_region in pool.index:
                            raise ValueError("target risk labels entered calibration")
                        for alpha in v140.ALPHAS:
                            n = len(pool)
                            rank = math.ceil((n + 1) * (1 - alpha))
                            radius = float(pool.sort_values().iloc[rank - 1]) if rank <= n else 1.0
                            if method == "physical":
                                radius = 0.0
                            scale = (row.physical_upper - row.physical_lower) if normalized else row.hours
                            lower = max(int(row.physical_lower), math.floor(a[j] - radius * scale + 1e-9))
                            upper = min(int(row.physical_upper), math.ceil(b[j] + radius * scale - 1e-9))
                            if not row.physical_lower <= lower <= upper <= row.physical_upper:
                                failures.append({"region": target_region, "month": row.month, "tau": tau,
                                                 "seed": seed, "method": method, "alpha": alpha,
                                                 "check": "physical_subset"})
                            outputs.append(record(row, seed, method, alpha, lower, upper))
                            for region, value in pool.items():
                                calibration_membership.append({"target_region": target_region, "month": row.month,
                                    "tau": tau, "seed": seed, "method": method, "alpha": alpha,
                                    "source_region": region, "score": value, "radius": radius,
                                    "calibration_regions": n, "rank": rank})
        print(f"target {target_region}", flush=True)
    rows = pd.DataFrame(outputs)
    target_months = panel.loc[panel.region.isin(("PACE", "LGEE")), ["region", "month"]].drop_duplicates()
    expected = len(target_months) * 2 * len(v140.SEEDS) * len(v140.ALPHAS) * len(METHODS)
    keys = ["region", "month", "tau", "seed", "method", "alpha"]
    if len(rows) != expected or rows.duplicated(keys).any():
        raise ValueError("target denominator failure")
    for key, group in rows.groupby(keys[:-1]):
        narrow, wide = group.sort_values("alpha", ascending=False).iloc[0], group.sort_values("alpha").iloc[0]
        if wide.lower > narrow.lower or wide.upper < narrow.upper:
            failures.append({"key": str(key), "check": "alpha_nesting"})
    frontier = rows.groupby(["region", "seed", "tau", "method", "alpha"]).agg(
        coverage=("covered", "mean"), width=("width", "mean"), robust_interval_score90=("robust_interval_score90", "mean"),
        months=("month", "size"), positive_01=("positive_0.01", "mean"), negative_01=("negative_0.01", "mean"),
        certificate_error_01=("error_0.01", "mean")).reset_index()
    if protocol_hash != v139.sha(PROTOCOL):
        raise RuntimeError("protocol changed during scoring")
    OUT.mkdir(parents=True)
    rows.to_csv(OUT / "target_rows.csv", index=False)
    frontier.to_csv(OUT / "target_frontier.csv", index=False)
    pd.DataFrame(roles).to_csv(OUT / "region_roles.csv", index=False)
    pd.DataFrame(calibration_membership).to_csv(OUT / "calibration_membership.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "assertion_failures.csv", index=False)
    paths = [PROTOCOL, V140_DIR / "monthly_feature_panel.csv", V140_DIR / "summary.json",
             PREPARED / "additional_monthly_panel.csv", PREPARED / "summary.json"]
    payload = {"status": "PACE_LGEE_OPENED_HISTORICAL_MONTHLY_TARGET_DEVELOPMENT", "target_region_months": len(target_months),
               "scored_rows": len(rows), "assertion_failures": len(failures), "methods": METHODS,
               "input_hashes": {str(p.relative_to(ROOT)): v139.sha(p) for p in paths},
               "script_sha256": v139.sha(Path(__file__)), "v140_helper_sha256": v139.sha(Path(v140.__file__)),
               "frontier": json.loads(frontier.to_json(orient="records")),
               "limitations": ["Already examined historical target identities/responses; not blind confirmation.",
                               "Source-expanded ordinary baselines, no new algorithm claim.",
                               "Target-month aggregates and assumed factor-box proxy labels, not Chinese CIF forecasts."]}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(frontier.loc[frontier.alpha.eq(.1)].to_string(index=False))
    print(f"assertion failures: {len(failures)}")


if __name__ == "__main__":
    main()
