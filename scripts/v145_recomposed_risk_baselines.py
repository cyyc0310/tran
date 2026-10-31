"""V145: ordinary recomposition and response-informed strong risk baselines."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

import v139_physical_reserve_calibration as v139
import v140_monthly_conditional_risk as v140
from monthly_tail_response import MonthlyTailResponses, RESPONSE_FEATURES
from v142_pace_lgee_monthly_holdout import record

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V145_RECOMPOSED_RISK_BASELINES_PROTOCOL.md"
INPUT = ROOT / "results/v144-canonical-profile-cache-r1"
RESPONSES = ROOT / "results/v144-composition-response-cache-r1"
V140 = ROOT / "results/v140-monthly-conditional-risk-r1"
V142 = ROOT / "results/v142-pace-lgee-monthly-holdout-r1"
ADDITIONS = ROOT / "results/v141-source-expansion-input-r1/donor_inventory.csv"
OUT = ROOT / "results/v145-recomposed-risk-baselines-r1"
METHODS = ("physical", "anchor_all_sources", "anchor_nearest12_sources", "direct_common_all", "direct_common_active",
           "direct_response_all", "residual_response_all", "residual_response_active",
           "recompose_pooled", "recompose_conditional", "recompose_conditional_scale")
LEARNED = METHODS[3:8]


def fit_heads(frame: pd.DataFrame, method: str, common: list[str]):
    features = common if method.startswith("direct_common") else common + list(RESPONSE_FEATURES)
    if any("factor_box_K" in c or "nominal_K" in c or "factor_floor" in c for c in features):
        raise ValueError("response labels entered feature whitelist")
    fit = frame.loc[frame.physical_upper.gt(frame.physical_lower)] if method.endswith("active") else frame
    weights = 1 / fit.groupby("region").region.transform("size").to_numpy(float)
    heads = []
    for quantile, factor in ((.05, "low"), (.95, "high")):
        y = fit[f"factor_box_K_{'lower' if factor == 'low' else 'upper'}"] / fit.hours
        if method.startswith("residual"):
            y = y - fit[f"rsp_conditional_{factor}_median"]
        model = GradientBoostingRegressor(alpha=quantile, **v140.PARAMS)
        model.fit(fit[features], y, sample_weight=weights)
        heads.append(model)
    return heads, features


def initial(frame: pd.DataFrame, method: str, fitted=None):
    hours = frame.hours.to_numpy(float)
    lo, hi = frame.physical_lower.to_numpy(float) / hours, frame.physical_upper.to_numpy(float) / hours
    scales = np.ones(len(frame))
    if method == "physical":
        a, b = lo.copy(), hi.copy()
    elif method.startswith("anchor"):
        a, b = frame.anchor_lower_fraction.to_numpy(float), frame.anchor_upper_fraction.to_numpy(float)
    elif method.startswith("recompose"):
        mode = "pooled" if method == "recompose_pooled" else "conditional"
        a, b = frame[f"rsp_{mode}_low_q05"].to_numpy(float), frame[f"rsp_{mode}_high_q95"].to_numpy(float)
        if method.endswith("scale"):
            scales = np.maximum.reduce([frame.rsp_conditional_low_sd.to_numpy(float),
                                        frame.rsp_conditional_high_sd.to_numpy(float), 1 / frame.hours.to_numpy(float)])
    else:
        heads, features = fitted
        a, b = heads[0].predict(frame[features]), heads[1].predict(frame[features])
        if method.startswith("residual"):
            a += frame.rsp_conditional_low_median.to_numpy(float)
            b += frame.rsp_conditional_high_median.to_numpy(float)
    a, b = np.minimum(a, b), np.maximum(a, b)
    return np.clip(a, lo, hi), np.clip(b, lo, hi), scales


def scores(frame: pd.DataFrame, a, b, scales):
    values = np.maximum.reduce([a - frame.factor_box_K_lower.to_numpy(float) / frame.hours,
                                frame.factor_box_K_upper.to_numpy(float) / frame.hours - b, np.zeros(len(frame))]) / scales
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("invalid calibration score")
    return pd.DataFrame({"region": frame.region.to_numpy(), "score": values}).groupby("region").score.max()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    protocol_hash = v139.sha(PROTOCOL)
    panel = pd.read_csv(INPUT / "canonical_monthly_panel.csv")
    metadata = panel.loc[panel.tau.eq(600), ["profile_id", "region", "month", "hours", *[f"fuelshare_{f}" for f in v140.v90.FUELS]]]
    cache_summary = json.loads((RESPONSES / "summary.json").read_text())
    if cache_summary["assertion_failures"] or v139.sha(RESPONSES / "response_counts.npz") != cache_summary["response_sha256"]:
        raise ValueError("response cache audit/hash failure")
    cache = np.load(RESPONSES / "response_counts.npz", allow_pickle=False)
    response_engine = MonthlyTailResponses(metadata, cache["low"], cache["high"])
    original_regions = sorted(pd.read_csv(V140 / "monthly_feature_panel.csv").region.unique())
    inventory = pd.read_csv(ADDITIONS)
    additional_regions = sorted(inventory.loc[inventory.eligible_donor, "region"])
    common = json.loads((V140 / "summary.json").read_text())["common_features"]
    old_sc_roles = pd.read_csv(V140 / "region_roles.csv")
    old_other_roles = pd.read_csv(V142 / "region_roles.csv")
    outputs, roles, provenance_rows, feature_rows, radii, failures = [], [], [], [], [], []
    for target_region in ("PACE", "LGEE", "SC"):
        for seed in v140.SEEDS:
            originals = [r for r in original_regions if r != target_region]
            ordered = sorted(originals, key=lambda r: hashlib.sha256(f"{seed}/{target_region}/{r}".encode()).hexdigest())
            cal_regions = ordered[:12]
            fit_regions = ordered[12:] + [r for r in additional_regions if r != target_region]
            if len(fit_regions) != 30 or len(cal_regions) != 12 or set(fit_regions) & set(cal_regions) or target_region in fit_regions + cal_regions:
                raise ValueError("role separation/count failure")
            reference = old_sc_roles if target_region == "SC" else old_other_roles
            archived_roles = reference.loc[reference.target_region.eq(target_region) & reference.seed.eq(seed)]
            if set(archived_roles.loc[archived_roles.role.eq("calibration"), "region"]) != set(cal_regions):
                raise ValueError("archived calibration roles changed")
            archived_fit = set(archived_roles.loc[archived_roles.role.eq("fit"), "region"])
            expected_fit = set(ordered[12:]) if target_region == "SC" else set(fit_regions)
            if archived_fit != expected_fit:
                raise ValueError("archived fitting roles changed")
            for role, roster in (("fit", fit_regions), ("calibration", cal_regions)):
                for region in roster:
                    roles.append({"target_region": target_region, "seed": seed, "role": role, "region": region})
            fold_features = []
            for row in panel.itertuples(index=False):
                features, provenance = response_engine.features(row.profile_id, row.tau, tuple(fit_regions))
                expected_regions = set(fit_regions) - {row.region}
                if set(provenance["donor_regions"]) != expected_regions or set(provenance["donor_regions"]) & set(cal_regions + [target_region]):
                    raise ValueError("response donor leakage or incomplete source bank")
                fold_features.append({"profile_id": row.profile_id, "tau": row.tau, **features})
                feature_rows.append({"target_region": target_region, "seed": seed, "query_region": row.region,
                                     "query_month": row.month, "profile_id": row.profile_id, "tau": row.tau, **features})
                if row.tau == 600:
                    provenance_rows.append({"target_region": target_region, "seed": seed, "query_region": row.region,
                        "query_month": row.month, "profile_id": row.profile_id, "donor_regions": json.dumps(provenance["donor_regions"]),
                        "donor_ids": json.dumps(provenance["donor_ids"]), "pooled_weights": json.dumps(provenance["pooled_weights"]),
                        "conditional_weights": json.dumps(provenance["conditional_weights"])})
            augmented = panel.merge(pd.DataFrame(fold_features), on=["profile_id", "tau"], validate="one_to_one")
            if not np.isfinite(augmented[common + list(RESPONSE_FEATURES)].to_numpy(float)).all():
                raise ValueError("nonfinite model features")
            for tau in (600, 900):
                frame = augmented.loc[augmented.tau.eq(tau)]
                fit = frame.loc[frame.region.isin(fit_regions)]
                calibration = frame.loc[frame.region.isin(cal_regions)]
                target = frame.loc[frame.region.eq(target_region)]
                all_sources = frame.loc[frame.region.ne(target_region)]
                aa, ab, asc = initial(all_sources, "anchor_all_sources")
                anchor_scores = scores(all_sources, aa, ab, asc)
                centers = all_sources.groupby("region")[[f"fuelshare_{f}" for f in v140.v90.FUELS]].mean()
                for method in METHODS:
                    fitted = fit_heads(fit, method, common) if method in LEARNED else None
                    if method.startswith("anchor"):
                        pool = anchor_scores
                    elif method == "physical":
                        pool = pd.Series(0.0, index=cal_regions)
                    else:
                        ca, cb, cs = initial(calibration, method, fitted)
                        pool = scores(calibration, ca, cb, cs)
                    a, b, scale = initial(target, method, fitted)
                    for j, row in enumerate(target.itertuples(index=False)):
                        chosen = pool
                        if method == "anchor_nearest12_sources":
                            x = np.asarray([getattr(row, f"fuelshare_{f}") for f in v140.v90.FUELS])
                            distance = (centers - x).abs().sum(axis=1)
                            nearest = sorted(distance.index, key=lambda r: (distance.loc[r], r))[:12]
                            chosen = anchor_scores.loc[nearest]
                        if target_region in chosen.index or (not method.startswith("anchor") and len(chosen) != 12):
                            raise ValueError("calibration role/denominator failure")
                        for alpha in v140.ALPHAS:
                            rank = math.ceil((len(chosen) + 1) * (1 - alpha))
                            if rank > len(chosen):
                                raise ValueError("nonfinite group quantile")
                            radius = float(chosen.sort_values().iloc[rank - 1])
                            lower = max(int(row.physical_lower), math.floor((a[j] - radius * scale[j]) * row.hours + 1e-9))
                            upper = min(int(row.physical_upper), math.ceil((b[j] + radius * scale[j]) * row.hours - 1e-9))
                            if not row.physical_lower <= lower <= upper <= row.physical_upper:
                                failures.append({"region": target_region, "month": row.month, "tau": tau, "seed": seed,
                                                 "method": method, "alpha": alpha, "check": "physical_subset"})
                            outputs.append({**record(row, seed, method, alpha, lower, upper),
                                            "factor_floor": row.factor_box_K_upper - row.factor_box_K_lower})
                            radii.append({"target_region": target_region, "month": row.month, "tau": tau, "seed": seed,
                                          "method": method, "alpha": alpha, "radius": radius, "scale": scale[j],
                                          "calibration_regions": len(chosen), "rank": rank,
                                          "calibration_roster": json.dumps(sorted(chosen.index))})
            print(f"risk responses target {target_region}, source seed {seed}", flush=True)
    rows = pd.DataFrame(outputs)
    keys = ["region", "month", "tau", "seed", "method", "alpha"]
    if len(rows) != 66 * 2 * 2 * 2 * len(METHODS) or rows.duplicated(keys).any():
        raise ValueError("complete target scoring denominator failure")
    for key, group in rows.groupby(keys[:-1]):
        narrow, wide = group.sort_values("alpha", ascending=False).iloc[0], group.sort_values("alpha").iloc[0]
        if wide.lower > narrow.lower or wide.upper < narrow.upper:
            failures.append({"key": str(key), "check": "alpha_nesting"})
    frontier = rows.groupby(["region", "seed", "tau", "method", "alpha"]).agg(
        coverage=("covered", "mean"), width=("width", "mean"), robust_interval_score90=("robust_interval_score90", "mean"),
        months=("month", "size"), factor_floor=("factor_floor", "mean"),
        positive_01=("positive_0.01", "mean"), negative_01=("negative_0.01", "mean"),
        certificate_error_01=("error_0.01", "mean")).reset_index()
    if protocol_hash != v139.sha(PROTOCOL):
        raise RuntimeError("protocol changed during run")
    OUT.mkdir(parents=True)
    rows.to_csv(OUT / "target_rows.csv", index=False)
    frontier.to_csv(OUT / "target_frontier.csv", index=False)
    pd.DataFrame(roles).to_csv(OUT / "region_roles.csv", index=False)
    pd.DataFrame(radii).to_csv(OUT / "calibration_radii.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(OUT / "query_response_features.csv", index=False)
    pd.DataFrame(provenance_rows).to_csv(OUT / "query_donor_provenance.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "assertion_failures.csv", index=False)
    paths = [PROTOCOL, INPUT / "summary.json", INPUT / "canonical_monthly_panel.csv", RESPONSES / "summary.json",
             RESPONSES / "response_counts.npz", V140 / "summary.json", V140 / "region_roles.csv", V142 / "region_roles.csv", ADDITIONS]
    payload = {"status": "RECOMPOSITION_AND_RESPONSE_INFORMED_BASELINE_DEVELOPMENT", "target_region_months": 66,
               "scored_rows": len(rows), "methods": METHODS, "assertion_failures": len(failures),
               "response_feature_rows": len(feature_rows), "donor_provenance_rows": len(provenance_rows),
               "input_hashes": {str(p.relative_to(ROOT)): v139.sha(p) for p in paths},
               "script_sha256": v139.sha(Path(__file__)), "response_helper_sha256": v139.sha(ROOT / "scripts/monthly_tail_response.py"),
               "common_features": common, "response_features": RESPONSE_FEATURES,
               "frontier": json.loads(frontier.to_json(orient="records")),
               "limitations": ["Previously examined development targets and retrospective monthly totals, not blind confirmation.",
                               "Recomposition, conditional weighting, residual CQR and scale normalization are established baselines.",
                               "Donor profile invariance and flat absent-fuel profiles are assumptions.",
                               "Proper-fit response features use 29-region teachers; calibration/target features use a fixed 30-region teacher.",
                               "Canonical assumed factors and proxy labels are not empirical Chinese uncertainty or hourly forecast accuracy."]}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(frontier.loc[frontier.alpha.eq(.1)].to_string(index=False))
    print(f"assertion failures: {len(failures)}")


if __name__ == "__main__":
    main()
