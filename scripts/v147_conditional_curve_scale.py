"""V147: same-budget source-trained scales for horizontal/vertical curve families."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from threadpoolctl import threadpool_limits

from critical_composition_response import CriticalCompositionResponses, DELTAS
from monthly_tail_response import MonthlyTailResponses, RESPONSE_FEATURES
from v139_physical_reserve_calibration import sha
from v142_pace_lgee_monthly_holdout import record
from v146_critical_composition_calibration import inverse_index, summarize

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V147_CONDITIONAL_CURVE_SCALE_PROTOCOL.md"
INPUT = ROOT / "results/v144-canonical-profile-cache-r1"
RESPONSES = ROOT / "results/v144-composition-response-cache-r1"
BASE = ROOT / "results/v145-recomposed-risk-baselines-r1"
CURVES = ROOT / "results/v146-critical-composition-calibration-r1"
OUT = ROOT / "results/v147-conditional-curve-scale-r1"
TAUS = (600, 900)
FAMILIES = ("horizontal_composition_joint", "vertical_hull_joint")
QUANTILES = (.5, .9)


def feature_matrix(frame: pd.DataFrame, archived: pd.DataFrame, common: list[str]) -> pd.DataFrame:
    """One monthly query, both thresholds' source responses; no endpoint labels."""
    matrix = frame.loc[frame.tau.eq(600), ["profile_id", *common]].set_index("profile_id")
    for tau in TAUS:
        features = archived.loc[archived.tau.eq(tau), ["profile_id", *RESPONSE_FEATURES]].set_index("profile_id")
        features = features.rename(columns={name: f"{name}_tau{tau}" for name in RESPONSE_FEATURES})
        matrix = matrix.join(features, validate="one_to_one")
    if (not matrix.index.is_unique or not np.isfinite(matrix.to_numpy(float)).all()
            or any("factor_box_K" in name or "factor_floor" in name or "nominal_K" in name for name in matrix.columns)):
        raise ValueError("invalid/leaking conditional-scale features")
    return matrix


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = [PROTOCOL, INPUT / "summary.json", INPUT / "canonical_monthly_panel.csv", INPUT / "profile_index.csv",
             INPUT / "profiles.npz", RESPONSES / "summary.json", RESPONSES / "response_counts.npz",
             BASE / "summary.json", BASE / "query_response_features.csv", BASE / "region_roles.csv", BASE / "target_rows.csv",
             CURVES / "summary.json", CURVES / "curves.npz", CURVES / "curve_index.csv",
             CURVES / "calibration_inverse_scores.csv", CURVES / "target_rows.csv",
             Path(__file__), ROOT / "scripts/critical_composition_response.py", ROOT / "scripts/monthly_tail_response.py",
             ROOT / "scripts/v146_critical_composition_calibration.py", ROOT / "scripts/v142_pace_lgee_monthly_holdout.py",
             ROOT / "scripts/v144_cache_composition_responses.py", ROOT / "scripts/v144_prepare_profile_cache.py"]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    canonical = json.loads((INPUT / "summary.json").read_text())
    response = json.loads((RESPONSES / "summary.json").read_text())
    baseline = json.loads((BASE / "summary.json").read_text())
    v146 = json.loads((CURVES / "summary.json").read_text())
    if any(summary.get("assertion_failures", 0) for summary in (canonical, response, baseline, v146)):
        raise ValueError("archived audit failed")
    checks = [(INPUT / "profiles.npz", canonical["profiles_sha256"]),
              (INPUT / "canonical_monthly_panel.csv", canonical["panel_sha256"]),
              (RESPONSES / "response_counts.npz", response["response_sha256"]),
              (ROOT / "scripts/monthly_tail_response.py", baseline["response_helper_sha256"])]
    checks += [(CURVES / name, digest) for name, digest in v146["output_hashes"].items()]
    if any(sha(path) != digest for path, digest in checks):
        raise ValueError("archived artifact hash mismatch")
    panel = pd.read_csv(INPUT / "canonical_monthly_panel.csv")
    profile_index = pd.read_csv(INPUT / "profile_index.csv")
    roles = pd.read_csv(BASE / "region_roles.csv")
    feature_rows = pd.read_csv(BASE / "query_response_features.csv")
    old_inverse = pd.read_csv(CURVES / "calibration_inverse_scores.csv")
    curve_index = pd.read_csv(CURVES / "curve_index.csv")
    curve_file = np.load(CURVES / "curves.npz", allow_pickle=False)
    if list(curve_file["methods"]) != list(FAMILIES) or not np.array_equal(curve_file["deltas"], DELTAS):
        raise ValueError("archived curve convention mismatch")
    cached_curves = curve_file["curves"]
    common = baseline["common_features"]
    monthly = panel.loc[panel.tau.eq(600)].set_index("profile_id", drop=False)
    metadata = monthly[["profile_id", "region", "month", "hours",
                        *[f"fuelshare_{f}" for f in ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")]]]
    if not profile_index[["profile_id", "region", "month", "hours"]].equals(
            metadata[["profile_id", "region", "month", "hours"]].reset_index(drop=True)):
        raise ValueError("canonical profile alignment failure")
    counts = np.load(RESPONSES / "response_counts.npz", allow_pickle=False)
    raw = np.load(INPUT / "profiles.npz", allow_pickle=False)
    engine = MonthlyTailResponses(metadata.reset_index(drop=True), counts["low"], counts["high"])
    critical = CriticalCompositionResponses(engine, profile_index, raw["generation"], raw["margin_fractions"])
    lookup = panel.set_index(["profile_id", "tau"], drop=False)
    training, provenance, scale_records, calibration_records, radii, outputs, archived_features = [], [], [], [], [], [], []
    with threadpool_limits(limits=2, user_api="blas"):
        for target in ("PACE", "LGEE", "SC"):
            for seed in (41, 83):
                roster = roles.loc[roles.target_region.eq(target) & roles.seed.eq(seed)]
                fit = tuple(sorted(roster.loc[roster.role.eq("fit"), "region"]))
                cal = tuple(sorted(roster.loc[roster.role.eq("calibration"), "region"]))
                if len(set(fit)) != 30 or len(set(cal)) != 12 or set(fit) & set(cal) or target in fit + cal:
                    raise ValueError("role separation failure")
                fold_features = feature_rows.loc[feature_rows.target_region.eq(target) & feature_rows.seed.eq(seed)]
                x = feature_matrix(panel, fold_features, common)
                archived_features.append(x.reset_index().assign(target_region=target, seed=seed))
                proper = monthly.loc[monthly.region.isin(fit)]
                fold_labels = []
                for position, query in enumerate(proper.itertuples(index=False)):
                    qid = int(query.profile_id)
                    observed = lookup.loc[[(qid, tau) for tau in TAUS]]
                    physical = observed[["physical_lower", "physical_upper"]].to_numpy(float) / query.hours
                    if query.fuelshare_coal >= 1:
                        raise ValueError("unexpected pure-coal fit query requires a new declared protocol")
                    horizontal, source = critical.curves(qid, fit)
                    if set(source["donor_regions"]) != set(fit) - {query.region}:
                        raise ValueError("fit-curve teacher contains own/calibration/target region")
                    horizontal = np.clip(horizontal, physical[:, None, 0, None], physical[:, None, 1, None])
                    initial = horizontal[:, 0, :]
                    vertical = initial[:, None, :] * (1 - DELTAS[None, :, None]) + physical[:, None, :] * DELTAS[None, :, None]
                    labels = observed[["factor_box_K_lower", "factor_box_K_upper"]].to_numpy(float) / query.hours
                    for family, curves in zip(FAMILIES, (horizontal, vertical)):
                        grid_index = inverse_index(curves, labels)
                        item = {"target_region": target, "seed": seed, "region": query.region,
                                "month": query.month, "profile_id": qid, "family": family,
                                "grid_index": grid_index, "supported": grid_index >= 0,
                                "training_delta": float(DELTAS[grid_index]) if grid_index >= 0 else 1.0}
                        fold_labels.append(item)
                        training.append(item)
                    provenance.append({"target_region": target, "seed": seed, "profile_id": qid,
                                       "region": query.region, "donor_regions": json.dumps(source["donor_regions"]),
                                       "donor_ids": json.dumps(source["donor_ids"]), "weights": json.dumps(source["weights"])})
                    if (position + 1) % 200 == 0:
                        print(f"proper-source inverse curves {target}/{seed}: {position + 1}/{len(proper)}", flush=True)
                labels_frame = pd.DataFrame(fold_labels)
                folded_inverse = old_inverse.loc[old_inverse.target_region.eq(target) & old_inverse.seed.eq(seed)]
                folded_index = curve_index.loc[curve_index.target_region.eq(target) & curve_index.seed.eq(seed)].set_index("profile_id")
                predicted_ids = folded_index.index.to_numpy(int)
                for family_index, family in enumerate(FAMILIES):
                    train = labels_frame.loc[labels_frame.family.eq(family)].set_index("profile_id")
                    if set(train.index) != set(proper.index) or len(train) != len(proper):
                        raise ValueError("proper-source training denominator mismatch")
                    weight = 1 / train.groupby("region").region.transform("size").to_numpy(float)
                    calibration = folded_inverse.loc[folded_inverse.method.eq(family)].set_index("profile_id")
                    expected_cal = set(monthly.loc[monthly.region.isin(cal)].index)
                    if set(calibration.index) != expected_cal or len(calibration) != len(expected_cal):
                        raise ValueError("calibration inverse denominator mismatch")
                    for quantile in QUANTILES:
                        method = f"{'horizontal' if family_index == 0 else 'vertical'}_scale_q{int(100 * quantile)}"
                        model = GradientBoostingRegressor(loss="quantile", alpha=quantile, n_estimators=150,
                            learning_rate=.05, max_depth=2, min_samples_leaf=15, random_state=20261028)
                        model.fit(x.loc[train.index], train.training_delta, sample_weight=weight)
                        scale = pd.Series(np.clip(model.predict(x.loc[predicted_ids]), 1 / 256, 1), index=predicted_ids)
                        for qid, value in scale.items():
                            scale_records.append({"target_region": target, "seed": seed, "method": method,
                                "profile_id": qid, "region": monthly.loc[qid, "region"], "month": monthly.loc[qid, "month"],
                                "role": folded_index.loc[qid, "role"], "predicted_scale": float(value)})
                        cal_scale = scale.loc[calibration.index].to_numpy(float)
                        scores = np.where(calibration.supported, calibration.delta.to_numpy(float) / cal_scale, np.inf)
                        region_scores = pd.DataFrame({"region": calibration.region.to_numpy(), "score": scores}).groupby("region").score.max()
                        for j, item in enumerate(calibration.itertuples()):
                            calibration_records.append({"target_region": target, "seed": seed, "method": method,
                                "profile_id": int(item.Index), "region": item.region, "month": item.month,
                                "scale": float(cal_scale[j]), "inverse_delta": float(item.delta) if item.supported else None,
                                "score": float(scores[j]) if np.isfinite(scores[j]) else None,
                                "supported": bool(item.supported)})
                        for alpha in (.1, .2):
                            rank = math.ceil(13 * (1 - alpha))
                            radius = float(region_scores.sort_values().iloc[rank - 1])
                            radii.append({"target_region": target, "seed": seed, "method": method, "alpha": alpha,
                                          "rank": rank, "radius": radius if np.isfinite(radius) else None,
                                          "infinite_radius": not np.isfinite(radius), "calibration_regions": len(region_scores)})
                            for row in panel.loc[panel.region.eq(target)].itertuples(index=False):
                                product = radius * float(scale.loc[row.profile_id])
                                fallback = not np.isfinite(product) or product > 1
                                grid_index = -1 if fallback else min(int(np.searchsorted(DELTAS, product, side="left")), len(DELTAS) - 1)
                                if fallback:
                                    lower, upper = int(row.physical_lower), int(row.physical_upper)
                                else:
                                    curve_id = int(folded_index.loc[row.profile_id, "curve_id"])
                                    a, b = cached_curves[curve_id, family_index, TAUS.index(row.tau), grid_index]
                                    lower = max(int(row.physical_lower), math.floor(a * row.hours + 1e-9))
                                    upper = min(int(row.physical_upper), math.ceil(b * row.hours - 1e-9))
                                if not row.physical_lower <= lower <= upper <= row.physical_upper:
                                    raise ValueError("predicted interval outside actual physical hull")
                                outputs.append({**record(row, seed, method, alpha, lower, upper),
                                    "profile_id": row.profile_id, "physical_fallback": fallback,
                                    "grid_index": grid_index, "predicted_scale": float(scale.loc[row.profile_id]),
                                    "calibration_scope": "joint_two_thresholds",
                                    "factor_floor": row.factor_box_K_upper - row.factor_box_K_lower})
                print(f"completed conditional scales {target}/{seed}", flush=True)
    rows = pd.DataFrame(outputs)
    keys = ["region", "month", "tau", "seed", "method", "alpha"]
    if len(rows) != 66 * 2 * 2 * 2 * 4 or rows.duplicated(keys).any():
        raise ValueError("target denominator failure")
    for (region, _, _, _, _), group in rows.groupby(["region", "tau", "seed", "method", "alpha"]):
        if len(group) != (18 if region == "PACE" else 24):
            raise ValueError("target per-cell denominator failure")
    for _, group in rows.groupby(keys[:-1]):
        wide, narrow = group.sort_values("alpha").itertuples(index=False)
        if wide.lower > narrow.lower or wide.upper < narrow.upper:
            raise ValueError("alpha nesting failure")
    old_rows = pd.read_csv(BASE / "target_rows.csv").assign(calibration_scope="per_threshold_archived_v145")
    joined = pd.concat([rows, pd.read_csv(CURVES / "target_rows.csv"), old_rows], ignore_index=True)
    frontier, joint = summarize(joined)
    if any(sha(ROOT / name) != value for name, value in hashes.items()):
        raise RuntimeError("frozen input/code/protocol changed during run")
    OUT.mkdir(parents=True)
    for name, frame in (("target_rows", rows), ("comparison_frontier", frontier), ("joint_two_threshold_coverage", joint),
        ("fit_inverse_labels", pd.DataFrame(training)), ("fit_donor_provenance", pd.DataFrame(provenance)),
        ("predicted_scales", pd.DataFrame(scale_records)), ("calibration_scores", pd.DataFrame(calibration_records)),
        ("calibration_radii", pd.DataFrame(radii)), ("scale_features", pd.concat(archived_features, ignore_index=True)),
        ("region_roles", roles)):
        frame.to_csv(OUT / f"{name}.csv", index=False)
    payload = {"status": "CONDITIONAL_SCALE_BASELINE_DEVELOPMENT", "target_region_months": 66,
        "scored_rows": len(rows), "proper_fit_queries": len(provenance), "fit_inverse_labels": len(training),
        "unsupported_fit_inverse_labels": int((~pd.DataFrame(training).supported).sum()),
        "donor_count_identity_checks": critical.identity_checks, "assertion_failures": 0,
        "noncoal_exact_decisions": critical.noncoal_exact_decisions,
        "critical_tie_corrections": critical.critical_tie_corrections,
        "feature_columns": list(x.columns), "input_hashes": hashes,
        "output_hashes": {path.name: sha(path) for path in sorted(OUT.iterdir()) if path.is_file()},
        "limitations": ["Conditional scale learning is an ordinary heteroscedastic baseline, not a new algorithm.",
            "Inverse-score training uses 29-region teachers; calibration/target curves use fixed 30-region teachers.",
            "Opened monthly-tail-count development only; original hourly prediction gate is untested."]}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(frontier.loc[frontier.alpha.eq(.1) & frontier.method.str.contains("_scale_q")].to_string(index=False))
    print("assertion failures: 0", flush=True)


if __name__ == "__main__":
    main()
