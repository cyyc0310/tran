"""V148: order sensitivity and explicit pinball-tie strong-baseline controls."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

from monthly_tail_response import RESPONSE_FEATURES
from strict_tie_quantile import StrictTieQuantileRegressor, audit_gradient_contract
from v139_physical_reserve_calibration import sha
from v142_pace_lgee_monthly_holdout import record
from v145_recomposed_risk_baselines import initial, scores

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V148_QUANTILE_STABILITY_PROTOCOL.md"
INPUT = ROOT / "results/v144-canonical-profile-cache-r1"
BASE = ROOT / "results/v145-recomposed-risk-baselines-r1"
OUT = ROOT / "results/v148-quantile-stability-r2"
BASES = ("direct_common_all", "direct_response_all", "residual_response_all")
VARIANTS = ("library_default", "split_guard", "strict_tie_guard")
ORDERS = ("canonical", "reverse")


def fit_heads(frame, base, variant, common):
    features = common if base == "direct_common_all" else common + list(RESPONSE_FEATURES)
    if any("factor_box_K" in name or "factor_floor" in name or "nominal_K" in name for name in features):
        raise ValueError("labels in feature whitelist")
    weight = 1 / frame.groupby("region").region.transform("size").to_numpy(float)
    model_class = StrictTieQuantileRegressor if variant == "strict_tie_guard" else GradientBoostingRegressor
    heads, diagnostics, gradients = [], [], []
    for q, side in ((.05, "low"), (.95, "high")):
        y = frame[f"factor_box_K_{'lower' if side == 'low' else 'upper'}"].to_numpy(float) / frame.hours.to_numpy(float)
        if base.startswith("residual"):
            y -= frame[f"rsp_conditional_{side}_median"].to_numpy(float)
        model = model_class(loss="quantile", alpha=q, n_estimators=150, learning_rate=.05,
            max_depth=2, min_samples_leaf=15, random_state=20261028,
            min_impurity_decrease=0.0 if variant == "library_default" else 1e-10)
        model.fit(frame[features], y, sample_weight=weight)
        raw = model.init_.predict(frame[features])
        gradient = -model._loss.gradient(y, raw)
        gradients.append(np.asarray(gradient, dtype=np.float64))
        first = model.estimators_[0, 0].tree_
        diagnostics.append({"side": side, "initial_prediction": float(raw[0]),
            "gradient_unique_values": int(np.unique(gradient).size), "first_tree_nodes": int(first.node_count),
            "first_root_feature": int(first.feature[0]), "first_root_impurity": float(first.impurity[0]),
            "first_min_impurity": float(first.impurity.min()), "training_rows": len(frame),
            "training_positive_rows": int((y > 0).sum()), "positive_weight": float(weight[y > 0].sum()),
            "total_weight": float(weight.sum())})
        heads.append(model)
    return (heads, features), diagnostics, gradients


def endpoint_summary(rows):
    data = rows.assign(lower_covered=rows.lower.le(rows.factor_box_K_lower),
                       upper_covered=rows.upper.ge(rows.factor_box_K_upper))
    return data.groupby(["region", "seed", "tau", "base", "variant", "order", "alpha"]).agg(
        coverage=("covered", "mean"), lower_coverage=("lower_covered", "mean"),
        upper_coverage=("upper_covered", "mean"), width=("width", "mean"),
        robust_interval_score90=("robust_interval_score90", "mean"), months=("month", "size"),
    ).reset_index()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if sklearn.__version__ != "1.8.0":
        raise RuntimeError("numerical controls are pinned to sklearn 1.8.0")
    audit_gradient_contract()
    paths = [PROTOCOL, INPUT / "canonical_monthly_panel.csv", INPUT / "summary.json", BASE / "summary.json",
        BASE / "query_response_features.csv", BASE / "region_roles.csv", BASE / "target_rows.csv",
        Path(__file__), ROOT / "scripts/strict_tie_quantile.py", ROOT / "scripts/monthly_tail_response.py",
        ROOT / "scripts/v145_recomposed_risk_baselines.py", ROOT / "scripts/v142_pace_lgee_monthly_holdout.py"]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    baseline = json.loads((BASE / "summary.json").read_text())
    if baseline["assertion_failures"] or sha(ROOT / "scripts/v145_recomposed_risk_baselines.py") != baseline["script_sha256"]:
        raise ValueError("baseline audit/script differs")
    panel = pd.read_csv(INPUT / "canonical_monthly_panel.csv")
    # Round-trip reading preserves stored derived feature floats used by V145.
    features = pd.read_csv(BASE / "query_response_features.csv", float_precision="round_trip")
    roles = pd.read_csv(BASE / "region_roles.csv")
    reference = pd.read_csv(BASE / "target_rows.csv")
    common = baseline["common_features"]
    outputs, diagnostics, radii, region_scores, fit_orders = [], [], [], [], []
    gradient_blocks, gradient_profiles, gradient_manifest = [], [], []
    gradient_offset = 0
    for target in ("PACE", "LGEE", "SC"):
        for seed in (41, 83):
            roster = roles.loc[roles.target_region.eq(target) & roles.seed.eq(seed)]
            fit = tuple(roster.loc[roster.role.eq("fit"), "region"])
            cal = tuple(roster.loc[roster.role.eq("calibration"), "region"])
            if len(set(fit)) != 30 or len(set(cal)) != 12 or set(fit) & set(cal) or target in fit + cal:
                raise ValueError("role separation failure")
            source_features = features.loc[features.target_region.eq(target) & features.seed.eq(seed),
                                          ["profile_id", "tau", *RESPONSE_FEATURES]]
            augmented = panel.merge(source_features, on=["profile_id", "tau"], validate="one_to_one")
            for tau in (600, 900):
                frame = augmented.loc[augmented.tau.eq(tau)]
                canonical = frame.loc[frame.region.isin(fit)].sort_values("profile_id")
                calibration = frame.loc[frame.region.isin(cal)]
                targets = frame.loc[frame.region.eq(target)]
                for order in ORDERS:
                    training = canonical if order == "canonical" else canonical.iloc[::-1]
                    ids = training.profile_id.to_numpy(dtype=np.int64)
                    fit_orders.append({"target_region": target, "seed": seed, "tau": tau, "order": order,
                        "profile_ids": json.dumps(ids.tolist()), "sha256": hashlib.sha256(ids.tobytes()).hexdigest()})
                    for base in BASES:
                        for variant in VARIANTS:
                            fitted, fit_diagnostics, gradients = fit_heads(training, base, variant, common)
                            spec = {"target_region": target, "seed": seed, "tau": tau,
                                    "base": base, "variant": variant, "order": order}
                            diagnostics.extend({**spec, **item} for item in fit_diagnostics)
                            for item, gradient in zip(fit_diagnostics, gradients):
                                gradient_manifest.append({**spec, "side": item["side"], "offset": gradient_offset,
                                    "stop": gradient_offset + len(gradient),
                                    "gradient_sha256": hashlib.sha256(gradient.tobytes()).hexdigest(),
                                    "profile_sha256": hashlib.sha256(ids.tobytes()).hexdigest()})
                                gradient_blocks.append(gradient)
                                gradient_profiles.append(ids.copy())
                                gradient_offset += len(gradient)
                            ca, cb, cs = initial(calibration, base, fitted)
                            pool = scores(calibration, ca, cb, cs)
                            if set(pool.index) != set(cal):
                                raise ValueError("calibration score denominator failure")
                            region_scores.extend({**spec, "calibration_region": region, "score": float(value)}
                                                 for region, value in pool.items())
                            a, b, scale = initial(targets, base, fitted)
                            for alpha in (.1, .2):
                                rank = math.ceil(13 * (1 - alpha))
                                radius = float(pool.sort_values().iloc[rank - 1])
                                radii.append({**spec, "alpha": alpha, "rank": rank, "radius": radius})
                                for j, row in enumerate(targets.itertuples(index=False)):
                                    lower = max(int(row.physical_lower), math.floor((a[j] - radius * scale[j]) * row.hours + 1e-9))
                                    upper = min(int(row.physical_upper), math.ceil((b[j] + radius * scale[j]) * row.hours - 1e-9))
                                    if not row.physical_lower <= lower <= upper <= row.physical_upper:
                                        raise ValueError("physical subset failure")
                                    outputs.append({**record(row, seed, base, alpha, lower, upper),
                                        "base": base, "variant": variant, "order": order,
                                        "calibration_scope": "per_threshold", "profile_id": row.profile_id})
            print(f"quantile controls completed {target}/{seed}", flush=True)
    rows = pd.DataFrame(outputs)
    keys = ["region", "month", "tau", "seed", "base", "variant", "order", "alpha"]
    if len(rows) != 66 * 2 * 2 * 2 * len(BASES) * len(VARIANTS) * len(ORDERS) or rows.duplicated(keys).any():
        raise ValueError("target scoring denominator failure")
    for _, group in rows.groupby(keys[:-1]):
        wide, narrow = group.sort_values("alpha").itertuples(index=False)
        if wide.lower > narrow.lower or wide.upper < narrow.upper:
            raise ValueError("alpha nesting failure")
    reproduced = rows.loc[rows.variant.eq("library_default") & rows.order.eq("canonical")]
    match_keys = ["region", "month", "tau", "seed", "method", "alpha"]
    comparison = reproduced.merge(reference.loc[reference.method.isin(BASES)], on=match_keys,
                                  suffixes=("_new", "_old"), validate="one_to_one")
    mismatch = comparison.lower_new.ne(comparison.lower_old) | comparison.upper_new.ne(comparison.upper_old)
    if len(comparison) != 66 * 2 * 2 * 2 * len(BASES) or mismatch.any():
        raise ValueError(f"canonical V145 replication failed: {int(mismatch.sum())} differences")
    frontier = endpoint_summary(rows)
    if any(sha(ROOT / name) != value for name, value in hashes.items()):
        raise RuntimeError("input/code/protocol changed during run")
    OUT.mkdir(parents=True)
    for name, frame in (("target_rows", rows), ("target_frontier", frontier), ("training_diagnostics", pd.DataFrame(diagnostics)),
        ("calibration_radii", pd.DataFrame(radii)), ("calibration_region_scores", pd.DataFrame(region_scores)),
        ("fit_order_manifest", pd.DataFrame(fit_orders)), ("region_roles", roles),
        ("gradient_manifest", pd.DataFrame(gradient_manifest))):
        frame.to_csv(OUT / f"{name}.csv", index=False)
    np.savez_compressed(OUT / "initial_source_gradients.npz", gradient=np.concatenate(gradient_blocks),
                        profile_id=np.concatenate(gradient_profiles))
    summary = {"status": "QUANTILE_OPTIMIZER_STABILITY_DEVELOPMENT", "scored_rows": len(rows),
        "target_region_months": 66, "canonical_baseline_reproduced_rows": len(comparison),
        "canonical_baseline_mismatches": int(mismatch.sum()), "assertion_failures": 0,
        "source_gradient_blocks": len(gradient_manifest), "source_gradient_values": gradient_offset,
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__, "input_hashes": hashes,
        "output_hashes": {path.name: sha(path) for path in sorted(OUT.iterdir()) if path.is_file()},
        "limitations": ["Numerical/optimization controls are established baseline choices, not new methodology.",
            "Both fitting orders are reported; no target-selected order is deployed.",
            "Monthly proxy tail-count endpoints only; no hourly forecast gate or unopened-region confirmation."]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(frontier.loc[frontier.alpha.eq(.1) & frontier.order.eq("canonical")].to_string(index=False))
    print("assertion failures: 0", flush=True)


if __name__ == "__main__":
    main()
