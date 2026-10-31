"""V146: calibrate fixed-donor risk curves jointly over two thresholds."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from critical_composition_response import CriticalCompositionResponses, DELTAS
from monthly_tail_response import MonthlyTailResponses
from v139_physical_reserve_calibration import sha
from v142_pace_lgee_monthly_holdout import record

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V146_CRITICAL_COMPOSITION_PROTOCOL.md"
INPUT = ROOT / "results/v144-canonical-profile-cache-r1"
RESPONSES = ROOT / "results/v144-composition-response-cache-r1"
REFERENCE = ROOT / "results/v145-recomposed-risk-baselines-r1"
OUT = ROOT / "results/v146-critical-composition-calibration-r1"
TAUS = (600, 900)
METHODS = ("horizontal_composition_joint", "vertical_hull_joint")
ALPHAS = (.1, .2)


def inverse_index(curves: np.ndarray, labels: np.ndarray) -> int:
    """First nested grid band containing both endpoints at both thresholds."""
    covered = ((curves[:, :, 0] <= labels[:, 0, None] + 1e-12)
               & (curves[:, :, 1] >= labels[:, 1, None] - 1e-12)).all(axis=0)
    indices = np.flatnonzero(covered)
    return int(indices[0]) if len(indices) else -1


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["region", "seed", "method", "alpha", "calibration_scope"]
    frontier = rows.groupby([*keys, "tau"]).agg(
        coverage=("covered", "mean"), width=("width", "mean"),
        robust_interval_score90=("robust_interval_score90", "mean"),
        months=("month", "size"), factor_floor=("factor_floor", "mean"),
    ).reset_index()
    monthly = rows.groupby([*keys, "month"]).agg(
        joint_covered=("covered", "all"), mean_threshold_width=("width", "mean"),
        threshold_count=("tau", "nunique"),
    ).reset_index()
    if not monthly.threshold_count.eq(2).all():
        raise ValueError("joint evaluation requires both registered thresholds")
    joint = monthly.groupby(keys).agg(
        joint_coverage=("joint_covered", "mean"),
        mean_threshold_width=("mean_threshold_width", "mean"), months=("month", "size"),
    ).reset_index()
    return frontier, joint


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = [PROTOCOL, INPUT / "summary.json", INPUT / "canonical_monthly_panel.csv",
             INPUT / "profile_index.csv", INPUT / "profiles.npz", RESPONSES / "summary.json",
             RESPONSES / "response_counts.npz", REFERENCE / "region_roles.csv",
             REFERENCE / "target_rows.csv", REFERENCE / "summary.json",
             Path(__file__), ROOT / "scripts/critical_composition_response.py",
             ROOT / "scripts/monthly_tail_response.py", ROOT / "scripts/v142_pace_lgee_monthly_holdout.py",
             ROOT / "scripts/v144_cache_composition_responses.py", ROOT / "scripts/v144_prepare_profile_cache.py"]
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    prepared = json.loads((INPUT / "summary.json").read_text())
    cached = json.loads((RESPONSES / "summary.json").read_text())
    reference_summary = json.loads((REFERENCE / "summary.json").read_text())
    if (cached["assertion_failures"] or reference_summary["assertion_failures"]
            or sha(INPUT / "profiles.npz") != prepared["profiles_sha256"]
            or sha(INPUT / "canonical_monthly_panel.csv") != prepared["panel_sha256"]
            or sha(RESPONSES / "response_counts.npz") != cached["response_sha256"]
            or sha(ROOT / "scripts/monthly_tail_response.py") != reference_summary["response_helper_sha256"]):
        raise ValueError("archived cache/helper hash or assertion failure")
    panel = pd.read_csv(INPUT / "canonical_monthly_panel.csv")
    index = pd.read_csv(INPUT / "profile_index.csv")
    roles = pd.read_csv(REFERENCE / "region_roles.csv")
    fuels = ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")
    metadata = panel.loc[panel.tau.eq(600), ["profile_id", "region", "month", "hours",
                                          *[f"fuelshare_{f}" for f in fuels]]]
    if not index[["profile_id", "region", "month", "hours"]].equals(
            metadata[["profile_id", "region", "month", "hours"]].reset_index(drop=True)):
        raise ValueError("profile and metadata index differ")
    if panel.duplicated(["profile_id", "tau"]).any():
        raise ValueError("duplicate query/threshold row")
    if not (panel.physical_lower.le(panel.factor_box_K_lower)
            & panel.factor_box_K_lower.le(panel.factor_box_K_upper)
            & panel.factor_box_K_upper.le(panel.physical_upper)).all():
        raise ValueError("proxy endpoints outside actual-margin physical hull")
    counts = np.load(RESPONSES / "response_counts.npz", allow_pickle=False)
    packed = np.load(INPUT / "profiles.npz", allow_pickle=False)
    engine = MonthlyTailResponses(metadata, counts["low"], counts["high"])
    critical = CriticalCompositionResponses(engine, index, packed["generation"], packed["margin_fractions"])
    lookup = panel.set_index(["profile_id", "tau"], drop=False)
    outputs, inverse_rows, radii, provenance, curve_index, saved_curves = [], [], [], [], [], []
    pure_coal_queries = 0
    with threadpool_limits(limits=2, user_api="blas"):
        for target in ("PACE", "LGEE", "SC"):
            for seed in (41, 83):
                fold = roles.loc[roles.target_region.eq(target) & roles.seed.eq(seed)]
                fit = tuple(sorted(fold.loc[fold.role.eq("fit"), "region"]))
                cal = tuple(sorted(fold.loc[fold.role.eq("calibration"), "region"]))
                if (len(set(fit)) != 30 or len(set(cal)) != 12 or set(fit) & set(cal)
                        or target in fit + cal or len(fold) != 42):
                    raise ValueError("frozen region-role separation failed")
                queries = metadata.loc[metadata.region.isin((*cal, target))]
                fold_curves = {}
                score_records = []
                for position, query in enumerate(queries.itertuples(index=False)):
                    qid = int(query.profile_id)
                    observed = lookup.loc[[(qid, tau) for tau in TAUS]]
                    physical = observed[["physical_lower", "physical_upper"]].to_numpy(float) / query.hours
                    if query.fuelshare_coal >= 1:
                        horizontal = np.broadcast_to(physical[:, None, :], (2, len(DELTAS), 2)).copy()
                        donor_info = {"donor_ids": [], "donor_regions": [], "weights": []}
                        pure_coal_queries += 1
                    else:
                        horizontal, donor_info = critical.curves(qid, fit)
                        if set(donor_info["donor_regions"]) != set(fit):
                            raise ValueError("curve source bank changed or query/calibration leakage")
                        horizontal = np.clip(horizontal, physical[:, None, 0, None], physical[:, None, 1, None])
                    initial = horizontal[:, 0, :]
                    vertical = (initial[:, None, :] * (1 - DELTAS[None, :, None])
                                + physical[:, None, :] * DELTAS[None, :, None])
                    curves = np.stack([horizontal, vertical])
                    if (not np.isfinite(curves).all()
                            or np.any(np.diff(curves[:, :, :, 0], axis=2) > 1e-12)
                            or np.any(np.diff(curves[:, :, :, 1], axis=2) < -1e-12)
                            or np.any(curves[:, :, :, 0] > curves[:, :, :, 1] + 1e-12)):
                        raise ValueError("nonfinite/non-nested joint response family")
                    fold_curves[qid] = curves
                    curve_index.append({"curve_id": len(saved_curves), "target_region": target, "seed": seed,
                                        "profile_id": qid, "query_region": query.region, "month": query.month,
                                        "role": "target" if query.region == target else "calibration"})
                    saved_curves.append(curves)
                    provenance.append({**curve_index[-1], **{k: json.dumps(v) for k, v in donor_info.items()}})
                    # Target labels enter only later scoring; radius selection sees calibration labels.
                    if query.region != target:
                        labels = observed[["factor_box_K_lower", "factor_box_K_upper"]].to_numpy(float) / query.hours
                        for method_index, method in enumerate(METHODS):
                            grid_index = inverse_index(curves[method_index], labels)
                            item = {"target_region": target, "seed": seed, "method": method,
                                    "region": query.region, "month": query.month, "profile_id": qid,
                                    "grid_index": grid_index, "supported": grid_index >= 0,
                                    "delta": float(DELTAS[grid_index]) if grid_index >= 0 else None}
                            score_records.append(item)
                            inverse_rows.append(item)
                    if (position + 1) % 100 == 0:
                        print(f"curves {target}/{seed}: {position + 1}/{len(queries)}", flush=True)
                scores = pd.DataFrame(score_records)
                for method_index, method in enumerate(METHODS):
                    selected = scores.loc[scores.method.eq(method)].copy()
                    selected["ordered_index"] = selected.grid_index.where(selected.supported, len(DELTAS))
                    region_scores = selected.groupby("region").ordered_index.max().sort_values()
                    if set(region_scores.index) != set(cal):
                        raise ValueError("inverse-score calibration denominator mismatch")
                    for alpha in ALPHAS:
                        rank = math.ceil((len(cal) + 1) * (1 - alpha))
                        chosen_index = int(region_scores.iloc[rank - 1])
                        fallback = chosen_index == len(DELTAS)
                        radii.append({"target_region": target, "seed": seed, "method": method, "alpha": alpha,
                                      "rank": rank, "calibration_regions": len(cal),
                                      "grid_index": -1 if fallback else chosen_index,
                                      "delta": None if fallback else float(DELTAS[chosen_index]),
                                      "physical_fallback": fallback,
                                      "unsupported_calibration_months": int((~selected.supported).sum()),
                                      "unsupported_calibration_regions": int(region_scores.eq(len(DELTAS)).sum())})
                        for row in panel.loc[panel.region.eq(target)].itertuples(index=False):
                            if fallback:
                                lower, upper = int(row.physical_lower), int(row.physical_upper)
                            else:
                                a, b = fold_curves[row.profile_id][method_index, TAUS.index(row.tau), chosen_index]
                                lower = max(int(row.physical_lower), math.floor(a * row.hours + 1e-9))
                                upper = min(int(row.physical_upper), math.ceil(b * row.hours - 1e-9))
                            if not row.physical_lower <= lower <= upper <= row.physical_upper:
                                raise ValueError("target interval outside physical hull")
                            outputs.append({**record(row, seed, method, alpha, lower, upper),
                                            "profile_id": row.profile_id, "physical_fallback": fallback,
                                            "calibration_scope": "joint_two_thresholds",
                                            "factor_floor": row.factor_box_K_upper - row.factor_box_K_lower})
                print(f"completed {target}/{seed}; {critical.identity_checks} donor-count identities", flush=True)
    rows = pd.DataFrame(outputs)
    keys = ["region", "month", "tau", "seed", "method", "alpha"]
    if len(rows) != 66 * 2 * 2 * 2 * len(METHODS) or rows.duplicated(keys).any():
        raise ValueError("target scoring denominator mismatch")
    for _, group in rows.groupby(keys[:-1]):
        wide, narrow = group.sort_values("alpha").itertuples(index=False)
        if wide.lower > narrow.lower or wide.upper < narrow.upper:
            raise ValueError("alpha nesting failure")
    archived = pd.read_csv(REFERENCE / "target_rows.csv")
    archived["calibration_scope"] = "per_threshold_archived_v145"
    combined = pd.concat([rows, archived], ignore_index=True)
    frontier, joint = summarize(combined)
    if any(sha(ROOT / name) != digest for name, digest in input_hashes.items()):
        raise RuntimeError("input/code/protocol changed during the run")
    OUT.mkdir(parents=True)
    rows.to_csv(OUT / "target_rows.csv", index=False)
    frontier.to_csv(OUT / "comparison_frontier.csv", index=False)
    joint.to_csv(OUT / "joint_two_threshold_coverage.csv", index=False)
    pd.DataFrame(inverse_rows).to_csv(OUT / "calibration_inverse_scores.csv", index=False)
    pd.DataFrame(radii).to_csv(OUT / "calibration_radii.csv", index=False)
    pd.DataFrame(provenance).to_csv(OUT / "donor_provenance.csv", index=False)
    pd.DataFrame(curve_index).to_csv(OUT / "curve_index.csv", index=False)
    roles.to_csv(OUT / "region_roles.csv", index=False)
    np.savez_compressed(OUT / "curves.npz", curves=np.stack(saved_curves), deltas=DELTAS,
                        methods=np.asarray(METHODS), taus=np.asarray(TAUS))
    payload = {"status": "JOINT_CRITICAL_COMPOSITION_DEVELOPMENT", "target_region_months": 66,
               "scored_rows": len(rows), "curve_queries": len(saved_curves), "assertion_failures": 0,
               "donor_count_identity_checks": critical.identity_checks,
               "noncoal_exact_decisions": critical.noncoal_exact_decisions,
               "critical_tie_corrections": critical.critical_tie_corrections,
               "pure_coal_queries": pure_coal_queries, "input_hashes": input_hashes,
               "output_hashes": {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()},
               "limitations": ["Opened development targets; no new region/time confirmation.",
                   "Virtual coal displacement measures model mismatch, not uncertainty in observed monthly shares.",
                   "Generic inverse conformal calibration is prior art; novelty is unproven.",
                   "Empirical monthly factor-box endpoint coverage is not hourly CIF forecast coverage.",
                   "Archived V145 references use separate per-threshold calibration; V146 uses a joint score."]}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(frontier.loc[frontier.alpha.eq(.1) & frontier.method.isin(METHODS)].to_string(index=False))
    print("assertion failures: 0", flush=True)


if __name__ == "__main__":
    main()
