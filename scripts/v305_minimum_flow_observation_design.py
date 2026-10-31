"""V305: enumerate minimal route-flow observations identifying import factors."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V305_MINIMUM_FLOW_OBSERVATION_DESIGN_PROTOCOL.md"
MARGINS = ROOT / "data/china/mee_regional_factor_trade_margins_2023.csv"
SUPPORT = ROOT / "data/china/cec_regional_trade_support_2023.csv"
OUT = ROOT / "results/v305-minimum-flow-observation-design-r1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identifies(matrix: np.ndarray, target: np.ndarray) -> bool:
    return np.linalg.matrix_rank(matrix) == np.linalg.matrix_rank(np.vstack([matrix, target]))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    margins = pd.read_csv(MARGINS)
    support = pd.read_csv(SUPPORT)
    regions = margins.region.tolist()
    edges = list(support[["origin_region", "destination_region"]].itertuples(index=False, name=None))
    n = len(edges)
    a_rows = []
    b_rows = []
    for origin in regions:
        a_rows.append([float(o == origin) for o, _ in edges])
        b_rows.append(float(margins.loc[margins.region.eq(origin), "outgoing_10k_kwh"].iloc[0]))
    for destination in regions:
        a_rows.append([float(d == destination) for _, d in edges])
        b_rows.append(float(margins.loc[margins.region.eq(destination), "incoming_10k_kwh"].iloc[0]))
    a = np.asarray(a_rows)
    b = np.asarray(b_rows)
    if np.linalg.matrix_rank(a) != np.linalg.matrix_rank(np.column_stack([a, b])):
        raise ValueError("margins are algebraically inconsistent")
    nullity = n - int(np.linalg.matrix_rank(a))
    if nullity > 2:
        raise ValueError(f"protocol only enumerates up to 2 observations; nullity is {nullity}")
    source_ef = margins.set_index("region").ef_kgco2_kwh.to_dict()
    targets = {}
    for destination in regions:
        if float(margins.loc[margins.region.eq(destination), "incoming_10k_kwh"].iloc[0]) == 0:
            continue
        targets[destination] = np.array([
            source_ef[o] if d == destination else 0.0 for o, d in edges
        ])

    subsets_by_k = {
        k: list(itertools.combinations(range(n), k))
        for k in range(3)
    }
    edge_rows = []
    for i, (origin, destination) in enumerate(edges):
        objective = np.eye(n)[i]
        lo = linprog(objective, A_eq=a, b_eq=b, bounds=(0, None), method="highs")
        hi = linprog(-objective, A_eq=a, b_eq=b, bounds=(0, None), method="highs")
        if not lo.success or not hi.success:
            raise RuntimeError("base flow polytope infeasible")
        edge_rows.append({"edge_index": i, "origin_region": origin,
                          "destination_region": destination,
                          "flow_lower_10k_kwh": float(lo.fun),
                          "flow_upper_10k_kwh": float(-hi.fun),
                          "flow_width_10k_kwh": float(-hi.fun - lo.fun)})

    per_destination = {}
    for destination, target in targets.items():
        found = None
        minimal_sets = []
        for k in range(3):
            candidates = []
            for subset in subsets_by_k[k]:
                extra = np.eye(n)[list(subset)] if subset else np.empty((0, n))
                if identifies(np.vstack([a, extra]), target):
                    candidates.append(subset)
            if candidates:
                found, minimal_sets = k, candidates
                break
        if found is None:
            raise RuntimeError(f"no identification subset within 2 measurements: {destination}")
        per_destination[destination] = {
            "minimum_measurements": found,
            "minimal_subsets": [[int(i) for i in subset] for subset in minimal_sets],
            "minimal_subsets_named": [
                [f"{edges[i][0]}->{edges[i][1]}" for i in subset]
                for subset in minimal_sets
            ],
        }

    joint_min = None
    joint_sets = []
    for k in range(3):
        for subset in subsets_by_k[k]:
            extra = np.eye(n)[list(subset)] if subset else np.empty((0, n))
            matrix = np.vstack([a, extra])
            if all(identifies(matrix, target) for target in targets.values()):
                joint_min, joint_sets = k, [subset]
                # Continue to capture all minimal subsets at this cardinality.
                joint_sets = [candidate for candidate in subsets_by_k[k]
                              if all(identifies(np.vstack([a, np.eye(n)[list(candidate)] if candidate else np.empty((0, n))]), target)
                                     for target in targets.values())]
                break
        if joint_min is not None:
            break
    if joint_min is None:
        raise RuntimeError("no joint identification subset within 2 observations")

    # Independent optimization check: after any minimal observation set, fix its
    # values at one feasible matrix and verify every identified functional collapses.
    feasible = linprog(np.zeros(n), A_eq=a, b_eq=b, bounds=(0, None), method="highs")
    if not feasible.success:
        raise RuntimeError(feasible.message)
    collapse_checks = []
    for destination, spec in per_destination.items():
        for subset in spec["minimal_subsets"]:
            obs_matrix = np.eye(n)[subset] if subset else np.empty((0, n))
            obs_values = feasible.x[subset] if subset else np.empty(0)
            aa = np.vstack([a, obs_matrix])
            bb = np.concatenate([b, obs_values])
            c = targets[destination]
            low = linprog(c, A_eq=aa, b_eq=bb, bounds=(0, None), method="highs")
            high = linprog(-c, A_eq=aa, b_eq=bb, bounds=(0, None), method="highs")
            if not low.success or not high.success:
                raise RuntimeError("conditional LP failed")
            width = float((-high.fun) - low.fun)
            if abs(width) > 1e-7:
                raise RuntimeError(f"functional did not collapse: {destination} width={width}")
            collapse_checks.append({"destination": destination, "subset": subset, "width": width})

    OUT.mkdir(parents=True)
    pd.DataFrame(edge_rows).to_csv(OUT / "route_flow_identification_ranges.csv", index=False)
    summary = {
        "status": "EXACT_MINIMUM_FLOW_SCALAR_IDENTIFICATION_CONDITIONAL_ON_2023_REGION_GRAPH",
        "n_routes": n,
        "constraint_rank": int(np.linalg.matrix_rank(a)),
        "flow_polytope_dimension": nullity,
        "minimum_bilateral_measurements_for_all_import_factor_functionals": joint_min,
        "joint_minimal_subsets_named": [
            [f"{edges[i][0]}->{edges[i][1]}" for i in subset] for subset in joint_sets
        ],
        "per_destination": per_destination,
        "lp_collapse_checks": collapse_checks,
        "protocol_sha256": sha(PROTOCOL), "margins_sha256": sha(MARGINS),
        "route_support_sha256": sha(SUPPORT), "script_sha256": sha(Path(__file__)),
        "limitations": [
            "Identification is conditional on exact 2023 margins, exact official regional factors, and the article's listed route support.",
            "This is a measurement-design calculation; the selected bilateral flows are not yet observed by this project.",
            "No extension to 30 province-level flows, measurement noise, other years, or hourly CIF is established.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
