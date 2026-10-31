"""V307 exploratory sensitivity: partial corridor flow lower bounds."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V307_PARTIAL_CORRIDOR_BOUND_PROTOCOL.md"
MARGINS = ROOT / "data/china/mee_regional_factor_trade_margins_2023.csv"
SUPPORT = ROOT / "data/china/cec_regional_trade_support_2023.csv"
OUT = ROOT / "results/v307-partial-corridor-lower-bound-r1"
OBSERVED_LOWER_100M_KWH = 801.02
RETAINED_SHARES = (0.0, 0.80, 0.90, 0.95, 1.0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUT}")
    margins = pd.read_csv(MARGINS)
    support = pd.read_csv(SUPPORT)
    regions = margins.region.tolist()
    edges = list(support[["origin_region", "destination_region"]].itertuples(index=False, name=None))
    n = len(edges)
    eq_rows, eq_rhs = [], []
    for origin in regions:
        eq_rows.append([float(o == origin) for o, _ in edges])
        eq_rhs.append(float(margins.loc[margins.region.eq(origin), "outgoing_10k_kwh"].iloc[0]))
    for destination in regions:
        eq_rows.append([float(d == destination) for _, d in edges])
        eq_rhs.append(float(margins.loc[margins.region.eq(destination), "incoming_10k_kwh"].iloc[0]))
    a_eq, b_eq = np.asarray(eq_rows), np.asarray(eq_rhs)
    if int(margins.outgoing_10k_kwh.sum()) != int(margins.incoming_10k_kwh.sum()):
        raise ValueError("regional margins do not close")
    edge_index = edges.index(("Southwest", "East"))
    a_ub = np.zeros((1, n))
    a_ub[0, edge_index] = -1.0
    ef = margins.set_index("region").ef_kgco2_kwh.to_dict()

    rows = []
    for retained in RETAINED_SHARES:
        kwargs = {}
        lower_bound_10k_kwh = retained * OBSERVED_LOWER_100M_KWH * 10_000
        if retained > 0:
            kwargs = {"A_ub": a_ub, "b_ub": np.array([-lower_bound_10k_kwh])}
        for destination in regions:
            inbound = float(margins.loc[margins.region.eq(destination), "incoming_10k_kwh"].iloc[0])
            if inbound <= 0:
                continue
            objective = np.array([ef[o] if d == destination else 0.0 for o, d in edges])
            low = linprog(objective, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs", **kwargs)
            high = linprog(-objective, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs", **kwargs)
            if not low.success or not high.success:
                raise RuntimeError(f"LP failed for retained={retained}, destination={destination}")
            lo, hi = float(low.fun / inbound), float(-high.fun / inbound)
            observed = float(low.x[edge_index])
            if observed + 1e-3 < lower_bound_10k_kwh:
                raise RuntimeError("solver output violates added corridor lower bound")
            for origin in regions:
                got = sum(low.x[k] for k, (o, _) in enumerate(edges) if o == origin)
                expected = float(margins.loc[margins.region.eq(origin), "outgoing_10k_kwh"].iloc[0])
                if abs(got - expected) > 1e-3:
                    raise RuntimeError(f"origin balance failed for {origin}")
            for dest in regions:
                got = sum(low.x[k] for k, (_, d) in enumerate(edges) if d == dest)
                expected = float(margins.loc[margins.region.eq(dest), "incoming_10k_kwh"].iloc[0])
                if abs(got - expected) > 1e-3:
                    raise RuntimeError(f"destination balance failed for {dest}")
            rows.append({
                "retained_share_of_line_sum": retained,
                "assumed_sw_to_east_lower_bound_100m_kwh": lower_bound_10k_kwh / 10_000,
                "destination": destination,
                "import_factor_lower": lo,
                "import_factor_upper": hi,
                "width": hi - lo,
                "sw_to_east_value_in_low_solution_100m_kwh": observed / 10_000,
            })

    OUT.mkdir(parents=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "conditional_import_factor_bounds.csv", index=False)
    base = result.loc[result.retained_share_of_line_sum.eq(0)].set_index("destination")
    full = result.loc[result.retained_share_of_line_sum.eq(1)].set_index("destination")
    summary = {
        "status": "POST_HOC_CONDITIONAL_PARTIAL_FLOW_BOUND_SENSITIVITY_NOT_EXTERNAL_VALIDATION",
        "line_flow_candidate_100m_kwh": OBSERVED_LOWER_100M_KWH,
        "retained_shares": RETAINED_SHARES,
        "full_assumption_width_ratio_vs_base": {
            d: float(full.loc[d, "width"] / base.loc[d, "width"])
            for d in base.index if base.loc[d, "width"] > 0
        },
        "protocol_sha256": sha(PROTOCOL),
        "margins_sha256": sha(MARGINS),
        "support_sha256": sha(SUPPORT),
        "script_sha256": sha(Path(__file__)),
        "limitations": [
            "All lower-bound interpretations are conditional on cross-source regional/accounting compatibility.",
            "Line totals do not establish the complete Southwest-East OD flow or forecast-time availability.",
            "Bounds concern imported source factor, not total provincial/regional hourly CIF.",
            "This is a post-hoc sensitivity using a previously opened 2023 target dataset, not confirmation.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(result.loc[result.destination.isin(["East", "Central"]), [
        "retained_share_of_line_sum", "destination", "import_factor_lower", "import_factor_upper", "width"
    ]].to_string(index=False))
    print(json.dumps(summary["full_assumption_width_ratio_vs_base"], indent=2))


if __name__ == "__main__":
    main()
