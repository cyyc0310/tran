"""V304: sharp LP bounds for import-weighted regional CO2 factors."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V304_SHARP_REGIONAL_IMPORT_FACTOR_BOUNDS_PROTOCOL.md"
MARGINS = ROOT / "data/china/mee_regional_factor_trade_margins_2023.csv"
SUPPORT = ROOT / "data/china/cec_regional_trade_support_2023.csv"
OUT = ROOT / "results/v304-sharp-regional-import-factor-bounds-r1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solve_bounds(margins: pd.DataFrame, edges: list[tuple[str, str]], destination: str) -> tuple[float, float]:
    regions = margins.region.tolist()
    edge_index = {edge: i for i, edge in enumerate(edges)}
    n = len(edges)
    a_eq = []
    b_eq = []
    for origin in regions:
        a_eq.append([float(o == origin) for o, _ in edges])
        b_eq.append(float(margins.loc[margins.region.eq(origin), "outgoing_10k_kwh"].iloc[0]))
    for dest in regions:
        a_eq.append([float(d == dest) for _, d in edges])
        b_eq.append(float(margins.loc[margins.region.eq(dest), "incoming_10k_kwh"].iloc[0]))
    source_ef = margins.set_index("region").ef_kgco2_kwh.to_dict()
    inbound = float(margins.loc[margins.region.eq(destination), "incoming_10k_kwh"].iloc[0])
    objective = np.array([source_ef[o] if d == destination else 0.0 for o, d in edges])
    kwargs = {"A_eq": np.asarray(a_eq), "b_eq": np.asarray(b_eq),
              "bounds": (0, None), "method": "highs"}
    low = linprog(objective, **kwargs)
    high = linprog(-objective, **kwargs)
    if not low.success or not high.success:
        raise RuntimeError(f"LP infeasible for {destination}: {low.message}; {high.message}")
    # Verify exact margin satisfaction in solver output, independently of status flags.
    flow = low.x
    for origin in regions:
        observed = sum(flow[k] for k, (o, _) in enumerate(edges) if o == origin)
        expected = float(margins.loc[margins.region.eq(origin), "outgoing_10k_kwh"].iloc[0])
        if abs(observed - expected) > 1e-4:
            raise RuntimeError(f"origin balance failed for {origin}: {observed} vs {expected}")
    for dest in regions:
        observed = sum(flow[k] for k, (_, d) in enumerate(edges) if d == dest)
        expected = float(margins.loc[margins.region.eq(dest), "incoming_10k_kwh"].iloc[0])
        if abs(observed - expected) > 1e-4:
            raise RuntimeError(f"destination balance failed for {dest}: {observed} vs {expected}")
    return float(low.fun / inbound), float(-high.fun / inbound)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    margins = pd.read_csv(MARGINS)
    if len(margins) != 7 or margins.region.nunique() != 7:
        raise ValueError("expected seven MEE regions")
    out_total = int(margins.outgoing_10k_kwh.sum())
    in_total = int(margins.incoming_10k_kwh.sum())
    if out_total != in_total:
        raise ValueError(f"regional margins do not close: out={out_total}, in={in_total}")
    support_df = pd.read_csv(SUPPORT)
    route_edges = list(support_df[["origin_region", "destination_region"]].itertuples(index=False, name=None))
    region_names = margins.region.tolist()
    if len(set(route_edges)) != len(route_edges) or any(o == d for o, d in route_edges):
        raise ValueError("route support contains duplicate or self edges")
    for row in margins.itertuples(index=False):
        if row.outgoing_10k_kwh > 0 and not any(o == row.region for o, _ in route_edges):
            raise ValueError(f"positive source has no route: {row.region}")
        if row.incoming_10k_kwh > 0 and not any(d == row.region for _, d in route_edges):
            raise ValueError(f"positive sink has no route: {row.region}")
    dense_edges = [(o, d) for o in region_names for d in region_names if o != d]

    rows = []
    for graph_name, edges in (("article_route_support", route_edges), ("dense_off_diagonal", dense_edges)):
        for row in margins.itertuples(index=False):
            if row.incoming_10k_kwh == 0:
                continue
            lo, hi = solve_bounds(margins, edges, row.region)
            rows.append({
                "graph": graph_name, "region": row.region, "region_zh": row.region_zh,
                "regional_ef_kgco2_kwh": row.ef_kgco2_kwh,
                "incoming_10k_kwh": row.incoming_10k_kwh,
                "import_factor_lower": lo, "import_factor_upper": hi,
                "import_factor_width": hi - lo,
                "source_factor_range": float(margins.ef_kgco2_kwh.max() - margins.ef_kgco2_kwh.min()),
            })
    result = pd.DataFrame(rows)
    OUT.mkdir(parents=True)
    result.to_csv(OUT / "regional_import_factor_bounds.csv", index=False)
    summary = {
        "status": "SHARP_IMPORT_FACTOR_BOUNDS_OVER_DECLARED_TRANSPORT_POLYTOPES_NOT_FULL_FACTOR_PREDICTION",
        "n_regions": len(margins), "route_support_edges": len(route_edges),
        "dense_off_diagonal_edges": len(dense_edges),
        "outgoing_total_10k_kwh": out_total, "incoming_total_10k_kwh": in_total,
        "regional_import_bounds": result.to_dict("records"),
        "protocol_sha256": sha(PROTOCOL), "margins_sha256": sha(MARGINS),
        "route_support_sha256": sha(SUPPORT), "script_sha256": sha(Path(__file__)),
        "limitations": [
            "Bounds concern imported-electricity source factor only, not the recipient region's full annual EF.",
            "The route-support graph treats unlisted routes as structural zeros; dense graph is reported as a sensitivity.",
            "All trade margins are 2023 aggregates and the exercise is retrospective, not a forecast-time experiment.",
            "Regional bounds cannot resolve within-region provincial flows or hourly source composition.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(result.to_string(index=False))
    print(f"closed margins: outgoing={out_total:,}; incoming={in_total:,} (10,000 kWh)")


if __name__ == "__main__":
    main()
