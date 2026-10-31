"""V265: summarize the EF precision needed to avoid threshold-side crossing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V265_EF_PRECISION_FRONTIER_PROTOCOL.md"
INPUT = ROOT / "results/v240-ef-uncertainty-temporal-risk-r1/threshold_side_certificate.csv"
OUT = ROOT / "results/v265-ef-precision-frontier-r1"
TOLERANCES = (0., .025, .05, .075, .10, .15, .20, .30, .40, .50, .60)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash = sha256(PROTOCOL)
    source = pd.read_csv(INPUT)
    required = {"region", "year", "month", "nominal_monthly_mean_cif",
                "critical_relative_ef_tolerance_to_reach_threshold", "threshold_star"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"V240 certificate missing columns: {sorted(missing)}")
    if source.duplicated(["region", "year", "month"]).any() or len(source) != 16:
        raise ValueError("expected 16 unique frozen V240 BA-month certificates")
    critical = source["critical_relative_ef_tolerance_to_reach_threshold"].to_numpy(float)
    rows = []
    for tolerance in TOLERANCES:
        for index, row in source.iterrows():
            crossed = tolerance + 1e-12 >= float(row.critical_relative_ef_tolerance_to_reach_threshold)
            rows.append({"region": row.region, "year": int(row.year), "month": int(row.month),
                         "threshold_star": float(row.threshold_star),
                         "nominal_monthly_mean_cif": float(row.nominal_monthly_mean_cif),
                         "critical_relative_ef_tolerance": float(row.critical_relative_ef_tolerance_to_reach_threshold),
                         "assumed_relative_ef_tolerance": tolerance,
                         "threshold_crossed": bool(crossed),
                         "threshold_side_gate_passed": not crossed})
    frontier = pd.DataFrame(rows)
    summary = frontier.groupby("assumed_relative_ef_tolerance", as_index=False).agg(
        ba_months=("region", "size"), threshold_crossed_count=("threshold_crossed", "sum"),
        threshold_crossed_fraction=("threshold_crossed", "mean"))
    critical_table = source[["region", "year", "month", "nominal_monthly_mean_cif",
                             "threshold_star", "critical_relative_ef_tolerance_to_reach_threshold"]].copy()
    OUT.mkdir(parents=True)
    frontier.to_csv(OUT / "ba_month_precision_frontier.csv", index=False)
    summary.to_csv(OUT / "tolerance_summary.csv", index=False)
    critical_table.to_csv(OUT / "critical_precision_by_ba_month.csv", index=False)
    metadata = {"status": "analytic_ef_precision_frontier_not_empirical_error_distribution",
                "protocol_sha256": protocol_hash, "script_sha256": sha256(Path(__file__)),
                "input_sha256": sha256(INPUT), "n_ba_months": int(len(source)),
                "threshold": float(source.threshold_star.iloc[0]),
                "relative_tolerances": list(TOLERANCES),
                "min_critical_tolerance": float(np.min(critical)),
                "max_critical_tolerance": float(np.max(critical)),
                "calculation": "critical r=abs(tau-mu)/mu for symmetric multiplicative uncertainty in positive fuel EFs",
                "caveat": "V240 fixed-EF EIA proxy cohort already opened; this is not a China EF uncertainty estimate"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(summary.to_string(index=False))
    print(critical_table.sort_values("critical_relative_ef_tolerance_to_reach_threshold").to_string(index=False))


if __name__ == "__main__":
    main()
