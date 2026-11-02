#!/usr/bin/env python3
"""Compare one-month versus six-month lagged coarse trajectories for risk."""

from __future__ import annotations

import json
from pathlib import Path

import v275_coarse_mix_conditional_risk as base
from run_lagged_coarse_risk_transfer import INPUT, ROOT, sha256, run_scope

OUTPUT = ROOT / "results/lagged-trajectory-risk-transfer-r1"


def write_monthly(path: Path, rows: list[dict]) -> None:
    # Single-month and trajectory variants record their feature periods in
    # different fields; normalize the CSV schema for this mixed comparison.
    for row in rows:
        row.setdefault("feature_period_end", "")
        row.setdefault("feature_periods", "")
    base.write_csv(path, rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows_by_ba = {ba: base.read_ba(INPUT / f"{ba}.xlsx", max_year=2025) for ba in base.BAS}
    generated = run_scope(rows_by_ba, "generated", ["lag2_cn5", "lag2to7_cn5"])
    consumed = run_scope(rows_by_ba, "consumed", [
        "lag2_cn5", "lag2_cn5_plus_net_import",
        "lag2to7_cn5", "lag2to7_cn5_plus_net_import",
    ])
    for scope, outputs in (("generated", generated), ("consumed", consumed)):
        by_ba, macro, monthly = outputs
        base.write_csv(OUTPUT / f"{scope}_by_ba_metrics.csv", by_ba)
        base.write_csv(OUTPUT / f"{scope}_macro_metrics.csv", macro)
        write_monthly(OUTPUT / f"{scope}_monthly_risk_predictions.csv", monthly)
    provenance = {
        "protocol": "docs/LAGGED_TRAJECTORY_RISK_PROTOCOL_20261101.md",
        "input_directory": str(INPUT.relative_to(ROOT)),
        "target_ba": list(base.BAS),
        "train_years": [2019, 2023],
        "calibration_year": 2024,
        "test_year": 2025,
        "single_month_features": "t-2 five group shares and log total generation",
        "trajectory_features": "ordered t-2 through t-7 five group shares and log total generation per month",
        "consumption_flow_extension": "lagged net imports / consumed electricity appended at each month",
        "risk_model": "standardized Ridge alpha=10; same fitting and leave-BA calibration as single-month r1",
        "label_tracks": "generation and consumption evaluated separately; monthly centers are target-month oracle",
        "source_vintage": "final EIA BA workbook histories; lag-simulated, not historical as-issued vintages",
        "input_sha256": {f"{ba}.xlsx": sha256(INPUT / f"{ba}.xlsx") for ba in base.BAS},
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "generated_macro": generated[1], "consumed_macro": consumed[1]}, indent=2))


if __name__ == "__main__":
    main()
