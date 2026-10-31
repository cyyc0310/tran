"""V268: separate factor-level shift from empirical CEMS dispersion."""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v225_partial_snapshot_sharp_risk_bounds as partial  # noqa: E402
import v226_frozen_four_field_confirmation as confirm  # noqa: E402

PROTOCOL = ROOT / "docs/V268_CEMS_LEVEL_AND_SPREAD_GATE_PROTOCOL.md"
RADIUS_INPUT = ROOT / "results/v267-cems-calibrated-threshold-gate-r2/cems_state_quarter_r90.csv"
INPUT_CERT = ROOT / "results/v240-ef-uncertainty-temporal-risk-r1/threshold_side_certificate.csv"
OUT = ROOT / "results/v268-cems-level-and-spread-gate-r2"
PROJECT_EF = {1000.0: "coal", 800.0: "oil", 400.0: "gas"}
THRESHOLD = 500.0001


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
    radius_frame = pd.read_csv(RADIUS_INPUT)
    radius = {(int(row.quarter), row.fuel_group): float(row.r90_state_quarter)
              for row in radius_frame.itertuples(index=False)}
    center = {(int(row.quarter), row.fuel_group): float(row.pooled_ef_g_kwh)
              for row in radius_frame.itertuples(index=False)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        native, schemas, input_hashes, helpers = confirm.load_native()
        series = confirm.build_series(native, schemas, helpers)
    certs = pd.read_csv(INPUT_CERT)
    if len(certs) != 16 or certs.duplicated(["region", "year", "month"]).any():
        raise RuntimeError("expected frozen 16 BA-month cohort")
    rows = []
    for cert in certs.itertuples(index=False):
        key = (cert.region, int(cert.year), int(cert.month))
        entry = series[key]
        amounts = np.asarray(entry["fuel_totals"], float)
        labels = np.asarray(partial.EF, float)
        grouped = {ef: float(amounts[labels == ef].sum()) for ef in PROJECT_EF}
        total = float(amounts.sum())  # include zero-direct-EF generation in the CIF denominator
        quarter = (int(cert.month) - 1) // 3 + 1
        spread_lo = spread_hi = level_lo = level_hi = nominal_total = measured_center_total = 0.0
        factors = {}
        for project_ef, fuel_group in PROJECT_EF.items():
            amount = grouped[project_ef]
            r = radius[(quarter, fuel_group)]
            c = center[(quarter, fuel_group)]
            spread_lo += amount * project_ef * max(0.0, 1.0 - r)
            spread_hi += amount * project_ef * (1.0 + r)
            level_lo += amount * c * max(0.0, 1.0 - r)
            level_hi += amount * c * (1.0 + r)
            nominal_total += amount * project_ef
            measured_center_total += amount * c
            factors[f"{fuel_group}_r90"] = r
            factors[f"{fuel_group}_cems_center_g_kwh"] = c
        spread_bounds = (spread_lo / total, spread_hi / total)
        level_bounds = (level_lo / total, level_hi / total)
        rows.append({"region": cert.region, "year": int(cert.year), "month": int(cert.month),
                     "quarter": quarter, "nominal_project_mean": nominal_total / total,
                     "cems_center_mean": measured_center_total / total,
                     "project_center_spread_low": spread_bounds[0], "project_center_spread_high": spread_bounds[1],
                     "cems_center_spread_low": level_bounds[0], "cems_center_spread_high": level_bounds[1],
                     "spread_only_crosses": spread_bounds[0] < THRESHOLD <= spread_bounds[1],
                     "level_plus_spread_crosses": level_bounds[0] < THRESHOLD <= level_bounds[1],
                     "uniform_10pct_crosses": float(cert.critical_relative_ef_tolerance_to_reach_threshold) <= .10,
                     **factors})
    cases = pd.DataFrame(rows)
    OUT.mkdir(parents=True)
    cases.to_csv(OUT / "ba_month_gate_scenarios.csv", index=False)
    summary = pd.DataFrame([
        {"scenario": "project nominal center + CEMS r90 spread", "ba_months": len(cases),
         "threshold_crossings": int(cases.spread_only_crosses.sum()),
         "mean_low": cases.project_center_spread_low.mean(), "mean_high": cases.project_center_spread_high.mean()},
        {"scenario": "CEMS measured center + CEMS r90 spread", "ba_months": len(cases),
         "threshold_crossings": int(cases.level_plus_spread_crosses.sum()),
         "mean_low": cases.cems_center_spread_low.mean(), "mean_high": cases.cems_center_spread_high.mean()},
        {"scenario": "project nominal center + uniform 10%", "ba_months": len(cases),
         "threshold_crossings": int(cases.uniform_10pct_crosses.sum()),
         "mean_low": np.nan, "mean_high": np.nan},
    ])
    summary.to_csv(OUT / "summary.csv", index=False)
    meta = {"status": "cems_level_and_dispersion_threshold_gate_transport_stress_not_china_ef_estimate",
            "protocol_sha256": protocol_hash, "script_sha256": sha256(Path(__file__)),
            "radius_input_sha256": sha256(RADIUS_INPUT), "certificate_sha256": sha256(INPUT_CERT),
            "eia_input_hashes": input_hashes, "threshold": THRESHOLD,
            "ba_months": len(cases), "spread_only_crossings": int(cases.spread_only_crosses.sum()),
            "level_plus_spread_crossings": int(cases.level_plus_spread_crosses.sum()),
            "uniform_10pct_crossings": int(cases.uniform_10pct_crosses.sum()),
            "warning": "CEMS fuel-quarter means and state-quarter dispersion are U.S. fossil gross-generation diagnostics transferred to already-open EIA mix; not China EF distribution or coverage"}
    (OUT / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(summary.to_string(index=False))
    print(cases[["region", "month", "nominal_project_mean", "cems_center_mean",
                 "spread_only_crosses", "level_plus_spread_crosses", "uniform_10pct_crosses"]].to_string(index=False))


if __name__ == "__main__":
    main()
