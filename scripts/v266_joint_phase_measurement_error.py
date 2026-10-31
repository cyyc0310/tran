"""V266: joint phase and measurement uncertainty in hourly total generation."""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v225_partial_snapshot_sharp_risk_bounds as partial  # noqa: E402
import v226_frozen_four_field_confirmation as confirm  # noqa: E402
import v237_nested_daily_hourly_risk as nested  # noqa: E402
import v264_full_cohort_phase_ambiguity as phase  # noqa: E402

PROTOCOL = ROOT / "docs/V266_JOINT_PHASE_AND_MEASUREMENT_ERROR_PROTOCOL.md"
OUT = ROOT / "results/v266-joint-phase-measurement-error-r1"
EPSILONS = (0.05, 0.10)
YEAR = 2025
TAU = 500.0


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        native, schemas, input_hashes, helpers = confirm.load_native()
        series = confirm.build_series(native, schemas, helpers)

    phase_path = ROOT / "results/v264-full-cohort-phase-ambiguity-r1/robust_risk_count_bounds.csv"
    phase_cases = pd.read_csv(phase_path)
    phase_cases = phase_cases[(phase_cases.regime == "one_hour_phase_envelope") & phase_cases.both_optimal]
    phase_cases = phase_cases[["region", "year", "month", "min_bound", "max_bound", "observed_tail_count"]]
    daily = pd.read_csv(ROOT / "results/v239-eia-seasonal-nested-risk-r1/risk_count_bounds.csv")
    daily = daily[(daily.regime == "daily_total_only") & (daily.tau == TAU)]
    if "year" not in daily:
        daily["year"] = YEAR
    daily = daily[["region", "year", "month", "min_bound", "max_bound"]].rename(
        columns={"min_bound": "daily_min", "max_bound": "daily_max"})

    endpoints: list[dict] = []
    for row in phase_cases.itertuples(index=False):
        key = (row.region, YEAR, row.month)
        entry = series[key]
        fuel_totals = np.asarray(entry["fuel_totals"], float)
        fuel = np.asarray(entry["fuel_profile"], float) * fuel_totals[None, :]
        true_total = fuel.sum(axis=1)
        daily_total = true_total.reshape(-1, 24).sum(axis=1)
        margins = phase.group_margins(fuel_totals)
        shifted = np.roll(true_total.reshape(-1, 24), 1, axis=1)
        phase_lo = np.minimum.reduce((np.roll(shifted, 1, axis=1), shifted,
                                      np.roll(shifted, -1, axis=1)))
        phase_hi = np.maximum.reduce((np.roll(shifted, 1, axis=1), shifted,
                                      np.roll(shifted, -1, axis=1)))
        if not np.allclose(phase_lo.sum(axis=1) <= daily_total + 1e-6, True):
            raise RuntimeError(f"invalid lower phase envelope for {key}")
        if not np.allclose(phase_hi.sum(axis=1) >= daily_total - 1e-6, True):
            raise RuntimeError(f"invalid upper phase envelope for {key}")
        if np.any(true_total < phase_lo.ravel() - 1e-8) or np.any(true_total > phase_hi.ravel() + 1e-8):
            raise RuntimeError(f"true hourly total outside phase envelope for {key}")
        for epsilon in EPSILONS:
            lower = (phase_lo * (1.0 - epsilon)).ravel()
            upper = (phase_hi * (1.0 + epsilon)).ravel()
            solved = phase.solve_uncertain_hourly_bounds(lower, upper, daily_total, margins)
            for result in solved:
                endpoints.append({
                    "region": row.region, "year": YEAR, "month": row.month, "tau": TAU,
                    "epsilon": epsilon, "endpoint": result["endpoint"], "bound": result["bound"],
                    "optimal": result["optimal"], "status": result["status"], "message": result["message"],
                    "mip_gap": result["mip_gap"], "observed_tail_count": row.observed_tail_count,
                    "true_hourly_total_inside": True,
                })
            print(f"solved {key}, epsilon={epsilon:.0%}, optimal={all(x['optimal'] for x in solved)}", flush=True)

    endpoint_df = pd.DataFrame(endpoints)
    cases = endpoint_df.groupby(["region", "year", "month", "tau", "epsilon"], as_index=False).agg(
        min_bound=("bound", "first"), max_bound=("bound", "last"), both_optimal=("optimal", "all"),
        observed_tail_count=("observed_tail_count", "first"), true_profile_inside=("true_hourly_total_inside", "all"))
    cases = cases.merge(phase_cases, on=["region", "year", "month"], suffixes=("", "_phase"), validate="many_to_one")
    cases = cases.merge(daily, on=["region", "year", "month"], validate="many_to_one")
    cases["joint_width"] = cases.max_bound - cases.min_bound
    cases["phase_width"] = cases.max_bound_phase - cases.min_bound_phase
    cases["daily_width"] = cases.daily_max - cases.daily_min
    cases["joint_width_penalty_vs_phase"] = cases.joint_width - cases.phase_width
    cases["daily_to_joint_shrink"] = cases.daily_width - cases.joint_width
    cases["witness_included"] = ((cases.min_bound <= cases.observed_tail_count)
                                 & (cases.max_bound >= cases.observed_tail_count))
    cases["nested_in_daily"] = ((cases.min_bound >= cases.daily_min) & (cases.max_bound <= cases.daily_max))
    proved = cases[cases.both_optimal].copy()
    if not proved.witness_included.all() or not proved.nested_in_daily.all():
        raise RuntimeError("witness containment or daily-set nesting failed")

    OUT.mkdir(parents=True)
    endpoint_df.to_csv(OUT / "milp_endpoints.csv", index=False)
    cases.to_csv(OUT / "joint_bounds.csv", index=False)
    summary = proved.groupby("epsilon", as_index=False).agg(
        candidate_cases=("region", "size"), mean_phase_width=("phase_width", "mean"),
        mean_joint_width=("joint_width", "mean"), mean_daily_width=("daily_width", "mean"),
        mean_joint_penalty_vs_phase=("joint_width_penalty_vs_phase", "mean"),
        mean_daily_to_joint_shrink=("daily_to_joint_shrink", "mean"),
        witness_rate=("witness_included", "mean"), nested_rate=("nested_in_daily", "mean"))
    summary.to_csv(OUT / "summary.csv", index=False)
    metadata = {
        "status": "joint_phase_measurement_error_identification_sensitivity_not_forecast_validation",
        "protocol_sha256": protocol_hash, "script_sha256": sha256(Path(__file__)),
        "input_hashes": input_hashes,
        "phase_reference_sha256": sha256(phase_path),
        "daily_reference_sha256": sha256(ROOT / "results/v239-eia-seasonal-nested-risk-r1/risk_count_bounds.csv"),
        "scipy_version": scipy.__version__, "epsilon": list(EPSILONS), "tau": TAU,
        "candidate_cases": int(len(phase_cases)), "endpoint_count": int(len(endpoint_df)),
        "optimal_endpoint_count": int(endpoint_df.optimal.sum()),
        "all_proved_witnesses_inside": bool(proved.witness_included.all()),
        "all_proved_nested_in_daily": bool(proved.nested_in_daily.all()),
        "warning": "EIA fixed-EF activity proxy only; existing opened cohort; not China measurement or forecast validation",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(summary.to_string(index=False))
    print(f"proved endpoints={metadata['optimal_endpoint_count']}/{metadata['endpoint_count']}")


if __name__ == "__main__":
    main()
