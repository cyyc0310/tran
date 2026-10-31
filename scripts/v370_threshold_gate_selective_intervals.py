"""V370: cross-fitted threshold-side abstention for monthly tail-count intervals."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v369_crossfit_risk_intervals import (  # noqa: E402
    ALPHAS,
    EF_HIGH,
    EF_LOW,
    EF_MID,
    REGIONS,
    THRESHOLDS,
    load_hourly,
    profiles,
    quantile,
)

OUT = ROOT / "results/v370-threshold-gate-selective-r1"


def count_above(profile: np.ndarray, threshold: float) -> int:
    return int(np.sum(profile > threshold))


def signed_interval(
    point_count: int,
    residual_fractions: np.ndarray,
    n_hours: int,
    alpha: float,
) -> tuple[int, int]:
    if residual_fractions.size == 0:
        return 0, n_hours
    low_error = quantile(residual_fractions, alpha / 2.0) * n_hours
    high_error = quantile(residual_fractions, 1.0 - alpha / 2.0) * n_hours
    lo = int(np.floor(max(0.0, point_count + low_error)))
    hi = int(np.ceil(min(float(n_hours), point_count + high_error)))
    return min(lo, hi), max(lo, hi)


def winkler(actual: int, lo: int, hi: int, alpha: float) -> float:
    return float((hi - lo) + (2.0 / alpha) * max(lo - actual, 0) + (2.0 / alpha) * max(actual - hi, 0))


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    hourly = load_hourly()
    by_key = {
        (region, int(year), int(month)): block.sort_values("time").copy()
        for (region, year, month), block in hourly.groupby(["region", "year", "month"], sort=True)
    }
    months = sorted({(year, month) for _, year, month in by_key})
    rows = []

    for outer_region in REGIONS:
        training_regions = [region for region in REGIONS if region != outer_region]
        calibration: dict[tuple[str, float, str], list[float]] = {}
        calibration_count: dict[tuple[str, float, str], int] = {}
        for tau in THRESHOLDS:
            for base in ("local", "pooled"):
                calibration[("global", tau, base)] = []
                calibration[("noncross", tau, base)] = []

        # Nested leave-one-region-out residuals; outer held-out region is absent.
        for calibration_region in training_regions:
            inner_donors = [region for region in training_regions if region != calibration_region]
            for year, month in months:
                target = by_key.get((calibration_region, year, month))
                donor_parts = [
                    by_key[(region, year, month)]
                    for region in inner_donors
                    if (region, year, month) in by_key
                ]
                if target is None or not donor_parts:
                    continue
                component = profiles(target, pd.concat(donor_parts, ignore_index=True))
                n_hours = len(np.asarray(component["actual"], float))
                actual_count_by_tau = {
                    tau: count_above(np.asarray(component["actual"], float), tau)
                    for tau in THRESHOLDS
                }
                local_profile = np.asarray(component["local_point"], float)
                pooled_profile = np.asarray(component["pooled_point"], float)
                crossing = {
                    tau: (EF_LOW * float(component["local_share"]) <= tau
                          and EF_HIGH * float(component["local_share"]) > tau)
                    for tau in THRESHOLDS
                }
                for tau in THRESHOLDS:
                    for base, profile in [("local", local_profile), ("pooled", pooled_profile)]:
                        residual = (actual_count_by_tau[tau] - count_above(profile, tau)) / n_hours
                        calibration[("global", tau, base)].append(float(residual))
                        if not crossing[tau]:
                            calibration[("noncross", tau, base)].append(float(residual))

        calibration_arrays = {key: np.asarray(values, dtype=float) for key, values in calibration.items()}

        for year, month in months:
            target = by_key.get((outer_region, year, month))
            donor_parts = [
                by_key[(region, year, month)]
                for region in training_regions
                if (region, year, month) in by_key
            ]
            if target is None or not donor_parts:
                continue
            component = profiles(target, pd.concat(donor_parts, ignore_index=True))
            actual = np.asarray(component["actual"], float)
            n_hours = len(actual)
            local_profile = np.asarray(component["local_point"], float)
            pooled_profile = np.asarray(component["pooled_point"], float)
            for tau in THRESHOLDS:
                observed = count_above(actual, tau)
                local_point_count = count_above(local_profile, tau)
                pooled_point_count = count_above(pooled_profile, tau)
                crossing = bool(
                    EF_LOW * float(component["local_share"]) <= tau
                    and EF_HIGH * float(component["local_share"]) > tau
                )
                for alpha in ALPHAS:
                    candidates: dict[str, tuple[tuple[int, int] | None, bool, str, int]] = {}
                    for base, point_count in [("local", local_point_count), ("pooled", pooled_point_count)]:
                        global_resid = calibration_arrays[("global", tau, base)]
                        noncross_resid = calibration_arrays[("noncross", tau, base)]
                        candidates[f"{base}_global_q"] = (
                            signed_interval(point_count, global_resid, n_hours, alpha), True, "global", len(global_resid)
                        )
                        conditional = signed_interval(point_count, noncross_resid, n_hours, alpha)
                        if crossing:
                            candidates[f"gate_{base}_full_range"] = ((0, n_hours), True, "crossing_full_range", len(noncross_resid))
                            candidates[f"gate_{base}_selective"] = (None, False, "abstain_crossing", len(noncross_resid))
                        else:
                            candidates[f"gate_{base}_full_range"] = (conditional, True, "conditional_noncrossing", len(noncross_resid))
                            candidates[f"gate_{base}_selective"] = (conditional, True, "conditional_noncrossing", len(noncross_resid))

                    for method, (bounds, issued, gate_status, n_cal) in candidates.items():
                        if bounds is None:
                            lo = hi = None
                            covered = np.nan
                            width = np.nan
                            width_fraction = np.nan
                            score = np.nan
                        else:
                            lo, hi = bounds
                            covered = float(lo <= observed <= hi)
                            width = int(hi - lo)
                            width_fraction = float(width / n_hours)
                            score = winkler(observed, lo, hi, alpha)
                        rows.append(
                            {
                                "outer_region": outer_region,
                                "year": year,
                                "month": month,
                                "threshold_g_kwh": tau,
                                "alpha": alpha,
                                "nominal_coverage": 1.0 - alpha,
                                "method": method,
                                "gate_crossing": crossing,
                                "gate_status": gate_status,
                                "issued_prediction": issued,
                                "calibration_count": n_cal,
                                "n_valid_hours": n_hours,
                                "local_thermal_share": float(component["local_share"]),
                                "observed_tail_count": observed,
                                "predicted_tail_count": local_point_count if "local" in method else pooled_point_count,
                                "interval_lower": lo,
                                "interval_upper": hi,
                                "coverage": covered,
                                "width_hours": width,
                                "width_fraction": width_fraction,
                                "winkler_score": score,
                            }
                        )

    detail = pd.DataFrame(rows)
    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "region_month_selective_scores.csv", index=False)

    summary_rows = []
    group_cols = ["alpha", "nominal_coverage", "threshold_g_kwh", "method"]
    for key, subset in detail.groupby(group_cols, sort=True):
        alpha, nominal, tau, method = key
        by_region = subset.groupby("outer_region")
        region_stats = by_region.apply(
            lambda block: pd.Series(
                {
                    "retention": float(block["issued_prediction"].mean()),
                    "issued_coverage": float(block.loc[block["issued_prediction"], "coverage"].mean())
                    if block["issued_prediction"].any() else np.nan,
                    "issued_width": float(block.loc[block["issued_prediction"], "width_fraction"].mean())
                    if block["issued_prediction"].any() else np.nan,
                    "issued_winkler": float(block.loc[block["issued_prediction"], "winkler_score"].mean())
                    if block["issued_prediction"].any() else np.nan,
                    "all_case_coverage": float(block["coverage"].mean()) if block["coverage"].notna().any() else np.nan,
                    "all_case_width": float(block["width_fraction"].mean()) if block["width_fraction"].notna().any() else np.nan,
                }
            ),
            include_groups=False,
        )
        summary_rows.append(
            {
                "alpha": alpha,
                "nominal_coverage": nominal,
                "threshold_g_kwh": tau,
                "method": method,
                "region_count": int(len(region_stats)),
                "equal_region_retention": float(region_stats["retention"].mean()),
                "equal_region_coverage_among_issued": float(region_stats["issued_coverage"].mean()),
                "worst_region_coverage_among_issued": float(region_stats["issued_coverage"].min()),
                "equal_region_width_among_issued_fraction": float(region_stats["issued_width"].mean()),
                "equal_region_winkler_among_issued": float(region_stats["issued_winkler"].mean()),
                "equal_region_all_case_coverage": float(region_stats["all_case_coverage"].mean()),
                "equal_region_all_case_width_fraction": float(region_stats["all_case_width"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "method_summary.csv", index=False)
    metadata = {
        "experiment": "V370 threshold-side selective prediction for tail-count intervals",
        "target": "EIA fixed-factor hourly generation CIF proxy, not measured emissions",
        "regions": list(REGIONS),
        "region_months": int(detail[["outer_region", "year", "month"]].drop_duplicates().shape[0]),
        "valid_hours": int(hourly.shape[0]),
        "thermal_ef_support_g_kwh": [EF_LOW, EF_HIGH],
        "point_ef_g_kwh": EF_MID,
        "thresholds_g_kwh_strict": list(THRESHOLDS),
        "interval_levels": [1.0 - alpha for alpha in ALPHAS],
        "summary": summary.to_dict(orient="records"),
        "caveats": [
            "All seven regions were previously inspected; leave-one-region-out is not an untouched regional confirmation.",
            "Coverage is empirical cross-validation; it has no formal conformal guarantee with serial dependence and seven region clusters.",
            "Hourly labels are fixed-factor generation proxies, not measured or full-grid emissions truth.",
            "Thermal EF support is illustrative and not locally calibrated, especially not for Chinese provinces.",
            "NBS monthly statistics cover above-designated-size enterprises; V370 does not repair that scope gap.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
