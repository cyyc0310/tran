"""V110: compare monthly-mean anchor and empirical intervals on V104 holdouts."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v90_coarse_capacity_identification as v90  # noqa: E402

REGIONS = ("CPLE", "FPC", "IPCO", "PNM", "SOCO")
RATES = (0.01, 0.05, 0.10)
V104_ROWS = ROOT / "results/v104-unseen-region-robust-confirmation-r1/region_month_scores.csv"
V105_ROWS = ROOT / "results/v105-unseen-region-capacity-ablation-r1/paired_rows.csv"
OUT = ROOT / "results/v110-unseen-region-monthly-anchor-baseline-r1"
PROTOCOL = ROOT / "docs/V110_UNSEEN_REGION_MONTHLY_ANCHOR_BASELINE_PROTOCOL.md"
CAL_START = pd.Timestamp("2024-01-01", tz="UTC")
CAL_STOP = pd.Timestamp("2026-01-01", tz="UTC")
TARGET_START = pd.Timestamp("2026-01-01", tz="UTC")
TARGET_STOP = pd.Timestamp("2026-07-01", tz="UTC")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    sorted_values, sorted_weights = values[order], weights[order]
    idx = min(int(np.searchsorted(np.cumsum(sorted_weights), quantile * sorted_weights.sum(), side="left")),
              len(sorted_values) - 1)
    return float(sorted_values[idx])


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def month_stats(group: pd.DataFrame, tau: int) -> dict:
    cols = [f"generation_{fuel}" for fuel in v90.FUELS]
    generation = group[cols].to_numpy(float)
    total_by_hour = generation.sum(axis=1)
    emissions_by_hour = generation @ np.array([v90.EF[fuel] for fuel in v90.FUELS], dtype=float)
    positive = np.isfinite(total_by_hour) & np.isfinite(emissions_by_hour) & (total_by_hour > 0)
    generation, total_by_hour, emissions_by_hour = generation[positive], total_by_hour[positive], emissions_by_hour[positive]
    margins = generation.sum(axis=0)
    total = float(margins.sum())
    mean_intensity = float(sum(margins[i] * v90.EF[fuel] for i, fuel in enumerate(v90.FUELS)) / total)
    tail_fraction = float(np.mean(emissions_by_hour > tau * total_by_hour)) if len(total_by_hour) else float("nan")
    return {"hours": int(len(total_by_hour)), "mean_intensity": mean_intensity,
            "anchor_fraction": float(mean_intensity > tau), "tail_fraction": tail_fraction}


def set_decision(lower: int, upper: int, required: int) -> tuple[bool, bool, bool]:
    positive = lower >= required
    negative = upper < required
    return positive, negative, not (positive or negative)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    v104 = pd.read_csv(V104_ROWS)
    v105 = pd.read_csv(V105_ROWS).drop_duplicates(["region", "month", "tau"])
    if set(v104.region.unique()) != set(REGIONS) or set(v105.region.unique()) != set(REGIONS):
        raise ValueError("frozen holdout roster mismatch")
    v90.source_data.ALL_REGIONS = tuple(dict.fromkeys(v90.source_data.ALL_REGIONS + v90.REGIONS + REGIONS))
    hourly, _, audits, hourly_hashes = v90.source_data.prepare_regions()

    calibration_rows = []
    for region in v90.REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge(CAL_START) & times.lt(CAL_STOP) & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        for month, group in frame.groupby("month", sort=True):
            expected = pd.Period(month, freq="M").days_in_month * 24
            if len(group) / expected < 0.80:
                continue
            for tau in (600, 900):
                stats = month_stats(group, tau)
                calibration_rows.append({"region": region, "month": month, "tau": tau,
                                         "coverage": len(group) / expected, **stats,
                                         "absolute_error": abs(stats["tail_fraction"] - stats["anchor_fraction"])})
    cal = pd.DataFrame(calibration_rows)
    if cal.empty or set(cal.region.unique()) != set(v90.REGIONS):
        raise ValueError("source calibration does not cover all frozen source regions")

    radii = {}
    for tau in (600, 900):
        subset = cal.loc[cal.tau.eq(tau)].copy()
        counts = subset.groupby("region").size()
        subset["weight"] = subset.region.map(lambda region: 1.0 / (len(counts) * counts[region]))
        radii[tau] = weighted_quantile(subset.absolute_error.to_numpy(float),
                                       subset.weight.to_numpy(float), 0.90)

    target_rows = []
    for region in REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge(TARGET_START) & times.lt(TARGET_STOP) & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        for month, group in frame.groupby("month", sort=True):
            expected = pd.Period(month, freq="M").days_in_month * 24
            if len(group) / expected < 0.80:
                raise ValueError(f"V104 target month fails frozen coverage: {region}/{month}")
            for tau in (600, 900):
                stats = month_stats(group, tau)
                radius = radii[tau]
                lower_fraction = max(0.0, stats["anchor_fraction"] - radius)
                upper_fraction = min(1.0, stats["anchor_fraction"] + radius)
                lower_count = max(0, min(stats["hours"], math.ceil(lower_fraction * stats["hours"] - 1e-12)))
                upper_count = max(0, min(stats["hours"], math.floor(upper_fraction * stats["hours"] + 1e-12)))
                if lower_count > upper_count:
                    raise ValueError("empty calibrated interval after integer conversion")
                robust = v104.loc[v104.region.eq(region) & v104.month.astype(str).str[:10].eq(month)
                                  & v104.tau.eq(tau)].iloc[0]
                monthly = v105.loc[v105.region.eq(region) & v105.month.astype(str).str[:10].eq(month)
                                   & v105.tau.eq(tau)].iloc[0]
                actual = int(robust.actual_nominal_K)
                if int(monthly.monthly_lower) > actual or int(monthly.monthly_upper) < actual:
                    raise ValueError(f"monthly identified set misses proxy K: {region}/{month}/{tau}")
                if int(monthly.capacity_robust_lower) > actual or int(monthly.capacity_robust_upper) < actual:
                    raise ValueError(f"capacity identified set misses proxy K: {region}/{month}/{tau}")
                point_count = int(round(stats["anchor_fraction"] * stats["hours"]))
                target_rows.append({"region": region, "month": month, "tau": tau, "hours": stats["hours"],
                                    "actual_K": actual, "anchor_fraction": stats["anchor_fraction"],
                                    "point_K": point_count,
                                    "point_absolute_error_hours": abs(point_count - actual),
                                    "calibration_q90_fraction": radius,
                                    "anchor_interval_lower": lower_count, "anchor_interval_upper": upper_count,
                                    "anchor_interval_width": upper_count - lower_count,
                                    "anchor_interval_covered": lower_count <= actual <= upper_count,
                                    "monthly_robust_lower": int(monthly.monthly_lower),
                                    "monthly_robust_upper": int(monthly.monthly_upper),
                                    "monthly_robust_width": int(monthly.monthly_upper - monthly.monthly_lower),
                                    "capacity_robust_lower": int(monthly.capacity_robust_lower),
                                    "capacity_robust_upper": int(monthly.capacity_robust_upper),
                                    "capacity_robust_width": int(monthly.capacity_robust_width)})

    detail = pd.DataFrame(target_rows)
    for rate in RATES:
        required = np.ceil(rate * detail.hours.to_numpy(float)).astype(int)
        detail[f"required_{rate:g}"] = required
        for prefix, lo_col, hi_col in (("anchor", "anchor_interval_lower", "anchor_interval_upper"),
                                       ("monthly", "monthly_robust_lower", "monthly_robust_upper"),
                                       ("capacity", "capacity_robust_lower", "capacity_robust_upper")):
            decisions = [set_decision(int(lo), int(hi), int(req))
                         for lo, hi, req in zip(detail[lo_col], detail[hi_col], required)]
            detail[f"{prefix}_positive_{rate:g}"] = [decision[0] for decision in decisions]
            detail[f"{prefix}_negative_{rate:g}"] = [decision[1] for decision in decisions]
            detail[f"{prefix}_abstain_{rate:g}"] = [decision[2] for decision in decisions]

    OUT.mkdir(parents=True, exist_ok=False)
    cal.to_csv(OUT / "source_calibration_rows.csv", index=False)
    detail.to_csv(OUT / "target_rows.csv", index=False)
    numeric = ["point_absolute_error_hours", "anchor_interval_width", "anchor_interval_covered",
               "monthly_robust_width", "capacity_robust_width"]
    per_region = detail.groupby(["region", "tau"])[numeric].mean().reset_index()
    for rate in RATES:
        for prefix in ("anchor", "monthly", "capacity"):
            for state in ("positive", "negative", "abstain"):
                col = f"{prefix}_{state}_{rate:g}"
                per_region[col] = detail.groupby(["region", "tau"])[col].mean().to_numpy()
    per_region.to_csv(OUT / "per_region_metrics.csv", index=False)
    macro = per_region.groupby("tau").mean(numeric_only=True).reset_index()
    macro.to_csv(OUT / "region_equal_metrics.csv", index=False)
    coverage = detail.groupby("tau").anchor_interval_covered.mean().rename("anchor_interval_coverage").reset_index()
    macro = macro.merge(coverage, on="tau")
    macro.to_csv(OUT / "region_equal_metrics.csv", index=False)
    summary = {"status": "UNSEEN_REGION_SAME_MONTH_MARGIN_ANCHOR_EMPIRICAL_INTERVAL_BASELINE",
               "target_regions": list(REGIONS), "calibration_regions": list(v90.REGIONS),
               "target_region_months": int(detail[["region", "month"]].drop_duplicates().shape[0]),
               "calibration_region_months_by_tau": cal.groupby("tau").size().to_dict(),
               "radius_q90_by_tau": radii, "region_equal_metrics": macro.to_dict("records"),
               "proxy_label_coverage_of_identified_sets": True,
               "hourly_source_sha256": hourly_hashes, "v104_rows_sha256": sha(V104_ROWS),
               "v105_rows_sha256": sha(V105_ROWS), "source_audits": audits,
               "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
               "limitations": ["Empirical region-balanced quantile is not a formal conformal guarantee under dependence or domain shift.",
                               "Target period is retrospectively observed and fixed-factor EIA proxy labeled.",
                               "Monthly anchor uses only target-month fuel margins; capacity set uses additional lagged capacity.",
                               "Results do not validate actual China CIF or marginal/consumption-based intensity."]}
    (OUT / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False,
                                                   allow_nan=False) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
