"""V111: evaluate a factor-robust calibrated monthly-mean anchor."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v90_coarse_capacity_identification as v90  # noqa: E402

REGIONS = ("CPLE", "FPC", "IPCO", "PNM", "SOCO")
RATES = (0.01, 0.05, 0.10)
V105_ROWS = ROOT / "results/v105-unseen-region-capacity-ablation-r1/paired_rows.csv"
OUT = ROOT / "results/v111-factor-robust-monthly-anchor-r1"
PROTOCOL = ROOT / "docs/V111_FACTOR_ROBUST_MONTHLY_ANCHOR_PROTOCOL.md"
CAL_START = pd.Timestamp("2024-01-01", tz="UTC")
CAL_STOP = pd.Timestamp("2026-01-01", tz="UTC")
TARGET_START = pd.Timestamp("2026-01-01", tz="UTC")
TARGET_STOP = pd.Timestamp("2026-07-01", tz="UTC")
DELTA = 0.10


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    x, w = values[order], weights[order]
    index = min(int(np.searchsorted(np.cumsum(w), quantile * w.sum(), side="left")), len(x) - 1)
    return float(x[index])


def factors_box():
    nominal = dict(v90.EF)
    low = {fuel: ef * (1 - DELTA) if fuel in ("coal", "gas", "oil", "other") else ef
           for fuel, ef in nominal.items()}
    high = {fuel: ef * (1 + DELTA) if fuel in ("coal", "gas", "oil", "other") else ef
            for fuel, ef in nominal.items()}
    return {"low": low, "nominal": nominal, "high": high}


def month_stats(group: pd.DataFrame, tau: int, factors: dict[str, float]) -> tuple[int, float, float]:
    fuel_cols = [f"generation_{fuel}" for fuel in v90.FUELS]
    matrix = group[fuel_cols].to_numpy(float)
    total_hour = matrix.sum(axis=1)
    emissions = matrix @ np.array([factors[fuel] for fuel in v90.FUELS], dtype=float)
    valid = np.isfinite(total_hour) & np.isfinite(emissions) & (total_hour > 0)
    matrix, total_hour, emissions = matrix[valid], total_hour[valid], emissions[valid]
    margins = matrix.sum(axis=0)
    mean_intensity = float(margins @ np.array([factors[fuel] for fuel in v90.FUELS]) / margins.sum())
    count = int(np.sum(emissions > tau * total_hour))
    return len(total_hour), mean_intensity, count / len(total_hour)


def add_interval_decision(prefix, lower_col, upper_col, frame):
    for rate in RATES:
        required = np.ceil(rate * frame.hours.to_numpy(float)).astype(int)
        positive = frame[lower_col].to_numpy(int) >= required
        negative = frame[upper_col].to_numpy(int) < required
        frame[f"{prefix}_positive_{rate:g}"] = positive
        frame[f"{prefix}_negative_{rate:g}"] = negative
        frame[f"{prefix}_abstain_{rate:g}"] = ~(positive | negative)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    v105 = pd.read_csv(V105_ROWS).drop_duplicates(["region", "month", "tau"])
    if set(v105.region.unique()) != set(REGIONS):
        raise ValueError("V104/V105 roster mismatch")
    v90.source_data.ALL_REGIONS = tuple(dict.fromkeys(v90.source_data.ALL_REGIONS + v90.REGIONS + REGIONS))
    hourly, _, audits, hourly_hashes = v90.source_data.prepare_regions()
    factor_vectors = factors_box()

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
                stats = {name: month_stats(group, tau, ef) for name, ef in factor_vectors.items()}
                anchor_low = min(float(value[1] > tau) for value in stats.values())
                anchor_high = max(float(value[1] > tau) for value in stats.values())
                error = max(max(anchor_low - value[2], 0.0, value[2] - anchor_high)
                            for value in stats.values())
                calibration_rows.append({"region": region, "month": month, "tau": tau,
                                         "anchor_lower_fraction": anchor_low,
                                         "anchor_upper_fraction": anchor_high,
                                         "worst_case_absolute_error": error})
    cal = pd.DataFrame(calibration_rows)
    if set(cal.region.unique()) != set(v90.REGIONS):
        raise ValueError("source calibration fails frozen region coverage")
    radii = {}
    for tau in (600, 900):
        subset = cal.loc[cal.tau.eq(tau)].copy()
        counts = subset.groupby("region").size()
        subset["weight"] = subset.region.map(lambda region: 1 / (len(counts) * counts[region]))
        radii[tau] = weighted_quantile(subset.worst_case_absolute_error.to_numpy(float),
                                       subset.weight.to_numpy(float), 0.90)

    target_rows = []
    for region in REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge(TARGET_START) & times.lt(TARGET_STOP) & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        for month, group in frame.groupby("month", sort=True):
            if len(group) / (pd.Period(month, freq="M").days_in_month * 24) < 0.80:
                raise ValueError(f"target month fails V104 quality: {region}/{month}")
            for tau in (600, 900):
                stats = {name: month_stats(group, tau, ef) for name, ef in factor_vectors.items()}
                hours = stats["nominal"][0]
                anchor_low = min(float(value[1] > tau) for value in stats.values())
                anchor_high = max(float(value[1] > tau) for value in stats.values())
                true_low = min(int(value[2] * hours) for value in stats.values())
                true_high = max(int(value[2] * hours) for value in stats.values())
                radius = radii[tau]
                lower_fraction = max(0.0, anchor_low - radius)
                upper_fraction = min(1.0, anchor_high + radius)
                lower = max(0, min(hours, math.ceil(lower_fraction * hours - 1e-12)))
                upper = max(0, min(hours, math.floor(upper_fraction * hours + 1e-12)))
                if lower > upper:
                    raise ValueError("empty robust anchor interval")
                factor_box_covered = lower <= true_low and upper >= true_high
                partial = v105.loc[v105.region.eq(region) & v105.month.astype(str).str[:10].eq(month)
                                   & v105.tau.eq(tau)].iloc[0]
                target_rows.append({"region": region, "month": month, "tau": tau, "hours": hours,
                                    "anchor_base_lower_fraction": anchor_low,
                                    "anchor_base_upper_fraction": anchor_high,
                                    "calibration_radius_fraction": radius,
                                    "anchor_robust_lower": lower, "anchor_robust_upper": upper,
                                    "anchor_robust_width": upper - lower,
                                    "factor_box_K_lower": true_low, "factor_box_K_upper": true_high,
                                    "factor_box_covered": factor_box_covered,
                                    "monthly_robust_lower": int(partial.monthly_lower),
                                    "monthly_robust_upper": int(partial.monthly_upper),
                                    "monthly_robust_width": int(partial.monthly_upper - partial.monthly_lower),
                                    "capacity_robust_lower": int(partial.capacity_robust_lower),
                                    "capacity_robust_upper": int(partial.capacity_robust_upper),
                                    "capacity_robust_width": int(partial.capacity_robust_width)})

    detail = pd.DataFrame(target_rows)
    add_interval_decision("anchor", "anchor_robust_lower", "anchor_robust_upper", detail)
    add_interval_decision("monthly", "monthly_robust_lower", "monthly_robust_upper", detail)
    add_interval_decision("capacity", "capacity_robust_lower", "capacity_robust_upper", detail)
    OUT.mkdir(parents=True, exist_ok=False)
    cal.to_csv(OUT / "source_calibration_rows.csv", index=False)
    detail.to_csv(OUT / "target_rows.csv", index=False)
    metrics = ["anchor_robust_width", "monthly_robust_width", "capacity_robust_width", "factor_box_covered"]
    per_region = detail.groupby(["region", "tau"])[metrics].mean().reset_index()
    for rate in RATES:
        for prefix in ("anchor", "monthly", "capacity"):
            for state in ("positive", "negative", "abstain"):
                col = f"{prefix}_{state}_{rate:g}"
                per_region[col] = detail.groupby(["region", "tau"])[col].mean().to_numpy()
    per_region.to_csv(OUT / "per_region_metrics.csv", index=False)
    macro = per_region.groupby("tau").mean(numeric_only=True).reset_index()
    macro.to_csv(OUT / "region_equal_metrics.csv", index=False)
    summary = {"status": "FACTOR_ROBUST_MONTHLY_MEAN_ANCHOR_EMPIRICAL_INTERVAL",
               "target_regions": list(REGIONS), "calibration_regions": list(v90.REGIONS),
               "target_region_months": int(detail[["region", "month"]].drop_duplicates().shape[0]),
               "calibration_region_months_by_tau": cal.groupby("tau").size().to_dict(),
               "worst_case_radius_q90_by_tau": radii,
               "region_equal_metrics": macro.to_dict("records"),
               "factor_vectors": factor_vectors, "target_factor_box_coverage": bool(detail.factor_box_covered.all()),
               "hourly_source_sha256": hourly_hashes, "v105_rows_sha256": sha(V105_ROWS),
               "source_audits": audits, "protocol_sha256": sha(PROTOCOL),
               "script_sha256": sha(Path(__file__)),
               "limitations": ["Region-balanced empirical radius is not formal conformal coverage under dependence/domain shift.",
                               "Calibration and target sets are EIA fixed-proxy outputs under a stress factor box, not real-CIF uncertainty.",
                               "Target period is retrospective; no future-time or actual-China validation."]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                   allow_nan=False) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
