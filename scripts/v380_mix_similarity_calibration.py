"""V380: test whether local thermal-share similarity can improve cross-region calibration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v369_crossfit_risk_intervals import (  # noqa: E402
    ALPHAS,
    REGIONS,
    THRESHOLDS,
    interval_score,
    load_hourly,
    profiles,
    quantile,
)

OUT = ROOT / "results/v380-mix-similarity-calibration-r1"
BANDWIDTHS = (0.10, 0.20, 0.40)
BLENDINGS = (0.25, 0.50, 0.75)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = probability * sorted_weights.sum()
    index = min(int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left")), len(values) - 1)
    return float(sorted_values[index])


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    hourly = load_hourly()
    by_key = {
        (region, int(year), int(month)): block.sort_values("time").copy()
        for (region, year, month), block in hourly.groupby(["region", "year", "month"], sort=True)
    }
    months = sorted({(year, month) for _, year, month in by_key})
    rows: list[dict] = []

    for outer_region in REGIONS:
        train_regions = [region for region in REGIONS if region != outer_region]
        calibration_blocks: list[dict] = []
        for cal_region in train_regions:
            donor_regions = [region for region in train_regions if region != cal_region]
            for year, month in months:
                target = by_key.get((cal_region, year, month))
                donor_parts = [by_key[(r, year, month)] for r in donor_regions if (r, year, month) in by_key]
                if target is None or not donor_parts:
                    continue
                component = profiles(target, pd.concat(donor_parts, ignore_index=True))
                residual = np.asarray(component["actual"], float) - np.asarray(component["local_point"], float)
                fuel_mass = target.loc[:, [
                    "generation_coal", "generation_gas", "generation_oil", "generation_nuclear",
                    "generation_hydro", "generation_wind", "generation_solar", "generation_other",
                ]].sum().to_numpy(float)
                fuel_mix = fuel_mass / fuel_mass.sum()
                calibration_blocks.append({
                    "region": cal_region,
                    "year": year,
                    "month": month,
                    "share": float(component["local_share"]),
                    "fuel_mix": fuel_mix,
                    "residual": residual,
                })
        if not calibration_blocks:
            raise RuntimeError(f"no cross-fitted calibration blocks for {outer_region}")

        global_residuals = np.concatenate([block["residual"] for block in calibration_blocks])
        for year, month in months:
            target = by_key.get((outer_region, year, month))
            donor_parts = [by_key[(r, year, month)] for r in train_regions if (r, year, month) in by_key]
            if target is None or not donor_parts:
                continue
            component = profiles(target, pd.concat(donor_parts, ignore_index=True))
            actual = np.asarray(component["actual"], float)
            point = np.asarray(component["local_point"], float)
            target_share = float(component["local_share"])
            target_fuel_mass = target.loc[:, [
                "generation_coal", "generation_gas", "generation_oil", "generation_nuclear",
                "generation_hydro", "generation_wind", "generation_solar", "generation_other",
            ]].sum().to_numpy(float)
            target_fuel_mix = target_fuel_mass / target_fuel_mass.sum()
            n_hours = len(actual)

            for alpha in ALPHAS:
                glo = quantile(global_residuals, alpha / 2.0)
                ghi = quantile(global_residuals, 1.0 - alpha / 2.0)
                methods = {
                    "global_local_asymmetric": (glo, ghi, np.nan),
                }
                selected_full_mix = None
                for bandwidth in BANDWIDTHS:
                    weights = []
                    residual_parts = []
                    block_weights = []
                    for block in calibration_blocks:
                        kernel = float(np.exp(-0.5 * ((target_share - block["share"]) / bandwidth) ** 2))
                        per_hour = kernel / len(block["residual"])
                        residual_parts.append(block["residual"])
                        weights.append(np.full(len(block["residual"]), per_hour, dtype=float))
                        block_weights.append(kernel)
                    values = np.concatenate(residual_parts)
                    sample_weights = np.concatenate(weights)
                    qlo = weighted_quantile(values, sample_weights, alpha / 2.0)
                    qhi = weighted_quantile(values, sample_weights, 1.0 - alpha / 2.0)
                    effective_blocks = float(np.sum(block_weights) ** 2 / np.sum(np.square(block_weights)))
                    methods[f"mix_kernel_bw_{bandwidth:.2f}"] = (qlo, qhi, effective_blocks)
                for bandwidth in BANDWIDTHS:
                    weights = []
                    residual_parts = []
                    block_weights = []
                    for block in calibration_blocks:
                        total_variation = 0.5 * float(np.abs(target_fuel_mix - block["fuel_mix"]).sum())
                        kernel = float(np.exp(-0.5 * (total_variation / bandwidth) ** 2))
                        per_hour = kernel / len(block["residual"])
                        residual_parts.append(block["residual"])
                        weights.append(np.full(len(block["residual"]), per_hour, dtype=float))
                        block_weights.append(kernel)
                    values = np.concatenate(residual_parts)
                    sample_weights = np.concatenate(weights)
                    qlo = weighted_quantile(values, sample_weights, alpha / 2.0)
                    qhi = weighted_quantile(values, sample_weights, 1.0 - alpha / 2.0)
                    effective_blocks = float(np.sum(block_weights) ** 2 / np.sum(np.square(block_weights)))
                    methods[f"full_mix_tv_kernel_bw_{bandwidth:.2f}"] = (qlo, qhi, effective_blocks)
                    if bandwidth == max(BANDWIDTHS):
                        selected_full_mix = (qlo, qhi, effective_blocks)
                if selected_full_mix is not None:
                    mix_lo, mix_hi, effective_blocks = selected_full_mix
                    for blend in BLENDINGS:
                        methods[f"global_full_mix_blend_{blend:.2f}"] = (
                            (1.0 - blend) * glo + blend * mix_lo,
                            (1.0 - blend) * ghi + blend * mix_hi,
                            effective_blocks,
                        )

                for method, (qlo, qhi, effective_blocks) in methods.items():
                    lo = np.clip(point + qlo, 0.0, 1000.0)
                    hi = np.clip(point + qhi, 0.0, 1000.0)
                    hourly_coverage = float(np.mean((actual >= lo) & (actual <= hi)))
                    hourly_width = float(np.mean(hi - lo))
                    iscore = float(np.mean(interval_score(actual, lo, hi, alpha)))
                    for threshold in THRESHOLDS:
                        observed = int(np.sum(actual > threshold))
                        lower = int(np.sum(lo > threshold))
                        upper = int(np.sum(hi > threshold))
                        rows.append({
                            "outer_region": outer_region,
                            "year": year,
                            "month": month,
                            "alpha": alpha,
                            "nominal_coverage": 1.0 - alpha,
                            "threshold_g_kwh": threshold,
                            "method": method,
                            "target_thermal_share": target_share,
                            "calibration_effective_blocks": effective_blocks,
                            "n_calibration_blocks": len(calibration_blocks),
                            "hourly_coverage": hourly_coverage,
                            "hourly_mean_width_g_kwh": hourly_width,
                            "hourly_interval_score": iscore,
                            "observed_tail_count": observed,
                            "tail_count_lower": lower,
                            "tail_count_upper": upper,
                            "tail_count_covered": lower <= observed <= upper,
                            "tail_count_width_hours": upper - lower,
                            "tail_count_width_fraction": (upper - lower) / n_hours,
                        })

    detail = pd.DataFrame(rows)
    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "region_month_scores.csv", index=False)
    region = detail.groupby(
        ["alpha", "nominal_coverage", "threshold_g_kwh", "method", "outer_region"], as_index=False
    ).agg(
        hourly_coverage=("hourly_coverage", "mean"),
        hourly_width=("hourly_mean_width_g_kwh", "mean"),
        hourly_interval_score=("hourly_interval_score", "mean"),
        tail_count_coverage=("tail_count_covered", "mean"),
        tail_count_width_hours=("tail_count_width_hours", "mean"),
        tail_count_width_fraction=("tail_count_width_fraction", "mean"),
        effective_blocks=("calibration_effective_blocks", "mean"),
    )
    summary = region.groupby(["alpha", "nominal_coverage", "threshold_g_kwh", "method"], as_index=False).agg(
        regions=("outer_region", "nunique"),
        region_equal_hourly_coverage=("hourly_coverage", "mean"),
        worst_region_hourly_coverage=("hourly_coverage", "min"),
        region_equal_hourly_width_g_kwh=("hourly_width", "mean"),
        region_equal_hourly_interval_score=("hourly_interval_score", "mean"),
        region_equal_tail_count_coverage=("tail_count_coverage", "mean"),
        worst_region_tail_count_coverage=("tail_count_coverage", "min"),
        region_equal_tail_count_width_hours=("tail_count_width_hours", "mean"),
        region_equal_tail_count_width_fraction=("tail_count_width_fraction", "mean"),
        mean_effective_calibration_blocks=("effective_blocks", "mean"),
    )
    summary.to_csv(OUT / "method_summary.csv", index=False)
    json_summary = summary.astype(object).where(pd.notna(summary), None).to_dict(orient="records")
    metadata = {
        "experiment": "V380 cross-fitted residual calibration weighted by target-vs-calibration local thermal-share similarity",
        "target": "EIA generation-based fixed-factor CIF proxy, not measured direct emissions",
        "regions": list(REGIONS),
        "region_months": int(detail[["outer_region", "year", "month"]].drop_duplicates().shape[0]),
        "valid_hourly_rows": int(hourly.shape[0]),
        "calibration_design": "For each outer held-out region, calibration scores are cross-fitted across the remaining six regions; each calibration residual is weighted by a Gaussian kernel on its local thermal share and the target month's share.",
        "bandwidths": list(BANDWIDTHS),
        "methods": [
            "global_local_asymmetric",
            *[f"mix_kernel_bw_{bw:.2f}" for bw in BANDWIDTHS],
            *[f"full_mix_tv_kernel_bw_{bw:.2f}" for bw in BANDWIDTHS],
            *[f"global_full_mix_blend_{blend:.2f}" for blend in BLENDINGS],
        ],
        "caveats": [
            "Seven already-inspected regions; not an untouched external confirmation.",
            "Only seven region clusters and serially dependent hourly scores; descriptive coverage is not a formal guarantee.",
            "Monthly mix and proxy labels are retrospective; as-of vintages unavailable.",
            "The factor support and fixed-factor target do not validate Chinese hourly CIF.",
        ],
        "summary": json_summary,
    }
    (OUT / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
