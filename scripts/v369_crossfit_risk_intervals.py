"""V369: cross-fitted risk intervals from local thermal share and donor hourly shape."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results/research/data"
OUT = ROOT / "results/v369-crossfit-risk-intervals-r1"
REGIONS = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
FUELS = (
    "generation_coal",
    "generation_gas",
    "generation_oil",
    "generation_nuclear",
    "generation_hydro",
    "generation_wind",
    "generation_solar",
    "generation_other",
)
THERMAL = FUELS[:3]
EF = np.array([1000.0, 400.0, 800.0, 0.0, 0.0, 0.0, 0.0, 0.0])
EF_LOW = 400.0
EF_MID = 700.0
EF_HIGH = 1000.0
ALPHAS = (0.10, 0.05)
THRESHOLDS = (300.0, 500.0, 700.0)


def load_hourly() -> pd.DataFrame:
    frames = []
    for region in REGIONS:
        frame = pd.read_csv(DATA_DIR / f"{region}_hourly.csv")
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
        frame["region"] = region
        valid = frame["valid"].astype(str).str.lower().eq("true")
        values = frame.loc[:, FUELS].to_numpy(float)
        valid &= frame["time"].notna() & np.isfinite(values).all(axis=1)
        frame = frame.loc[valid].copy()
        values = frame.loc[:, FUELS].to_numpy(float)
        frame = frame.loc[(values >= -1e-9).all(axis=1)].copy()
        frame.loc[:, FUELS] = frame.loc[:, FUELS].clip(lower=0.0)
        frame["total_mwh"] = frame.loc[:, FUELS].sum(axis=1)
        frame = frame.loc[frame["total_mwh"].gt(0)].copy()
        values = frame.loc[:, FUELS].to_numpy(float)
        frame["cif_proxy"] = values @ EF / frame["total_mwh"].to_numpy(float)
        frame["year"] = frame["time"].dt.year
        frame["month"] = frame["time"].dt.month
        frames.append(frame[["time", "region", "year", "month", *FUELS, "total_mwh", "cif_proxy"]])
    return pd.concat(frames, ignore_index=True)


def profiles(target: pd.DataFrame, donor: pd.DataFrame) -> dict[str, np.ndarray | float]:
    donor_margin = donor.loc[:, FUELS].sum().to_numpy(float)
    donor_share = float(donor.loc[:, THERMAL].to_numpy(float).sum() / donor_margin.sum())
    local_margin = target.loc[:, FUELS].sum().to_numpy(float)
    local_share = float(target.loc[:, THERMAL].to_numpy(float).sum() / local_margin.sum())

    donor_by_time = donor.groupby("time")["cif_proxy"].mean()
    donor_shape_center = float(donor["cif_proxy"].mean())
    deviations = (donor_by_time - donor_shape_center).reindex(
        pd.DatetimeIndex(target["time"])
    ).fillna(0.0).to_numpy(float)
    local_point = np.clip(EF_MID * local_share + deviations, 0.0, 1000.0)
    pooled_point = np.clip(EF_MID * donor_share + deviations, 0.0, 1000.0)
    local_ef_box = np.column_stack(
        [
            np.clip(EF_LOW * local_share + deviations, 0.0, 1000.0),
            np.clip(EF_HIGH * local_share + deviations, 0.0, 1000.0),
        ]
    )
    pooled_ef_box = np.column_stack(
        [
            np.clip(EF_LOW * donor_share + deviations, 0.0, 1000.0),
            np.clip(EF_HIGH * donor_share + deviations, 0.0, 1000.0),
        ]
    )
    return {
        "local_share": local_share,
        "pooled_share": donor_share,
        "local_point": local_point,
        "pooled_point": pooled_point,
        "local_ef_box": local_ef_box,
        "pooled_ef_box": pooled_ef_box,
        "actual": target["cif_proxy"].to_numpy(float),
    }


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="higher"))


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> np.ndarray:
    return (hi - lo) + (2.0 / alpha) * np.maximum(lo - y, 0.0) + (2.0 / alpha) * np.maximum(y - hi, 0.0)


def make_intervals(
    components: dict[str, np.ndarray | float],
    residuals: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, np.ndarray]:
    result = {
        "local_ef_box": np.asarray(components["local_ef_box"], float),
        "pooled_ef_box": np.asarray(components["pooled_ef_box"], float),
    }
    for level_name, residual_name in [("local", "local"), ("pooled", "pooled")]:
        point = np.asarray(components[f"{level_name}_point"], float)
        resid = residuals[residual_name]
        radius = quantile(np.abs(resid), 1.0 - alpha)
        lo_q = quantile(resid, alpha / 2.0)
        hi_q = quantile(resid, 1.0 - alpha / 2.0)
        result[f"{level_name}_symmetric_q"] = np.column_stack(
            [np.clip(point - radius, 0.0, 1000.0), np.clip(point + radius, 0.0, 1000.0)]
        )
        result[f"{level_name}_asymmetric_q"] = np.column_stack(
            [np.clip(point + lo_q, 0.0, 1000.0), np.clip(point + hi_q, 0.0, 1000.0)]
        )
    return result


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    hourly = load_hourly()
    by_key = {
        (region, int(year), int(month)): block.sort_values("time").copy()
        for (region, year, month), block in hourly.groupby(["region", "year", "month"], sort=True)
    }
    months = sorted({(year, month) for _, year, month in by_key})
    result_rows = []

    for outer_region in REGIONS:
        training_regions = [region for region in REGIONS if region != outer_region]
        calibration_residuals = {"local": [], "pooled": []}

        # Cross-fit calibration scores inside the outer training fold.
        for calibration_region in training_regions:
            inner_donors = [region for region in training_regions if region != calibration_region]
            for year, month in months:
                target = by_key.get((calibration_region, year, month))
                donor_parts = [by_key[(region, year, month)] for region in inner_donors if (region, year, month) in by_key]
                if target is None or not donor_parts:
                    continue
                donor = pd.concat(donor_parts, ignore_index=True)
                component = profiles(target, donor)
                actual = np.asarray(component["actual"], float)
                calibration_residuals["local"].append(actual - np.asarray(component["local_point"], float))
                calibration_residuals["pooled"].append(actual - np.asarray(component["pooled_point"], float))

        residuals = {
            key: np.concatenate(parts) if parts else np.array([], dtype=float)
            for key, parts in calibration_residuals.items()
        }
        if any(values.size == 0 for values in residuals.values()):
            raise RuntimeError(f"empty calibration residuals for outer region {outer_region}")

        for year, month in months:
            target = by_key.get((outer_region, year, month))
            donor_parts = [by_key[(region, year, month)] for region in training_regions if (region, year, month) in by_key]
            if target is None or not donor_parts:
                continue
            donor = pd.concat(donor_parts, ignore_index=True)
            component = profiles(target, donor)
            actual = np.asarray(component["actual"], float)
            n_hours = len(actual)
            expected_hours = int(pd.Period(f"{year}-{month:02d}").days_in_month * 24)
            for alpha in ALPHAS:
                intervals = make_intervals(component, residuals, alpha)
                for method, bounds in intervals.items():
                    lo, hi = bounds[:, 0], bounds[:, 1]
                    hourly_covered = (actual >= lo) & (actual <= hi)
                    for threshold in THRESHOLDS:
                        observed_count = int(np.sum(actual > threshold))
                        lower_count = int(np.sum(lo > threshold))
                        upper_count = int(np.sum(hi > threshold))
                        count_covered = lower_count <= observed_count <= upper_count
                        count_width = upper_count - lower_count
                        winkler = interval_score(
                            np.array([observed_count], dtype=float),
                            np.array([lower_count], dtype=float),
                            np.array([upper_count], dtype=float),
                            alpha,
                        )[0]
                        result_rows.append(
                            {
                                "outer_region": outer_region,
                                "year": year,
                                "month": month,
                                "alpha": alpha,
                                "nominal_coverage": 1.0 - alpha,
                                "threshold_g_kwh": threshold,
                                "method": method,
                                "n_valid_hours": n_hours,
                                "expected_hours": expected_hours,
                                "local_thermal_share": float(component["local_share"]),
                                "pooled_thermal_share": float(component["pooled_share"]),
                                "calibration_residual_count": int(len(residuals["local"])),
                                "hourly_coverage": float(hourly_covered.mean()),
                                "hourly_mean_width_g_kwh": float(np.mean(hi - lo)),
                                "hourly_interval_score": float(np.mean(interval_score(actual, lo, hi, alpha))),
                                "observed_tail_count": observed_count,
                                "tail_count_lower": lower_count,
                                "tail_count_upper": upper_count,
                                "tail_count_coverage": bool(count_covered),
                                "tail_count_width_hours": int(count_width),
                                "tail_count_width_fraction": float(count_width / n_hours),
                                "tail_count_winkler_score": float(winkler),
                            }
                        )

    detail = pd.DataFrame(result_rows)
    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "region_month_interval_scores.csv", index=False)

    metric_columns = [
        "hourly_coverage",
        "hourly_mean_width_g_kwh",
        "hourly_interval_score",
        "tail_count_coverage",
        "tail_count_width_hours",
        "tail_count_width_fraction",
        "tail_count_winkler_score",
    ]
    region_means = detail.groupby(
        ["alpha", "nominal_coverage", "threshold_g_kwh", "method", "outer_region"],
        as_index=False,
    )[metric_columns].mean()
    summary = region_means.groupby(
        ["alpha", "nominal_coverage", "threshold_g_kwh", "method"], as_index=False
    ).agg(
        regions=("outer_region", "nunique"),
        region_equal_hourly_coverage=("hourly_coverage", "mean"),
        worst_region_hourly_coverage=("hourly_coverage", "min"),
        region_equal_hourly_width_g_kwh=("hourly_mean_width_g_kwh", "mean"),
        region_equal_hourly_interval_score=("hourly_interval_score", "mean"),
        region_equal_tail_count_coverage=("tail_count_coverage", "mean"),
        worst_region_tail_count_coverage=("tail_count_coverage", "min"),
        region_equal_tail_count_width_hours=("tail_count_width_hours", "mean"),
        region_equal_tail_count_width_fraction=("tail_count_width_fraction", "mean"),
        region_equal_tail_count_winkler_score=("tail_count_winkler_score", "mean"),
    )
    summary.to_csv(OUT / "method_summary.csv", index=False)

    metadata = {
        "experiment": "V369 cross-fitted local-thermal-share hourly and tail-count intervals",
        "target": "EIA fixed-factor hourly generation CIF proxy; no direct emissions truth",
        "regions": list(REGIONS),
        "n_region_months": int(detail[["outer_region", "year", "month"]].drop_duplicates().shape[0]),
        "valid_hours": int(hourly.shape[0]),
        "factor_support_g_kwh": [EF_LOW, EF_HIGH],
        "point_factor_g_kwh": EF_MID,
        "interval_levels": [1.0 - alpha for alpha in ALPHAS],
        "thresholds_g_kwh_strict": list(THRESHOLDS),
        "methods": ["local_ef_box", "pooled_ef_box", "local_symmetric_q", "local_asymmetric_q", "pooled_symmetric_q", "pooled_asymmetric_q"],
        "outer_holdout_unit": "one balancing authority; no hourly CIF from this region enters donor profile or calibration residuals",
        "inner_calibration": "cross-fitted region-month residuals among the six other BAs; held-out calibration region excluded from its donor shape",
        "summary": summary.to_dict(orient="records"),
        "caveats": [
            "Coverage is descriptive and empirical; hourly residual dependence and only seven region clusters preclude formal split-conformal claims.",
            "The region data were previously inspected; outer folds are not a sealed confirmation.",
            "EIA monthly margins are retrospectively reconstructed from the same files; publication vintages are unknown.",
            "The target is a generated fixed-factor proxy and cannot validate China's all-plant CIF.",
            "The `[400,1000]` thermal EF box is a scenario; its validity for Chinese provinces is unverified.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"region_months={metadata['n_region_months']} valid_hours={metadata['valid_hours']}")


if __name__ == "__main__":
    main()
