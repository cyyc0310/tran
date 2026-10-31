"""V367: test local monthly mix value under leave-one-region-out hourly-shape transfer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results/research/data"
OUT = ROOT / "results/v367-spatial-coarse-mix-proxy-r1"
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
EF = np.array([1000.0, 400.0, 800.0, 0.0, 0.0, 0.0, 0.0, 0.0])
THRESHOLDS = (300.0, 500.0, 700.0)
BOOTSTRAPS = 10_000
SEED = 367


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


def bootstrap_region_mean(values: pd.DataFrame, column: str) -> dict[str, float]:
    region_means = values.groupby("region")[column].mean().reindex(REGIONS).dropna().to_numpy(float)
    if not len(region_means):
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(SEED)
    draws = rng.choice(region_means, size=(BOOTSTRAPS, len(region_means)), replace=True).mean(axis=1)
    return {
        "mean": float(region_means.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")

    hourly = load_hourly()
    margins: dict[tuple[str, int, int], np.ndarray] = {}
    groups: dict[tuple[str, int, int], pd.DataFrame] = {}
    for (region, year, month), block in hourly.groupby(["region", "year", "month"], sort=True):
        margins[(region, int(year), int(month))] = block.loc[:, FUELS].sum().to_numpy(float)
        groups[(region, int(year), int(month))] = block.sort_values("time").copy()

    detail_rows = []
    for (region, year, month), target in groups.items():
        donor = hourly.loc[
            hourly["year"].eq(year)
            & hourly["month"].eq(month)
            & hourly["region"].ne(region)
        ].copy()
        if donor.empty:
            continue
        donor_margin = donor.loc[:, FUELS].sum().to_numpy(float)
        donor_level = float(donor_margin @ EF / donor_margin.sum())
        local_margin = margins[(region, year, month)]
        local_level = float(local_margin @ EF / local_margin.sum())

        donor_by_time = donor.groupby("time")["cif_proxy"].mean()
        donor_shape_mean = float(donor["cif_proxy"].mean())
        donor_deviation = (donor_by_time - donor_shape_mean)
        target_times = pd.DatetimeIndex(target["time"])
        deviations = donor_deviation.reindex(target_times).fillna(0.0).to_numpy(float)

        actual = target["cif_proxy"].to_numpy(float)
        n_hours = len(actual)
        forecasts = {
            "pooled_flat_monthly_anchor": np.full(n_hours, donor_level),
            "local_flat_monthly_anchor": np.full(n_hours, local_level),
            "pooled_level_plus_donor_shape": donor_level + deviations,
            "local_level_plus_donor_shape": local_level + deviations,
        }
        forecasts = {name: np.clip(value, 0.0, 1000.0) for name, value in forecasts.items()}
        for threshold in THRESHOLDS:
            observed_tail = int(np.sum(actual > threshold))
            for method, prediction in forecasts.items():
                predicted_tail = int(np.sum(prediction > threshold))
                detail_rows.append(
                    {
                        "region": region,
                        "year": year,
                        "month": month,
                        "n_valid_hours": n_hours,
                        "expected_hours": int(pd.Period(f"{year}-{month:02d}").days_in_month * 24),
                        "threshold_g_kwh": threshold,
                        "local_monthly_anchor_g_kwh": local_level,
                        "leave_one_region_out_pooled_anchor_g_kwh": donor_level,
                        "method": method,
                        "hourly_cif_mae_g_kwh": float(np.mean(np.abs(prediction - actual))),
                        "observed_tail_count": observed_tail,
                        "predicted_tail_count": predicted_tail,
                        "absolute_tail_count_error": abs(predicted_tail - observed_tail),
                        "absolute_tail_fraction_error": abs(predicted_tail - observed_tail) / n_hours,
                    }
                )

    detail = pd.DataFrame(detail_rows)
    primary = detail.loc[
        detail["method"].isin(
            ["pooled_level_plus_donor_shape", "local_level_plus_donor_shape"]
        )
    ].copy()
    index_cols = ["region", "year", "month", "threshold_g_kwh"]
    paired = primary.pivot(index=index_cols, columns="method", values=[
        "hourly_cif_mae_g_kwh", "absolute_tail_count_error", "absolute_tail_fraction_error"
    ]).reset_index()
    paired.columns = [
        "_".join(str(piece) for piece in column if str(piece)) if isinstance(column, tuple) else str(column)
        for column in paired.columns
    ]
    paired["hourly_mae_gain_local_vs_pooled"] = (
        paired["hourly_cif_mae_g_kwh_pooled_level_plus_donor_shape"]
        - paired["hourly_cif_mae_g_kwh_local_level_plus_donor_shape"]
    )
    paired["tail_count_error_gain_local_vs_pooled"] = (
        paired["absolute_tail_count_error_pooled_level_plus_donor_shape"]
        - paired["absolute_tail_count_error_local_level_plus_donor_shape"]
    )
    paired["tail_fraction_error_gain_local_vs_pooled"] = (
        paired["absolute_tail_fraction_error_pooled_level_plus_donor_shape"]
        - paired["absolute_tail_fraction_error_local_level_plus_donor_shape"]
    )

    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "region_month_method_scores.csv", index=False)
    paired.to_csv(OUT / "paired_local_vs_pooled.csv", index=False)

    metric_summary = {}
    for threshold, subset in paired.groupby("threshold_g_kwh"):
        metric_summary[str(int(threshold))] = {}
        for metric, column in [
            ("hourly_mae_gain_g_kwh", "hourly_mae_gain_local_vs_pooled"),
            ("tail_count_error_gain_hours", "tail_count_error_gain_local_vs_pooled"),
            ("tail_fraction_error_gain", "tail_fraction_error_gain_local_vs_pooled"),
        ]:
            gain = bootstrap_region_mean(subset, column)
            gain["regions_won"] = int((subset.groupby("region")[column].mean() > 0).sum())
            gain["region_count"] = int(subset["region"].nunique())
            metric_summary[str(int(threshold))][metric] = gain

    method_summary = (
        detail.groupby(["threshold_g_kwh", "method"], as_index=False)
        .agg(
            region_months=("region", "size"),
            mean_hourly_cif_mae_g_kwh=("hourly_cif_mae_g_kwh", "mean"),
            mean_absolute_tail_count_error=("absolute_tail_count_error", "mean"),
            mean_absolute_tail_fraction_error=("absolute_tail_fraction_error", "mean"),
        )
    )
    method_summary.to_csv(OUT / "method_summary.csv", index=False)

    metadata = {
        "experiment": "V367 leave-one-region-out information value of local monthly generation mix",
        "input_files": [f"results/research/data/{region}_hourly.csv" for region in REGIONS],
        "target": "positive_reported_generation_eight_bucket_proxy; derived CIF, not measured emissions",
        "fuel_columns": list(FUELS),
        "fixed_emission_factors_g_kwh": EF.tolist(),
        "thresholds_g_kwh_strict": list(THRESHOLDS),
        "regions": list(REGIONS),
        "n_region_months": int(detail[["region", "year", "month"]].drop_duplicates().shape[0]),
        "n_scored_month_threshold_methods": int(len(detail)),
        "valid_hours": int(hourly.shape[0]),
        "methods": [
            "pooled_flat_monthly_anchor",
            "local_flat_monthly_anchor",
            "pooled_level_plus_donor_shape",
            "local_level_plus_donor_shape",
        ],
        "primary_pair": "local_level_plus_donor_shape vs pooled_level_plus_donor_shape; identical donor shape, differing only in month-level CIF anchor",
        "metrics": metric_summary,
        "bootstrap": {
            "unit": "held-out region; region-months averaged within region before resampling",
            "resamples": BOOTSTRAPS,
            "seed": SEED,
            "regions": int(paired["region"].nunique()),
            "interpretation": "descriptive cluster interval with only seven region clusters",
        },
        "limitations": [
            "Hourly labels are fixed-factor generation proxies, not direct CO2 emissions or full-grid truth.",
            "All regions were previously opened; leave-one-region-out here means no target-region hourly outcomes enter its donor curve, not a sealed external holdout.",
            "Monthly generation margins are computed retrospectively from the same source as proxy labels; original release vintages are unknown.",
            "The experiment tests regional level information on a US analog and cannot establish China transfer or interval coverage.",
        ],
    }
    (OUT / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata["metrics"], ensure_ascii=False, indent=2))
    print(f"region_months={metadata['n_region_months']} valid_hours={metadata['valid_hours']}")


if __name__ == "__main__":
    main()
