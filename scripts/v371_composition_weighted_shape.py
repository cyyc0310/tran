"""V371: select donor hourly-shape profiles by monthly generation composition."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v369_crossfit_risk_intervals import FUELS, REGIONS, THRESHOLDS, load_hourly  # noqa: E402

OUT = ROOT / "results/v371-composition-weighted-shape-r1"
LOCAL_EF = 700.0
BANDWIDTHS = (0.15, 0.30)
BOOTSTRAPS = 10_000
SEED = 371


def composition(block: pd.DataFrame) -> np.ndarray:
    margins = block.loc[:, FUELS].sum().to_numpy(float)
    total = float(margins.sum())
    coal, gas, oil, nuclear, hydro, wind, solar, other = margins
    return np.array([coal + gas + oil, hydro, nuclear, wind, solar, other], dtype=float) / total


def weighted_shape(
    target_times: pd.DatetimeIndex,
    donor_shapes: pd.DataFrame,
    donor_weights: np.ndarray,
) -> np.ndarray:
    aligned = donor_shapes.reindex(target_times).to_numpy(float)
    available = np.isfinite(aligned)
    w = np.broadcast_to(donor_weights[None, :], aligned.shape).copy()
    w[~available] = 0.0
    aligned[~available] = 0.0
    denominator = w.sum(axis=1)
    return np.divide(
        (aligned * w).sum(axis=1),
        denominator,
        out=np.zeros(len(target_times), dtype=float),
        where=denominator > 0,
    )


def boot_gain(values: pd.DataFrame, column: str) -> dict[str, float | int]:
    region_means = values.groupby("region")[column].mean().reindex(REGIONS).dropna().to_numpy(float)
    rng = np.random.default_rng(SEED)
    draws = rng.choice(region_means, size=(BOOTSTRAPS, len(region_means)), replace=True).mean(axis=1)
    return {
        "mean_gain": float(region_means.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "regions_won": int((region_means > 0).sum()),
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    hourly = load_hourly()
    blocks = {
        (region, int(year), int(month)): block.sort_values("time").copy()
        for (region, year, month), block in hourly.groupby(["region", "year", "month"], sort=True)
    }
    month_keys = sorted({(year, month) for _, year, month in blocks})
    rows = []

    for held_region in REGIONS:
        donor_regions = [region for region in REGIONS if region != held_region]
        for year, month in month_keys:
            target = blocks.get((held_region, year, month))
            available_donors = [
                region for region in donor_regions if (region, year, month) in blocks
            ]
            if target is None or not available_donors:
                continue

            target_mix = composition(target)
            target_times = pd.DatetimeIndex(target["time"])
            actual = target["cif_proxy"].to_numpy(float)
            thermal_share = float(target_mix[0])
            anchor = LOCAL_EF * thermal_share
            donor_mix = []
            centered_profiles = {}
            for donor_region in available_donors:
                donor = blocks[(donor_region, year, month)]
                donor_mix.append(composition(donor))
                series = donor.set_index("time")["cif_proxy"]
                centered_profiles[donor_region] = series - float(series.mean())
            donor_mix_array = np.vstack(donor_mix)
            distances = np.linalg.norm(donor_mix_array - target_mix[None, :], axis=1)
            shape_matrix = pd.concat(
                [centered_profiles[region].rename(region) for region in available_donors],
                axis=1,
                sort=True,
            ).sort_index()

            method_weights = {"equal_all": np.ones(len(available_donors), dtype=float)}
            nearest = np.argsort(distances)[: min(3, len(available_donors))]
            nearest_weights = np.zeros(len(available_donors), dtype=float)
            nearest_weights[nearest] = 1.0
            method_weights["nearest3"] = nearest_weights
            for bandwidth in BANDWIDTHS:
                method_weights[f"kernel_{int(bandwidth * 100):03d}"] = np.exp(
                    -(distances**2) / (2.0 * bandwidth**2)
                )

            predictions = {"local_flat": np.full(len(actual), anchor)}
            for method, weights in method_weights.items():
                weights = weights / weights.sum()
                delta = weighted_shape(target_times, shape_matrix, weights)
                predictions[f"local_level_{method}"] = np.clip(anchor + delta, 0.0, 1000.0)

            for threshold in THRESHOLDS:
                observed_count = int(np.sum(actual > threshold))
                for method, predicted in predictions.items():
                    predicted_count = int(np.sum(predicted > threshold))
                    rows.append(
                        {
                            "region": held_region,
                            "year": year,
                            "month": month,
                            "threshold_g_kwh": threshold,
                            "n_valid_hours": len(actual),
                            "local_thermal_share": thermal_share,
                            "composition_distance_to_nearest_donor": float(np.min(distances)),
                            "method": method,
                            "hourly_cif_mae_g_kwh": float(np.mean(np.abs(predicted - actual))),
                            "observed_tail_count": observed_count,
                            "predicted_tail_count": predicted_count,
                            "absolute_tail_count_error": abs(predicted_count - observed_count),
                            "absolute_tail_fraction_error": abs(predicted_count - observed_count) / len(actual),
                        }
                    )

    detail = pd.DataFrame(rows)
    baseline = detail.loc[detail["method"].eq("local_level_equal_all")].copy()
    candidates = detail.loc[~detail["method"].eq("local_level_equal_all")].copy()
    keys = ["region", "year", "month", "threshold_g_kwh"]
    paired = candidates.merge(
        baseline[keys + ["hourly_cif_mae_g_kwh", "absolute_tail_count_error", "absolute_tail_fraction_error"]],
        on=keys,
        suffixes=("", "_equal_all"),
        validate="many_to_one",
    )
    paired["hourly_mae_gain_vs_equal_all"] = (
        paired["hourly_cif_mae_g_kwh_equal_all"] - paired["hourly_cif_mae_g_kwh"]
    )
    paired["tail_count_error_gain_vs_equal_all"] = (
        paired["absolute_tail_count_error_equal_all"] - paired["absolute_tail_count_error"]
    )
    paired["tail_fraction_error_gain_vs_equal_all"] = (
        paired["absolute_tail_fraction_error_equal_all"] - paired["absolute_tail_fraction_error"]
    )

    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "region_month_method_scores.csv", index=False)
    paired.to_csv(OUT / "paired_shape_method_comparison.csv", index=False)
    method_summary = (
        detail.groupby(["method", "threshold_g_kwh"], as_index=False)
        .agg(
            region_months=("region", "size"),
            mean_hourly_cif_mae_g_kwh=("hourly_cif_mae_g_kwh", "mean"),
            mean_absolute_tail_count_error=("absolute_tail_count_error", "mean"),
            mean_absolute_tail_fraction_error=("absolute_tail_fraction_error", "mean"),
        )
    )
    method_summary.to_csv(OUT / "method_summary.csv", index=False)

    gain_summary = {}
    for (method, threshold), subset in paired.groupby(["method", "threshold_g_kwh"]):
        gain_summary[f"{method}_tau{int(threshold)}"] = {
            "hourly_mae_gain_g_kwh": boot_gain(subset, "hourly_mae_gain_vs_equal_all"),
            "tail_count_error_gain_hours": boot_gain(subset, "tail_count_error_gain_vs_equal_all"),
            "tail_fraction_error_gain": boot_gain(subset, "tail_fraction_error_gain_vs_equal_all"),
        }
    metadata = {
        "experiment": "V371 composition-weighted donor hourly-shape transfer",
        "target": "EIA fixed-factor generation CIF proxy, not measured emissions",
        "input_features": ["thermal", "hydro", "nuclear", "wind", "solar", "other residual"],
        "regions": list(REGIONS),
        "region_months": int(detail[["region", "year", "month"]].drop_duplicates().shape[0]),
        "valid_hours": int(hourly.shape[0]),
        "local_thermal_ef_g_kwh": LOCAL_EF,
        "thresholds_g_kwh_strict": list(THRESHOLDS),
        "gain_summary_vs_equal_all": gain_summary,
        "bootstrap": {"unit": "region", "resamples": BOOTSTRAPS, "seed": SEED, "region_clusters": int(detail.region.nunique())},
        "limitations": [
            "All seven balancing authorities have previously been inspected; folds are not a sealed external holdout.",
            "The target is an hourly fixed-factor generation proxy, not measured emissions or China validation.",
            "Monthly composition is reconstructed from retrospective source data; original availability vintages are unknown.",
            "Other is retained as a residual feature although the Chinese NBS table lists five main generation classes.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gain_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
