"""V368: leave-region-out transfer using only local monthly thermal share."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results/research/data"
OUT = ROOT / "results/v368-thermal-share-only-transfer-r1"
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
EF_SCENARIOS = (500.0, 700.0, 900.0)
THRESHOLDS = (300.0, 500.0, 700.0)
BOOTSTRAPS = 10_000
SEED = 368


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


def paired_bootstrap(values: pd.DataFrame, column: str) -> dict[str, float | int]:
    region_means = values.groupby("region")[column].mean().reindex(REGIONS).dropna().to_numpy(float)
    if not len(region_means):
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "regions_won": 0}
    rng = np.random.default_rng(SEED)
    draws = rng.choice(region_means, size=(BOOTSTRAPS, len(region_means)), replace=True).mean(axis=1)
    return {
        "mean": float(region_means.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "regions_won": int((region_means > 0).sum()),
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    hourly = load_hourly()
    month_groups = {
        (region, int(year), int(month)): block.sort_values("time").copy()
        for (region, year, month), block in hourly.groupby(["region", "year", "month"], sort=True)
    }
    rows = []
    for (region, year, month), target in month_groups.items():
        donor = hourly.loc[
            hourly["year"].eq(year)
            & hourly["month"].eq(month)
            & hourly["region"].ne(region)
        ].copy()
        if donor.empty:
            continue
        local_total = float(target["total_mwh"].sum())
        donor_total = float(donor["total_mwh"].sum())
        local_share = float(target.loc[:, THERMAL].to_numpy(float).sum() / local_total)
        donor_share = float(donor.loc[:, THERMAL].to_numpy(float).sum() / donor_total)
        donor_by_time = donor.groupby("time")["cif_proxy"].mean()
        donor_shape_center = float(donor["cif_proxy"].mean())
        donor_deviation = donor_by_time - donor_shape_center
        target_times = pd.DatetimeIndex(target["time"])
        deviations = donor_deviation.reindex(target_times).fillna(0.0).to_numpy(float)
        actual = target["cif_proxy"].to_numpy(float)
        n_hours = len(actual)
        expected_hours = int(pd.Period(f"{year}-{month:02d}").days_in_month * 24)

        for ef in EF_SCENARIOS:
            pooled_level = ef * donor_share
            local_level = ef * local_share
            predictions = {
                "pooled_flat_thermal_anchor": np.full(n_hours, pooled_level),
                "local_flat_thermal_anchor": np.full(n_hours, local_level),
                "pooled_thermal_level_plus_donor_shape": pooled_level + deviations,
                "local_thermal_level_plus_donor_shape": local_level + deviations,
            }
            predictions = {name: np.clip(pred, 0.0, 1000.0) for name, pred in predictions.items()}
            for threshold in THRESHOLDS:
                observed_tail = int(np.sum(actual > threshold))
                for method, prediction in predictions.items():
                    predicted_tail = int(np.sum(prediction > threshold))
                    rows.append(
                        {
                            "region": region,
                            "year": year,
                            "month": month,
                            "valid_hours": n_hours,
                            "expected_hours": expected_hours,
                            "valid_fraction": n_hours / expected_hours,
                            "threshold_g_kwh": threshold,
                            "thermal_ef_scenario_g_kwh": ef,
                            "local_thermal_share": local_share,
                            "leave_one_region_out_pooled_thermal_share": donor_share,
                            "local_anchor_g_kwh": local_level,
                            "pooled_anchor_g_kwh": pooled_level,
                            "method": method,
                            "hourly_cif_mae_g_kwh": float(np.mean(np.abs(prediction - actual))),
                            "observed_tail_count": observed_tail,
                            "predicted_tail_count": predicted_tail,
                            "absolute_tail_count_error": abs(predicted_tail - observed_tail),
                            "absolute_tail_fraction_error": abs(predicted_tail - observed_tail) / n_hours,
                        }
                    )

    detail = pd.DataFrame(rows)
    primary_methods = [
        "pooled_thermal_level_plus_donor_shape",
        "local_thermal_level_plus_donor_shape",
    ]
    primary = detail.loc[detail["method"].isin(primary_methods)]
    keys = ["region", "year", "month", "threshold_g_kwh", "thermal_ef_scenario_g_kwh"]
    paired = primary.pivot(
        index=keys,
        columns="method",
        values=["hourly_cif_mae_g_kwh", "absolute_tail_count_error", "absolute_tail_fraction_error"],
    ).reset_index()
    paired.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
        for col in paired.columns
    ]
    pairs = [
        ("hourly_cif_mae_g_kwh", "hourly_mae_gain_local_vs_pooled"),
        ("absolute_tail_count_error", "tail_count_error_gain_local_vs_pooled"),
        ("absolute_tail_fraction_error", "tail_fraction_error_gain_local_vs_pooled"),
    ]
    for metric, gain_name in pairs:
        pooled_col = f"{metric}_pooled_thermal_level_plus_donor_shape"
        local_col = f"{metric}_local_thermal_level_plus_donor_shape"
        paired[gain_name] = paired[pooled_col] - paired[local_col]

    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "region_month_method_scores.csv", index=False)
    paired.to_csv(OUT / "paired_local_vs_pooled.csv", index=False)
    method_summary = (
        detail.groupby(["thermal_ef_scenario_g_kwh", "threshold_g_kwh", "method"], as_index=False)
        .agg(
            region_months=("region", "size"),
            mean_hourly_cif_mae_g_kwh=("hourly_cif_mae_g_kwh", "mean"),
            mean_absolute_tail_count_error=("absolute_tail_count_error", "mean"),
            mean_absolute_tail_fraction_error=("absolute_tail_fraction_error", "mean"),
        )
    )
    method_summary.to_csv(OUT / "method_summary.csv", index=False)

    gains = {}
    for (ef, threshold), subset in paired.groupby(["thermal_ef_scenario_g_kwh", "threshold_g_kwh"]):
        key = f"ef{int(ef)}_tau{int(threshold)}"
        gains[key] = {}
        for metric, column in [
            ("hourly_mae_gain_g_kwh", "hourly_mae_gain_local_vs_pooled"),
            ("tail_count_error_gain_hours", "tail_count_error_gain_local_vs_pooled"),
            ("tail_fraction_error_gain", "tail_fraction_error_gain_local_vs_pooled"),
        ]:
            gains[key][metric] = paired_bootstrap(subset, column)

    metadata = {
        "experiment": "V368 leave-region-out transfer with only local monthly thermal share",
        "input_files": [f"results/research/data/{region}_hourly.csv" for region in REGIONS],
        "target": "fixed-factor hourly eight-bucket generation CIF proxy, not measured emissions",
        "local_input_contract": "only held-out local monthly coal+gas+oil share; no held-out hourly CIF or source-specific local factor is used by the forecast",
        "donor_contract": "same-year-month other-region hourly CIF shape; one target region fully excluded from donor values",
        "thermal_ef_scenarios_g_kwh": list(EF_SCENARIOS),
        "thresholds_g_kwh_strict": list(THRESHOLDS),
        "regions": list(REGIONS),
        "n_region_months": int(detail[["region", "year", "month"]].drop_duplicates().shape[0]),
        "valid_hours": int(hourly.shape[0]),
        "primary_pair": "local_thermal_level_plus_donor_shape vs pooled_thermal_level_plus_donor_shape",
        "paired_gains": gains,
        "bootstrap": {
            "unit": "held-out region; average region-month paired gains within region before resampling",
            "resamples": BOOTSTRAPS,
            "seed": SEED,
            "region_clusters": int(paired["region"].nunique()),
            "interpretation": "descriptive cluster interval based on seven regions",
        },
        "caveats": [
            "EIA generation/CIF series are a source-domain proxy, not direct measured emissions.",
            "Regions were previously opened; CV prevents target-hour leakage into its donor shape but is not a sealed unseen-region test.",
            "Monthly inputs are reconstructed from retrospective hourly source data; publication vintages are unknown.",
            "Thermal-factor scenarios are sensitivity values, not locally calibrated estimates.",
            "China's NBS monthly series is above-designated-size and has an unobserved all-plant remainder; that scope issue is not addressed here.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gains, ensure_ascii=False, indent=2))
    print(f"region_months={metadata['n_region_months']} valid_hours={metadata['valid_hours']}")


if __name__ == "__main__":
    main()
