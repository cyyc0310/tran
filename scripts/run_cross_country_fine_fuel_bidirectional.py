"""Bidirectional US/GB transfer with a five-group budget and fine-fuel upper budget."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcif.research import cross_country as cc

OUT = ROOT / "results/cross-country-fine-fuel-bidirectional-r1"
PROTOCOL = ROOT / "docs/CROSS_COUNTRY_FINE_FUEL_BIDIRECTIONAL_PROTOCOL_20261101.md"
US_REGIONS = tuple(cc.ZONES)
COMMON_MONTHS = ("2024-02", "2024-04", "2024-06")
INPUTS = [ROOT / f"results/research/data/{region}_hourly.csv" for region in US_REGIONS]
INPUTS += [ROOT / "data/neso-gb/df_fuel_ckan.csv"]
CODE_INPUTS = [Path(__file__).resolve(), Path(cc.__file__), PROTOCOL]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fine_monthly_features(frame: pd.DataFrame, lag: int = 2) -> pd.DataFrame:
    """Attach complete, lagged eight-fuel shares without using target-month totals."""
    if lag < 2:
        raise ValueError("fine-fuel monthly features require lag >= 2")
    out = frame[["time", "clock", "region"]].copy()
    periods = pd.to_datetime(out.clock).dt.to_period("M")
    out["period"] = periods
    shares_by_region = []
    for region, part in frame.groupby("region", sort=False):
        period = pd.to_datetime(part.clock).dt.to_period("M")
        values = part[list(cc.FUELS)].copy()
        values["period"] = period.to_numpy()
        sums = values.groupby("period")[list(cc.FUELS)].sum()
        counts = values.groupby("period").size()
        expected = part.assign(period=period.to_numpy()).groupby("period").expected_hours.first()
        complete = counts / expected >= .99
        sums = sums.loc[complete]
        totals = sums.sum(axis=1)
        valid = totals.gt(0) & np.isfinite(totals)
        shares = sums.loc[valid].div(totals.loc[valid], axis=0)
        shares.index += lag
        shares = shares.add_prefix("fine_")
        shares["region"] = region
        shares_by_region.append(shares)
    panel = pd.concat(shares_by_region).reset_index(names="period")
    out = out.merge(panel, on=["region", "period"], how="left", validate="many_to_one")
    return out.drop(columns=["region", "period"])


def combined_features(frame: pd.DataFrame) -> pd.DataFrame:
    coarse = cc.monthly_inputs(frame, lag=2)
    fine = fine_monthly_features(frame, lag=2)
    return coarse.join(fine.drop(columns=["time", "clock"]), how="left")


def supported(frame: pd.DataFrame) -> np.ndarray:
    columns = [*cc.COARSE, *(f"fine_{fuel}" for fuel in cc.FUELS)]
    values = frame[columns].to_numpy(float)
    return np.isfinite(values).all(axis=1)


def source_fine_profiles(raw: pd.DataFrame, train_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    profiles, support = [], {}
    source = raw.loc[train_mask].copy()
    for region, part in source.groupby("region", sort=True):
        hour = pd.to_datetime(part.clock).dt.hour.to_numpy(int)
        by_hour = (part.assign(hour=hour).groupby("hour")[list(cc.FUELS)].mean()
                   .reindex(range(24)).fillna(0.))
        means = by_hour.mean(axis=0)
        one = np.ones((24, len(cc.FUELS)), dtype=float)
        active = means.to_numpy(float) > 0
        one[:, active] = by_hour.loc[:, means.index[active]].to_numpy(float) / means.loc[means.index[active]].to_numpy(float)
        profiles.append(one)
        support[str(region)] = {
            fuel: {"active_hours": int((part[fuel].to_numpy(float) > 0).sum()),
                   "flat_fallback_used": bool(not active[i])}
            for i, fuel in enumerate(cc.FUELS)}
    if not profiles:
        raise ValueError("source profiles need at least one region")
    return np.mean(profiles, axis=0), support


def fine_profile_prediction(profile: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    hour = pd.to_datetime(features.clock).dt.hour.to_numpy(int)
    shares = features[[f"fine_{fuel}" for fuel in cc.FUELS]].to_numpy(float)
    activity = profile[hour] * shares
    totals = activity.sum(axis=1)
    if np.any(totals <= 0) or not np.isfinite(activity).all():
        raise ValueError("fine-profile prediction has invalid activity")
    return activity @ cc.EF / totals


def static_fine_prediction(features: pd.DataFrame) -> np.ndarray:
    shares = features[[f"fine_{fuel}" for fuel in cc.FUELS]].to_numpy(float)
    return shares @ cc.EF


def predict_all(models: dict, profile: np.ndarray, features: pd.DataFrame) -> dict[str, np.ndarray]:
    baseline = cc.predict_models(models, features)
    return {
        "anchor700": baseline["anchor700"],
        "source_thermal_ef": baseline["source_thermal_ef"],
        "M5_coarse_recomposition": baseline["coarse_recomposition"],
        "MplusF_static_EF": np.clip(static_fine_prediction(features), 0., 1000.),
        "MplusF_fuel_profile": np.clip(fine_profile_prediction(profile, features), 0., 1000.),
    }


def calibration(models: dict, profile: np.ndarray, frame: pd.DataFrame,
                features: pd.DataFrame, labels: np.ndarray) -> dict[str, tuple[float, float]]:
    mask = (pd.to_datetime(features.clock).dt.year.eq(2023).to_numpy()
            & supported(features))
    if mask.sum() < 1000:
        raise ValueError(f"insufficient source calibration rows: {int(mask.sum())}")
    rows = features.loc[mask].copy()
    y = labels[mask]
    raw = predict_all(models, profile, rows)
    result = {}
    for method, prediction in raw.items():
        residual = y - prediction
        bias = float(residual.mean())
        sigma = max(float(np.sqrt(np.mean((residual - bias) ** 2))), 1.)
        result[method] = (bias, sigma)
    return result


def run_direction(raw_all: dict[str, pd.DataFrame], direction: str,
                  source_regions: tuple[str, ...], target_regions: tuple[str, ...]) -> tuple[list[pd.DataFrame], dict]:
    source_raw = pd.concat([raw_all[r].assign(_origin_region=r) for r in source_regions], ignore_index=True)
    source_raw["region"] = source_raw._origin_region.astype(str)
    source_features = combined_features(source_raw)
    joint = supported(source_features)
    train_mask = joint & pd.to_datetime(source_features.clock).dt.year.eq(2022).to_numpy()
    if train_mask.sum() < 1000 or source_features.loc[train_mask, "region"].nunique() < 1:
        raise ValueError(f"insufficient 2022 source support in {direction}")
    train = source_features.loc[train_mask].copy()
    train_fuels = source_raw.loc[train_mask, list(cc.FUELS)].to_numpy(float)
    models = cc.fit_models(train, train_fuels)
    profile, profile_support = source_fine_profiles(source_raw, train_mask)
    source_y = cc.labels(source_raw)
    calibration_rows = calibration(models, profile, source_raw, source_features, source_y)
    detail: list[pd.DataFrame] = []
    target_support = {}
    for target_region in target_regions:
        target_raw = raw_all[target_region].copy().reset_index(drop=True)
        target_features = combined_features(target_raw)
        year = pd.to_datetime(target_features.clock).dt.year
        months = pd.to_datetime(target_features.clock).dt.strftime("%Y-%m")
        use = (year.eq(2024) & months.isin(COMMON_MONTHS)).to_numpy() & supported(target_features)
        target_support[target_region] = int(use.sum())
        if not use.any():
            continue
        rows = target_features.loc[use].copy()
        y = cc.labels(target_raw)[use]
        points = predict_all(models, profile, rows)
        for month in COMMON_MONTHS:
            month_mask = pd.to_datetime(rows.clock).dt.strftime("%Y-%m").eq(month).to_numpy()
            if not month_mask.any():
                continue
            block = rows.loc[month_mask].copy()
            y_block = y[month_mask]
            for method, point in points.items():
                bias, sigma = calibration_rows[method]
                score = cc.metrics(block, y_block, point[month_mask], bias, sigma)
                detail.append(pd.DataFrame({
                    "direction": direction, "target_region": target_region, "month": month,
                    "method": method, "time": block.time.astype(str),
                    "label": y_block, "prediction_raw": point[month_mask],
                    "prediction_corrected": point[month_mask] + bias,
                    "thermal_share": block.thermal.to_numpy(float),
                    "fine_coal_share": block.fine_coal.to_numpy(float),
                    "fine_gas_share": block.fine_gas.to_numpy(float),
                    "mae": score["mae"], "shape_mae": score["shape_mae"],
                    "crps": score["crps_gaussian"], "coverage90": score["coverage90"],
                    "width90": score["width90"], "interval_score90": score["interval_score90"],
                    "hours": score["hours"],
                }))
    metadata = {
        "direction": direction, "source_regions": list(source_regions),
        "target_regions": list(target_regions), "train_year": 2022,
        "calibration_year": 2023, "target_common_months": list(COMMON_MONTHS),
        "source_train_hours": int(train_mask.sum()),
        "source_train_regions": int(source_features.loc[train_mask, "region"].nunique()),
        "calibration": {method: {"bias": bias, "sigma": sigma}
                        for method, (bias, sigma) in calibration_rows.items()},
        "source_fuel_profile_support": profile_support,
        "target_scored_hours": target_support,
    }
    return detail, metadata


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    raw, load_audit = cc.load_inputs(ROOT)
    direction_runs = [
        ("US_to_GB", tuple(r for r in raw if r != "GB"), ("GB",)),
        ("GB_to_US", ("GB",), tuple(r for r in raw if r != "GB")),
    ]
    details, metadata = [], []
    for direction, sources, targets in direction_runs:
        rows, info = run_direction(raw, direction, sources, targets)
        details.extend(rows)
        metadata.append(info)
    if not details:
        raise ValueError("no paired target predictions")
    hourly = pd.concat(details, ignore_index=True)
    cell = (hourly.groupby(["direction", "target_region", "month", "method"], as_index=False)
            .agg(hours=("hours", "first"), mae=("mae", "first"),
                 shape_mae=("shape_mae", "first"), crps=("crps", "first"),
                 coverage90=("coverage90", "first"), width90=("width90", "first"),
                 interval_score90=("interval_score90", "first")))
    macro = (cell.groupby(["direction", "method"], as_index=False)
             .agg(target_region_months=("month", "size"), target_regions=("target_region", "nunique"),
                  macro_mae=("mae", "mean"), macro_shape_mae=("shape_mae", "mean"),
                  macro_crps=("crps", "mean"), macro_coverage90=("coverage90", "mean"),
                  mean_width90=("width90", "mean"),
                  macro_interval_score90=("interval_score90", "mean")))
    OUT.mkdir(parents=True)
    hourly.to_csv(OUT / "hourly_predictions.csv.gz", index=False, compression="gzip")
    cell.to_csv(OUT / "region_month_metrics.csv", index=False)
    macro.to_csv(OUT / "macro_metrics.csv", index=False)
    provenance = {
        "status": "POSTHOC_BIDIRECTIONAL_FINE_FUEL_DEVELOPMENT_NOT_UNSEEN_CONFIRMATION",
        "label": "fixed-EF domestic-generation intensity proxy",
        "monthly_inputs": "revised hourly-derived shares with exact t-2 lag; not as-issued official vintage",
        "target_fine_fuel_budget": "eight fuel shares; upper budget beyond China five-group M",
        "source_years": {"train": 2022, "calibration": 2023},
        "target_year": 2024, "common_months": list(COMMON_MONTHS),
        "direction_metadata": metadata, "load_audit": load_audit,
        "input_hashes": {str(p.relative_to(ROOT)): sha(p) for p in INPUTS},
        "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in CODE_INPUTS},
        "runtime_versions": {"python": platform.python_version(), "pandas": pd.__version__,
                             "numpy": np.__version__, "scipy": scipy.__version__,
                             "scikit_learn": sklearn.__version__},
        "limitations": ["2024 targets were previously viewed in other development experiments",
                        "only three common target months are in the primary paired cohort",
                        "target monthly composition is reconstructed from target hourly histories",
                        "fine-fuel budget is not the China five-group deployment budget",
                        "fixed-EF proxy is not official consumption-side CIF",
                        "source oil-profile support may be absent and is explicitly recorded"],
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
