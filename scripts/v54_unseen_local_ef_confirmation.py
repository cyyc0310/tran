"""Frozen confirmation of local annual thermal EF on six unseen EIA regions."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_external_v6 import default_native_paths  # noqa: E402
from transcif.data.eia930 import normalize_balance  # noqa: E402
from transcif.research.eia_proxy import FUELS, _monthly, _proxy_rows  # noqa: E402

SOURCE_REGIONS = ("AVA", "BANC", "BPAT", "FPL", "JEA", "MISO", "NWMT", "NYIS", "SC", "SCEG", "SEC")
TARGET_REGIONS = ("FMPP", "GRID", "IID", "LDWP", "PSEI", "TEC")
ALL_REGIONS = SOURCE_REGIONS + TARGET_REGIONS
THERMAL = ("coal", "gas", "oil", "other")
EF = np.array([1000.0, 400.0, 800.0, 0.0, 0.0, 0.0, 0.0, 400.0])
START = pd.Timestamp("2022-01-01T00:00:00Z")
STOP = pd.Timestamp("2026-07-01T00:00:00Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fuel_totals(row: pd.Series) -> np.ndarray:
    return float(row.total_mwh) * np.array([float(row[f"fuelshare_{fuel}"]) for fuel in FUELS])


def thermal_ef(values: np.ndarray) -> float:
    indexes = [FUELS.index(fuel) for fuel in THERMAL]
    denominator = float(values[indexes].sum())
    return float(values[indexes] @ EF[indexes] / denominator) if denominator > 0 else float("nan")


def coarse_totals(values: np.ndarray) -> np.ndarray:
    return np.array([sum(values[FUELS.index(fuel)] for fuel in THERMAL),
                     values[FUELS.index("nuclear")], values[FUELS.index("hydro")],
                     values[FUELS.index("wind")], values[FUELS.index("solar")]])


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    vals, wts = values[order], weights[order]
    return float(vals[min(np.searchsorted(np.cumsum(wts), q * wts.sum()), len(vals) - 1)])


def prepare_regions() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, dict], dict[str, str]]:
    paths = [Path(path).resolve() for path in default_native_paths()]
    input_hashes = {str(path): sha256(path) for path in paths}
    schemas, frames = {}, []
    seen: dict[tuple[str, int], set[pd.Timestamp]] = {}
    for snapshot, path in enumerate(paths):
        raw = pd.read_csv(path, low_memory=False)
        raw = raw.loc[raw["Balancing Authority"].isin(ALL_REGIONS)].copy()
        native = normalize_balance(raw)
        schemas[str(snapshot)] = {"fuel_columns": [c for c in native if c.startswith("generation_")]}
        native = native.loc[native.region.isin(ALL_REGIONS)
                            & native.valid_at.ge(START) & native.valid_at.lt(STOP)].copy()
        native["source_snapshot"] = snapshot
        keep = []
        for idx, row in native.iterrows():
            key = (str(row.region), int(row.valid_at.year))
            hours = seen.setdefault(key, set())
            if row.valid_at not in hours:
                hours.add(row.valid_at)
                keep.append(idx)
        if keep:
            frames.append(native.loc[keep])
        del raw, native
    combined = pd.concat(frames, ignore_index=True)
    hourly: dict[str, pd.DataFrame] = {}
    monthly_rows = []
    release_audits = {}
    for region in ALL_REGIONS:
        rows = combined.loc[combined.region.eq(region)].copy()
        rows = rows.sort_values("valid_at")
        rows.index = pd.DatetimeIndex(rows.valid_at)
        region_hourly, _ = _proxy_rows(rows, region, schemas)
        # Month-end +30d is after all rows of that month. With frozen earliest
        # snapshots and per-snapshot schemas, a single proxy pass is equivalent
        # to replaying every historical prefix for that month, while avoiding
        # repeated full-history scans for each of 54 monthly releases.
        month_rows, dropped = _monthly(region_hourly)
        hourly[region] = region_hourly
        monthly_rows.extend(month_rows)
        release_audits[region] = {"eligible_months": len(month_rows), "dropped_months": dropped}
    monthly = pd.DataFrame(monthly_rows)
    monthly["period_start"] = pd.to_datetime(monthly.period_start, utc=True)
    monthly["period_end"] = pd.to_datetime(monthly.period_end, utc=True)
    monthly["available_at"] = pd.to_datetime(monthly.available_at, utc=True)
    if monthly.duplicated(["region", "period_start"]).any():
        raise ValueError("Duplicate release-frozen monthly observations")
    return hourly, monthly, release_audits, input_hashes


def main() -> None:
    protocol = ROOT / "docs/V54_UNSEEN_LOCAL_EF_CONFIRMATION_PROTOCOL.md"
    protocol_hash = sha256(protocol)
    hourly, monthly, release_audits, input_hashes = prepare_regions()
    if sha256(protocol) != protocol_hash:
        raise RuntimeError("Frozen protocol changed during run")

    def row_for(region: str, period: pd.Timestamp, origin: pd.Timestamp) -> pd.Series | None:
        rows = monthly.loc[monthly.region.eq(region) & monthly.period_start.eq(period)
                           & monthly.period_end.le(origin) & monthly.available_at.le(origin)]
        return rows.iloc[-1] if len(rows) else None

    def coarse_for(region: str, origin: pd.Timestamp) -> np.ndarray | None:
        rows = monthly.loc[monthly.region.eq(region) & monthly.period_start.dt.month.eq(origin.month)
                           & monthly.period_start.lt(origin) & monthly.period_end.le(origin)
                           & monthly.available_at.le(origin)]
        if rows.empty:
            return None
        return coarse_totals(fuel_totals(rows.sort_values(["period_start", "available_at"]).iloc[-1]))

    def local_complete_year_ef(region: str, origin: pd.Timestamp) -> float | None:
        rows = monthly.loc[monthly.region.eq(region) & monthly.period_end.le(origin)
                           & monthly.available_at.le(origin)]
        for year in sorted(rows.period_start.dt.year.unique(), reverse=True):
            if year >= origin.year:
                continue
            annual = rows.loc[rows.period_start.dt.year.eq(year)]
            if annual.period_start.dt.month.nunique() == 12:
                value = thermal_ef(np.sum([fuel_totals(row) for _, row in annual.iterrows()], axis=0))
                if np.isfinite(value):
                    return value
        return None

    def donor_ef(donors: tuple[str, ...], origin: pd.Timestamp, year: int) -> float | None:
        period = pd.Timestamp(f"{year - 1}-{origin.month:02d}-01", tz="UTC")
        values = []
        for donor in donors:
            row = row_for(donor, period, origin)
            if row is not None:
                value = thermal_ef(fuel_totals(row))
                if np.isfinite(value):
                    values.append(value)
        return float(np.mean(values)) if values else None

    shape_cache: dict[tuple[str, int, int], np.ndarray] = {}

    def region_shape(region: str, month_no: int, year: int) -> np.ndarray:
        key = (region, month_no, year)
        if key in shape_cache:
            return shape_cache[key]
        frame = hourly[region]
        times = pd.to_datetime(frame.time, utc=True)
        selected = frame.loc[times.dt.year.eq(year) & times.dt.month.eq(month_no) & frame.valid]
        matrix = np.zeros((24, 5))
        if not selected.empty:
            work = selected.copy()
            work["hour"] = pd.to_datetime(work.time, utc=True).dt.hour
            cols = [f"generation_{fuel}" for fuel in FUELS]
            means = work.groupby("hour")[cols].mean()
            for hour, values in means.iterrows():
                matrix[int(hour), 0] = sum(float(values[f"generation_{fuel}"]) for fuel in THERMAL)
                for idx, fuel in enumerate(("nuclear", "hydro", "wind", "solar"), 1):
                    matrix[int(hour), idx] = float(values[f"generation_{fuel}"])
        totals = matrix.sum(axis=0)
        matrix = np.divide(matrix, totals[None, :], out=np.ones_like(matrix) / 24, where=totals[None, :] > 0)
        shape_cache[key] = matrix
        return matrix

    def predict(region: str, donors: tuple[str, ...], origin: pd.Timestamp, year: int,
                local_factor: bool) -> dict | None:
        coarse = coarse_for(region, origin)
        factor = local_complete_year_ef(region, origin) if local_factor else donor_ef(donors, origin, year)
        if coarse is None or factor is None:
            return None
        frame = hourly[region]
        times = pd.to_datetime(frame.time, utc=True)
        current = frame.loc[times.dt.year.eq(origin.year) & times.dt.month.eq(origin.month) & frame.valid].copy()
        if current.empty:
            return None
        actual = np.stack([current[f"generation_{fuel}"].to_numpy(float) for fuel in FUELS], axis=1)
        valid = np.isfinite(actual).all(axis=1) & (actual >= 0).all(axis=1) & (actual.sum(axis=1) > 0)
        current, actual = current.loc[valid], actual[valid]
        if len(current) < 480:
            return None
        profile = np.mean([region_shape(donor, origin.month, year - 1) for donor in donors], axis=0)
        profile /= np.maximum(profile.sum(axis=0, keepdims=True), 1e-12)
        times = pd.to_datetime(current.time, utc=True)
        predicted = np.stack([coarse[j] * profile[times.dt.hour.to_numpy(), j] / origin.days_in_month
                              for j in range(5)], axis=1)
        estimate = predicted[:, 0] * factor / np.maximum(predicted.sum(axis=1), 1e-12)
        truth = actual @ EF / actual.sum(axis=1)
        return {"truth": truth, "prediction": estimate, "residual": truth - estimate}

    def months(year: int, half: str) -> list[pd.Timestamp]:
        start, stop = (1, 7) if half == "H1" else (7, 13)
        return [pd.Timestamp(f"{year}-{month:02d}-01", tz="UTC") for month in range(start, stop)]

    cal = {}
    for half in ("H1", "H2"):
        for region in SOURCE_REGIONS:
            donors = tuple(candidate for candidate in SOURCE_REGIONS if candidate != region)
            for method, local in (("donor_same_month", False), ("local_complete_year", True)):
                cal[(half, region, method)] = [predict(region, donors, month, 2024, local)
                                               for month in months(2024, half)]

    scores = []
    for target in TARGET_REGIONS:
        donors = SOURCE_REGIONS
        for half in ("H1", "H2"):
            for method, local in (("donor_same_month", False), ("local_complete_year", True)):
                by_region = {}
                for source in SOURCE_REGIONS:
                    rows = [row for row in cal[(half, source, method)] if row is not None]
                    if rows:
                        by_region[source] = rows
                if len(by_region) < 9:
                    raise ValueError(f"Insufficient calibration source regions: {half}/{method}/{len(by_region)}")
                residuals = {source: np.concatenate([row["residual"] for row in rows])
                             for source, rows in by_region.items()}
                values, weights = [], []
                for residual in residuals.values():
                    absolute = np.abs(residual)
                    values.extend(absolute.tolist())
                    weights.extend(np.full(len(absolute), 1 / (len(residuals) * len(absolute))).tolist())
                pooled_radius = weighted_quantile(np.asarray(values), np.asarray(weights), .90)
                group_radius = float(np.max([np.quantile(np.abs(value), .90) for value in residuals.values()]))
                for month in months(2025, half):
                    result = predict(target, donors, month, 2025, local)
                    if result is None:
                        continue
                    for calibration, radius in (("pooled_q90", pooled_radius), ("max_region_q90", group_radius)):
                        truth, prediction = result["truth"], result["prediction"]
                        low, high = prediction - radius, prediction + radius
                        winkler = 2 * radius + 20 * np.maximum(low - truth, 0) + 20 * np.maximum(truth - high, 0)
                        scores.append({"target_region": target, "half": half, "target_month": str(month.date()),
                                       "ef_method": method, "calibration": calibration,
                                       "calibration_regions": len(residuals), "n_hours": len(truth),
                                       "point_mae": float(np.mean(np.abs(result["residual"]))),
                                       "point_bias": float(np.mean(result["residual"])),
                                       "coverage": float(np.mean((truth >= low) & (truth <= high))),
                                       "width": 2 * radius, "winkler": float(np.mean(winkler)),
                                       "radius": radius})

    score_frame = pd.DataFrame(scores)
    if score_frame.empty:
        raise ValueError("No eligible unseen-region scores")
    output = ROOT / "results/v54-unseen-local-ef-confirmation-r1"
    output.mkdir(parents=True, exist_ok=False)
    score_frame.to_csv(output / "region_month_scores.csv", index=False)
    region_scores = (score_frame.groupby(["half", "ef_method", "calibration", "target_region"])
                     .apply(lambda frame: pd.Series({
                         "coverage": np.average(frame.coverage, weights=frame.n_hours),
                         "width": np.average(frame.width, weights=frame.n_hours),
                         "winkler": np.average(frame.winkler, weights=frame.n_hours),
                         "point_mae": np.average(frame.point_mae, weights=frame.n_hours),
                         "bias": np.average(frame.point_bias, weights=frame.n_hours),
                         "hours": frame.n_hours.sum()}), include_groups=False).reset_index())
    region_scores.to_csv(output / "region_scores.csv", index=False)
    macro = (region_scores.groupby(["half", "ef_method", "calibration"])
             .agg(regions=("target_region", "nunique"), macro_coverage=("coverage", "mean"),
                  worst_region_coverage=("coverage", "min"), macro_width=("width", "mean"),
                  macro_winkler=("winkler", "mean"), macro_point_mae=("point_mae", "mean"),
                  macro_bias=("bias", "mean")).reset_index())
    macro.to_csv(output / "macro.csv", index=False)
    script = Path(__file__).resolve()
    summary = {"status": "SEALED_UNSEEN_CONFIRMATION", "target_regions": list(TARGET_REGIONS),
               "source_regions": list(SOURCE_REGIONS), "target_year": 2025,
               "never_scored_regions_read_before_freeze": False,
               "frozen_method": "target-local latest complete prior calendar-year thermal EF; five-category lagged monthly totals; fixed source-only hourly shapes; source-equal pooled q90",
               "label_definition": "EIA fixed-factor positive reported-generation proxy; not full-grid measured CIF",
               "script_sha256": sha256(script), "protocol_sha256": protocol_hash,
               "raw_input_sha256": input_hashes, "monthly_release_audit": release_audits,
               "macro": macro.to_dict("records")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
