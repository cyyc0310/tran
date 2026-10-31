"""Controlled-information EIA generation proxies, never issued-public CIF truth.

The target is the positive part of reported generation in eight named buckets.
Standalone storage and unknown fuel are excluded. Legacy hydro cannot separate
pumped storage; modern hydro can. These are explicit proxy boundary limitations,
not a reconstruction of complete production emissions.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FUELS = ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")
REGIONS = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
MIN_MONTH_COVERAGE = 0.8
RELEASE_LAG_DAYS = 30
_PREFIX = "generation_"
_SUFFIX = "_mw"


def _bucket(column):
    fuel = column.removeprefix(_PREFIX).removesuffix(_SUFFIX)
    if fuel in FUELS:
        return fuel
    if fuel == "hydro_and_pumped_storage":
        return "hydro"
    if fuel.startswith("wind_"):
        return "wind"
    if fuel.startswith("solar_"):
        return "solar"
    if fuel == "geothermal":
        return "other"
    return None


def _read_region(path):
    rows = pd.read_csv(path)
    required = {"region", "valid_at", "source_snapshot", "net_generation_mw"}
    if not required.issubset(rows):
        raise ValueError(f"missing staging columns in {path.name}")
    if rows.empty or not rows.region.eq(path.stem).all():
        raise ValueError(f"empty or inconsistent region in {path.name}")
    times = pd.to_datetime(rows.valid_at, utc=True, format="mixed", errors="raise")
    if times.isna().any() or not times.eq(times.dt.floor("h")).all() or times.duplicated().any():
        raise ValueError(f"unique UTC hour-start timestamps required in {path.name}")
    rows.index = pd.DatetimeIndex(times)
    rows = rows.sort_index()
    rows = rows.loc[(rows.index.year >= 2022) & (rows.index.year <= 2024)]
    if rows.empty:
        raise ValueError(f"no 2022-2024 observations in {path.name}")
    return rows


def _proxy_rows(rows, region, schemas):
    """Determine observed columns and quality using exactly the supplied rows."""
    generation = pd.DataFrame(0.0, index=rows.index, columns=FUELS)
    quality = pd.Series(False, index=rows.index)
    snapshot_audit = {}
    for snapshot, frame in rows.groupby("source_snapshot", sort=True):
        if not float(snapshot).is_integer():
            raise ValueError("integer source_snapshot required")
        key = str(int(snapshot))
        if key not in schemas:
            raise ValueError(f"source snapshot {key} lacks a native schema")
        columns = schemas[key]["fuel_columns"]
        if not columns or len(set(columns)) != len(columns) or not set(columns).issubset(frame):
            raise ValueError(f"missing or duplicate native fuel columns for snapshot {key}")
        native = frame[columns].apply(pd.to_numeric, errors="raise")
        if np.isinf(native.to_numpy()).any():
            raise ValueError("infinite native generation")
        active = native.columns[native.notna().any()].tolist()
        unobserved = native.columns[~native.notna().any()].tolist()
        net = pd.to_numeric(frame.net_generation_mw, errors="raise")
        signed_total = native.sum(axis=1, min_count=1)
        residual = (signed_total - net).abs()
        complete = native[active].notna().all(axis=1) if active else pd.Series(False, index=frame.index)
        quality.loc[frame.index] = (complete & np.isfinite(net) & (net > 0)
            & residual.le(np.maximum(10.0, 0.01 * net)))
        # All-unobserved categories are excluded from this proxy, not certified
        # absent from the grid. A missing value in an active category invalidates
        # its entire forecast day below, rather than becoming a zero label.
        for column in active:
            bucket = _bucket(column)
            if bucket is not None:
                generation.loc[frame.index, bucket] += native[column].clip(lower=0).fillna(0)
        excluded = [column for column in columns if _bucket(column) is None]
        snapshot_audit[key] = {
            "native_columns": columns,
            "unobserved_columns_excluded_not_certified_zero": unobserved,
            "active_columns": active,
            "excluded_storage_or_unknown_columns": excluded,
            "rows_with_sporadic_missing": int((~complete).sum()),
            "negative_native_energy_mwh_clipped": {
                column: float(-native[column].clip(upper=0).sum()) for column in active
                if _bucket(column) is not None and native[column].lt(0).any()},
            "excluded_signed_energy_mwh": {
                column: float(native[column].sum(min_count=1)) if native[column].notna().any() else None
                for column in excluded},
            "balance_failure_hours": int((~residual.le(np.maximum(10.0, 0.01 * net))).sum()),
            "legacy_hydro_includes_pumped_storage": "generation_hydro_and_pumped_storage_mw" in columns,
        }
    quality &= generation.sum(axis=1).gt(0)
    # Freeze the scored population to complete UTC days, with no interpolation.
    day = rows.index.floor("D")
    valid = quality & quality.groupby(day).transform("sum").eq(24)
    valid &= quality.groupby(day).transform("size").eq(24)
    generation.loc[~valid] = np.nan
    hourly = generation.rename(columns=lambda fuel: f"generation_{fuel}").copy()
    hourly.insert(0, "region", region)
    hourly.insert(0, "time", hourly.index)
    hourly["valid"] = valid
    hourly = hourly.reset_index(drop=True)
    audit = {
        "snapshots": snapshot_audit,
        "hours_in_2022_2024": len(hourly),
        "retained_hours": int(valid.sum()),
        "retained_full_utc_days_by_year": {
            str(year): int(valid.loc[valid.index.year == year].sum() // 24)
            for year in (2022, 2023, 2024)},
    }
    return hourly, audit


def _region_proxy(path, schemas):
    """Retrospective quality masks for evaluation labels, not input releases."""
    hourly, audit = _proxy_rows(_read_region(path), path.stem, schemas)
    audit.update({"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_path": str(path.resolve())})
    return hourly, audit


def _monthly(hourly):
    rows = hourly.copy()
    rows.index = pd.DatetimeIndex(rows.time)
    columns = [f"generation_{fuel}" for fuel in FUELS]
    observations = []
    dropped = 0
    for start, frame in rows.groupby(pd.Grouper(freq="MS")):
        end = start + pd.offsets.MonthBegin(1)
        expected = int((end - start) / pd.Timedelta(hours=1))
        selected = frame.loc[frame.valid, columns]
        observed = len(selected)
        energy = selected.sum(axis=0)
        total = float(energy.sum())
        if observed / expected < MIN_MONTH_COVERAGE or total <= 0:
            dropped += 1
            continue
        record = {"region": str(frame.region.iloc[0]), "period_start": start,
            "period_end": end, "available_at": end + pd.Timedelta(days=RELEASE_LAG_DAYS),
            "observed_hours": observed, "expected_hours": expected,
            "coverage": observed / expected, "total_mwh": total}
        record.update({f"fuelshare_{fuel}": float(energy[f"generation_{fuel}"] / total) for fuel in FUELS})
        observations.append(record)
    return observations, dropped


def _monthly_at_release(rows, region, schemas):
    """Freeze each monthly input using the raw prefix at its assumed release.

    A fuel that first reports after this release cannot retroactively invalidate
    the earlier month. Downloaded revisions still lack original vintages, so
    this is a controlled information budget rather than an issued-public claim.
    """
    first = rows.index.min().normalize().replace(day=1)
    last = rows.index.max().normalize().replace(day=1)
    observations, eligibility = [], []
    for start in pd.date_range(first, last, freq="MS"):
        end = start + pd.offsets.MonthBegin(1)
        release = end + pd.Timedelta(days=RELEASE_LAG_DAYS)
        prefix = rows.loc[rows.index < release]
        hourly, prefix_audit = _proxy_rows(prefix, region, schemas)
        month = hourly.loc[hourly.time.ge(start) & hourly.time.lt(end)]
        if month.empty:
            monthly = []
        else:
            monthly, _ = _monthly(month)
        observations.extend(monthly)
        snapshots = rows.loc[(rows.index >= start) & (rows.index < end), "source_snapshot"].unique()
        eligibility.append({
            "period_start": start.isoformat(), "available_at": release.isoformat(),
            "observed_hours": int(month.valid.sum()),
            "expected_hours": int((end - start) / pd.Timedelta(hours=1)),
            "eligible": bool(monthly),
            "active_columns_at_release": {
                str(int(snapshot)): prefix_audit["snapshots"][str(int(snapshot))]["active_columns"]
                for snapshot in snapshots},
        })
    return observations, eligibility


def prepare(source: Path, output: Path) -> dict:
    """Write seven-region retrospective proxies and artificial-lag observations.

    A fixture/subset may contain fewer registered regions; missing regions are
    recorded. Existing files are not overwritten. No raw demand or forecast
    columns are exported into the controlled-information model input table.
    """
    source, output = Path(source), Path(output)
    metadata_path = source / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    paths = [source / f"{region}.csv" for region in REGIONS if (source / f"{region}.csv").exists()]
    if not paths:
        raise ValueError("no supported EIA region files")
    names = [f"{path.stem}_hourly.csv" for path in paths] + ["monthly.csv", "audit.json"]
    if any((output / name).exists() for name in names):
        raise FileExistsError("proxy output files already exist")
    tables, observations, regions = {}, [], {}
    for path in paths:
        rows = _read_region(path)
        hourly, audit = _proxy_rows(rows, path.stem, metadata["native_schemas"])
        monthly, eligibility = _monthly_at_release(rows, path.stem, metadata["native_schemas"])
        audit.update({"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_path": str(path.resolve()), "released_months": len(monthly),
            "months_below_coverage_or_zero_energy": sum(not item["eligible"] for item in eligibility),
            "monthly_eligibility_at_assumed_release": eligibility})
        tables[path.stem] = hourly
        regions[path.stem] = audit
        observations.extend(monthly)
    report = {
        "protocol": "controlled_information_budget",
        "target": "positive_reported_generation_eight_bucket_proxy",
        "eligible_for_issued_public": False,
        "full_grid_emissions_truth": False,
        "fuels": list(FUELS),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "excluded_regions": {"PJM": "large native fuel/net-generation accounting discrepancy"},
        "missing_supported_regions": [region for region in REGIONS if region not in tables],
        "assumptions": [
            "Retrospective downloaded revisions; original publication vintages are unknown.",
            "Month-end plus 30 days is an artificial information budget, not a verified release date.",
            "Nonreported native categories are excluded, not certified zero generation.",
            "Negative native generation is clipped to zero only for the explicitly defined proxy.",
            "Standalone pumped, battery, other and unknown storage, and unknown fuel are excluded.",
            "Legacy hydro includes inseparable pumped storage; modern hydro excludes explicit pumped storage.",
            "Integrated-battery wind/solar are combined with their corresponding generation categories.",
            "Geothermal is combined into other; other requires emission-factor sensitivity analysis.",
            "No capacity release, demand forecast or future weather availability is inferred.",
            "Monthly totals describe retained observed hours, not imputed full-month totals.",
            "Monthly active columns and quality use only valid_at before each assumed release; released rows are frozen.",
            "Evaluation hourly masks remain retrospective full-snapshot masks and are never used to form monthly inputs.",
        ],
        "quality_rule": "active native columns complete; signed all-fuel/storage balance error <= max(10MW,1%); positive proxy; full UTC day",
        "monthly_minimum_coverage": MIN_MONTH_COVERAGE,
        "artificial_release_lag_days": RELEASE_LAG_DAYS,
        "monthly_quality_method": "prefix_at_assumed_release",
        "evaluation_hourly_quality_method": "retrospective_full_snapshot",
        "regions": regions,
    }
    output.mkdir(parents=True, exist_ok=True)
    for region, table in tables.items():
        table.to_csv(output / f"{region}_hourly.csv", index=False)
    monthly_columns = ["region", "period_start", "period_end", "available_at", "observed_hours",
        "expected_hours", "coverage", "total_mwh", *[f"fuelshare_{fuel}" for fuel in FUELS]]
    pd.DataFrame(observations, columns=monthly_columns).to_csv(output / "monthly.csv", index=False)
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def latest_observation(monthly, region, origin):
    """Return the newest completed, released month at a timezone-aware origin."""
    origin = pd.Timestamp(origin)
    if origin.tzinfo is None:
        raise ValueError("timezone-aware forecast origin required")
    rows = monthly.loc[monthly.region.eq(region)].copy()
    if rows.empty:
        return None
    for column in ("period_start", "period_end", "available_at"):
        rows[column] = pd.to_datetime(rows[column], utc=True, format="mixed", errors="raise")
    admissible = (rows.available_at.le(origin) & rows.period_end.le(origin)
        & rows.available_at.ge(rows.period_end) & rows.period_start.lt(rows.period_end)
        & rows.coverage.ge(MIN_MONTH_COVERAGE))
    rows = rows.loc[admissible].sort_values(["period_end", "available_at"])
    return None if rows.empty else rows.iloc[-1]
