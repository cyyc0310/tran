"""Lossless staging of EIA-930 balance files; not an as-of model adapter."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FUEL_COLUMNS = {
    "coal": ("Coal",), "gas": ("Natural Gas",), "nuclear": ("Nuclear",),
    "oil": ("All Petroleum Products", "Petroleum"),
    "hydro": ("Hydropower and Pumped Storage",), "solar": ("Solar",), "wind": ("Wind",),
    "other": ("Other Fuel Sources",), "unknown": ("Unknown Fuel Sources",)}
MODERN_FUEL_COLUMNS = {**{fuel: aliases for fuel, aliases in FUEL_COLUMNS.items()
                         if fuel not in ("hydro", "solar", "wind")},
    "hydro": ("Hydropower Excluding Pumped Storage",), "pumped_storage": ("Pumped Storage",),
    "solar_without_battery": ("Solar without Integrated Battery Storage",),
    "solar_with_battery": ("Solar with Integrated Battery Storage",),
    "wind_without_battery": ("Wind without Integrated Battery Storage",),
    "wind_with_battery": ("Wind with Integrated Battery Storage",),
    "battery_storage": ("Battery Storage",), "other_storage": ("Other Energy Storage",),
    "unknown_storage": ("Unknown Energy Storage",), "geothermal": ("Geothermal",)}
VALUE_COLUMNS = {"Demand (MW)": "demand_mw", "Demand Forecast (MW)": "native_demand_forecast_mw",
                 "Net Generation (MW)": "net_generation_mw", "Total Interchange (MW)": "net_export_mw"}
REFERENCE = "https://www.eia.gov/electricity/930-content/about.html"


def normalize_balance(frame, regions=None):
    """Keep all native fuels and negative net output, and convert hour END to START.

    A demand forecast in the balance file does not identify an issuance vintage.
    No publication timestamps, absent fuel values, EFs or CIFs are invented.
    """
    prefix = "Net Generation (MW) from "
    names = {}
    modern = prefix + "Hydropower Excluding Pumped Storage" in frame
    mapping = MODERN_FUEL_COLUMNS if modern else FUEL_COLUMNS
    for fuel, aliases in mapping.items():
        columns = [prefix + alias for alias in aliases if prefix + alias in frame]
        if len(columns) != 1:
            raise ValueError(f"exactly one native EIA column required for {fuel}: {columns}")
        output_fuel = "hydro_and_pumped_storage" if fuel == "hydro" and not modern else fuel
        names[columns[0]] = f"generation_{output_fuel}_mw"
    additional = [name for name in frame if name.startswith(prefix) and name not in names
                  and not name.endswith((" (Adjusted)", " (Imputed)"))]
    if additional:
        raise ValueError(f"unmapped native fuel columns must be reviewed: {additional}")
    required = ["Balancing Authority", "UTC Time at End of Hour", *VALUE_COLUMNS, *names]
    if any(column not in frame for column in required):
        raise ValueError("EIA balance metadata/operating columns missing")
    frame = frame[required].copy()
    if regions is not None:
        frame = frame[frame["Balancing Authority"].isin(regions)].copy()
        if set(regions) - set(frame["Balancing Authority"]):
            raise ValueError("requested EIA balancing authorities absent")
    end = pd.to_datetime(frame["UTC Time at End of Hour"], utc=True, format="mixed", errors="raise")
    if end.isna().any() or not (end == end.dt.floor("h")).all():
        raise ValueError("finite hour-aligned EIA UTC ending times required")
    frame["valid_at"] = end - pd.Timedelta(hours=1)
    frame = frame.rename(columns={"Balancing Authority": "region", **VALUE_COLUMNS, **names})
    frame = frame.drop(columns="UTC Time at End of Hour")
    numeric = [*VALUE_COLUMNS.values(), *names.values()]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if np.isinf(frame[numeric].to_numpy(dtype=float)).any():
        raise ValueError("infinite EIA measurements")
    if frame.duplicated(["region", "valid_at"]).any():
        raise ValueError("duplicate EIA region/hour; vintage resolution is required")
    return frame[["region", "valid_at", *numeric]].sort_values(["region", "valid_at"]).reset_index(drop=True)


def stage_balance(paths, regions, output):
    paths, output = [Path(path) for path in paths], Path(output)
    if not paths or not regions or len(set(regions)) != len(regions) or (output.exists() and any(output.iterdir())):
        raise ValueError("input files, unique regions and empty output required")
    frames, schemas = [], {}
    for i, path in enumerate(paths):
        native = normalize_balance(pd.read_csv(path))
        schemas[str(i)] = {"input": str(path.resolve()), "fuel_columns": [
            column for column in native if column.startswith("generation_")]}
        native["source_snapshot"] = i
        frames.append(native)
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame.region.isin(regions)].sort_values(["region", "valid_at"]).reset_index(drop=True)
    if set(regions) - set(frame.region) or frame.duplicated(["region", "valid_at"]).any():
        raise ValueError("requested regions absent or duplicate EIA region/hour across input files")
    report = {"provider": "EIA-930", "reference": REFERENCE,
        "native_schemas": schemas, "interval": "UTC hour start converted from hour end",
        "input_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "eligible_for_issued_public": False,
        "missing_evidence": ["forecast issuance and original publication vintage",
            "label publication/revision vintage", "published capacity/mix/cost/EF releases",
            "compatible weather forecasts and full fuel/storage accounting"],
        "measurement_note": "signed net measurements; legacy hydro includes pumped storage; integrated-battery types kept separate",
        "regions": {}}
    output.mkdir(parents=True, exist_ok=True)
    fuels = [column for column in frame if column.startswith("generation_")]
    extra = [column for column in fuels if column not in [f"generation_{fuel}_mw" for fuel in
                                                        ("coal", "gas", "nuclear", "hydro", "solar", "wind")]]
    for region, rows in frame.groupby("region", sort=True):
        values = rows[fuels].to_numpy(dtype=float)
        complete = np.ones(len(rows), dtype=bool)
        for snapshot, group in rows.groupby("source_snapshot"):
            mask = rows.source_snapshot.to_numpy() == snapshot
            complete[mask] = np.isfinite(group[schemas[str(snapshot)]["fuel_columns"]].to_numpy()).all(1)
        discrepancy = rows.loc[complete, fuels].sum(axis=1) - rows.loc[complete, "net_generation_mw"]
        discrepancy = discrepancy[np.isfinite(discrepancy)]
        report["regions"][region] = {"rows": len(rows), "first_hour": str(rows.valid_at.min()),
            "last_hour": str(rows.valid_at.max()), "missing_by_column": rows.isna().sum().astype(int).to_dict(),
            "negative_generation_rows": int((values < 0).any(1).sum()),
            "reported_additional_component_nonzero_rows": int((rows[extra].abs() > 0).any(axis=1).sum()),
            "complete_fuel_rows": int(complete.sum()),
            "fuel_balance_max_abs_mw": float(discrepancy.abs().max()) if len(discrepancy) else None}
        rows.to_csv(output / f"{region}.csv", index=False)
    (output / "metadata.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
