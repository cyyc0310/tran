#!/usr/bin/env python
"""Inventory legacy CSV evidence without assigning missing times, units or provenance."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def inspect_csv(path):
    columns = list(pd.read_csv(path, nrows=0).columns)
    time = next((name for name in ("hour", "valid_at", "time", "datetime", "timestamp") if name in columns), None)
    cif = next((name for name in ("cif_real_gco2_per_kwh", "cif_g_per_kwh") if name in columns), None)
    selected = [name for name in (time, cif, "total_energy_so", "cif_source") if name is not None and name in columns]
    frame = pd.read_csv(path, usecols=selected) if selected else None
    issued = {"issued_at", "available_at", "valid_at", "member"}.issubset(columns)
    record = {"columns": columns, "rows": len(frame) if frame is not None else None,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "kind": "aggregate_cif" if cif else ("issued_forecast_schema" if issued else "other"),
        "issued_run_fields_present": issued, "findings": []}
    if time is not None and len(frame):
        raw = frame[time].astype(str)
        parsed = pd.to_datetime(raw, errors="coerce", utc=True, format="mixed")
        record.update(first_timestamp=raw.iloc[0], last_timestamp=raw.iloc[-1],
            invalid_timestamps=int(parsed.isna().sum()), duplicate_timestamps=int(raw.duplicated().sum()),
            timezone_explicit=bool(raw.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).all()))
        if not record["timezone_explicit"]:
            record["findings"].append("timestamp timezone unverified; no timezone has been assigned")
    if cif is not None:
        values = pd.to_numeric(frame[cif], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
        record.update(invalid_cif=int((~valid).sum()), negative_cif=int((values[valid] < 0).sum()),
                      cif_min=float(values[valid].min()) if valid.any() else None,
                      cif_max=float(values[valid].max()) if valid.any() else None)
        if "cif_source" not in columns:
            record["findings"].append("CIF observation/reconstruction/forecast provenance absent")
        else:
            record["label_sources"] = frame.cif_source.fillna("missing").astype(str).value_counts().astype(int).to_dict()
        if "total_energy_so" in columns:
            energy = pd.to_numeric(frame.total_energy_so, errors="coerce").to_numpy(dtype=float)
            valid_energy = energy[np.isfinite(energy)]
            around_100 = float((np.abs(valid_energy - 100) <= .5).mean()) if len(valid_energy) else None
            record["energy_near_100_fraction"] = around_100
            if around_100 is not None and around_100 >= .95:
                record["findings"].append("energy near 100 throughout: possible percentage units, not verified MW")
        missing = ["generation_" + fuel + "_mw" for fuel in
                   ("coal", "gas", "nuclear", "hydro", "wind", "solar")
                   if "generation_" + fuel + "_mw" not in columns]
        record["missing_six_fuel_supervision"] = missing
    return record


def audit(directory):
    directory = Path(directory)
    files = {}
    for path in sorted(directory.rglob("*.csv")):
        try:
            files[str(path.relative_to(directory))] = inspect_csv(path)
        except (ValueError, OSError, pd.errors.ParserError, UnicodeError) as error:
            files[str(path.relative_to(directory))] = {"error": str(error)}
    return {"directory": str(directory.resolve()), "files": files,
        "summary": {"csv_files": len(files), "aggregate_cif_files": sum(
            r.get("kind") == "aggregate_cif" for r in files.values()),
            "issued_run_schema_files": sum(r.get("issued_run_fields_present", False) for r in files.values())},
        "limitations": ["schema/quality inventory, not certification of historical publication or units",
                        "source files unchanged; no target inputs or labels have been manufactured"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.directory.is_dir() or args.output.exists():
        parser.error("existing data directory and fresh output required")
    result = audit(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
