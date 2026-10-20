#!/usr/bin/env python
"""Audit monthly SCADA unit inventories against explicit retrospective registry snapshots."""

import argparse
import json
from pathlib import Path
import re
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transcif.data.aemo import sha256
from transcif.data.aemo_registry import registry_at


def audit(staging, registration_month):
    staging = Path(staging)
    metadata = staging / "metadata.json"
    report = json.loads(metadata.read_text())
    snapshots, used = {}, {str(metadata.resolve()): sha256(metadata)}
    for record in report["snapshots"]:
        month = re.search(r"_(\d{6})010000\.CSV$", Path(record["input"]).name)
        if month is None:
            raise ValueError("reviewed native monthly DVD filenames required")
        key = (record["table"][1], month[1])
        if key in snapshots:
            raise ValueError("ambiguous duplicate table/month snapshots")
        snapshots[key] = record

    def read_record(table, month, data=True):
        record = snapshots[(table, month)]
        file = staging / record["output"]
        if file.parent.resolve() != staging.resolve() or sha256(file) != record["output_sha256"]:
            raise ValueError("staged AEMO snapshot hash/path mismatch")
        used[str(file.resolve())] = record["output_sha256"]
        return pd.read_csv(file, dtype=str, keep_default_na=False) if data else record

    details = read_record("DUDETAILSUMMARY", registration_month)
    allocations = read_record("DUALLOC", registration_month)
    months = sorted(month for table, month in snapshots if table == "UNIT_SCADA")
    cases = []
    for month in months:
        scada = read_record("UNIT_SCADA", month, data=False)
        at = f"{month[:4]}-{month[4:]}-15T12:00:00+10:00"
        case = registry_at(scada["duids"], at, details, allocations, read_record("GENUNITS", month))
        case["measurement_month"] = month
        cases.append(case)
    return {"registration_snapshot_month": registration_month, "cases": cases, "input_sha256": used,
        "eligible_for_issued_public": False, "retrospective_only": True,
        "interpretation": "each month's DUID inventory audited at its midpoint; counts are neither generation/energy-weighted nor a full within-month mapping history",
        "ef_note": "native CO2E_EMISSIONS_FACTOR values retained without conversion to direct CO2 or CIF labels"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--registration-month", required=True, help="explicit DUALLOC/DUDETAILSUMMARY snapshot YYYYMM")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("existing registry audit is never overwritten")
    result = audit(args.staging, args.registration_month)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for case in result["cases"]:
        print(case["measurement_month"], case["counts"])


if __name__ == "__main__":
    main()
