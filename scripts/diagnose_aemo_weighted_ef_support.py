#!/usr/bin/env python
"""Measure approximate positive-SCADA support with complete AEMO EF mappings.

This is a registration/support diagnostic only. It does not integrate SCADA
readings into energy, construct emissions, or produce a CIF label.
"""

import argparse
import json
from pathlib import Path
import re

import pandas as pd

from transcif.data.aemo import sha256
from transcif.data.aemo_registry import registry_at


STATUSES = ("complete_ef", "incomplete_ef", "no_allocation", "ambiguous_allocation")


def _snapshot_index(staging):
    metadata_path = staging / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    records = {}
    for record in metadata["snapshots"]:
        match = re.search(r"_(\d{6})010000\.CSV$", Path(record["input"]).name)
        if match is None:
            raise ValueError("reviewed native monthly DVD filenames required")
        key = (record["table"][1], match.group(1))
        if key in records:
            raise ValueError("ambiguous duplicate AEMO table/month snapshots")
        records[key] = record
    return metadata, metadata_path, records


def _read_snapshot(staging, record):
    path = staging / record["output"]
    if path.parent.resolve() != staging.resolve() or sha256(path) != record["output_sha256"]:
        raise ValueError("staged AEMO snapshot hash/path mismatch")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _read_scada_sample_mass(path):
    """Sum positive instantaneous MW samples by DUID without energy conversion."""
    chunks = []
    for chunk in pd.read_csv(path, usecols=["duid", "scada_mw"], dtype={"duid": str}, chunksize=500_000):
        values = pd.to_numeric(chunk["scada_mw"], errors="coerce")
        chunk = chunk.assign(_positive_mw=values.clip(lower=0))
        chunks.append(chunk.groupby("duid", as_index=False)["_positive_mw"].sum())
    if not chunks:
        raise ValueError("empty SCADA snapshot")
    return pd.concat(chunks).groupby("duid", as_index=False)["_positive_mw"].sum()


def audit(staging, registration_month):
    staging = Path(staging).resolve()
    if not staging.is_dir():
        raise ValueError("AEMO staging directory required")
    metadata, metadata_path, snapshots = _snapshot_index(staging)
    if metadata.get("eligible_for_issued_public") is not False:
        raise ValueError("expected staging marked ineligible for issued-public use")

    details = _read_snapshot(staging, snapshots[("DUDETAILSUMMARY", registration_month)])
    allocations = _read_snapshot(staging, snapshots[("DUALLOC", registration_month)])
    used_hashes = {str(metadata_path): sha256(metadata_path)}
    registration_records = [snapshots[(table, registration_month)] for table in ("DUDETAILSUMMARY", "DUALLOC")]
    for record in registration_records:
        used_hashes[str((staging / record["output"]).resolve())] = record["output_sha256"]

    month_rows = []
    measurement_months = sorted(month for table, month in snapshots if table == "UNIT_SCADA")
    for month in measurement_months:
        scada_record = snapshots[("UNIT_SCADA", month)]
        genset_record = snapshots[("GENUNITS", month)]
        scada_path = staging / scada_record["output"]
        genset_path = staging / genset_record["output"]
        for record, path in ((scada_record, scada_path), (genset_record, genset_path)):
            if path.parent.resolve() != staging.resolve() or sha256(path) != record["output_sha256"]:
                raise ValueError("staged AEMO snapshot hash/path mismatch")
            used_hashes[str(path.resolve())] = record["output_sha256"]

        sample_mass = _read_scada_sample_mass(scada_path)
        gensets = pd.read_csv(genset_path, dtype=str, keep_default_na=False)
        registry_time = pd.Timestamp(f"{month[:4]}-{month[4:]}-15T12:00:00+10:00")
        registry = registry_at(sample_mass.duid.tolist(), registry_time, details, allocations, gensets)
        by_duid = {row["duid"]: row for row in registry["records"]}
        records = []
        for _, row in sample_mass.iterrows():
            unit = by_duid[row["duid"]]
            regions = unit["region_candidates"]
            region = regions[0] if len(regions) == 1 else "UNRESOLVED_REGION"
            if not unit["allocation_candidates"]:
                status = "no_allocation"
            elif unit["allocation_version_ambiguous"]:
                status = "ambiguous_allocation"
            elif unit["native_ef_min"] is None:
                status = "incomplete_ef"
            else:
                status = "complete_ef"
            records.append((region, status, float(row["_positive_mw"])))

        frame = pd.DataFrame(records, columns=["region", "status", "positive_mw_sample_sum"])
        group = frame.groupby(["region", "status"])["positive_mw_sample_sum"].sum()
        region_totals = frame.groupby("region")["positive_mw_sample_sum"].sum()
        for region, total in region_totals.items():
            entry = {"month": month, "region": region, "positive_mw_sample_sum": float(total)}
            for status in STATUSES:
                mass = float(group.get((region, status), 0.0))
                entry[f"{status}_share"] = mass / float(total) if total else None
                entry[f"{status}_positive_mw_sample_sum"] = mass
            entry["duid_count"] = int(frame.loc[frame.region == region, "status"].count())
            month_rows.append(entry)

    monthly = pd.DataFrame(month_rows)
    region_summary = []
    for region, frame in monthly.groupby("region", sort=True):
        record = {"region": region, "months": int(frame.month.nunique())}
        for status in STATUSES:
            record[f"mean_monthly_{status}_share"] = float(frame[f"{status}_share"].mean())
            record[f"max_monthly_{status}_share"] = float(frame[f"{status}_share"].max())
            total_mass = float(frame["positive_mw_sample_sum"].sum())
            record[f"annual_sample_mass_{status}_share"] = (
                float(frame[f"{status}_positive_mw_sample_sum"].sum()) / total_mass if total_mass else None
            )
        region_summary.append(record)

    return {
        "purpose": "retrospective AEMO registration support diagnostic; not a label builder",
        "staging": str(staging),
        "registration_snapshot_month": registration_month,
        "registration_asof_note": "one explicit December snapshot is used across 2023; it is not an as-issued or historical monthly registration panel",
        "measurement_months": measurement_months,
        "weight_definition": "positive instantaneous SCADA MW sample sum by DUID; not MWh, energy, emissions, or a CIF",
        "monthly_region_rows": monthly.sort_values(["month", "region"]).to_dict(orient="records"),
        "region_summary": region_summary,
        "input_sha256": used_hashes,
        "limitations": [
            "SCADA readings are instantaneous MW; no interval shift or integration is applied",
            "registration and allocation versions are retrospective and do not establish original public availability",
            "a candidate mapping is called EF-complete only when every mapped genset has a native factor; ambiguous mapping versions remain unresolved",
            "native factors are CO2e and the CDEII boundary includes fugitive emissions; no conversion to direct CO2 is made",
            "this diagnostic does not establish official hourly labels or consumption-side CIF",
        ],
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--registration-month", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing output")
    result = _json_safe(audit(args.staging, args.registration_month))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for row in result["region_summary"]:
        print(row["region"], row["mean_monthly_complete_ef_share"], row["max_monthly_incomplete_ef_share"])


if __name__ == "__main__":
    main()
