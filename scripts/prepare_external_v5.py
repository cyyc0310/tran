"""Prepare the frozen V5 external regions using native data quality only.

Hourly references are exported for later sealed evaluation, never model fitting
or calibration. No emission factors, CIF summaries, or prediction scores enter
this module. Each UTC year has its own retrospective quality and release prefix.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcif.data.eia930 import normalize_balance
from transcif.research.eia_proxy import FUELS, _monthly_at_release, _proxy_rows


REGIONS = ("SCL", "WACM", "SEC", "SCEG")
YEARS = (2022, 2023, 2024, 2025)
MONTHLY_COLUMNS = ["region", "period_start", "period_end", "available_at",
                   "observed_hours", "expected_hours", "coverage", "total_mwh",
                   *[f"fuelshare_{fuel}" for fuel in FUELS]]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_native_paths():
    historical = ROOT.parent / "tran-stable/data_2023/raw_eia930"
    return [historical / f"gen_{half}_{year}.csv"
            for year in YEARS[:-1] for half in ("jan_jun", "jul_dec")] + [
        ROOT / "data/native-eia-2025" / f"EIA930_BALANCE_2025_{half}.csv"
        for half in ("Jan_Jun", "Jul_Dec")]


def _quality_only(audit):
    excluded = {"negative_native_energy_mwh_clipped", "excluded_signed_energy_mwh"}
    return {key: {name: value for name, value in record.items() if name not in excluded}
            for key, record in audit["snapshots"].items()}


def prepare_external(native_paths, output, protocol):
    """Write fixed four-region, 2022–2025 references and as-of coarse inputs.

    Input files retain distinct source IDs even when native schemas match.
    Year boundaries are clipped before both quality and monthly processing:
    January of the next year cannot activate a prior December native field.
    """
    output, protocol = Path(output), Path(protocol)
    paths = [Path(path).resolve() for path in native_paths]
    if output.exists():
        raise FileExistsError("Use a fresh output directory for external V5 data")
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Unique nonempty native paths required")
    protocol_before = sha(protocol)
    source_paths = [Path(__file__).resolve(), ROOT / "src/transcif/data/eia930.py",
                    ROOT / "src/transcif/research/eia_proxy.py"]
    source_hashes = {str(path): sha(path) for path in source_paths}
    sources, schemas, frames = [], {}, []
    for snapshot, path in enumerate(paths):
        raw_hash = sha(path)
        raw = pd.read_csv(path)
        raw = raw.loc[raw["Balancing Authority"].isin(REGIONS)].copy()
        normalized = normalize_balance(raw, REGIONS)
        normalized["source_snapshot"] = snapshot
        schemas[str(snapshot)] = {
            "input": str(path), "fuel_columns": [
                column for column in normalized if column.startswith("generation_")]}
        frames.append(normalized)
        sources.append({"path": str(path), "sha256": raw_hash,
                        "source_snapshot": snapshot,
                        "selected_native_rows": len(normalized)})
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.loc[combined.valid_at.dt.year.isin(YEARS)].copy()
    if combined.duplicated(["region", "valid_at"]).any():
        raise ValueError("Duplicate native region/hour across snapshots")
    tables, observations, audits = {}, [], {}
    for region in REGIONS:
        hourly_parts, region_audit = [], {}
        for year in YEARS:
            rows = combined.loc[combined.region.eq(region)
                                & combined.valid_at.dt.year.eq(year)].copy()
            if rows.empty:
                raise ValueError(f"No {year} native observations for {region}")
            rows.index = pd.DatetimeIndex(rows.valid_at)
            rows = rows.sort_index()
            hourly, quality = _proxy_rows(rows, region, schemas)
            monthly, eligibility = _monthly_at_release(rows, region, schemas)
            hourly_parts.append(hourly)
            observations.extend(monthly)
            region_audit[str(year)] = {
                "native_hours": len(rows), "valid_full_utc_days": int(hourly.valid.sum() // 24),
                "eligible_months": len(monthly),
                "first_hour": rows.index.min().isoformat(),
                "last_hour": rows.index.max().isoformat(),
                "source_snapshots": sorted(int(value) for value in rows.source_snapshot.unique()),
                "quality_snapshots": _quality_only(quality),
                "monthly_prefix_eligibility": eligibility}
        tables[region] = pd.concat(hourly_parts, ignore_index=True)
        audits[region] = region_audit
    monthly = pd.DataFrame(observations, columns=MONTHLY_COLUMNS)
    if monthly.duplicated(["region", "period_start"]).any():
        raise ValueError("Duplicate external monthly observations")
    if sha(protocol) != protocol_before:
        raise RuntimeError("Frozen protocol changed during external preparation")
    if any(sha(path) != value for path, value in source_hashes.items()):
        raise RuntimeError("Preparation source changed during execution")
    if any(sha(item["path"]) != item["sha256"] for item in sources):
        raise RuntimeError("Native source changed during execution")
    output.mkdir(parents=True)
    for region, hourly in tables.items():
        hourly.to_csv(output / f"{region}_hourly.csv", index=False)
    monthly.sort_values(["region", "period_start"]).to_csv(output / "monthly.csv", index=False)
    report = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "controlled_information_budget; month-end plus30d",
        "protocol_path": str(protocol.resolve()), "protocol_sha256_before": protocol_before,
        "protocol_sha256_after": sha(protocol), "preparation_source_hashes": source_hashes,
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "numpy": np.__version__,
                        "pandas": pd.__version__},
        "native_sources": sources, "native_schemas": schemas,
        "external_regions": list(REGIONS), "years": list(YEARS),
        "final_external_year": 2025, "regions": audits,
        "derived_hashes": {path.name: sha(path) for path in sorted(output.glob("*.csv"))},
        "cif_scores_inspected": False, "cif_summaries_computed": False,
        "target_hourly_used_for_fit_or_calibration": False,
        "issued_public": False, "full_grid_emissions_truth": False,
        "notes": [
            "Regions frozen by record availability and quality, not CIF or forecast outcomes.",
            "Hourly references are solely for later evaluation after all prediction seals.",
            "Quality is retrospective within each UTC year, using native per-file schemas.",
            "Monthly active fields use only same-year valid_at before assumed release.",
            "No next-year native hours or active fields enter prior-year monthly releases.",
            "Missing boundary hours are not filled; incomplete UTC days are excluded.",
            "Monthly inputs require 80% retained full-day hourly coverage.",
            "Downloaded revisions have no original publication vintages; release lag is artificial.",
            "Standalone storage/unknown excluded; legacy hydro includes pumped storage."]}
    if report["protocol_sha256_after"] != protocol_before:
        raise RuntimeError("Frozen protocol changed before audit publication")
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", nargs="+", type=Path, default=default_native_paths())
    parser.add_argument("--output", type=Path, default=ROOT / "results/external-v5-data")
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/RISK_TRANSFER_V5_PROTOCOL.md")
    args = parser.parse_args()
    report = prepare_external(args.native, args.output, args.protocol)
    print(json.dumps({region: {year: {key: values[key] for key in
                     ("native_hours", "valid_full_utc_days", "eligible_months")}
                     for year, values in years.items()}
                     for region, years in report["regions"].items()}, indent=2))


if __name__ == "__main__":
    main()
