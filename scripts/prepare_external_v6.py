"""Prepare V6 regions using predeclared native-record quality, never CIF scores.

The first six qualifying candidates in CANDIDATES are frozen after QA. Hourly
references are exported solely for later sealed scoring, including 2026 H1.
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


CANDIDATES = ("AZPS", "DUK", "LGEE", "PACW", "PACE", "PSCO", "SRP", "TVA")
PRIOR_REGIONS = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS",
                 "SWPP", "AVA", "BANC", "JEA", "SC", "NWMT",
                 "SCL", "WACM", "SEC", "SCEG")
YEARS = (2022, 2023, 2024, 2025, 2026)
MIN_PERIOD_COVERAGE = 0.8
SELECTION_SIZE = 6
# The candidate order and quality rule are fixed before native QA.
# Frozen after native-record QA; no model scores or CIF summaries were computed.
REGIONS = ("AZPS", "DUK", "LGEE", "PACE", "PSCO", "SRP")
START = pd.Timestamp("2022-01-01T00:00Z")
STOP = pd.Timestamp("2026-07-01T00:00Z")
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
            for year in YEARS[:3] for half in ("jan_jun", "jul_dec")] + [
        ROOT / "data/native-eia-2025" / f"EIA930_BALANCE_2025_{half}.csv"
        for half in ("Jan_Jun", "Jul_Dec")] + [
        ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jan_Jun.csv"]


def _quality_only(audit):
    excluded = {"negative_native_energy_mwh_clipped", "excluded_signed_energy_mwh"}
    return {key: {name: value for name, value in record.items() if name not in excluded}
            for key, record in audit["snapshots"].items()}


def _load_native(native_paths):
    paths = [Path(path).resolve() for path in native_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Unique nonempty native paths required")
    sources, schemas, frames = [], {}, []
    for snapshot, path in enumerate(paths):
        raw_hash = sha(path)
        raw = pd.read_csv(path)
        raw = raw.loc[raw["Balancing Authority"].isin(CANDIDATES)].copy()
        normalized = normalize_balance(raw)
        normalized["source_snapshot"] = snapshot
        schemas[str(snapshot)] = {
            "input": str(path), "fuel_columns": [
                column for column in normalized if column.startswith("generation_")]}
        frames.append(normalized)
        sources.append({"path": str(path), "sha256": raw_hash,
                        "source_snapshot": snapshot,
                        "candidate_native_rows": len(normalized)})
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.loc[combined.valid_at.ge(START) & combined.valid_at.lt(STOP)].copy()
    if combined.duplicated(["region", "valid_at"]).any():
        raise ValueError("Duplicate native region/hour across snapshots")
    return combined, schemas, sources


def _candidate_quality(combined, schemas):
    """Use completeness and accounting QA only; do not calculate CIF or shape."""
    if set(CANDIDATES) & set(PRIOR_REGIONS):
        raise ValueError("V6 candidates overlap prior evaluation regions")
    audits = {}
    for region in CANDIDATES:
        periods = {}
        for year in YEARS:
            start = pd.Timestamp(f"{year}-01-01T00:00Z")
            stop = min(pd.Timestamp(f"{year + 1}-01-01T00:00Z"), STOP)
            expected = int((stop - start) / pd.Timedelta(hours=1))
            rows = combined.loc[combined.region.eq(region)
                                & combined.valid_at.ge(start) & combined.valid_at.lt(stop)].copy()
            rows.index = pd.DatetimeIndex(rows.valid_at)
            rows = rows.sort_index()
            if rows.empty:
                retained, quality = 0, {}
            else:
                hourly, quality_audit = _proxy_rows(rows, region, schemas)
                retained, quality = int(hourly.valid.sum()), _quality_only(quality_audit)
            periods[str(year)] = {
                "expected_hours": expected, "native_hours": len(rows),
                "retained_hours": retained, "valid_full_utc_days": retained // 24,
                "coverage": retained / expected,
                "native_record_coverage": len(rows) / expected,
                "retained_coverage_ge_80pct": retained / expected >= MIN_PERIOD_COVERAGE,
                "first_hour": rows.index.min().isoformat() if len(rows) else None,
                "last_hour": rows.index.max().isoformat() if len(rows) else None,
                "source_snapshots": sorted(int(value) for value in rows.source_snapshot.unique()),
                "quality_snapshots": quality}
        # Historical coarse months retain V4/V5's independent 80% monthly gate.
        # A historical full-year 80% retained-label gate did not exist there.
        qualifies = (all(item["native_record_coverage"] >= MIN_PERIOD_COVERAGE
                         for item in periods.values())
                     and periods["2026"]["retained_coverage_ge_80pct"])
        audits[region] = {"qualifies": qualifies,
                          "periods": periods}
    qualifying = [region for region in CANDIDATES if audits[region]["qualifies"]]
    return {"candidate_order": list(CANDIDATES), "prior_regions": list(PRIOR_REGIONS),
            "prior_region_overlap": [], "minimum_period_coverage": MIN_PERIOD_COVERAGE,
            "selection_rule": "native record coverage >=80% each year/H1; retained complete UTC hours >=80% in confirmation 2026H1; historical monthly quality inherited",
            "selection_size": SELECTION_SIZE, "qualifying_regions": qualifying,
            "selected_regions": qualifying[:SELECTION_SIZE], "candidates": audits}


def candidate_quality(native_paths):
    combined, schemas, sources = _load_native(native_paths)
    report = _candidate_quality(combined, schemas)
    report["native_sources"] = sources
    report["cif_scores_inspected"] = False
    report["cif_summaries_computed"] = False
    return report


def prepare_external(native_paths, output, protocol):
    """Export frozen regions with year-isolated quality and release prefixes."""
    output, protocol = Path(output), Path(protocol)
    if output.exists():
        raise FileExistsError("Use a fresh output directory for external V6 data")
    protocol_before = sha(protocol)
    source_paths = [Path(__file__).resolve(), ROOT / "src/transcif/data/eia930.py",
                    ROOT / "src/transcif/research/eia_proxy.py"]
    source_hashes = {str(path): sha(path) for path in source_paths}
    combined, schemas, sources = _load_native(native_paths)
    selection = _candidate_quality(combined, schemas)
    if len(REGIONS) != SELECTION_SIZE or tuple(selection["selected_regions"]) != REGIONS:
        raise ValueError("Frozen V6 regions must equal first six quality-eligible candidates")
    tables, observations, audits = {}, [], {}
    for region in REGIONS:
        hourly_parts, region_audit = [], {}
        for year in YEARS:
            rows = combined.loc[combined.region.eq(region)
                                & combined.valid_at.dt.year.eq(year)].copy()
            rows.index = pd.DatetimeIndex(rows.valid_at)
            rows = rows.sort_index()
            hourly, _ = _proxy_rows(rows, region, schemas)
            monthly, eligibility = _monthly_at_release(rows, region, schemas)
            hourly_parts.append(hourly)
            observations.extend(monthly)
            region_audit[str(year)] = {
                **selection["candidates"][region]["periods"][str(year)],
                "eligible_months": len(monthly), "monthly_prefix_eligibility": eligibility}
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
        "native_sources": sources, "native_schemas": schemas, "candidate_selection": selection,
        "external_regions": list(REGIONS), "years": list(YEARS),
        "final_external_year": 2026, "confirmation_start": "2026-01-01T00:00:00+00:00",
        "confirmation_stop_exclusive": STOP.isoformat(),
        "time_convention": "UTC hour start = native UTC hour end minus one hour",
        "regions": audits,
        "derived_hashes": {path.name: sha(path) for path in sorted(output.glob("*.csv"))},
        "cif_scores_inspected": False, "cif_summaries_computed": False,
        "target_hourly_used_for_fit_or_calibration": False,
        "issued_public": False, "full_grid_emissions_truth": False,
        "notes": [
            "First six passing native coverage in each period and retained-hour coverage in 2026H1, in declared order.",
            "Historical years have no new annual retained-label gate; V4/V5 monthly 80% gate is inherited.",
            "Selection uses native completeness/accounting and timestamps only, never CIF or outcomes.",
            "Hourly references are solely for later evaluation after all prediction seals.",
            "Only 2026H1 hourly labels may enter V6 final scoring; older hourly labels are not model inputs.",
            "Monthly inputs may use 2022-2026H1 only after available_at <= origin.",
            "Quality is retrospective within each UTC year using per-file native schemas.",
            "Monthly active fields use only same-year valid_at before assumed release.",
            "June 2026 coarse inputs are exported with July 31 release, so unusable during 2026H1.",
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
    parser.add_argument("--output", type=Path, default=ROOT / "results/external-v6-data")
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/EXTERNAL_V6_DATA_PROTOCOL.md")
    parser.add_argument("--qa-only", action="store_true")
    args = parser.parse_args()
    report = (candidate_quality(args.native) if args.qa_only else
              prepare_external(args.native, args.output, args.protocol))
    if args.qa_only:
        compact = {region: {year: {key: item[key] for key in
                   ("native_hours", "valid_full_utc_days", "coverage", "native_record_coverage")}
                   for year, item in values["periods"].items()}
                   for region, values in report["candidates"].items()}
        print(json.dumps({"qualifying_regions": report["qualifying_regions"],
                          "selected_regions": report["selected_regions"], "qa": compact}, indent=2))
    else:
        print(json.dumps({region: {year: {key: values[key] for key in
                         ("native_hours", "valid_full_utc_days", "eligible_months")}
                         for year, values in years.items()}
                         for region, years in report["regions"].items()}, indent=2))


if __name__ == "__main__":
    main()
