"""Quality-only census of EIA-930 balancing authorities for source-pool expansion.

The report contains timestamps, completeness and schema metadata only. It does
not emit CIF values, fuel shares, errors, predictions or region rankings.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
YEARS = (2022, 2023, 2024, 2025, 2026)
STOP = pd.Timestamp("2026-07-01T00:00:00Z")
MIN_COVERAGE = 0.80


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from transcif.data.eia930 import normalize_balance
    from transcif.research.eia_proxy import _proxy_rows

    from prepare_external_v6 import default_native_paths

    paths = [Path(p).resolve() for p in default_native_paths()]
    records: dict[str, dict[int, dict]] = {}
    schemas = {}
    source_manifest = []
    duplicates = 0
    seen_hours: dict[tuple[str, int], set[pd.Timestamp]] = {}

    for snapshot, path in enumerate(paths):
        raw = pd.read_csv(path, low_memory=False)
        native = normalize_balance(raw)
        schemas[str(snapshot)] = {
            "fuel_columns": [c for c in native if c.startswith("generation_")],
            "path": str(path),
        }
        source_manifest.append({"path": str(path), "sha256": sha256(path),
                                "rows": len(native), "snapshot": snapshot})
        native["source_snapshot"] = snapshot
        native = native.loc[native.valid_at.lt(STOP)]

        for (region, year), group in native.groupby(
            ["region", native.valid_at.dt.year], sort=True
        ):
            start = pd.Timestamp(f"{year}-01-01T00:00:00Z")
            stop = min(pd.Timestamp(f"{year + 1}-01-01T00:00:00Z"), STOP)
            group = group.loc[group.valid_at.ge(start) & group.valid_at.lt(stop)].copy()
            if group.empty:
                continue
            duplicate_count = int(group.valid_at.duplicated().sum())
            key = (region, int(year))
            seen = seen_hours.setdefault(key, set())
            cross_snapshot = group.valid_at.isin(seen)
            duplicate_count += int(cross_snapshot.sum())
            duplicates += duplicate_count
            if duplicate_count:
                group = group.loc[~group.valid_at.duplicated() & ~cross_snapshot].copy()
            if group.empty:
                continue
            seen.update(group.valid_at.tolist())
            group.index = pd.DatetimeIndex(group.valid_at)
            hourly, audit = _proxy_rows(group, region, schemas)
            entry = records.setdefault(region, {}).setdefault(year, {
                "native_rows": 0, "unique_hours": 0, "valid_hours": 0,
                "active_fuel_columns": set(), "snapshots": [],
            })
            entry["native_rows"] += len(group)
            entry["unique_hours"] += group.valid_at.nunique()
            entry["valid_hours"] += int(hourly.valid.sum())
            entry["active_fuel_columns"].update(
                audit["snapshots"][str(snapshot)]["active_columns"]
            )
            entry["snapshots"].append(snapshot)

        del native, raw

    summary = {}
    for region, periods in sorted(records.items()):
        summary[region] = {}
        for year in YEARS:
            expected = int((min(pd.Timestamp(f"{year + 1}-01-01T00:00:00Z"), STOP)
                            - pd.Timestamp(f"{year}-01-01T00:00:00Z"))
                           / pd.Timedelta(hours=1))
            entry = periods.get(year, {"native_rows": 0, "unique_hours": 0,
                                       "valid_hours": 0, "active_fuel_columns": set(),
                                       "snapshots": []})
            summary[region][str(year)] = {
                "expected_hours": expected,
                "native_rows": entry["native_rows"],
                "unique_hours": entry["unique_hours"],
                "native_coverage": entry["unique_hours"] / expected if expected else 0,
                "complete_proxy_hours": entry["valid_hours"],
                "complete_proxy_coverage": entry["valid_hours"] / expected if expected else 0,
                "active_fuel_columns_union": sorted(entry["active_fuel_columns"]),
                "snapshots": sorted(entry["snapshots"]),
            }

    qualified = []
    for region, periods in summary.items():
        if all(periods[str(year)]["native_coverage"] >= MIN_COVERAGE
               and periods[str(year)]["complete_proxy_coverage"] >= MIN_COVERAGE
               for year in YEARS):
            qualified.append(region)

    output = ROOT / "results/eia-ba-quality-census-v52-r1"
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "QUALITY_ONLY_NO_CIF_SCORES_OR_REGION_RANKING",
        "source_pool_candidate_count": len(summary),
        "quality_qualified_all_periods_count": len(qualified),
        "quality_qualified_regions_alphabetical": qualified,
        "selection_rule": "native timestamp and complete proxy-label coverage >=80% in each 2022-2025 calendar year and 2026H1; no CIF summaries or performance-based selection",
        "periods": summary,
        "overlapping_region_hour_records_across_files": duplicates,
        "sources": source_manifest,
        "schema_by_snapshot": schemas,
        "limitations": [
            "EIA-930 fixed-EF positive reported-generation proxy is not full-grid measured CIF.",
            "EIA balancing authorities are not necessarily independent exchangeable grid groups; many share interconnections or markets.",
            "Coverage eligibility is not a claim of fuel-boundary comparability or historical issue-time availability.",
            "No held-out region CIF metric, score, shape summary, or method comparison was computed.",
        ],
    }
    (output / "quality_census.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"candidate_count": len(summary), "qualified_count": len(qualified),
                      "qualified_regions": qualified,
                      "overlap_records": duplicates}, indent=2))


if __name__ == "__main__":
    main()
