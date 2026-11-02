#!/usr/bin/env python3
"""Compare one month's state fuel tables across two archived EIA EPM issues."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import statistics
import zipfile
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


TABLES = {
    "total_generation": "Table_1_03_A.xlsx",
    "coal": "Table_1_04_A.xlsx",
    "petroleum_liquids": "Table_1_05_A.xlsx",
    "petroleum_coke": "Table_1_06_A.xlsx",
    "natural_gas": "Table_1_07_A.xlsx",
    "other_fossil_gas": "Table_1_08_A.xlsx",
    "nuclear": "Table_1_09_A.xlsx",
    "hydro_conventional": "Table_1_10_A.xlsx",
    "renewables_ex_hydro": "Table_1_11_A.xlsx",
    "pumped_storage": "Table_1_12_A.xlsx",
    "other": "Table_1_13_A.xlsx",
}
NON_STATE_ROWS = {
    "New England",
    "Middle Atlantic",
    "East North Central",
    "West North Central",
    "South Atlantic",
    "East South Central",
    "West South Central",
    "Mountain",
    "Pacific Contiguous",
    "Pacific Noncontiguous",
    "U.S. Total",
}
CORE_SHARE_FIELDS = (
    "coal",
    "natural_gas",
    "nuclear",
    "hydro_conventional",
    "renewables_ex_hydro",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sheet(archive: zipfile.ZipFile, member: str) -> list[tuple[Any, ...]]:
    workbook = load_workbook(io.BytesIO(archive.read(member)), read_only=True, data_only=True)
    try:
        return list(workbook.active.iter_rows(values_only=True))
    finally:
        workbook.close()


def as_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def extract_release(path: Path, observation_year: int, target_month: str) -> dict[str, dict[str, float | None]]:
    month_col_name = f"{target_month} {observation_year}"
    output: dict[str, dict[str, float | None]] = {}
    with zipfile.ZipFile(path) as archive:
        for field, member in TABLES.items():
            rows = read_sheet(archive, member)
            header_idx = next(
                (
                    i
                    for i, row in enumerate(rows)
                    if row and isinstance(row[0], str) and row[0].strip() == "Census Division\nand State"
                ),
                None,
            )
            if header_idx is None:
                raise ValueError(f"Could not find state header in {member}")
            headers = rows[header_idx]
            if month_col_name not in headers:
                raise ValueError(f"{month_col_name} missing from {member}: {headers[:4]}")
            col = headers.index(month_col_name)
            for row in rows[header_idx + 1 :]:
                label = row[0] if row else None
                if not isinstance(label, str):
                    continue
                state = " ".join(label.split())
                if not state or state in NON_STATE_ROWS or state.startswith("Displayed values"):
                    continue
                output.setdefault(state, {})[field] = as_number(row[col] if col < len(row) else None)
    return output


def summarize(
    earlier: dict[str, dict[str, float | None]],
    later: dict[str, dict[str, float | None]],
) -> dict[str, Any]:
    states = sorted(set(earlier) & set(later))
    cells: list[tuple[str, str, float, float]] = []
    by_field: dict[str, dict[str, Any]] = {}
    for field in TABLES:
        field_cells = []
        for state in states:
            a = earlier[state].get(field)
            b = later[state].get(field)
            if a is not None and b is not None:
                field_cells.append((state, field, a, b))
                cells.append((state, field, a, b))
        deltas = [b - a for _, _, a, b in field_cells]
        by_field[field] = {
            "numeric_pairs": len(field_cells),
            "changed_pairs": sum(delta != 0 for delta in deltas),
            "median_absolute_delta_gwh": statistics.median(map(abs, deltas)) if deltas else None,
            "p90_absolute_delta_gwh": _quantile(list(map(abs, deltas)), 0.9),
            "max_absolute_delta_gwh": max(map(abs, deltas)) if deltas else None,
        }

    total_a = {s: earlier[s].get("total_generation") for s in states}
    total_b = {s: later[s].get("total_generation") for s in states}
    share_l1: list[float] = []
    for state in states:
        denom_a, denom_b = total_a[state], total_b[state]
        if denom_a is None or denom_b is None or denom_a <= 0 or denom_b <= 0:
            continue
        if any(earlier[state].get(f) is None or later[state].get(f) is None for f in CORE_SHARE_FIELDS):
            continue
        share_l1.append(
            sum(
                abs(earlier[state][f] / denom_a - later[state][f] / denom_b)  # type: ignore[operator]
                for f in CORE_SHARE_FIELDS
            )
        )
    total_deltas = [
        total_b[s] - total_a[s]
        for s in states
        if total_a[s] is not None and total_b[s] is not None
    ]
    return {
        "state_rows": len(states),
        "paired_numeric_state_fuel_cells": len(cells),
        "changed_paired_cells": sum(a != b for _, _, a, b in cells),
        "changed_paired_cell_fraction": sum(a != b for _, _, a, b in cells) / len(cells) if cells else None,
        "generation_total_numeric_pairs": len(total_deltas),
        "generation_total_median_absolute_revision_gwh": statistics.median(map(abs, total_deltas)) if total_deltas else None,
        "generation_total_p90_absolute_revision_gwh": _quantile(list(map(abs, total_deltas)), 0.9),
        "generation_total_max_absolute_revision_gwh": max(map(abs, total_deltas)) if total_deltas else None,
        "core_share_states": len(share_l1),
        "core_share_fields": list(CORE_SHARE_FIELDS),
        "core_share_l1_median_percentage_points": 100 * statistics.median(share_l1) if share_l1 else None,
        "core_share_l1_mean_percentage_points": 100 * statistics.mean(share_l1) if share_l1 else None,
        "core_share_l1_p90_percentage_points": 100 * _quantile(share_l1, 0.9) if share_l1 else None,
        "core_share_l1_max_percentage_points": 100 * max(share_l1) if share_l1 else None,
        "by_field": by_field,
    }


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--earlier", type=Path, required=True, help="Earlier EIA monthly issue ZIP")
    parser.add_argument("--earlier-year", type=int, required=True, help="Issue year of --earlier")
    parser.add_argument("--later", type=Path, required=True, help="Later EIA monthly issue ZIP")
    parser.add_argument("--later-year", type=int, required=True, help="Issue year of --later")
    parser.add_argument("--observation-year", type=int, required=True, help="Year of the month being compared")
    parser.add_argument("--month", default="April", help="Reported month present in both state tables")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    a = extract_release(args.earlier, args.observation_year, args.month)
    b = extract_release(args.later, args.observation_year, args.month)
    states = sorted(set(a) & set(b))
    fields = list(TABLES)
    rows = []
    for state in states:
        for field in fields:
            value_a = a[state].get(field)
            value_b = b[state].get(field)
            rows.append(
                {
                    "state": state,
                    "measure": field,
                    "earlier_release_year": args.earlier_year,
                    "earlier_value_gwh": value_a,
                    "later_release_year": args.later_year,
                    "later_value_gwh": value_b,
                    "revision_gwh": value_b - value_a if value_a is not None and value_b is not None else None,
                    "status": "paired_numeric" if value_a is not None and value_b is not None else "missing_or_suppressed",
                }
            )

    import csv

    csv_path = args.output_dir / "state_month_fuel_vintage_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(a, b)
    summary.update(
        {
            "earlier_issue_year": args.earlier_year,
            "later_issue_year": args.later_year,
            "observation_year": args.observation_year,
            "target_observation_month": args.month,
            "earlier_archive_sha256": sha256(args.earlier),
            "later_archive_sha256": sha256(args.later),
            "csv_sha256": sha256(csv_path),
            "interpretation": "Preliminary EIA state fuel-table revision diagnostic; not a CIF forecast experiment.",
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
