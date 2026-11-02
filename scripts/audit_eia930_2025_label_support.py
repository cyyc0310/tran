#!/usr/bin/env python3
"""Audit whether 2025 EIA-930 hourly fuel labels are supported by EIA-923 final totals.

This is a label-support diagnostic. EIA-923 2025 final was released in 2026 and is
never treated as a prediction-time feature. Missing EIA-930 categories are not
zero-filled unless the corresponding annual EIA-923 category total is exactly zero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_eia923_ba_mix_alignment import DEFAULT_BAS, MIX_FIELDS, parse_eia923, file_hash

COMPONENTS = {
    "coal": ("Coal",),
    "gas": ("Natural Gas",),
    "oil": ("All Petroleum Products",),
    "nuclear": ("Nuclear",),
    "hydro": ("Hydropower Excluding Pumped Storage",),
    "wind": ("Wind without Integrated Battery Storage", "Wind with Integrated Battery Storage"),
    "solar": ("Solar without Integrated Battery Storage", "Solar with Integrated Battery Storage"),
    "other": ("Other Fuel Sources",),
}


def audit(hourly_files: list[Path], eia923_zip: Path, bas: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = parse_eia923(eia923_zip, bas, 2025)
    totals = monthly.groupby("ba", as_index=True)[[f"eia923_{fuel}_mwh" for fuel in MIX_FIELDS]].sum()
    hourly = pd.concat([pd.read_csv(path, low_memory=False) for path in hourly_files], ignore_index=True)
    hourly = hourly.loc[hourly["Balancing Authority"].isin(bas)].copy()
    hourly["UTC Time at End of Hour"] = pd.to_datetime(hourly["UTC Time at End of Hour"], utc=True, format="mixed", errors="raise")
    hourly = hourly.sort_values(["Balancing Authority", "UTC Time at End of Hour"])

    rows = []
    summary = []
    for ba in bas:
        group = hourly.loc[hourly["Balancing Authority"].eq(ba)]
        support_failures = []
        supported_categories = 0
        for fuel, names in COMPONENTS.items():
            annual_mwh = float(totals.loc[ba, f"eia923_{fuel}_mwh"])
            columns = [f"Net Generation (MW) from {name} (Adjusted)" for name in names]
            values = group[columns] if all(column in group for column in columns) else pd.DataFrame(index=group.index)
            columns_present = len(columns) > 0 and all(column in group for column in columns)
            complete = bool(columns_present and values.notna().all(axis=None))
            status = "hourly_reported" if complete else ("structural_zero_supported" if annual_mwh == 0 else "unidentified_missing_hourly_component")
            if status != "unidentified_missing_hourly_component":
                supported_categories += 1
            else:
                support_failures.append(fuel)
            rows.append({
                "ba": ba,
                "fuel": fuel,
                "eia923_2025_final_mwh": annual_mwh,
                "eia923_final_released_after_target_year": True,
                "hourly_components": "|".join(names),
                "hourly_adjusted_components_present": columns_present,
                "hourly_adjusted_component_complete": complete,
                "missing_component_hours": int(values.isna().sum().sum()) if columns_present else len(group),
                "status": status,
            })
        unique_hours = int(group["UTC Time at End of Hour"].nunique())
        summary.append({
            "ba": ba,
            "hour_rows": int(len(group)),
            "unique_utc_hours": unique_hours,
            "hourly_categories_supported_of_8": supported_categories,
            "complete_8_category_label_support": supported_categories == len(COMPONENTS) and unique_hours == 8760,
            "unsupported_nonzero_categories": "|".join(support_failures),
        })
    return pd.DataFrame(rows), pd.DataFrame(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eia930", nargs="+", type=Path, required=True)
    parser.add_argument("--eia923", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bas", nargs="+", default=list(DEFAULT_BAS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail, summary = audit(args.eia930, args.eia923, tuple(args.bas))
    detail.to_csv(args.output_dir / "fuel_support_by_ba.csv", index=False)
    summary.to_csv(args.output_dir / "ba_support_summary.csv", index=False)
    metadata = {
        "purpose": "post hoc 2025 target-label support audit only; no labels imputed and no future information used as a predictor",
        "hourly_source": "EIA-930 native Balance files; (Adjusted) fuel generation columns",
        "monthly_support_source": "EIA-923 2025 Final, used only to distinguish structurally zero annual categories from missing nonzero support",
        "eia923_sha256": file_hash(args.eia923),
        "eia930_sha256": {str(path): file_hash(path) for path in args.eia930},
        "eia930_target_rows": int(summary.hour_rows.sum()),
        "balancing_authorities": list(args.bas),
        "complete_ba_count": int(summary.complete_8_category_label_support.sum()),
        "ba_count": int(len(summary)),
        "result": "NO_COMMON_COMPLETE_2025_HOURLY_FUEL_LABEL_COHORT" if not summary.complete_8_category_label_support.any() else "SUPPORTED_BA_SUBSET_AVAILABLE",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
