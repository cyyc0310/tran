#!/usr/bin/env python3
"""Audit EIA-923 monthly BA mixes against EIA-930 hourly mix aggregates."""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from statistics import mean, median
from zipfile import ZipFile

import pandas as pd
from openpyxl import load_workbook


DEFAULT_BAS = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
FUEL_MAP = {
    "COL": "coal",
    "NG": "gas",
    "DFO": "oil",
    "RFO": "oil",
    "PC": "oil",
    "NUC": "nuclear",
    "WAT": "hydro",
    "HYC": "hydro",
    "HPS": "hydro",
    "WND": "wind",
    "SUN": "solar",
}
MIX_FIELDS = ("coal", "gas", "oil", "other", "nuclear", "hydro", "wind", "solar")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_eia923(path: Path, bas: tuple[str, ...], year: int) -> pd.DataFrame:
    with ZipFile(path) as archive:
        member = next(
            name
            for name in archive.namelist()
            if name.startswith("EIA923_Schedules_2_3_4_5_M_") and name.endswith(".xlsx")
        )
        workbook = load_workbook(BytesIO(archive.read(member)), read_only=True, data_only=True)
        try:
            sheet = workbook["Page 1 Generation and Fuel Data"]
            rows = sheet.iter_rows(min_row=6, values_only=True)
            header = next(rows)
            columns = {
                str(value).replace("\n", " ").strip(): index
                for index, value in enumerate(header)
                if value is not None
            }
            required = ("Balancing Authority Code", "MER Fuel Type Code")
            if any(field not in columns for field in required):
                raise ValueError("EIA-923 plant/BA/fuel header is missing")
            month_names = tuple(calendar.month_name[month] for month in range(1, 13))
            month_cols = {month: columns[f"Netgen {month_name}"] for month, month_name in enumerate(month_names, 1)}
            sums: dict[tuple[str, str, str], float] = defaultdict(float)
            row_counts: dict[tuple[str, str], int] = defaultdict(int)
            for row in rows:
                if not row or row[0] is None:
                    continue
                ba = str(row[columns["Balancing Authority Code"]] or "").strip()
                if ba not in bas:
                    continue
                source = str(row[columns["MER Fuel Type Code"]] or "").strip()
                category = FUEL_MAP.get(source, "other")
                for month, col in month_cols.items():
                    try:
                        value = float(row[col])
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(value):
                        continue
                    key = (ba, f"{year}-{month:02d}")
                    sums[(ba, f"{year}-{month:02d}", category)] += value
                    row_counts[key] += 1
            records = []
            for ba in bas:
                for month in range(1, 13):
                    key = (ba, f"{year}-{month:02d}")
                    record = {"ba": ba, "month": key[1], "eia923_rows": row_counts.get(key, 0)}
                    record.update({f"eia923_{field}_mwh": sums.get((ba, key[1], field), 0.0) for field in MIX_FIELDS})
                    record["eia923_total_mwh"] = sum(record[f"eia923_{field}_mwh"] for field in MIX_FIELDS)
                    records.append(record)
            return pd.DataFrame(records)
        finally:
            workbook.close()


def parse_eia930(feature_dir: Path, bas: tuple[str, ...], year: int) -> pd.DataFrame:
    records = []
    for ba in bas:
        path = feature_dir / f"{ba}_generation_2022_2024.csv.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, parse_dates=["time_utc"])
        frame = frame.loc[frame.time_utc.dt.year.eq(year)].copy()
        frame["month"] = frame.time_utc.dt.strftime("%Y-%m")
        expected_fields = [f"generation_{field}" for field in MIX_FIELDS]
        if any(field not in frame for field in expected_fields):
            raise ValueError(f"EIA-930 mix fields missing for {ba}")
        for month, group in frame.groupby("month", sort=True):
            expected_hours = calendar.monthrange(year, int(month[-2:]))[1] * 24
            records.append(
                {
                    "ba": ba,
                    "month": month,
                    "eia930_unique_hours": int(group.time_utc.nunique()),
                    "eia930_complete_fuel_hours": int(group[expected_fields].notna().all(axis=1).sum()),
                    **{
                        f"eia930_{field}_mwh": float(group[f"generation_{field}"].sum(min_count=expected_hours))
                        if group[f"generation_{field}"].notna().sum() == expected_hours
                        else None
                        for field in MIX_FIELDS
                    },
                    "expected_hours": expected_hours,
                }
            )
    return pd.DataFrame(records)


def summarize(frame: pd.DataFrame) -> dict:
    available = frame.dropna(subset=[f"eia930_{field}_mwh" for field in MIX_FIELDS]).copy()
    available = available.loc[
        available.eia930_unique_hours.eq(available.expected_hours)
        & available.eia930_complete_fuel_hours.eq(available.expected_hours)
    ].copy()
    metrics = []
    for row in available.to_dict("records"):
        a = [float(row[f"eia923_{field}_mwh"]) for field in MIX_FIELDS]
        b = [float(row[f"eia930_{field}_mwh"]) for field in MIX_FIELDS]
        total_a, total_b = sum(a), sum(b)
        if total_a <= 0 or total_b <= 0:
            continue
        share_diffs = [abs(x / total_a - y / total_b) for x, y in zip(a, b)]
        metrics.append(
            {
                "ba": row["ba"],
                "month": row["month"],
                "eia923_mwh": total_a,
                "eia930_mwh": total_b,
                "eia923_to_eia930_total_ratio": total_a / total_b,
                "mix_share_l1": sum(share_diffs),
                "max_single_source_share_gap": max(share_diffs),
            }
        )
    detail = pd.DataFrame(metrics)
    by_ba = []
    for ba, group in detail.groupby("ba", sort=True):
        by_ba.append(
            {
                "ba": ba,
                "paired_complete_months": int(len(group)),
                "median_total_ratio": float(group.eia923_to_eia930_total_ratio.median()),
                "median_mix_share_l1": float(group.mix_share_l1.median()),
            }
        )
    return {
        "available_eia923_ba_months": int(len(frame)),
        "paired_complete_eia930_ba_months": int(len(detail)),
        "paired_ba_count": int(detail.ba.nunique()) if len(detail) else 0,
        "paired_month_count": int(detail.month.nunique()) if len(detail) else 0,
        "median_eia923_to_eia930_total_ratio": float(detail.eia923_to_eia930_total_ratio.median()),
        "median_mix_share_l1": float(detail.mix_share_l1.median()),
        "mean_mix_share_l1": float(detail.mix_share_l1.mean()),
        "p90_mix_share_l1": float(detail.mix_share_l1.quantile(0.9)),
        "max_mix_share_l1": float(detail.mix_share_l1.max()),
        "mix_fields": list(MIX_FIELDS),
        "fuel_code_map": FUEL_MAP,
        "by_ba": by_ba,
        "scope_note": "2024 final EIA-923 monthly plant reports aggregated by their reported BA code; comparison only, not a vintage-certified forecast input.",
    }, detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eia923-zip", type=Path, required=True)
    parser.add_argument("--eia930-feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--bas", nargs="+", default=list(DEFAULT_BAS))
    args = parser.parse_args()
    bas = tuple(args.bas)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    e923 = parse_eia923(args.eia923_zip, bas, args.year)
    e930 = parse_eia930(args.eia930_feature_dir, bas, args.year)
    merged = e923.merge(e930, on=["ba", "month"], how="left", validate="one_to_one")
    summary, detail = summarize(merged)
    merged.to_csv(args.output_dir / "ba_month_fuel_comparison_all.csv", index=False)
    detail.to_csv(args.output_dir / "ba_month_fuel_comparison_complete.csv", index=False)
    summary.update(
        {
            "year": args.year,
            "balancing_authorities": list(bas),
            "eia923_archive_sha256": file_hash(args.eia923_zip),
            "eia930_input_sha256": {
                ba: file_hash(args.eia930_feature_dir / f"{ba}_generation_2022_2024.csv.gz") for ba in bas
            },
            "complete_comparison_csv_sha256": file_hash(args.output_dir / "ba_month_fuel_comparison_complete.csv"),
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
