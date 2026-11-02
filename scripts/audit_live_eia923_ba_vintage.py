#!/usr/bin/env python3
"""Audit an as-issued EIA-923 monthly release against current EIA-930 BA hours."""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transcif.data.eia930 import normalize_balance  # noqa: E402
from transcif.research.eia_proxy import FUELS, _proxy_rows  # noqa: E402

DEFAULT_BAS = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
FUEL_MAP = {"COL": "coal", "NG": "gas", "DFO": "oil", "RFO": "oil", "PC": "oil",
            "NUC": "nuclear", "WAT": "hydro", "HYC": "hydro", "HPS": "hydro",
            "WND": "wind", "SUN": "solar"}
PROTOCOL = ROOT / "docs/LIVE_EIA923_BA_VINTAGE_AUDIT_PROTOCOL_20261101.md"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_header(value) -> str:
    return str(value).replace("\n", " ").strip() if value is not None else ""


def read_sheet(workbook, title: str, header_row: int) -> pd.DataFrame:
    iterator = workbook[title].iter_rows(min_row=header_row, values_only=True)
    header = next(iterator)
    names, seen = [], {}
    for position, value in enumerate(header):
        name = clean_header(value)
        if name in seen:
            seen[name] += 1
            name = f"{name}__duplicate_{seen[name]}_at_{position}"
        else:
            seen[name] = 1
        names.append(name)
    return pd.DataFrame(iterator, columns=names)


def parse_monthly_generation(path: Path, year: int, months: range, bas: tuple[str, ...]):
    with ZipFile(path) as archive:
        members = [name for name in archive.namelist()
                   if name.startswith("EIA923_Schedules_2_3_4_5_M_") and name.endswith(".xlsx")]
        if len(members) != 1:
            raise ValueError(f"expected one monthly EIA-923 workbook, got {members}")
        workbook = load_workbook(BytesIO(archive.read(members[0])), read_only=True, data_only=True)
        try:
            page1 = read_sheet(workbook, "Page 1 Generation and Fuel Data", 6)
            plant_frame = read_sheet(workbook, "Page 6 Plant Frame", 4)
            if "2026 August" not in str(workbook["Page 1 Generation and Fuel Data"]["A2"].value):
                raise ValueError("Page 1 release title does not match the frozen 2026 August vintage")
        finally:
            workbook.close()

    required_p1 = {"Plant Id", "MER Fuel Type Code", *[f"Netgen {calendar.month_name[m]}" for m in months]}
    required_frame = {"YEAR", "MONTH", "Plant Id", "Reporting Frequency", "Balancing Authority Code"}
    if not required_p1.issubset(page1) or not required_frame.issubset(plant_frame):
        raise ValueError("required monthly generation or plant-frame fields missing")
    page1["Plant Id"] = pd.to_numeric(page1["Plant Id"], errors="coerce")
    page1 = page1.loc[page1["Plant Id"].notna()].copy()
    plant_frame["Plant Id"] = pd.to_numeric(plant_frame["Plant Id"], errors="coerce")
    plant_frame["YEAR"] = pd.to_numeric(plant_frame["YEAR"], errors="coerce")
    plant_frame["MONTH"] = pd.to_numeric(plant_frame["MONTH"], errors="coerce")
    plant_frame = plant_frame.loc[
        plant_frame.YEAR.eq(year) & plant_frame["Reporting Frequency"].eq("M")
    ].copy()
    if plant_frame.duplicated(["Plant Id", "YEAR", "MONTH"]).any():
        raise ValueError("duplicate plant-month BA rows in monthly plant frame")

    records = []
    page1["MER Fuel Type Code"] = page1["MER Fuel Type Code"].astype(str).str.strip()
    for month in months:
        column = f"Netgen {calendar.month_name[month]}"
        part = page1[["Plant Id", "MER Fuel Type Code", column]].rename(columns={column: "mwh"}).copy()
        part["mwh"] = pd.to_numeric(part["mwh"], errors="coerce")
        mapped = part.merge(
            plant_frame.loc[plant_frame.MONTH.eq(month), ["Plant Id", "Balancing Authority Code"]],
            on="Plant Id", how="inner", validate="many_to_one",
        )
        mapped = mapped.loc[mapped["Balancing Authority Code"].astype(str).str.strip().isin(bas)]
        mapped["fuel"] = mapped["MER Fuel Type Code"].map(FUEL_MAP).fillna("other")
        for ba in bas:
            group = mapped.loc[mapped["Balancing Authority Code"].eq(ba)]
            numeric = group.mwh.notna() & np.isfinite(group.mwh.fillna(0))
            sums = group.loc[numeric].groupby("fuel").mwh.sum()
            record = {"ba": ba, "month": f"{year}-{month:02d}",
                      "eia923_plant_fuel_rows": int(len(group)),
                      "eia923_numeric_rows": int(numeric.sum()),
                      "eia923_numeric_row_fraction": float(numeric.mean()) if len(group) else 0.0,
                      "eia923_complete": bool(len(group) > 0 and numeric.all())}
            record.update({f"eia923_{fuel}_mwh": float(sums.get(fuel, 0.0)) for fuel in FUELS})
            record["eia923_total_mwh"] = sum(record[f"eia923_{fuel}_mwh"] for fuel in FUELS)
            records.append(record)
    return pd.DataFrame(records), {"page1_rows": int(len(page1)),
                                   "page1_unique_plants": int(page1["Plant Id"].nunique()),
                                   "monthly_plant_frame_rows": int(len(plant_frame)),
                                   "unmapped_page1_plant_ids": sorted(
                                       int(value) for value in set(page1["Plant Id"])
                                       - set(plant_frame["Plant Id"]))}


def parse_hourly(paths: list[Path], year: int, months: range, bas: tuple[str, ...]) -> pd.DataFrame:
    native = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    native = native.loc[native["Balancing Authority"].isin(bas)].copy()
    normalized = normalize_balance(native, regions=bas)
    fuel_columns = [column for column in normalized if column.startswith("generation_")]
    records = []
    for ba, rows in normalized.groupby("region", sort=True):
        rows = rows.loc[rows.valid_at.dt.year.eq(year)].copy()
        rows["source_snapshot"] = 0
        rows.index = pd.DatetimeIndex(rows.valid_at)
        rows = rows.sort_index()
        hourly, _ = _proxy_rows(rows, ba, {"0": {"fuel_columns": fuel_columns}})
        hourly["month"] = pd.to_datetime(hourly.time, utc=True).dt.strftime("%Y-%m")
        for month in months:
            key = f"{year}-{month:02d}"
            group = hourly.loc[hourly.month.eq(key)]
            expected = calendar.monthrange(year, month)[1] * 24
            valid = int(group.valid.sum())
            record = {"ba": ba, "month": key, "eia930_native_hours": int(len(group)),
                      "eia930_valid_hours": valid, "eia930_expected_hours": expected,
                      "eia930_complete": bool(len(group) == expected and valid == expected)}
            if record["eia930_complete"]:
                record.update({f"eia930_{fuel}_mwh": float(group[f"generation_{fuel}"].sum())
                               for fuel in FUELS})
                record["eia930_total_mwh"] = sum(record[f"eia930_{fuel}_mwh"] for fuel in FUELS)
            else:
                record.update({f"eia930_{fuel}_mwh": np.nan for fuel in FUELS})
                record["eia930_total_mwh"] = np.nan
            records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eia923-zip", type=Path, required=True)
    parser.add_argument("--eia930", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    year, months, bas = 2026, range(1, 9), DEFAULT_BAS
    monthly, mapping_audit = parse_monthly_generation(args.eia923_zip, year, months, bas)
    hourly = parse_hourly(args.eia930, year, months, bas)
    detail = monthly.merge(hourly, on=["ba", "month"], validate="one_to_one")
    scored = detail.loc[detail.eia923_complete & detail.eia930_complete
                        & detail.eia923_total_mwh.gt(0) & detail.eia930_total_mwh.gt(0)].copy()
    for fuel in FUELS:
        scored[f"eia923_share_{fuel}"] = scored[f"eia923_{fuel}_mwh"] / scored.eia923_total_mwh
        scored[f"eia930_share_{fuel}"] = scored[f"eia930_{fuel}_mwh"] / scored.eia930_total_mwh
    scored["total_ratio_eia923_to_eia930"] = scored.eia923_total_mwh / scored.eia930_total_mwh
    scored["mix_share_l1"] = sum(
        (scored[f"eia923_share_{fuel}"] - scored[f"eia930_share_{fuel}"]).abs() for fuel in FUELS)
    detail = detail.merge(scored[["ba", "month", "total_ratio_eia923_to_eia930", "mix_share_l1"]],
                          on=["ba", "month"], how="left", validate="one_to_one")
    detail.to_csv(output / "ba_month_alignment.csv", index=False)
    by_ba = detail.groupby("ba").agg(
        months=("month", "count"), eia923_complete_months=("eia923_complete", "sum"),
        eia930_complete_months=("eia930_complete", "sum"), paired_complete_months=("mix_share_l1", "count")
    ).reset_index()
    by_ba.to_csv(output / "by_ba_coverage.csv", index=False)
    summary = {
        "release_vintage": "EIA-923 2026 August monthly release, announced 2026-10-23",
        "observation_months": [f"2026-{m:02d}" for m in months],
        "balancing_authorities": list(bas), "fuel_fields": list(FUELS),
        "mapping_audit": mapping_audit,
        "candidate_ba_months": int(len(detail)),
        "eia923_complete_ba_months": int(detail.eia923_complete.sum()),
        "eia930_complete_ba_months": int(detail.eia930_complete.sum()),
        "paired_complete_ba_months": int(len(scored)),
        "median_total_ratio": float(scored.total_ratio_eia923_to_eia930.median()) if len(scored) else None,
        "median_mix_share_l1": float(scored.mix_share_l1.median()) if len(scored) else None,
        "mean_mix_share_l1": float(scored.mix_share_l1.mean()) if len(scored) else None,
        "p90_mix_share_l1": float(scored.mix_share_l1.quantile(.9)) if len(scored) else None,
        "max_mix_share_l1": float(scored.mix_share_l1.max()) if len(scored) else None,
        "limitations": ["EIA-930 is a current revised snapshot, not matched to each EIA-923 release vintage",
                        "EIA generation proxy is not measured CIF", "no prediction model was evaluated"],
    }
    inputs = [args.eia923_zip.resolve(), *[path.resolve() for path in args.eia930]]
    code = [Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "src/transcif/data/eia930.py",
            Path(__file__).resolve().parents[1] / "src/transcif/research/eia_proxy.py", PROTOCOL]
    summary["input_sha256"] = {str(path): sha(path) for path in inputs}
    summary["code_sha256"] = {str(path): sha(path) for path in code}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
