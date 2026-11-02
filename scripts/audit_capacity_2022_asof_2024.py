#!/usr/bin/env python3
"""Extract EIA-860 2022 US-BA coal/gas nameplate and DUKES revised 2022 values."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/evidence/public-capacity-2022-asof-2024-r1"
EIA_ZIP = OUT / "raw/eia8602022.zip"
DUKES_XLSX = ROOT / "docs/evidence/public-capacity-2025-asof-nov2026-r1/raw/DUKES_5.7.xlsx"
BAS = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
COAL = frozenset(("ANT", "BIT", "LIG", "RC", "SGC", "SUB", "WC"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def workbook_rows(archive: ZipFile, member: str):
    workbook = load_workbook(BytesIO(archive.read(member)), read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    rows = sheet.iter_rows(values_only=True)
    next(rows)
    headers = next(rows)
    for row in rows:
        yield dict(zip(headers, row))
    workbook.close()


def main() -> None:
    capacities = {ba: {"coal_nameplate_mw": 0.0, "gas_nameplate_mw": 0.0} for ba in BAS}
    counts: defaultdict[str, int] = defaultdict(int)
    with ZipFile(EIA_ZIP) as archive:
        plants = {
            str(row["Plant Code"]).strip(): str(row["Balancing Authority Code"]).strip()
            for row in workbook_rows(archive, "2___Plant_Y2022.xlsx")
            if row.get("Plant Code") is not None
        }
        workbook = load_workbook(BytesIO(archive.read("3_1_Generator_Y2022.xlsx")),
                                read_only=True, data_only=True)
        sheet = workbook["Operable"]
        rows = sheet.iter_rows(values_only=True)
        next(rows)
        headers = next(rows)
        ix = {field: headers.index(field) for field in
              ("Plant Code", "Energy Source 1", "Nameplate Capacity (MW)")}
        for row in rows:
            ba = plants.get(str(row[ix["Plant Code"]]).strip())
            energy = row[ix["Energy Source 1"]]
            column = ("coal_nameplate_mw" if energy in COAL else
                      "gas_nameplate_mw" if energy == "NG" else None)
            if ba in capacities and column:
                capacities[ba][column] += float(row[ix["Nameplate Capacity (MW)"]] or 0.)
                counts[ba] += 1
        workbook.close()

    dukes = load_workbook(DUKES_XLSX, read_only=True, data_only=True)
    sheet = dukes["5.7"]
    revised = {}
    for row in sheet.iter_rows(min_row=7, values_only=True):
        if row[0] == "All generating companies" and row[1] in ("Coal fired", "Gas fired"):
            revised[row[1]] = float(row[28])  # 2022 column in the current revised workbook.
    dukes.close()

    with (OUT / "capacity_2022_by_ba.csv").open("w", newline="") as handle:
        fields = ("region", "capacity_year", "coal_nameplate_mw", "gas_nameplate_mw",
                  "coal_share", "included_operable_generator_rows")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ba, values in capacities.items():
            total = sum(values.values())
            writer.writerow({"region": ba, "capacity_year": 2022,
                             **{key: round(value, 3) for key, value in values.items()},
                             "coal_share": round(values["coal_nameplate_mw"] / total, 8) if total else "",
                             "included_operable_generator_rows": counts[ba]})

    summary = {
        "status": "STAT_YEAR_2022_ANNUAL_CAPACITY_HISTORICAL_DEVELOPMENT_BUDGET",
        "statistic_year": 2022,
        "eia860_release_date": "2023-09-19",
        "dukes_2023_release_date": "2023-07-27",
        "eia860_source_vintage": "archived original eia8602022.zip; member timestamps 2023-09-19",
        "dukes_source_vintage": "2026 revised DUKES 5.7 historical row for 2022; original 2023 input workbook unavailable",
        "eia860_by_ba": {
            ba: {**{key: round(value, 3) for key, value in values.items()},
                 "coal_share": round(values["coal_nameplate_mw"] / sum(values.values()), 8)
                 if sum(values.values()) else None,
                 "included_operable_generator_rows": counts[ba]}
            for ba, values in capacities.items()},
        "dukes_current_revised_2022_mw": {
            "coal": revised.get("Coal fired"), "gas": revised.get("Gas fired")},
        "limitations": [
            "EIA registered BA and nameplate semantics",
            "DUKES GB proxy is UK aggregate including Northern Ireland",
            "DUKES 2022 values are revised in the 2026 workbook, not authenticated as values published in 2023",
            "development score uses already viewed 2024 target labels"],
        "sha256": {
            "eia8602022.zip": sha(EIA_ZIP), "dukes_2026_table57.xlsx": sha(DUKES_XLSX),
            "capacity_2022_by_ba.csv": sha(OUT / "capacity_2022_by_ba.csv")},
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
