"""Audit the published 2019-2020 China provincial daily power-emissions workbook.

Exploratory data-lineage analysis only; emissions are not CIF labels.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
RECORD_ID = "4730175"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
FILE_KEY = "Dataset_Daily CO2_31Provincial_MonPow_MonInd_YearVeh_2019_2020_week_holiday.xlsx"
OUT = ROOT / "results/china-daily-power-emissions-audit-r1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str) -> tuple[bytes, dict]:
    request = urllib.request.Request(url, headers={"User-Agent": "TransCIF research audit"})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
        metadata = {"url": url, "status": response.status,
                    "last_modified": response.headers.get("Last-Modified"),
                    "content_type": response.headers.get("Content-Type"),
                    "bytes": len(body)}
    return body, metadata


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    api_bytes, api_http = fetch(API_URL)
    api_metadata = json.loads(api_bytes)
    record_file = next(file for file in api_metadata["files"] if file["key"] == FILE_KEY)
    body, file_http = fetch(record_file["links"]["self"])
    workbook_path = OUT / "china_provincial_daily_emissions_2019_2020.xlsx"
    workbook_path.write_bytes(body)
    (OUT / "zenodo_record.json").write_bytes(api_bytes)
    md5 = hashlib.md5(body).hexdigest()
    expected_md5 = record_file["checksum"].removeprefix("md5:")
    if md5 != expected_md5:
        raise ValueError(f"Zenodo MD5 mismatch: expected {expected_md5}, got {md5}")

    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    all_rows = []
    summary_by_year = []
    province_codes: list[str] | None = None
    for year in (2019, 2020):
        sheet_name = f"Power{year}"
        if sheet_name not in workbook.sheetnames:
            raise ValueError(f"Missing expected sheet {sheet_name}")
        sheet = workbook[sheet_name]
        rows = [row for row in sheet.iter_rows(min_row=2, max_col=36, values_only=True)
                if isinstance(row[0], (pd.Timestamp,)) or hasattr(row[0], "year")]
        if not rows:
            raise ValueError(f"No date rows in {sheet_name}")
        dates = pd.DatetimeIndex(pd.to_datetime([row[0] for row in rows]))
        national = np.asarray([float(row[1]) for row in rows], dtype=float)
        province_sum = np.asarray([float(row[2]) for row in rows], dtype=float)
        province_values = np.asarray([[float(value) for value in row[3:34]] for row in rows], dtype=float)
        codes = [str(sheet.cell(1, col).value) for col in range(4, 35)]
        if len(codes) != 31 or not np.isfinite(province_values).all():
            raise ValueError("Expected 31 finite province-level daily series")
        if province_codes is None:
            province_codes = codes
        elif codes != province_codes:
            raise ValueError("Province order differs across years")
        expected_days = 366 if year % 4 == 0 else 365
        if len(rows) != expected_days or dates.year.nunique() != 1 or int(dates.year[0]) != year:
            raise ValueError(f"Unexpected date coverage in {sheet_name}: {len(rows)} rows")
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError(f"Duplicate or unsorted dates in {sheet_name}")

        national_sum_abs_error = np.abs(province_values.sum(axis=1) - province_sum)
        local_share_of_national = province_values / national[:, None]
        for month in range(1, 13):
            indexes = np.flatnonzero(dates.month == month)
            local_month_profile = province_values[indexes] / province_values[indexes].sum(axis=0, keepdims=True)
            national_month_profile = national[indexes] / national[indexes].sum()
            ratio_cv = local_share_of_national[indexes].std(axis=0) / local_share_of_national[indexes].mean(axis=0)
            correlations = np.asarray([
                np.corrcoef(local_month_profile[:, i], national_month_profile)[0, 1]
                for i in range(province_values.shape[1])
            ])
            for i, code in enumerate(codes):
                all_rows.append({"year": year, "month": month, "province_code": code,
                                 "n_days": int(len(indexes)),
                                 "local_share_national_ratio_cv": float(ratio_cv[i]),
                                 "local_vs_national_normalized_daily_profile_corr": float(correlations[i]),
                                 "local_emissions_total": float(province_values[indexes, i].sum()),
                                 "national_emissions_total": float(national[indexes].sum())})
            summary_by_year.append({"year": year, "month": month, "days": int(len(indexes)),
                                    "max_local_national_share_ratio_cv": float(ratio_cv.max()),
                                    "median_local_national_share_ratio_cv": float(np.median(ratio_cv)),
                                    "min_daily_profile_correlation": float(correlations.min()),
                                    "mean_daily_profile_correlation": float(correlations.mean()),
                                    "max_abs_sum_provinces_vs_sum_column": float(national_sum_abs_error.max())})

    profiles = pd.DataFrame(all_rows)
    profiles.to_csv(OUT / "province_month_profile_inheritance.csv", index=False)
    by_month = pd.DataFrame(summary_by_year)
    by_month.to_csv(OUT / "year_month_summary.csv", index=False)
    ratios = profiles["local_share_national_ratio_cv"].to_numpy(float)
    correlations = profiles["local_vs_national_normalized_daily_profile_corr"].to_numpy(float)
    result = {
        "status": "EXPLORATORY_DAILY_EMISSIONS_LINEAGE_AUDIT_NOT_CIF_VALIDATION",
        "dataset_title": api_metadata["metadata"]["title"],
        "dataset_doi": api_metadata["metadata"].get("doi"),
        "dataset_page": f"https://zenodo.org/records/{RECORD_ID}",
        "associated_preprint": "https://doi.org/10.5194/essd-2021-153",
        "associated_preprint_status": "The ESSD discussion page states the preprint revision was not accepted.",
        "api_http": api_http,
        "file_http": file_http,
        "file_name": FILE_KEY,
        "file_bytes": len(body),
        "file_md5": md5,
        "zenodo_expected_md5": expected_md5,
        "file_sha256": sha256(workbook_path),
        "sheet_names": workbook.sheetnames,
        "power_years": [2019, 2020],
        "province_count": len(province_codes or []),
        "province_codes": province_codes,
        "daily_rows": int(sum(summary["days"] for summary in summary_by_year)),
        "province_month_rows": int(len(profiles)),
        "max_local_national_within_month_share_ratio_cv": float(ratios.max()),
        "median_local_national_within_month_share_ratio_cv": float(np.median(ratios)),
        "min_province_month_local_vs_national_daily_profile_corr": float(correlations.min()),
        "mean_province_month_local_vs_national_daily_profile_corr": float(correlations.mean()),
        "all_month_profiles_identical_to_national_within_numeric_precision": bool(
            correlations.min() > 0.999999 and ratios.max() < 1e-10),
        "interpretation": "Within each month, the provincial daily power-emissions profile inherits the same national daily profile; provincial weights set scale, not local day-to-day shape.",
        "limitations": [
            "The workbook contains daily estimated CO2 emissions, not generation amounts or CIF intensities.",
            "The associated ESSD discussion manuscript says provincial daily power-emissions variation is scaled from national daily estimates using monthly provincial thermal shares and acknowledges weak representation of province-specific daily changes.",
            "The associated preprint revision was not accepted; treat the dataset as a methodological artifact, not ground truth.",
            "No hourly CIF, local hourly generation denominator, or independent provincial daily generation series is provided by this workbook."
        ]
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
