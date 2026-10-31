"""Audit provenance and effective temporal variation of a public China hourly-load workbook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/china-hourly-load-dataset-audit-v64-r1/Data output.xlsx"
OUT = ROOT / "results/china-hourly-load-dataset-audit-v64-r1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    xls = pd.ExcelFile(INPUT)
    yearly = [s for s in xls.sheet_names if s.endswith("load curve")]
    annual, province_rows, monthly_load_rows = [], [], []
    for sheet in yearly:
        year = int(sheet[:4])
        frame = pd.read_excel(INPUT, sheet_name=sheet)
        time_col = frame.columns[0]
        frame = frame.dropna(subset=[time_col]).copy()
        frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
        frame = frame.dropna(subset=[time_col]).sort_values(time_col)
        provinces = [c for c in frame.columns[1:] if frame[c].notna().any()]
        timestamps = pd.DatetimeIndex(frame[time_col])
        expected = 8784 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 8760
        duplicate_count = int(timestamps.duplicated().sum())
        complete_hourly = (len(timestamps) == expected and duplicate_count == 0
                           and timestamps.equals(pd.date_range(f"{year}-01-01", periods=expected, freq="h")))
        values = frame[provinces].apply(pd.to_numeric, errors="coerce")
        missing_cells = int(values.isna().sum().sum())
        negative_cells = int((values < 0).sum().sum())
        annual.append({"year": year, "rows": int(len(frame)), "expected_hours": expected,
                       "province_columns": len(provinces), "duplicate_timestamps": duplicate_count,
                       "complete_hourly_calendar": bool(complete_hourly),
                       "missing_cells": missing_cells, "negative_cells": negative_cells,
                       "timestamp_first": str(timestamps.min()), "timestamp_last": str(timestamps.max())})

        work = values.copy()
        work.index = timestamps
        monthly_totals = work.groupby([work.index.year, work.index.month]).sum()
        for (month_year, month_no), row in monthly_totals.iterrows():
            for province in provinces:
                monthly_load_rows.append({"Province": str(province).split(" (")[0],
                                          "Date": f"{int(month_year):04d}-{int(month_no):02d}-01",
                                          "load_TWh": float(row[province]) / 1000.0})
        day_sum = work.groupby(work.index.date).transform("sum")
        norm = work / day_sum
        for province in provinces:
            x = work[province].to_numpy(float)
            q = norm[province].to_numpy(float)
            usable = np.isfinite(q).all() and np.isfinite(x).all() and len(x) >= expected
            if not usable:
                continue
            daily = q.reshape(-1, 24)
            mean_shape = daily.mean(axis=0)
            corr = [np.corrcoef(row, mean_shape)[0, 1] for row in daily]
            cv = daily.std(axis=0) / np.maximum(mean_shape, 1e-12)
            peak_hours = np.argmax(daily, axis=1)
            province_rows.append({"year": year, "province": str(province),
                                  "daily_profile_corr_to_annual_mean_median": float(np.nanmedian(corr)),
                                  "daily_profile_corr_to_annual_mean_p10": float(np.nanquantile(corr, .10)),
                                  "mean_hourly_shape_cv_across_days": float(np.mean(cv)),
                                  "daily_peak_hour_unique_count": int(np.unique(peak_hours).size),
                                  "daily_peak_hour_entropy_bits": float(-np.sum(
                                      np.bincount(peak_hours, minlength=24) / len(peak_hours)
                                      * np.log2(np.maximum(np.bincount(peak_hours, minlength=24) / len(peak_hours), 1e-15))))})

    annual_df = pd.DataFrame(annual)
    province_df = pd.DataFrame(province_rows)
    monthly_load = pd.DataFrame(monthly_load_rows)
    ember_path = ROOT / "results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
    ember = pd.read_csv(ember_path)
    ember = ember.loc[ember["Electricity source"].eq("Total generation"),
                      ["Province", "Date", "Generation (TWh)"]].copy()
    monthly_compare = monthly_load.merge(ember, on=["Province", "Date"], how="inner")
    monthly_compare["load_to_generation_ratio"] = monthly_compare.load_TWh / monthly_compare["Generation (TWh)"]
    monthly_compare.to_csv(OUT / "load_vs_ember_generation.csv", index=False)
    province_year = monthly_compare.assign(year=monthly_compare.Date.str[:4]).groupby(
        ["Province", "year"], as_index=False).agg(load_TWh=("load_TWh", "sum"),
                                                   generation_TWh=("Generation (TWh)", "sum"))
    province_year["load_to_generation_ratio"] = province_year.load_TWh / province_year.generation_TWh
    province_year.to_csv(OUT / "province_year_load_generation_ratio.csv", index=False)
    zenodo_csv = OUT / "Appendix 1_Hourly electric power load final.csv"
    zenodo_raw = pd.read_csv(zenodo_csv, sep=";")
    zvalues = zenodo_raw.iloc[:, 1:].to_numpy(dtype=float)
    if zvalues.shape != (365 * 24, 31):
        raise ValueError(f"Unexpected Zenodo 2018 load shape: {zvalues.shape}")
    zday = zvalues.reshape(365, 24, 31).transpose(0, 2, 1)
    zrange = zday.max(axis=2, keepdims=True) - zday.min(axis=2, keepdims=True)
    zshape = (zday - zday.min(axis=2, keepdims=True)) / np.maximum(zrange, 1e-12)
    zrows = []
    for j, province in enumerate(zenodo_raw.columns[1:]):
        mean_shape = zshape[:, j, :].mean(axis=0)
        correlations = np.array([np.corrcoef(curve, mean_shape)[0, 1]
                                 for curve in zshape[:, j, :]])
        zrows.append({"province_code": str(province),
                      "median_daily_profile_corr_to_annual_mean": float(np.nanmedian(correlations)),
                      "p10_daily_profile_corr_to_annual_mean": float(np.nanquantile(correlations, .10))})
    zshape_df = pd.DataFrame(zrows)
    zshape_df.to_csv(OUT / "zenodo_2018_daily_shape_audit.csv", index=False)
    zenodo_metrics = {"record_doi": "10.5281/zenodo.8322210",
                      "csv_sha256": sha256(zenodo_csv),
                      "csv_expected_md5": "8066da6f9f86a68f3a35633c5eb2029c",
                      "csv_observed_md5": hashlib.md5(zenodo_csv.read_bytes()).hexdigest(),
                      "rows": int(len(zenodo_raw)), "province_columns": int(zvalues.shape[1]),
                      "median_daily_profile_corr_to_annual_mean": float(
                          zshape_df.median_daily_profile_corr_to_annual_mean.median()),
                      "median_p10_daily_profile_corr_to_annual_mean": float(
                          zshape_df.p10_daily_profile_corr_to_annual_mean.median()),
                      "interpretation": "2018 curves are reconstructed by affine rescaling of a representative workday or holiday template to each day's reported minimum and maximum; this is not independently observed hourly load."}
    annual_df.to_csv(OUT / "annual_structure.csv", index=False)
    province_df.to_csv(OUT / "province_shape_variation.csv", index=False)
    data = {"status": "PUBLIC_MODEL_DERIVED_HOURLY_DEMAND_AUDIT_NOT_CIF_OR_GENERATION_VALIDATION",
            "source_doi": "10.6084/m9.figshare.29832701.v3",
            "article_doi": "10.1038/s41597-026-07327-8",
            "workbook_sha256": sha256(INPUT), "sheets": xls.sheet_names,
            "annual": annual_df.to_dict("records"),
            "shape_variation_summary": {
                "province_year_rows": int(len(province_df)),
                "median_daily_profile_corr": float(province_df.daily_profile_corr_to_annual_mean_median.median()),
                "median_p10_daily_profile_corr": float(province_df.daily_profile_corr_to_annual_mean_p10.median()),
                "median_mean_hourly_shape_cv": float(province_df.mean_hourly_shape_cv_across_days.median()),
                "median_peak_hour_entropy_bits": float(province_df.daily_peak_hour_entropy_bits.median())},
            "load_vs_generation_comparison": {
                "matched_province_month_rows": int(len(monthly_compare)),
                "matched_provinces": int(monthly_compare.Province.nunique()),
                "period_start": str(monthly_compare.Date.min()), "period_end": str(monthly_compare.Date.max()),
                "monthly_load_to_generation_ratio_median": float(monthly_compare.load_to_generation_ratio.median()),
                "monthly_load_to_generation_ratio_p10": float(monthly_compare.load_to_generation_ratio.quantile(.1)),
                "monthly_load_to_generation_ratio_p90": float(monthly_compare.load_to_generation_ratio.quantile(.9)),
                "province_year_rows": int(len(province_year)),
                "province_year_ratio_median": float(province_year.load_to_generation_ratio.median()),
                "province_year_ratio_p10": float(province_year.load_to_generation_ratio.quantile(.1)),
                "province_year_ratio_p90": float(province_year.load_to_generation_ratio.quantile(.9)),
                "median_within_province_monthly_ratio_cv": float(
                    monthly_compare.groupby("Province").load_to_generation_ratio.std().div(
                        monthly_compare.groupby("Province").load_to_generation_ratio.mean()).median())},
            "zenodo_2018_load_audit": zenodo_metrics,
            "interpretation": "This is estimated provincial electricity demand, not generation or measured CIF. Subdaily variation is model-generated from released aggregate statistics, typical-day profiles and weather-driven adjustments; it can be a sensitivity input only, not independent hourly truth.",
            "limits": ["The dataset describes demand/load, not provincial generation, net imports, or CIF.",
                       "The paper states that hourly profiles are reconstructed from typical-day load curves, annual/load statistics, and meteorological variables.",
                       "High nominal time resolution does not make this an independent measured hourly observation."]}
    (OUT / "audit_summary.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(data["shape_variation_summary"], indent=2))
    print(json.dumps(data["load_vs_generation_comparison"], indent=2))
    print(json.dumps(data["zenodo_2018_load_audit"], indent=2))
    print(annual_df.to_string(index=False))


if __name__ == "__main__":
    main()
