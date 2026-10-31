"""Snapshot and schema-audit Ember's public China provincial monthly electricity CSV.

This is a data-lineage audit only. It does not provide hourly CIF labels or validate
any target-local hourly forecast.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/china-monthly-ember-audit-r1"
DATA_URL = (
    "https://storage.googleapis.com/emb-prod-bkt-publicdata/public-downloads/"
    "generation/outputs/release_chn_subnational_generation_monthly.csv"
)
METHOD_URL = (
    "https://storage.googleapis.com/emb-prod-bkt-publicdata/public-downloads/"
    "ember_china_generation_methodology.pdf"
)
PAGE_URL = "https://ember-energy.org/data/china-electricity-data/"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, destination: Path) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "TransCIF research data audit"})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
        metadata = {"url": url, "http_status": response.status,
                    "last_modified": response.headers.get("Last-Modified"),
                    "content_type": response.headers.get("Content-Type"),
                    "content_length_header": response.headers.get("Content-Length")}
    destination.write_bytes(body)
    metadata.update({"bytes": len(body), "sha256": sha256(destination), "file": destination.name})
    return metadata


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    monthly_meta = download(DATA_URL, OUT / "ember_china_province_monthly.csv")
    method_meta = download(METHOD_URL, OUT / "ember_china_methodology.pdf")
    frame = pd.read_csv(OUT / monthly_meta["file"], low_memory=False)
    expected = {"Province", "Province code", "Date", "Electricity source", "Is aggregated source",
                "Generation (TWh)", "Emissions (MtCO2e)", "Emissions intensity (gCO2e/kWh)"}
    missing = sorted(expected.difference(frame.columns))
    if missing:
        raise ValueError(f"Unexpected Ember schema; missing {missing}")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame["Generation (TWh)"] = pd.to_numeric(frame["Generation (TWh)"], errors="coerce")
    frame["Emissions (MtCO2e)"] = pd.to_numeric(frame["Emissions (MtCO2e)"], errors="coerce")
    frame["Emissions intensity (gCO2e/kWh)"] = pd.to_numeric(
        frame["Emissions intensity (gCO2e/kWh)"], errors="coerce")
    frame["Is aggregated source"] = frame["Is aggregated source"].astype(str).str.lower().eq("true")

    province_list = sorted(frame["Province"].dropna().unique().tolist())
    source_rows = frame.loc[~frame["Is aggregated source"]].copy()
    province_month_source_counts = (source_rows.groupby(["Province", "Date"])["Electricity source"]
                                    .nunique().rename("source_categories").reset_index())
    province_month_total_rows = frame.loc[frame["Electricity source"].eq("Total generation")].copy()
    province_month_total_rows = province_month_total_rows.sort_values(["Province", "Date"])
    province_month_total_rows["recomputed_intensity"] = (
        province_month_total_rows["Emissions (MtCO2e)"]
        / province_month_total_rows["Generation (TWh)"] * 1000.0
    )
    province_month_total_rows["intensity_difference"] = (
        province_month_total_rows["recomputed_intensity"]
        - province_month_total_rows["Emissions intensity (gCO2e/kWh)"]
    )

    pivot = source_rows.pivot_table(index=["Province", "Date"], columns="Electricity source",
                                    values="Generation (TWh)", aggfunc="first")
    fuel_sources = ["Coal", "Gas", "Other fossil", "Bioenergy", "Wind", "Solar", "Hydro", "Nuclear"]
    present_fuels = [fuel for fuel in fuel_sources if fuel in pivot.columns]
    total = province_month_total_rows.set_index(["Province", "Date"])["Generation (TWh)"]
    summed = pivot[present_fuels].sum(axis=1, min_count=1)
    aligned = pd.concat([total.rename("total"), summed.rename("fuel_sum")], axis=1).dropna()
    aligned["difference_twh"] = aligned["total"] - aligned["fuel_sum"]

    monthly_coverage = (province_month_total_rows.groupby(province_month_total_rows["Date"].dt.year)
                        .agg(provinces=("Province", "nunique"), province_months=("Province", "size"),
                             first_month=("Date", "min"), last_month=("Date", "max"))
                        .reset_index(names="year"))
    monthly_coverage.to_csv(OUT / "year_coverage.csv", index=False)
    source_inventory = (frame.groupby(["Is aggregated source", "Electricity source"], dropna=False)
                        .agg(rows=("Province", "size"), provinces=("Province", "nunique"),
                             first_month=("Date", "min"), last_month=("Date", "max"))
                        .reset_index())
    source_inventory.to_csv(OUT / "source_inventory.csv", index=False)
    province_month_source_counts.to_csv(OUT / "fuel_category_counts.csv", index=False)

    valid_recomputed = province_month_total_rows["recomputed_intensity"].notna()
    intensity_diff = province_month_total_rows.loc[valid_recomputed, "intensity_difference"].dropna()
    discrepancies = aligned["difference_twh"].abs()
    result = {
        "status": "PUBLIC_MONTHLY_DATA_SCHEMA_AND_INTERNAL_CONSISTENCY_AUDIT_ONLY",
        "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "landing_page": PAGE_URL,
        "monthly_data": monthly_meta,
        "methodology_pdf": method_meta,
        "rows": int(len(frame)),
        "province_count": int(len(province_list)),
        "province_names": province_list,
        "date_min": frame["Date"].min().date().isoformat(),
        "date_max": frame["Date"].max().date().isoformat(),
        "source_categories": sorted(frame["Electricity source"].dropna().unique().tolist()),
        "aggregated_source_rows": int(frame["Is aggregated source"].sum()),
        "non_aggregated_rows": int((~frame["Is aggregated source"]).sum()),
        "total_generation_rows": int(frame["Electricity source"].eq("Total generation").sum()),
        "generation_fuel_categories_present": present_fuels,
        "complete_eight_category_months": int(province_month_source_counts["source_categories"].eq(8).sum()),
        "province_months_total_rows": int(len(province_month_total_rows)),
        "recomputed_total_intensity_max_abs_difference_g_per_kwh": float(intensity_diff.abs().max()),
        "fuel_sum_vs_total_generation_abs_difference_twh": {
            "median": float(discrepancies.median()), "p95": float(discrepancies.quantile(0.95)),
            "max": float(discrepancies.max()), "rows_compared": int(len(aligned))},
        "limitations": [
            "This is the current downloaded revision; the public CSV is not a historical as-of vintage archive.",
            "The data are assembled and corrected estimates, not a single raw official monthly series.",
            "Monthly emissions and intensity are not hourly labels and cannot independently validate hourly CIF forecasts.",
            "Internal arithmetic consistency does not establish independent accuracy or correct temporal/spatial accounting scope."
        ]
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
