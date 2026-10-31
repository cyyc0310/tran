"""V294: audit lineage and coverage of Ember's public China monthly fuel CSV."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V294_CHINA_MONTHLY_SOURCE_LINEAGE_PROTOCOL.md"
OUT = ROOT / "results/v294-china-ember-lineage-r4"
CSV_URL = "https://files.ember-energy.org/public-downloads/generation/outputs/release_chn_subnational_generation_monthly.csv"
METHOD_URL = "https://storage.googleapis.com/emb-prod-bkt-publicdata/public-downloads/ember_china_generation_methodology.pdf"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    out = OUT
    out.mkdir(parents=True)
    request = Request(CSV_URL, headers={"User-Agent": "TransCIF research audit/1.0"})
    with urlopen(request, timeout=60) as response:
        csv_bytes = response.read()
        response_headers = dict(response.headers.items())
    method_request = Request(METHOD_URL, headers={"User-Agent": "TransCIF research audit/1.0"})
    with urlopen(method_request, timeout=60) as response:
        method_bytes = response.read()
    snapshot = out / "ember_china_monthly_generation.csv"
    snapshot.write_bytes(csv_bytes)
    (out / "ember_china_generation_methodology.pdf").write_bytes(method_bytes)
    df = pd.read_csv(snapshot, low_memory=False)
    required = {"Province", "Date", "Electricity source", "Is aggregated source",
                "Generation (TWh)", "Emissions (MtCO2e)",
                "Emissions intensity (gCO2e/kWh)"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"missing expected columns: {missing}")
    dates = pd.to_datetime(df["Date"], errors="raise")
    agg = df["Is aggregated source"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}
    )
    if agg.isna().any():
        raise RuntimeError("unrecognized aggregate-source values")
    raw = df.loc[~agg].copy()
    duplicate_keys = int(raw.duplicated(["Province", "Date", "Electricity source"]).sum())
    if duplicate_keys:
        raise RuntimeError(f"duplicate primary source keys: {duplicate_keys}")

    provincial = raw.loc[raw["Province"].str.strip().ne("China (total)")].copy()
    national = raw.loc[raw["Province"].str.strip().eq("China (total)")].copy()
    source_counts = provincial.groupby("Electricity source", dropna=False).agg(
        rows=("Generation (TWh)", "size"), provinces=("Province", "nunique"),
        months=("Date", "nunique"), generation_missing=("Generation (TWh)", lambda x: int(x.isna().sum())),
        emissions_missing=("Emissions (MtCO2e)", lambda x: int(x.isna().sum())),
        intensity_missing=("Emissions intensity (gCO2e/kWh)", lambda x: int(x.isna().sum())),
    ).reset_index()
    source_counts.to_csv(out / "primary_source_coverage.csv", index=False)
    aggregate_counts = df.loc[agg].groupby("Electricity source", dropna=False).size()
    aggregate_counts.rename("rows").reset_index().to_csv(out / "aggregate_rows.csv", index=False)

    latest = dates.max()
    latest_df = provincial.loc[dates.loc[provincial.index].eq(latest)]
    latest_coverage = latest_df.groupby("Electricity source")["Province"].nunique()
    expected_provinces = 31
    expected_sources = 8
    expected_rows_per_source = expected_provinces * dates.nunique()
    if provincial["Province"].nunique() != expected_provinces:
        raise RuntimeError(f"expected 31 provinces, found {provincial['Province'].nunique()}")
    if provincial["Electricity source"].nunique() != expected_sources:
        raise RuntimeError(f"expected 8 primary fuel sources, found {provincial['Electricity source'].nunique()}")
    if not source_counts["rows"].eq(expected_rows_per_source).all():
        raise RuntimeError("not every primary source has a complete province-month panel")
    if not latest_coverage.eq(expected_provinces).all():
        raise RuntimeError("latest month does not cover all provinces for every source")
    latest_coverage.rename("n_provinces").reset_index().to_csv(
        out / "latest_month_source_coverage.csv", index=False
    )

    gen = pd.to_numeric(provincial["Generation (TWh)"], errors="coerce")
    emissions = pd.to_numeric(provincial["Emissions (MtCO2e)"], errors="coerce")
    intensity = pd.to_numeric(provincial["Emissions intensity (gCO2e/kWh)"], errors="coerce")
    valid = gen.gt(0) & emissions.notna() & intensity.notna()
    # Mt/TWh is numerically t/MWh; multiply by 1,000 to obtain g/kWh.
    ratio = 1000. * emissions.loc[valid] / gen.loc[valid]
    intensity_error = float(np.max(np.abs(ratio.to_numpy() - intensity.loc[valid].to_numpy()))) if valid.any() else None
    if intensity_error is not None and intensity_error > 0.01:
        raise RuntimeError(f"emissions/generation does not reproduce intensity: {intensity_error}")

    metadata = {
        "protocol_sha256": sha256_file(PROTOCOL),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "csv_url": CSV_URL, "methodology_url": METHOD_URL,
        "csv_sha256": sha256_bytes(csv_bytes), "csv_bytes": len(csv_bytes),
        "methodology_pdf_sha256": sha256_bytes(method_bytes),
        "methodology_pdf_bytes": len(method_bytes),
        "http_content_type": response_headers.get("Content-Type"),
        "columns": list(df.columns),
        "date_min": dates.min().strftime("%Y-%m-%d"),
        "date_max": latest.strftime("%Y-%m-%d"),
        "n_rows": int(len(df)), "n_primary_rows": int(len(raw)),
        "n_aggregated_rows": int(agg.sum()),
        "n_geographies_primary_including_china": int(raw["Province"].nunique()),
        "n_provincial_units_excluding_national_china": int(provincial["Province"].nunique()),
        "province_names": sorted(provincial["Province"].unique().tolist()),
        "n_national_rows": int(len(national)),
        "n_primary_sources": int(raw["Electricity source"].nunique()),
        "n_months": int(dates.nunique()),
        "duplicate_primary_keys": duplicate_keys,
        "max_abs_emissions_per_generation_minus_intensity": intensity_error,
        "interpretation": "Ember's current provincial source-specific generation and corresponding emissions/intensity are methodologically estimated; not independent measured hourly CIF labels.",
        "methodology_version_as_displayed": "China Methodology v1, last updated August 2026",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    if metadata["protocol_sha256"] != sha256_file(PROTOCOL):
        raise RuntimeError("protocol changed during run")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(source_counts.to_string(index=False))
    print("latest month source coverage:")
    print(latest_coverage.to_string())


if __name__ == "__main__":
    main()
