"""Preflight regional AEMO SCADA support for a strict five-group transfer test."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import validate_aemo_daily_proxy as validation  # noqa: E402
from transcif.data.aemo import sha256  # noqa: E402
from transcif.data.aemo_registry import registry_at  # noqa: E402

DEFAULT_STAGING = Path("/Users/cyyc0310/Desktop/tran/data/staging/aemo-native-2023")
PROXY = ROOT / "results/aemo-cdeii-validation-r3/hourly_scada_ef_proxy_2023.csv"
REGION_INDEX = {region: i for i, region in enumerate(sorted(validation.REGIONS))}
FUEL_GROUPS = ("thermal", "hydro", "nuclear", "wind", "solar")
SOURCE_GROUP = {
    "solar": "solar", "wind": "wind", "hydro": "hydro", "nuclear": "nuclear",
    "natural gas (pipeline)": "thermal", "black coal": "thermal", "brown coal": "thermal",
    "diesel oil": "thermal", "kerosene - non aviation": "thermal",
    "coal seam methane": "thermal", "coal seam methane ": "thermal",
    "coal mine waste gas": "thermal", "landfill biogas methane": "thermal",
    "bagasse": "thermal", "biomass and industrial materials": "thermal",
    "other biofuels": "thermal", "primary solid biomass fuels": "thermal",
    "other solid fossil fuels": "thermal", "coal seam methane": "thermal",
    "battery storage": "storage",
}


def category(source: str) -> str | None:
    key = source.strip().lower()
    if not key:
        return None
    if key not in SOURCE_GROUP:
        raise ValueError(f"unreviewed AEMO energy source: {source!r}")
    return SOURCE_GROUP[key]


def month_record(staging: Path, month: str, records: dict, details: pd.DataFrame,
                 allocations: pd.DataFrame, gensets: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    instant = pd.Timestamp(f"{month[:4]}-{month[4:]}-15T12:00:00+10:00")
    audit = registry_at(sorted(details.DUID.unique()), instant, details, allocations, gensets)
    region_by_duid, category_by_duid = {}, {}
    for unit in audit["records"]:
        if unit["dispatch_type_candidates"] != ["GENERATOR"]:
            continue
        if len(unit["region_candidates"]) != 1 or unit["region_candidates"][0] not in REGION_INDEX:
            continue
        duid = unit["duid"]
        region_by_duid[duid] = unit["region_candidates"][0]
        categories = {category(value) for value in unit["fuel_candidates"]}
        categories.discard(None)
        # A DUID with no fuel declaration or competing broad technologies is
        # deliberately not assigned to a CN5 group.
        if len(categories) == 1:
            category_by_duid[duid] = next(iter(categories))
    regions, factors = validation._mapping_for_month(
        sorted(details.DUID.unique()), instant, details, allocations, gensets)
    selected = {key: value for key, value in records.items() if key == ("SCADA", month)}
    if len(selected) != 1:
        raise ValueError(f"missing unique SCADA snapshot for {month}")
    scada = staging / selected[("SCADA", month)]["output"]
    if sha256(scada) != selected[("SCADA", month)]["output_sha256"]:
        raise ValueError(f"staging SHA mismatch: {scada}")

    seen = np.zeros((367, len(REGION_INDEX), 24, 12), dtype=bool)
    grouped_chunks = []
    for chunk in pd.read_csv(scada, usecols=["dispatch_interval_timestamp_utc", "duid", "scada_mw"],
                             dtype={"duid": str}, chunksize=500_000):
        region = chunk.duid.map(region_by_duid)
        assigned = region.notna()
        if not assigned.any():
            continue
        utc = pd.to_datetime(chunk.loc[assigned, "dispatch_interval_timestamp_utc"], utc=True)
        local = utc.dt.tz_convert("Etc/GMT-10")
        within = local.dt.strftime("%Y-%m").eq(f"{month[:4]}-{month[4:]}").to_numpy()
        if not within.any():
            continue
        positions = assigned[assigned].index[within]
        local = local.iloc[np.flatnonzero(within)]
        duid = chunk.loc[positions, "duid"]
        reg = region.loc[positions].to_numpy()
        day = local.dt.dayofyear.to_numpy()
        hour = local.dt.hour.to_numpy()
        slots = (local.dt.minute.to_numpy() % 60) // 5
        ri = pd.Series(reg).map(REGION_INDEX).to_numpy()
        seen[day, ri, hour, slots] = True
        mw = pd.to_numeric(chunk.loc[positions, "scada_mw"], errors="coerce").fillna(0.).clip(lower=0.).to_numpy()
        fuel_group = duid.map(category_by_duid)
        label_ef = pd.to_numeric(duid.map(factors), errors="coerce").to_numpy()
        coarse_mw = np.where(fuel_group.isin(FUEL_GROUPS).to_numpy(), mw, 0.)
        storage_mw = np.where(fuel_group.eq("storage").to_numpy(), mw, 0.)
        mapped_mw = np.where(np.isfinite(label_ef), mw, 0.)
        frame = pd.DataFrame({
            "date": local.dt.strftime("%Y-%m-%d").to_numpy(), "region": reg,
            "hour": hour, "positive_mw": mw, "coarse_mw": coarse_mw,
            "storage_mw": storage_mw, "ef_mapped_mw": mapped_mw,
            "thermal": np.where(fuel_group.eq("thermal").to_numpy(), mw, 0.),
            "hydro": np.where(fuel_group.eq("hydro").to_numpy(), mw, 0.),
            "nuclear": np.where(fuel_group.eq("nuclear").to_numpy(), mw, 0.),
            "wind": np.where(fuel_group.eq("wind").to_numpy(), mw, 0.),
            "solar": np.where(fuel_group.eq("solar").to_numpy(), mw, 0.),
        })
        grouped_chunks.append(frame.groupby(["date", "region", "hour"], as_index=False).sum())
    if not grouped_chunks:
        raise ValueError(f"no assigned positive generation for {month}")
    hourly = pd.concat(grouped_chunks, ignore_index=True).groupby(
        ["date", "region", "hour"], as_index=False).sum()
    hourly["coarse_support"] = hourly.coarse_mw / hourly.positive_mw.replace(0., np.nan)
    hourly["label_support"] = hourly.ef_mapped_mw / hourly.positive_mw.replace(0., np.nan)
    day = pd.to_datetime(hourly.date).dt.dayofyear.to_numpy()
    reg_i = hourly.region.map(REGION_INDEX).to_numpy()
    slot_count = seen[day, reg_i, hourly.hour.to_numpy()].sum(axis=1)
    hourly["interval_support"] = slot_count / 12.
    hourly["eligible_cn5_hour"] = (hourly.interval_support.ge(.95) &
                                   hourly.label_support.ge(.95) &
                                   hourly.coarse_support.ge(.99))
    hourly["month"] = month
    # Keep only compact diagnostics; full rows are built by the model runner.
    by_region = []
    for region, part in hourly.groupby("region", sort=True):
        by_region.append({
            "region": region, "hours": len(part),
            "eligible_cn5_hours": int(part.eligible_cn5_hour.sum()),
            "eligible_fraction": float(part.eligible_cn5_hour.mean()),
            "coarse_support_min": float(part.coarse_support.min()),
            "coarse_support_p01": float(part.coarse_support.quantile(.01)),
            "storage_positive_mw_share": float(part.storage_mw.sum() / part.positive_mw.sum()),
            "label_support_p01": float(part.label_support.quantile(.01)),
            "interval_support_p01": float(part.interval_support.quantile(.01)),
        })
    return hourly, {"month": month, "region_support": by_region,
                    "active_generator_duids": len(region_by_duid),
                    "cn5_mapped_duids": len(category_by_duid)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    metadata_path, snapshots = validation._snapshot_index(args.staging)
    details = validation._read_staged(args.staging, snapshots[("DETAIL", "202312")])
    allocations = validation._read_staged(args.staging, snapshots[("ALLOC", "202312")])
    reports, hourly = [], []
    hashes = {str(metadata_path.resolve()): sha256(metadata_path)}
    for month in sorted(m for table, m in snapshots if table == "SCADA"):
        if ("GENUNITS", month) not in snapshots:
            raise ValueError(f"missing monthly GENUNITS snapshot: {month}")
        gensets = validation._read_staged(args.staging, snapshots[("GENUNITS", month)])
        frame, summary = month_record(args.staging, month, snapshots, details, allocations, gensets)
        reports.append(summary)
        hourly.append(frame)
        for key in (("SCADA", month), ("GENUNITS", month)):
            record = snapshots[key]
            path = args.staging / record["output"]
            hashes[str(path.resolve())] = record["output_sha256"]
    if sorted(row["month"] for row in reports) != [f"2023{i:02d}" for i in range(1, 13)]:
        raise ValueError("complete 2023 monthly generation archive required")
    for key in (("DETAIL", "202312"), ("ALLOC", "202312")):
        path = args.staging / snapshots[key]["output"]
        hashes[str(path.resolve())] = snapshots[key]["output_sha256"]
    compact = pd.concat(hourly, ignore_index=True)
    by_region = compact.groupby("region").agg(
        hours=("eligible_cn5_hour", "size"), eligible_hours=("eligible_cn5_hour", "sum"),
        coarse_support_min=("coarse_support", "min"),
        coarse_support_p01=("coarse_support", lambda x: x.quantile(.01)),
        label_support_p01=("label_support", lambda x: x.quantile(.01)),
        interval_support_p01=("interval_support", lambda x: x.quantile(.01)),
        storage_share=("storage_mw", "sum"), positive_mw=("positive_mw", "sum"))
    by_region["eligible_fraction"] = by_region.eligible_hours / by_region.hours
    by_region["storage_positive_mw_share"] = by_region.storage_share / by_region.positive_mw
    by_region.reset_index().to_csv(out / "region_support.csv", index=False)
    with (out / "monthly_audit.jsonl").open("w") as stream:
        for report in reports:
            for region_row in report["region_support"]:
                stream.write(json.dumps({"month": report["month"], **region_row}) + "\n")
    (out / "audit.json").write_text(json.dumps({
        "status": "PREMODEL_CN5_SUPPORT_PREFLIGHT",
        "purpose": "determine whether AEMO regional hourly proxy can support a five-group generation-transfer development test",
        "staging_eligible_for_issued_public": False,
        "label_is_official_hourly_cif": False,
        "regions": sorted(validation.REGIONS), "months": [r["month"] for r in reports],
        "fuel_group_contract": {"thermal": "all reviewed combustible sources including biomass and waste fuels",
                                "hydro": "Hydro", "nuclear": "structural zero in 2023 AEMO registered source inventory",
                                "wind": "Wind", "solar": "Solar", "storage": "tracked separately and excluded from CN5 numerator"},
        "monthly": reports,
        "provenance_sha256": hashes,
        "limitations": ["retrospective registration and source-factor snapshot; not issue-time metadata",
                        "SCADA MW is instantaneous sampled generation, not metered sent-out MWh",
                        "hourly label is the existing DUID EF CO2e proxy, not official hourly CIF",
                        "AEMO CDEII is an official daily generation-area index and cannot certify hourly proxy shape"]
    }, indent=2) + "\n")
    print(by_region[["hours", "eligible_hours", "eligible_fraction", "coarse_support_min",
                     "coarse_support_p01", "label_support_p01", "interval_support_p01",
                     "storage_positive_mw_share"]].to_string())
    print(f"wrote preflight: {out}", flush=True)


if __name__ == "__main__":
    main()
