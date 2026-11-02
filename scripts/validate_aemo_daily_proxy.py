#!/usr/bin/env python
"""Compare a retrospective DUID-SCADA/EF proxy with official daily AEMO CDEII.

This is a data-construction diagnostic. It does not create an hourly CIF label,
claim an as-issued input history, or validate the consumption-side target.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transcif.data.aemo import sha256
from transcif.data.aemo_registry import registry_at


REGIONS = {"NSW1", "QLD1", "SA1", "TAS1", "VIC1"}
REGION_INDEX = {region: index for index, region in enumerate(sorted(REGIONS))}
EXPECTED_INTERVALS_PER_DAY = 288  # 2023 uses fixed UTC+10 NEM time, no DST.


def _snapshot_index(staging):
    metadata_path = staging / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("eligible_for_issued_public") is not False:
        raise ValueError("expected retrospective AEMO staging, not issued-public data")
    snapshots = {}
    for record in metadata["snapshots"]:
        name = Path(record["output"]).name
        if "UNIT_SCADA" in name:
            month = name.split("_SCADA_")[1][:6]
            table = "SCADA"
        elif "GENUNITS" in name:
            month = name.split("GENUNITS_")[1][:6]
            table = "GENUNITS"
        elif "DUDETAILSUMMARY_202312" in name:
            month, table = "202312", "DETAIL"
        elif "DUALLOC_202312" in name:
            month, table = "202312", "ALLOC"
        else:
            continue
        key = (table, month)
        if key in snapshots:
            raise ValueError(f"duplicate AEMO snapshot key: {key}")
        snapshots[key] = record
    return metadata_path, snapshots


def _read_staged(staging, record):
    path = staging / record["output"]
    if path.parent.resolve() != staging.resolve() or sha256(path) != record["output_sha256"]:
        raise ValueError(f"AEMO staging hash/path mismatch: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _read_official_cdeii(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if not row or row[0] != "D":
                continue
            if len(row) != 11 or row[1:4] != ["CO2EII", "PUBLISHING", "1"]:
                raise ValueError("unreviewed AEMO CDEII record schema")
            region = row[7]
            if region not in REGIONS:
                continue
            rows.append({
                "date": pd.Timestamp(row[6]).date().isoformat(),
                "region": region,
                "official_cdeii_tco2e_mwh": float(row[10]),
            })
    frame = pd.DataFrame(rows)
    if frame.empty or frame.duplicated(["date", "region"]).any():
        raise ValueError("unique official regional CDEII day rows required")
    return frame


def _mapping_for_month(duids, at, details, allocations, gensets):
    audit = registry_at(duids, at, details, allocations, gensets)
    regions, factors = {}, {}
    for unit in audit["records"]:
        if unit["dispatch_type_candidates"] != ["GENERATOR"]:
            continue
        if len(unit["region_candidates"]) != 1 or unit["region_candidates"][0] not in REGIONS:
            continue
        duid = unit["duid"]
        regions[duid] = unit["region_candidates"][0]
        if (unit["allocation_candidates"] and not unit["allocation_version_ambiguous"]
                and unit["native_ef_min"] is not None
                and unit["native_ef_min"] == unit["native_ef_max"]):
            factors[duid] = float(unit["native_ef_min"])
    return regions, factors


def _month_proxy(staging, scada_record, regions, factors, chunksize=500_000):
    month = Path(scada_record["output"]).name.split("_SCADA_")[1][:6]
    local_month = f"{month[:4]}-{month[4:]}"
    path = staging / scada_record["output"]
    if path.parent.resolve() != staging.resolve() or sha256(path) != scada_record["output_sha256"]:
        raise ValueError(f"SCADA snapshot hash/path mismatch: {path}")

    daily_sums, hourly_sums = [], []
    seen_hours = np.zeros((367, len(REGION_INDEX), 24, 12), dtype=bool)
    for chunk in pd.read_csv(
        path,
        usecols=["dispatch_interval_timestamp_utc", "duid", "scada_mw"],
        dtype={"duid": str},
        chunksize=chunksize,
    ):
        region = chunk["duid"].map(regions)
        assigned = region.notna()
        if not assigned.any():
            continue
        local_time = pd.to_datetime(
            chunk.loc[assigned, "dispatch_interval_timestamp_utc"], utc=True
        ).dt.tz_convert("Etc/GMT-10")
        date = local_time.dt.strftime("%Y-%m-%d")
        within_month = date.str.slice(0, 7).eq(local_month).to_numpy()
        if not within_month.any():
            continue
        positions = assigned[assigned].index[within_month]
        date_values = date.to_numpy()[within_month]
        region_values = region.loc[positions].to_numpy()
        minutes = local_time.dt.minute.to_numpy()[within_month]
        day_indices = local_time.dt.dayofyear.to_numpy()[within_month]
        region_indices = pd.Series(region_values).map(REGION_INDEX).to_numpy()
        hour_indices = local_time.dt.hour.to_numpy()[within_month]
        slots_in_hour = (minutes % 60) // 5
        seen_hours[day_indices, region_indices, hour_indices, slots_in_hour] = True

        mw = pd.to_numeric(chunk.loc[positions, "scada_mw"], errors="coerce").fillna(0.0).clip(lower=0.0)
        ef = chunk.loc[positions, "duid"].map(factors)
        utc_hour = pd.to_datetime(
            chunk.loc[positions, "dispatch_interval_timestamp_utc"], utc=True
        ).dt.floor("h").astype(str).to_numpy()
        frame = pd.DataFrame({
            "date": date_values,
            "region": region_values,
            "time_utc": utc_hour,
            "positive_mw_sample": mw.to_numpy(),
            "ef": ef.to_numpy(),
        })
        frame = frame[frame.positive_mw_sample > 0]
        frame["mapped_mw_sample"] = frame.positive_mw_sample.where(frame.ef.notna(), 0.0)
        frame["proxy_emission_mass"] = frame.mapped_mw_sample * frame.ef.fillna(0.0)
        daily_sums.append(frame.groupby(["date", "region"], as_index=False)[
            ["positive_mw_sample", "mapped_mw_sample", "proxy_emission_mass"]
        ].sum())
        hourly_sums.append(frame.groupby(["time_utc", "date", "region"], as_index=False)[
            ["positive_mw_sample", "mapped_mw_sample", "proxy_emission_mass"]
        ].sum())

    if not daily_sums or not hourly_sums:
        raise ValueError(f"no mapped regional SCADA rows for {month}")
    result = pd.concat(daily_sums).groupby(["date", "region"], as_index=False)[
        ["positive_mw_sample", "mapped_mw_sample", "proxy_emission_mass"]
    ].sum()
    result["mapped_support"] = result.mapped_mw_sample / result.positive_mw_sample
    result["proxy_cdeii_tco2e_mwh"] = result.proxy_emission_mass / result.mapped_mw_sample.replace(0.0, np.nan)
    day_indices = pd.to_datetime(result.date).dt.dayofyear.to_numpy()
    region_indices = result.region.map(REGION_INDEX).to_numpy()
    result["observed_5min_stamps"] = seen_hours[day_indices, region_indices].sum(axis=(1, 2))
    result["interval_support"] = result.observed_5min_stamps / EXPECTED_INTERVALS_PER_DAY

    hourly = pd.concat(hourly_sums).groupby(["time_utc", "date", "region"], as_index=False)[
        ["positive_mw_sample", "mapped_mw_sample", "proxy_emission_mass"]
    ].sum()
    hourly["mapped_support"] = hourly.mapped_mw_sample / hourly.positive_mw_sample
    hourly["proxy_cdeii_tco2e_mwh"] = hourly.proxy_emission_mass / hourly.mapped_mw_sample.replace(0.0, np.nan)
    time = pd.to_datetime(hourly.time_utc, utc=True)
    day_indices = pd.to_datetime(hourly.date).dt.dayofyear.to_numpy()
    region_indices = hourly.region.map(REGION_INDEX).to_numpy()
    hour_indices = time.dt.tz_convert("Etc/GMT-10").dt.hour.to_numpy()
    hourly["observed_5min_stamps"] = seen_hours[
        day_indices, region_indices, hour_indices
    ].sum(axis=1)
    hourly["interval_support"] = hourly.observed_5min_stamps / 12
    return result, hourly


def _scores(frame):
    error = frame.proxy_cdeii_tco2e_mwh - frame.official_cdeii_tco2e_mwh
    return {
        "region_days": int(len(frame)),
        "mean_mapped_support": float(frame.mapped_support.mean()),
        "minimum_mapped_support": float(frame.mapped_support.min()),
        "mean_5min_interval_support": float(frame.interval_support.mean()),
        "minimum_5min_interval_support": float(frame.interval_support.min()),
        "bias_tco2e_mwh": float(error.mean()),
        "mae_tco2e_mwh": float(error.abs().mean()),
        "rmse_tco2e_mwh": float(np.sqrt(np.mean(error ** 2))),
        "correlation": float(frame[["proxy_cdeii_tco2e_mwh", "official_cdeii_tco2e_mwh"]].corr().iloc[0, 1]),
        "centered_rmse_tco2e_mwh": float(np.sqrt(np.mean(((error - error.mean())) ** 2))),
    }


def validate(staging, official_csv, output):
    staging, official_csv, output = map(Path, (staging, official_csv, output))
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    metadata_path, snapshots = _snapshot_index(staging)
    required = [("DETAIL", "202312"), ("ALLOC", "202312")]
    required += [(kind, month) for kind in ("SCADA", "GENUNITS") for month in
                 sorted(m for table, m in snapshots if table == kind)]
    if any(key not in snapshots for key in required):
        raise ValueError("complete 2023 SCADA/GENUNITS and December registry snapshots required")

    details = _read_staged(staging, snapshots[("DETAIL", "202312")])
    allocations = _read_staged(staging, snapshots[("ALLOC", "202312")])
    duids = sorted(details.DUID.unique())
    pieces, hourly_pieces = [], []
    input_hashes = {str(metadata_path.resolve()): sha256(metadata_path)}
    for key in [("DETAIL", "202312"), ("ALLOC", "202312")]:
        record = snapshots[key]
        input_hashes[str((staging / record["output"]).resolve())] = record["output_sha256"]

    months = sorted(m for table, m in snapshots if table == "SCADA")
    if months != [f"2023{i:02d}" for i in range(1, 13)]:
        raise ValueError("complete January–December 2023 SCADA snapshots required")
    for month in months:
        genset_record = snapshots[("GENUNITS", month)]
        gensets = _read_staged(staging, genset_record)
        scada_record = snapshots[("SCADA", month)]
        at = pd.Timestamp(f"{month[:4]}-{month[4:]}-15T12:00:00+10:00")
        regions, factors = _mapping_for_month(duids, at, details, allocations, gensets)
        daily_proxy, hourly_proxy = _month_proxy(staging, scada_record, regions, factors)
        pieces.append(daily_proxy)
        hourly_pieces.append(hourly_proxy)
        for record in (genset_record, scada_record):
            input_hashes[str((staging / record["output"]).resolve())] = record["output_sha256"]

    proxy = pd.concat(pieces, ignore_index=True)
    hourly = pd.concat(hourly_pieces, ignore_index=True)
    if proxy.duplicated(["date", "region"]).any():
        raise ValueError("local-month filtering must yield unique region-day proxy rows")
    if hourly.duplicated(["time_utc", "region"]).any():
        raise ValueError("hour proxy must have unique UTC-hour/region rows")
    expected_hours = 365 * 24
    if set(hourly.groupby("region").size().astype(int)) != {expected_hours}:
        raise ValueError("each NEM region must have all 8,760 hourly proxy rows")
    hourly["hourly_label_eligible"] = (
        (hourly.interval_support >= 0.95) & (hourly.mapped_support >= 0.95)
    )
    hourly_daily = hourly.groupby(["date", "region"], as_index=False)[
        ["mapped_mw_sample", "proxy_emission_mass"]
    ].sum()
    hourly_daily["hourly_rollup_cdeii_tco2e_mwh"] = (
        hourly_daily.proxy_emission_mass / hourly_daily.mapped_mw_sample
    )
    daily_check = proxy.merge(
        hourly_daily[["date", "region", "hourly_rollup_cdeii_tco2e_mwh"]],
        on=["date", "region"], validate="one_to_one",
    )
    rollup_diff = float((daily_check.proxy_cdeii_tco2e_mwh -
                         daily_check.hourly_rollup_cdeii_tco2e_mwh).abs().max())
    if rollup_diff > 1e-10:
        raise ValueError("hourly proxy energy-weighted daily rollup differs from direct SCADA daily aggregation")
    official = _read_official_cdeii(official_csv)
    paired = proxy.merge(official, on=["date", "region"], how="inner", validate="one_to_one")
    # The official file stops at Dec 30, 2023; score exactly its matched days.
    if paired.empty or paired.duplicated(["date", "region"]).any():
        raise ValueError("nonempty unique official/proxy region-day intersection required")
    if len(paired) != len(official):
        raise ValueError("every official regional day must have exactly one proxy row")
    if (paired.interval_support < 0.95).any():
        raise ValueError("at least 95% of fixed-UTC+10 five-minute stamps must be observed each region-day")

    overall = _scores(paired)
    per_region = [{"region": region, **_scores(rows)}
                  for region, rows in paired.groupby("region", sort=True)]
    hourly_support = [{
        "region": region,
        "hours": int(len(rows)),
        "eligible_hours": int(rows.hourly_label_eligible.sum()),
        "eligible_fraction": float(rows.hourly_label_eligible.mean()),
        "minimum_interval_support": float(rows.interval_support.min()),
        "minimum_mapped_support": float(rows.mapped_support.min()),
    } for region, rows in hourly.groupby("region", sort=True)]
    paired.to_csv(output / "daily_scada_ef_proxy_vs_official.csv", index=False)
    hourly = hourly.sort_values(["time_utc", "region"]).reset_index(drop=True)
    hourly.to_csv(output / "hourly_scada_ef_proxy_2023.csv", index=False)
    input_hashes[str(official_csv.resolve())] = sha256(official_csv)
    report = {
        "purpose": "retrospective daily label-construction check; not an hourly CIF validation",
        "proxy_definition": "positive instantaneous SCADA MW sample-weighted native CO2e EF ratio for mapped regional GENERATOR DUIDs",
        "target_definition": "official AEMO regional daily CDEII, t CO2e/MWh",
        "registration": "retrospective December 2023 DUDETAILSUMMARY/DUALLOC; month-specific 2023 GENUNITS snapshot; no original issue-time reconstruction",
        "time": "fixed UTC+10 NEM local date; monthly boundary rows excluded; five-minute stamp support divided by 288",
        "known_limitations": [
            "CDEII is daily and CO2e; this does not create an official hourly CIF or a consumption-side target",
            "SCADA is instantaneous MW, not sent-out metered MWh; equal-sample MW weighting is an approximation",
            "native emission factors and registry are retrospective; they do not prove original public availability",
            "only unique-region generator DUIDs with unambiguous, single-valued EF mappings enter the proxy numerator",
            "the official CDEII procedure and DUID-derived proxy may differ in sent-out boundaries, factor vintage, fugitive emissions, and unmapped generation",
        ],
        "official_region_days": int(len(official)),
        "paired_region_days": int(len(paired)),
        "hourly_proxy_rows": int(len(hourly)),
        "hours_per_region": int(hourly.groupby("region").size().min()),
        "minimum_hour_interval_support": float(hourly.interval_support.min()),
        "hourly_label_eligibility": "interval_support >= 0.95 and positive-SCADA-sample EF mapped_support >= 0.95; excluded hours are retained with an explicit false flag",
        "hourly_support_by_region": hourly_support,
        "max_daily_rollup_difference": rollup_diff,
        "date_min": str(paired.date.min()),
        "date_max": str(paired.date.max()),
        "overall": overall,
        "per_region": per_region,
        "input_sha256": input_hashes,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--official-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.staging, args.official_csv, args.output)
    print(json.dumps({"paired_region_days": result["paired_region_days"],
                      "overall": result["overall"], "per_region": result["per_region"]}, indent=2))


if __name__ == "__main__":
    main()
