#!/usr/bin/env python3
"""Score sealed US/GB-to-Taiwan predictions against public Q2 2026 unit output."""
from __future__ import annotations

import gzip
import hashlib
import json
from json import JSONDecodeError, JSONDecoder
from pathlib import Path
import re
import sys
from typing import Iterator

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transcif.research import cross_country as source  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402

PREDICTION_DIR = ROOT / "results/taipower-cn5-external-transfer-r1"
PREDICTION_FILE = PREDICTION_DIR / "sealed_predictions.csv.gz"
PREDICTION_SEAL = PREDICTION_DIR / "prediction_seal.json"
PROTOCOL = ROOT / "docs/TAIPOWER_CN5_EXTERNAL_TRANSFER_PROTOCOL_20261102.md"
INPUT_DIR = ROOT / "data/taipower-public-hourly-target-r1"
INPUT_CSV = INPUT_DIR / "target_monthly_cn5_inputs.csv"
MONTHLY_API = INPUT_DIR / "monthly_generation_api_snapshot.json"
TARGET_URL = "https://service.taipower.com.tw/data/opendata/apply/file/d006010/001.json"
TARGET_RAW_GZIP = INPUT_DIR / "target_generation_10min_snapshot.json.gz"
OUTPUT_DIR = PREDICTION_DIR / "scored"
METHODS = (*model.METHODS, "fuelshare_history")
SOURCE_CONFIGS = ("US-only", "GB-only", "US+GB")
FINE_COLUMNS = list(model.FINE)
SCORE_COLUMNS = [name for name in FINE_COLUMNS if name != "other_renewable"]
FINE_FACTORS = dict(zip(model.FINE, model.FACTORS.tolist()))
COARSE_TO_FINE = {
    "thermal": ("coal", "gas", "oil", "other"),
    "hydro": ("hydro",),
    "nuclear": ("nuclear",),
    "wind": ("wind",),
    "solar": ("solar",),
}
EXPECTED_STAMPS = pd.date_range(
    "2026-04-01 00:00", "2026-06-30 23:50", freq="10min"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_fuel(value: object) -> str | None:
    """Map only explicitly recognized fuel labels; ambiguous types stay unknown."""
    text = str(value).strip().lower().replace(" ", "")
    if any(token in text for token in ("抽蓄", "抽水蓄能", "pumped")):
        return "pumped_storage"
    if any(token in text for token in ("燃煤", "煤炭", "煤電", "coal")):
        return "coal"
    if any(token in text for token in ("燃氣", "燃气", "天然氣", "天然气", "lng", "naturalgas")):
        return "gas"
    if any(token in text for token in (
        "燃油", "燃料油", "重油", "輕油", "轻油", "柴油", "oil-fired", "heavyoil"
    )):
        return "oil"
    if any(token in text for token in ("核能", "核電", "核电", "nuclear")):
        return "nuclear"
    if any(token in text for token in ("水力", "水電", "水电", "hydro")):
        return "hydro"
    if any(token in text for token in ("風力", "风力", "wind")):
        return "wind"
    if any(token in text for token in ("太陽", "太阳", "光電", "光电", "光伏", "solar")):
        return "solar"
    if any(token in text for token in ("地熱", "地热", "geothermal")):
        return "other_renewable"
    if any(token in text for token in (
        "生質", "生质", "廢棄物", "废弃物", "biomass", "waste",
        "其他燃燒", "其他燃烧", "otherconventional"
    )):
        return "other"
    return None


def download_target_snapshot() -> dict:
    response = requests.get(TARGET_URL, stream=True, timeout=(30, 300))
    response.raise_for_status()
    raw_hash = hashlib.sha256()
    raw_bytes = 0
    with gzip.open(TARGET_RAW_GZIP, "wb", compresslevel=6) as compressed:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if not chunk:
                continue
            raw_hash.update(chunk)
            raw_bytes += len(chunk)
            compressed.write(chunk)
    return {
        "url": TARGET_URL,
        "http_status": response.status_code,
        "headers": {key.lower(): value for key, value in response.headers.items()},
        "uncompressed_bytes": raw_bytes,
        "uncompressed_sha256": raw_hash.hexdigest(),
        "stored_gzip_sha256": sha256(TARGET_RAW_GZIP),
        "stored_gzip_bytes": TARGET_RAW_GZIP.stat().st_size,
        "retrieved_after_prediction_seal": True,
    }


def iter_netp_records(path: Path) -> Iterator[dict]:
    """Stream NET_P JSON objects without materializing the 194 MB array."""
    decoder = JSONDecoder()
    marker = re.compile(r'"NET_P"\s*:\s*\[')
    with gzip.open(path, "rt", encoding="utf-8-sig") as stream:
        buffer = ""
        started = False
        eof = False
        while True:
            if not started:
                match = marker.search(buffer)
                if match:
                    buffer = buffer[match.end():]
                    started = True
                elif eof:
                    raise ValueError("NET_P array not found in target JSON")
                else:
                    chunk = stream.read(1 << 20)
                    if not chunk:
                        eof = True
                    buffer += chunk
                    continue

            buffer = buffer.lstrip()
            if not buffer:
                chunk = stream.read(1 << 20)
                if not chunk:
                    raise ValueError("unexpected end of target JSON array")
                buffer = chunk
                continue
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            if buffer.startswith("]"):
                return
            try:
                record, end = decoder.raw_decode(buffer)
            except JSONDecodeError:
                chunk = stream.read(1 << 20)
                if not chunk:
                    raise ValueError("truncated target JSON record")
                buffer += chunk
                continue
            if not isinstance(record, dict):
                raise ValueError("NET_P array contained a non-object record")
            yield record
            buffer = buffer[end:]


def parse_target_hourly(path: Path) -> tuple[pd.DataFrame, dict]:
    by_stamp: dict[pd.Timestamp, np.ndarray] = {}
    stamp_record_counts: dict[pd.Timestamp, int] = {}
    fuel_counts: dict[str, int] = {}
    unknown_positive_mw: dict[str, float] = {}
    unknown_positive_rows: dict[str, int] = {}
    negative_rows: dict[str, int] = {}
    negative_mw: dict[str, float] = {}
    negative_unknown_rows: dict[str, int] = {}
    negative_unknown_mw: dict[str, float] = {}
    invalid_value_rows = 0
    invalid_value_stamps: set[pd.Timestamp] = set()
    excluded_storage_rows = 0
    excluded_storage_mw = 0.0
    total_rows = 0

    for record in iter_netp_records(path):
        total_rows += 1
        stamp = pd.Timestamp(record["DATETIME"])
        if stamp not in EXPECTED_STAMPS:
            continue
        fuel_type = str(record.get("FUEL_TYPE", "")).strip()
        raw_value = record.get("NET_P")
        try:
            value = float(str(raw_value).replace(",", ""))
        except (TypeError, ValueError):
            value = float("nan")
        if raw_value is None or not str(raw_value).strip() or not np.isfinite(value):
            invalid_value_rows += 1
            invalid_value_stamps.add(stamp)
            continue
        mapped = classify_fuel(fuel_type)
        fuel_counts[fuel_type] = fuel_counts.get(fuel_type, 0) + 1
        stamp_record_counts[stamp] = stamp_record_counts.get(stamp, 0) + 1
        if mapped == "pumped_storage":
            excluded_storage_rows += 1
            excluded_storage_mw += max(value, 0.0)
            continue
        if mapped is None:
            if value > 0:
                unknown_positive_mw[fuel_type] = unknown_positive_mw.get(fuel_type, 0.0) + value
                unknown_positive_rows[fuel_type] = unknown_positive_rows.get(fuel_type, 0) + 1
            if value < 0:
                negative_unknown_rows[fuel_type] = negative_unknown_rows.get(fuel_type, 0) + 1
                negative_unknown_mw[fuel_type] = negative_unknown_mw.get(fuel_type, 0.0) + value
            continue
        if value < 0:
            negative_rows[mapped] = negative_rows.get(mapped, 0) + 1
            negative_mw[mapped] = negative_mw.get(mapped, 0.0) + value
        values = by_stamp.setdefault(stamp, np.zeros(len(FINE_COLUMNS), dtype=float))
        values[FINE_COLUMNS.index(mapped)] += value

    rows = [
        {"clock": stamp, **dict(zip(FINE_COLUMNS, values))}
        for stamp, values in by_stamp.items()
    ]
    if not rows:
        raise ValueError("no target generation rows found for frozen Q2 2026 period")
    ten_minute = pd.DataFrame(rows).set_index("clock").sort_index()
    ten_minute = ten_minute.reindex(EXPECTED_STAMPS, fill_value=0.0)
    stamp_complete = pd.Series(
        EXPECTED_STAMPS.isin(stamp_record_counts)
        & ~EXPECTED_STAMPS.isin(invalid_value_stamps),
        index=EXPECTED_STAMPS,
    )
    hourly = ten_minute.resample("1h").mean()
    stamp_counts = stamp_complete.astype(int).resample("1h").sum()
    hourly["complete_six_slots"] = stamp_counts.eq(6)
    hourly = hourly.reset_index(names="clock")

    positive_mapped = float(ten_minute.clip(lower=0).sum().sum())
    unknown_positive = float(sum(unknown_positive_mw.values()))
    total_known_or_unknown = positive_mapped + unknown_positive
    mapped_support = (positive_mapped / total_known_or_unknown
                      if total_known_or_unknown > 0 else 0.0)
    audit = {
        "target_period": "2026-04-01..2026-06-30 Asia/Taipei",
        "raw_records_total": total_rows,
        "expected_10min_stamps": int(len(EXPECTED_STAMPS)),
        "observed_10min_stamps": int(len(stamp_record_counts)),
        "complete_hour_rows": int(hourly.complete_six_slots.sum()),
        "expected_hour_rows": 2184,
        "fuel_type_record_counts": fuel_counts,
        "unknown_positive_rows_by_type": unknown_positive_rows,
        "unknown_positive_mw_by_type": unknown_positive_mw,
        "mapped_positive_mw": positive_mapped,
        "unknown_positive_mw": unknown_positive,
        "mapped_generation_support_fraction": mapped_support,
        "negative_rows_by_mapped_type": negative_rows,
        "negative_mw_by_mapped_type": negative_mw,
        "negative_rows_by_unknown_type": negative_unknown_rows,
        "negative_mw_by_unknown_type": negative_unknown_mw,
        "invalid_net_p_rows": invalid_value_rows,
        "pumped_storage_excluded_rows": excluded_storage_rows,
        "pumped_storage_positive_mw_excluded": excluded_storage_mw,
    }
    return hourly, audit


def scope_audit(hourly: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    raw = json.loads(MONTHLY_API.read_text(encoding="utf-8"))
    nationwide = raw["全國"]
    rows = {}
    for record in nationwide:
        period = record.get("Column29")
        if not isinstance(period, str):
            continue
        if not re.fullmatch(r"20\d{2}/\d{2}", period):
            continue
        try:
            coal, oil, gas = (float(record[column])
                              for column in ("Column7", "Column9", "Column11"))
            hydro, nuclear = float(record["Column17"]), float(record["Column13"])
            solar, wind = float(record["Column21"]), float(record["Column23"])
            biomass, waste = float(record["Column25"]), float(record["Column27"])
        except (KeyError, TypeError, ValueError):
            continue
        rows[period] = {
            "thermal": coal + oil + gas + biomass + waste,
            "hydro": hydro,
            "nuclear": nuclear,
            "wind": wind,
            "solar": solar,
        }

    observed = hourly.copy()
    observed["month"] = pd.to_datetime(observed.clock).dt.strftime("%Y/%m")
    observed["thermal"] = observed[["coal", "gas", "oil", "other"]].sum(axis=1)
    grouped = observed.groupby("month")[[*model.GROUPS]].sum()
    audit_rows = []
    passed = True
    for month, group in grouped.iterrows():
        source_month = month.replace("-", "/")
        official_values = rows[source_month]
        observed_total = float(group.sum()) / 1000.0
        reference_total = float(sum(official_values.values()))
        total_gap = abs(observed_total - reference_total) / max(reference_total, 1e-9)
        group_gaps = {}
        for name in model.GROUPS:
            official_value = official_values[name]
            measured_gwh = float(group[name]) / 1000.0
            group_gaps[name] = (
                0.0 if official_value == 0 and measured_gwh == 0
                else abs(measured_gwh - official_value) / max(abs(official_value), 1e-9)
            )
            if group_gaps[name] > 0.10:
                passed = False
        if total_gap > 0.10:
            passed = False
        audit_rows.append({
            "month": month,
            "official_nationwide_total_GWh": float(sum(official_values.values())),
            "hourly_mapped_total_GWh": observed_total,
            "hourly_to_official_total_relative_gap": total_gap,
            **{f"official_{name}_GWh": official_values[name] for name in model.GROUPS},
            **{f"hourly_{name}_GWh": float(group[name]) / 1000.0 for name in model.GROUPS},
            **{f"hourly_to_official_{name}_relative_gap": value
               for name, value in group_gaps.items()},
        })
    return pd.DataFrame(audit_rows), passed


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite score output: {OUTPUT_DIR}")
    seal = json.loads(PREDICTION_SEAL.read_text(encoding="utf-8"))
    if sha256(PREDICTION_FILE) != seal["prediction_sha256"]:
        raise ValueError("sealed prediction hash mismatch")
    if sha256(PROTOCOL) != seal["protocol_sha256"]:
        raise ValueError("frozen protocol changed since prediction seal")
    if seal["prediction_rows"] != 6552 or seal["target_hours_per_source_config"] != 2184:
        raise ValueError("unexpected frozen target prediction support")

    predictions = pd.read_csv(PREDICTION_FILE, compression="gzip")
    predictions["time_local"] = pd.to_datetime(predictions.time_local, errors="raise")
    expected_times = pd.DatetimeIndex(
        pd.date_range("2026-04-01 00:00", "2026-06-30 23:00", freq="h")
    )
    if set(predictions.source.unique()) != set(SOURCE_CONFIGS):
        raise ValueError("sealed predictions do not contain the exact source configurations")
    if predictions.duplicated(["source", "time_local"]).any():
        raise ValueError("sealed predictions contain duplicate source/time keys")
    for source_config in SOURCE_CONFIGS:
        source_times = pd.DatetimeIndex(
            predictions.loc[predictions.source.eq(source_config), "time_local"]
        ).sort_values()
        if not source_times.equals(expected_times):
            raise ValueError(f"sealed prediction grid is incomplete for {source_config}")

    snapshot = download_target_snapshot()
    hourly, response_audit = parse_target_hourly(TARGET_RAW_GZIP)
    hourly["clock"] = pd.to_datetime(hourly.clock, errors="raise")
    joined = predictions.merge(
        hourly,
        left_on="time_local",
        right_on="clock",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    joined = joined.loc[joined._merge.eq("both")].copy()
    joined["mapped_total"] = joined[SCORE_COLUMNS].sum(axis=1)
    joined["nonnegative_fuel_support"] = joined[FINE_COLUMNS].ge(0).all(axis=1)
    joined["y_fixed_ef_proxy"] = (
        joined[SCORE_COLUMNS].to_numpy(float)
        @ np.asarray([FINE_FACTORS[name] for name in SCORE_COLUMNS])
        / joined["mapped_total"].replace(0, np.nan).to_numpy(float)
    )
    scored_support = joined.loc[
        joined.complete_six_slots.astype(bool)
        & joined.nonnegative_fuel_support
        & joined.mapped_total.gt(0)
    ].copy()

    scope_metrics, scope_pass = scope_audit(hourly)
    complete_hour_fraction = float(hourly.complete_six_slots.mean())
    mapping_pass = (
        response_audit["mapped_generation_support_fraction"] >= 0.99
    )
    expected_by_month = pd.Series(expected_times).groupby(
        pd.Series(expected_times).dt.strftime("%Y-%m")
    ).size()
    valid_hourly = hourly.loc[
        hourly.complete_six_slots.astype(bool)
        & hourly[[*FINE_COLUMNS]].ge(0).all(axis=1)
        & hourly[SCORE_COLUMNS].sum(axis=1).gt(0)
    ].copy()
    valid_hourly["month"] = valid_hourly.clock.dt.strftime("%Y-%m")
    valid_by_month = valid_hourly.groupby("month").size()
    monthly_support = {
        month: {
            "valid_hours": int(valid_by_month.get(month, 0)),
            "expected_hours": int(expected),
            "valid_fraction": float(valid_by_month.get(month, 0) / expected),
        }
        for month, expected in expected_by_month.items()
    }
    scored_support["month"] = scored_support.time_local.dt.strftime("%Y-%m")
    per_source_month_counts = scored_support.groupby(["source", "month"]).size()
    score_grid_pass = all(
        per_source_month_counts.get((source_config, month), 0)
        >= 0.99 * expected
        for source_config in SOURCE_CONFIGS
        for month, expected in expected_by_month.items()
    )
    support_pass = (
        all(item["valid_fraction"] >= 0.99 for item in monthly_support.values())
        and score_grid_pass
    )
    quality_pass = mapping_pass and support_pass and scope_pass

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    scope_metrics.to_csv(OUTPUT_DIR / "monthly_scope_audit.csv", index=False)
    score_rows = []
    if quality_pass:
        for (source_config, month), group in scored_support.groupby(
            ["source", "month"], sort=True
        ):
            inputs = pd.DataFrame({"clock": group.time_local})
            y = group.y_fixed_ef_proxy.to_numpy(float)
            for method in METHODS:
                metrics = source.metrics(
                    inputs,
                    y,
                    group[method].to_numpy(float),
                    float(group[f"{method}_bias"].iloc[0]),
                    float(group[f"{method}_sigma"].iloc[0]),
                )
                score_rows.append({
                    "source": source_config,
                    "month": month,
                    "method": method,
                    "scored_hours": len(group),
                    **metrics,
                })
    monthly_metrics = pd.DataFrame(score_rows)
    monthly_metrics.to_csv(OUTPUT_DIR / "monthly_metrics.csv", index=False)
    if quality_pass:
        pooled_metrics = []
        for source_config in SOURCE_CONFIGS:
            for method in METHODS:
                group = scored_support.loc[scored_support.source.eq(source_config)]
                inputs = pd.DataFrame({"clock": group.time_local})
                result = source.metrics(
                    inputs,
                    group.y_fixed_ef_proxy.to_numpy(float),
                    group[method].to_numpy(float),
                    float(group[f"{method}_bias"].iloc[0]),
                    float(group[f"{method}_sigma"].iloc[0]),
                )
                pooled_metrics.append({
                    "source": source_config,
                    "method": method,
                    "scored_hours": len(group),
                    **result,
                })
        pd.DataFrame(pooled_metrics).to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
    else:
        pd.DataFrame(columns=["source", "method", "scored_hours"]).to_csv(
            OUTPUT_DIR / "summary_metrics.csv", index=False
        )

    # Keep the joined labels as a separate post-seal artifact for auditability.
    joined[["time_local", "source", *FINE_COLUMNS, "y_fixed_ef_proxy",
            "complete_six_slots"]].to_csv(
        OUTPUT_DIR / "target_fixed_ef_labels.csv.gz", index=False, compression="gzip"
    )
    status = {
        "status": "SCORED_AFTER_SEAL" if quality_pass else "RESPONSE_OPENED_QUALITY_GATE_FAILED",
        "target_estimand": "generation-side fixed-EF proxy, not official/measured CIF",
        "target_response_sample_values_opened_before_seal": True,
        "full_target_response_series_opened_after_seal": True,
        "target_hours_expected_per_source": seal["target_hours_per_source_config"],
        "prediction_rows": seal["prediction_rows"],
        "joined_prediction_rows": int(len(joined)),
        "scoreable_rows": int(len(scored_support)),
        "complete_hour_fraction": complete_hour_fraction,
        "monthly_hour_support": monthly_support,
        "score_grid_pass": bool(score_grid_pass),
        "mapped_generation_support_fraction": response_audit["mapped_generation_support_fraction"],
        "unknown_positive_mw_by_type": response_audit["unknown_positive_mw_by_type"],
        "negative_mapped_rows": sum(response_audit["negative_rows_by_mapped_type"].values()),
        "negative_unknown_rows": sum(response_audit["negative_rows_by_unknown_type"].values()),
        "scope_match_pass": bool(scope_pass),
        "mapping_pass": bool(mapping_pass),
        "support_pass": bool(support_pass),
        "quality_gate_pass": bool(quality_pass),
        "response_source": snapshot,
        "response_audit": response_audit,
        "warning": (
            "A quality-gate failure forbids a cross-domain accuracy conclusion; any stored metrics are omitted."
            if not quality_pass else
            "A passing quarter is a frozen external-system stress test, not a blind confirmation or China validation."
        ),
    }
    (OUTPUT_DIR / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    files = ["monthly_metrics.csv", "summary_metrics.csv", "monthly_scope_audit.csv",
             "target_fixed_ef_labels.csv.gz", "status.json"]
    score_seal = {
        "status": status["status"],
        "prediction_sha256": seal["prediction_sha256"],
        "prediction_seal_sha256": sha256(PREDICTION_SEAL),
        "protocol_sha256": seal["protocol_sha256"],
        "target_response_sha256": snapshot["uncompressed_sha256"],
        "target_response_gzip_sha256": snapshot["stored_gzip_sha256"],
        "scoring_code_sha256": sha256(Path(__file__).resolve()),
        "quality_gate_pass": bool(quality_pass),
        "score_files_sha256": {name: sha256(OUTPUT_DIR / name) for name in files},
    }
    (OUTPUT_DIR / "score_seal.json").write_text(
        json.dumps(score_seal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": status["status"],
        "scope_match_pass": bool(scope_pass),
        "mapping_pass": bool(mapping_pass),
        "complete_hour_fraction": complete_hour_fraction,
        "mapped_generation_support_fraction": response_audit["mapped_generation_support_fraction"],
        "scored_rows": int(len(scored_support)) if quality_pass else 0,
        "score_seal_sha256": sha256(OUTPUT_DIR / "score_seal.json"),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
