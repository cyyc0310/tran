#!/usr/bin/env python3
"""Audit archived GFS object timestamps against the frozen forecast origins."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/weather-transfer-expanded-r1"
PROTOCOL = ROOT / "docs/GFS_PREORIGIN_PUBLIC_ARCHIVE_AUDIT_PROTOCOL_20261101.md"
BUCKET_PREFIX = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def audit(input_dir: Path) -> tuple[list[dict], dict[str, str]]:
    completion_path = input_dir / "collection_complete.json"
    completion = json.loads(completion_path.read_text())
    if not completion.get("complete") or len(completion.get("windows", {})) != 24:
        raise ValueError("expected the sealed 24-window collection")
    hashes = {str(completion_path): sha(completion_path), str(PROTOCOL): sha(PROTOCOL),
              str(Path(__file__)): sha(Path(__file__))}
    rows: list[dict] = []
    for date_name, expected_manifest_hash in sorted(completion["windows"].items()):
        window_name = f"window-{date_name}"
        window = input_dir / window_name
        manifest_path = window / "manifest.json"
        if sha(manifest_path) != expected_manifest_hash:
            raise ValueError(f"window manifest seal mismatch: {window_name}")
        hashes[str(manifest_path)] = expected_manifest_hash
        manifest = json.loads(manifest_path.read_text())
        manifest_files = {record["file"]: record["sha256"] for record in manifest["files"]}
        origin = dt(manifest["forecast_origin_at"])
        initialization = dt(manifest["initialization_at"])
        if initialization != origin.replace(hour=0, minute=0, second=0, microsecond=0) - pd.Timedelta(days=1):
            raise ValueError(f"unexpected initialization/origin relation: {window_name}")
        metadata_files = sorted((window / "evidence").glob("f*_metadata.json"))
        if len(metadata_files) != 49:
            raise ValueError(f"expected 49 leads in {window_name}; found {len(metadata_files)}")
        for metadata_path in metadata_files:
            relative_metadata = str(metadata_path.relative_to(window))
            expected_metadata_hash = manifest_files.get(relative_metadata)
            if expected_metadata_hash is None or sha(metadata_path) != expected_metadata_hash:
                raise ValueError(f"metadata file is not bound to the sealed window manifest: {metadata_path}")
            metadata = json.loads(metadata_path.read_text())
            lead = int(metadata["lead_hours"])
            if lead not in range(24, 73):
                raise ValueError(f"unexpected lead {lead} in {metadata_path}")
            object_info = metadata.get("grib_object", {})
            last_modified = object_info.get("last_modified")
            url = metadata.get("source_url", "")
            if not last_modified or not url.startswith(BUCKET_PREFIX):
                raise ValueError(f"missing timestamp or public bucket URL in {metadata_path}")
            modified = dt(last_modified)
            margin = (origin - modified).total_seconds() / 3600
            if margin <= 0:
                raise ValueError(f"object timestamp is not before forecast origin: {metadata_path}")
            if not object_info.get("etag"):
                raise ValueError(f"missing S3 ETag in {metadata_path}")
            messages = metadata.get("messages", {})
            expected_variables = {"temperature_2m", "wind_u_100m", "wind_v_100m", "shortwave_surface"}
            if set(messages) != expected_variables:
                raise ValueError(f"weather variable set changed in {metadata_path}")
            rows.append({
                "window": date_name,
                "forecast_origin_at": origin.isoformat(),
                "initialization_at": initialization.isoformat(),
                "lead_hours": lead,
                "last_modified_at": modified.isoformat(),
                "hours_before_origin": margin,
                "source_url": url,
                "etag": object_info["etag"],
                "metadata_file": str(metadata_path.relative_to(ROOT)),
                "metadata_sha256": sha(metadata_path),
                "weather_variables": ",".join(sorted(messages)),
                "preorigin_public_archive_evidence": True,
            })
            hashes[str(metadata_path)] = sha(metadata_path)
    if len(rows) != 24 * 49:
        raise ValueError("unexpected total number of archived lead objects")
    return rows, hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows, hashes = audit(args.input.resolve())
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "object_timestamps.csv", index=False)
    summary = {
        "status": "PASS_PREORIGIN_PUBLIC_ARCHIVE_OBJECT_TIMESTAMPS",
        "windows": int(frame.window.nunique()),
        "objects": int(len(frame)),
        "objects_after_origin": int((frame.hours_before_origin <= 0).sum()),
        "minimum_hours_before_origin": float(frame.hours_before_origin.min()),
        "maximum_hours_before_origin": float(frame.hours_before_origin.max()),
        "all_objects_from_noaa_gfs_public_s3_bucket": bool(frame.source_url.str.startswith(BUCKET_PREFIX).all()),
        "interpretation": "Object-level evidence that archived GFS products were stored in the NOAA public S3 bucket before the frozen forecast origins; not per-client download logs, a public ACL history, or a real-time latency SLA.",
        "target_labels_read": False,
        "model_predictions_or_metrics_read": False,
        "input_sha256": hashes,
        "output_sha256": sha(output / "object_timestamps.csv"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
