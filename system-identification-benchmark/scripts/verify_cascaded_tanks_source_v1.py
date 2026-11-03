#!/usr/bin/env python3
"""Verify source hashes and data contracts for the Cascaded Tanks archive."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs/algorithmic-control-v1/external-source/cascaded_tanks"
MANIFEST_PATH = SOURCE_DIR / "acquisition_manifest.json"


def digest_bytes(data: bytes, algorithm: str) -> str:
    return hashlib.new(algorithm, data).hexdigest()


def verify() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest.get("doi") != "10.4121/12960104":
        raise ValueError("unexpected dataset DOI")
    if manifest.get("license") != "CC BY-SA 4.0":
        raise ValueError("dataset license does not match the reviewed record")

    archive_path = SOURCE_DIR / manifest["archive"]["path"]
    archive_bytes = archive_path.read_bytes()
    receipt = manifest["archive"]
    if len(archive_bytes) != receipt["bytes"] or len(archive_bytes) != receipt["expected_bytes_from_official_dataloader"]:
        raise ValueError("archive byte count mismatch")
    if digest_bytes(archive_bytes, "md5") != receipt["md5"]:
        raise ValueError("archive MD5 mismatch")
    if receipt["md5"] != receipt["expected_md5_from_4tu_record"]:
        raise ValueError("archive MD5 differs from the recorded 4TU value")
    archive_sha256 = digest_bytes(archive_bytes, "sha256")
    if archive_sha256 != receipt["sha256"]:
        raise ValueError("archive SHA-256 mismatch")

    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        members = archive.namelist()
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC check failed")
        for member in members:
            path = Path(member)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe ZIP member path: {member}")

        checked_members = 0
        member_bytes: dict[str, bytes] = {}
        for name, expected in manifest["verified_members"].items():
            if name not in members:
                raise ValueError(f"missing required archive member: {name}")
            data = archive.read(name)
            if len(data) != expected["bytes"] or digest_bytes(data, "sha256") != expected["sha256"]:
                raise ValueError(f"archive member hash mismatch: {name}")
            member_bytes[name] = data
            checked_members += 1

    mat = loadmat(io.BytesIO(member_bytes["CascadedTanksFiles/dataBenchmark.mat"]))
    expected_keys = ("uEst", "uVal", "yEst", "yVal")
    arrays: dict[str, np.ndarray] = {}
    for key in expected_keys:
        if key not in mat:
            raise ValueError(f"MAT file missing array {key}")
        values = np.asarray(mat[key], dtype=float).reshape(-1)
        if len(values) != 1024 or not np.isfinite(values).all():
            raise ValueError(f"{key} has unexpected length or non-finite values")
        arrays[key] = values
    sampling_seconds = float(np.asarray(mat["Ts"]).reshape(-1)[0])
    if sampling_seconds != 4.0:
        raise ValueError(f"unexpected sampling interval: {sampling_seconds}")

    table = pd.read_csv(io.BytesIO(member_bytes["CascadedTanksFiles/dataBenchmark.csv"]))
    expected_columns = ["uEst", "uVal", "yEst", "yVal", "Ts", "Unnamed: 5"]
    if table.shape != (1024, 6) or table.columns.tolist() != expected_columns:
        raise ValueError("CSV schema or row count differs from the official archive")
    max_abs_difference: dict[str, float] = {}
    csv_mat_tolerance = float(manifest["observed_data_summary"]["csv_to_mat_absolute_tolerance"])
    for key in expected_keys:
        difference = float(np.max(np.abs(table[key].to_numpy(dtype=float) - arrays[key])))
        max_abs_difference[key] = difference
        if difference > csv_mat_tolerance:
            raise ValueError(f"CSV/MAT numeric difference exceeds CSV rounding tolerance for {key}")
    recorded_csv_mat_difference = float(manifest["observed_data_summary"]["maximum_csv_to_mat_absolute_difference"])
    if not np.isclose(max(max_abs_difference.values()), recorded_csv_mat_difference, rtol=0.0, atol=1e-15):
        raise ValueError("recorded CSV/MAT maximum difference does not match the source files")
    if table["Ts"].iloc[0] != sampling_seconds or not table["Ts"].iloc[1:].isna().all():
        raise ValueError("CSV sampling interval field has an unexpected encoding")
    if not table["Unnamed: 5"].isna().all():
        raise ValueError("unexpected non-empty trailing CSV column")

    observed = manifest["observed_data_summary"]
    for key in expected_keys:
        entry = observed[key]
        values = arrays[key]
        if entry["count"] != len(values) or not np.isclose(entry["min"], values.min()) or not np.isclose(entry["max"], values.max()):
            raise ValueError(f"observed summary mismatch for {key}")
        if not np.isclose(entry["mean"], values.mean()):
            raise ValueError(f"observed mean mismatch for {key}")
        if "count_equal_10" in entry and entry["count_equal_10"] != int(np.sum(values == 10.0)):
            raise ValueError(f"top-value count mismatch for {key}")

    return {
        "status": "VERIFIED_CASCADED_TANKS_SOURCE_ARCHIVE",
        "doi": manifest["doi"],
        "license": manifest["license"],
        "archive_bytes": len(archive_bytes),
        "archive_md5": receipt["md5"],
        "archive_sha256": archive_sha256,
        "archive_members_checked": len(members),
        "key_members_hashed": checked_members,
        "mat_and_csv_arrays_checked": list(expected_keys),
        "max_csv_to_mat_absolute_difference": max_abs_difference,
        "samples_per_record": len(arrays["uEst"]),
        "sampling_interval_seconds": sampling_seconds,
        "model_scores_generated": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
