#!/usr/bin/env python3
"""Run the frozen InTaSet and Nano-Drone control benchmarks from raw inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTROL_DOCS = ROOT / "docs/algorithmic-control-v1"
LOCK_PATH = ROOT / "requirements-control-benchmark-v1.lock.txt"
INTASET_MANIFEST = CONTROL_DOCS / "source_manifest.json"
INTASET_ARCHIVE = CONTROL_DOCS / "source/InTaSet.zip"
NANODRONE_MANIFEST = CONTROL_DOCS / "nanodrone_data_manifest_20261103.json"
NANODRONE_AUTHOR_ROOT = ROOT / "data/research_control_v1/external/nanodrone"
NANODRONE_DATA_ROOT = ROOT / "data/research_control_v1/external/nanodrone_lfs"
EXPECTED_NANODRONE_COMMIT = "2d921b57d166fe2debe08a5d39bd07297c5abc39"
EXPECTED_INTASET_SOURCE_MANIFEST_SHA256 = "f52712fb927870b1492190d8f163ddddb5f68f9a772c746fc3bcb325e1f73b81"
DEFAULT_OUTPUT_DIR = CONTROL_DOCS / "results/control_benchmark_v1_reproduction"
RESULTS_ROOT = CONTROL_DOCS / "results"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def locked_packages() -> tuple[str, dict[str, str]]:
    text = LOCK_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"^# Control benchmark v1 package-version lock; CPython ([0-9.]+)\.$",
        text,
        re.MULTILINE,
    )
    if match is None:
        raise ValueError("runtime lock does not declare its CPython version")
    packages: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, expected = line.partition("==")
        if not separator or not name or not expected:
            raise ValueError(f"invalid pinned requirement in runtime lock: {line}")
        packages[name] = expected
    return match.group(1), packages


def verify_runtime() -> dict[str, str]:
    python_expected, packages = locked_packages()
    python_actual = platform.python_version()
    if sys.implementation.name != "cpython":
        raise RuntimeError(f"runtime lock requires CPython, found {sys.implementation.name}")
    if python_actual != python_expected:
        raise RuntimeError(f"Python runtime {python_actual} differs from lock {python_expected}")
    actual: dict[str, str] = {}
    for name, expected in packages.items():
        try:
            found = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"locked package is missing: {name}=={expected}") from exc
        if found != expected:
            raise RuntimeError(f"package version mismatch: {name}=={found}; lock requires {expected}")
        actual[name] = found
    return actual


def verify_intaset_input() -> dict[str, Any]:
    if file_hash(INTASET_MANIFEST) != EXPECTED_INTASET_SOURCE_MANIFEST_SHA256:
        raise ValueError("InTaSet source receipt hash differs from the reviewed manifest")
    manifest = read_json(INTASET_MANIFEST)
    if INTASET_ARCHIVE.stat().st_size != int(manifest["size_bytes"]):
        raise ValueError("InTaSet archive size differs from its source receipt")
    if file_hash(INTASET_ARCHIVE, "md5") != manifest["zenodo_md5_expected"]:
        raise ValueError("InTaSet archive MD5 differs from Zenodo receipt")
    actual_sha256 = file_hash(INTASET_ARCHIVE)
    if actual_sha256 != manifest["sha256_actual"]:
        raise ValueError("InTaSet archive SHA-256 differs from source receipt")
    return {
        "path": INTASET_ARCHIVE.relative_to(ROOT).as_posix(),
        "md5": manifest["zenodo_md5_expected"],
        "sha256": actual_sha256,
        "size_bytes": INTASET_ARCHIVE.stat().st_size,
    }


def git_blob(commit: str, relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(NANODRONE_AUTHOR_ROOT), "show", f"{commit}:{relative_path}"],
        check=True,
        capture_output=True,
    ).stdout


def verify_nanodrone_inputs() -> dict[str, Any]:
    manifest = read_json(NANODRONE_MANIFEST)
    commit = manifest.get("source_commit")
    if commit != EXPECTED_NANODRONE_COMMIT:
        raise ValueError(f"Nano-Drone manifest does not pin the reviewed commit: {commit}")
    actual_head = subprocess.run(
        ["git", "-C", str(NANODRONE_AUTHOR_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_head != EXPECTED_NANODRONE_COMMIT:
        raise ValueError(f"Nano-Drone author checkout is {actual_head}, expected {EXPECTED_NANODRONE_COMMIT}")

    expected_columns = manifest["observed"].get("columns")
    if not isinstance(expected_columns, list) or not expected_columns:
        raise ValueError("Nano-Drone manifest has no expected CSV schema")
    data_root = NANODRONE_DATA_ROOT.resolve(strict=True)
    if not data_root.is_relative_to(ROOT):
        raise ValueError("Nano-Drone data directory resolves outside the repository")
    observed_rows = {"train": 0, "test": 0}
    observed_files = {"train": 0, "test": 0}
    seen: set[str] = set()
    input_files = []
    for entry in manifest["files"]:
        relative = entry["path"]
        relative_path = Path(relative)
        split = entry["split"]
        if (relative in seen or relative_path.is_absolute() or ".." in relative_path.parts
                or "\\" in relative or len(relative_path.parts) != 3
                or relative_path.parts[0] != "data" or relative_path.parts[1] != split
                or split not in observed_rows or relative_path.suffix.lower() != ".csv"):
            raise ValueError(f"unsafe or duplicate Nano-Drone path: {relative}")
        seen.add(relative)

        pointer = git_blob(EXPECTED_NANODRONE_COMMIT, relative).decode("utf-8")
        oid_match = re.search(r"^oid sha256:([0-9a-f]{64})$", pointer, re.MULTILINE)
        size_match = re.search(r"^size ([0-9]+)$", pointer, re.MULTILINE)
        expected_sha = entry["lfs_sha256"]
        expected_size = int(entry["size_bytes"])
        if (oid_match is None or size_match is None or oid_match.group(1) != expected_sha
                or int(size_match.group(1)) != expected_size or not SHA256_RE.fullmatch(expected_sha)):
            raise ValueError(f"Nano-Drone LFS receipt differs from pinned pointer: {relative}")

        raw_path = NANODRONE_DATA_ROOT / relative
        resolved = raw_path.resolve(strict=True)
        if not resolved.is_relative_to(data_root):
            raise ValueError(f"Nano-Drone data path escapes its data directory: {relative}")
        if raw_path.is_symlink() or resolved.stat().st_size != expected_size:
            raise ValueError(f"Nano-Drone data object is a symlink or has wrong size: {relative}")
        actual_sha = file_hash(resolved)
        if actual_sha != expected_sha:
            raise ValueError(f"Nano-Drone raw data hash mismatch: {relative}")

        with resolved.open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if header != expected_columns:
                raise ValueError(f"Nano-Drone CSV columns differ from manifest: {relative}")
            rows = 0
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(expected_columns):
                    raise ValueError(f"wrong field count in {relative}:{line_number}")
                rows += 1
        if rows != int(entry["rows"]):
            raise ValueError(f"Nano-Drone row count differs from manifest: {relative}")
        observed_rows[split] += rows
        observed_files[split] += 1
        input_files.append({"path": relative, "sha256": actual_sha, "size_bytes": expected_size, "rows": rows})

    if observed_files != {"train": 12, "test": 3}:
        raise ValueError(f"unexpected Nano-Drone file split: {observed_files}")
    for split in ("train", "test"):
        if observed_rows[split] != int(manifest["observed"][f"{split}_rows"]):
            raise ValueError(f"Nano-Drone {split} row total differs from manifest")
    return {
        "manifest_sha256": file_hash(NANODRONE_MANIFEST),
        "source_commit": commit,
        "csv_files": input_files,
        "rows": observed_rows,
    }


def run_child(name: str, command: list[str], log_dir: Path) -> None:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    (log_dir / f"{name}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (log_dir / f"{name}.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, command)
    print("\n".join(result.stdout.rstrip().splitlines()[-12:]))
    if result.stderr:
        sys.stderr.write(result.stderr)


def expected_script_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/run_intaset_aligned_budget_v1.py",
        ROOT / "scripts/intaset_causal_transfer_benchmark_v1.py",
        ROOT / "scripts/intaset_sparse_target_calibration_v1.py",
        ROOT / "scripts/intaset_neural_ode_transfer_benchmark_v1.py",
        ROOT / "scripts/nanodrone_external_control_benchmark_v1.py",
    )
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in paths}


def resolve_run_directory(path: Path, *, must_exist: bool = False) -> Path:
    path = path.expanduser()
    candidate = path if path.is_absolute() else ROOT / path
    results_root = RESULTS_ROOT.resolve(strict=True)
    try:
        relative = candidate.relative_to(results_root)
    except ValueError as exc:
        raise ValueError(f"run directory must be under {results_root}: {path}") from exc
    if ".." in relative.parts:
        raise ValueError(f"run directory must not contain '..': {path}")
    current = results_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"run directory uses a symbolic link: {current}")
    resolved = candidate.resolve(strict=must_exist)
    if not resolved.is_relative_to(results_root):
        raise ValueError(f"resolved run directory escapes {results_root}: {path}")
    return resolved


def reject_symlink_tree(root: Path) -> None:
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(parent) / name
            if path.is_symlink():
                raise ValueError(f"run output contains a symbolic link: {path}")


def verify_saved_run(output_dir: Path) -> dict[str, Any]:
    output_dir = resolve_run_directory(output_dir, must_exist=True)
    manifest_path = output_dir / "control_benchmark_run_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("run manifest is missing or is a symbolic link")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "RAW_TO_RESULTS_REPRODUCTION_COMPLETE":
        raise ValueError("saved run manifest has an unexpected status")

    runtime = verify_runtime()
    if manifest.get("runtime", {}).get("locked_packages") != runtime:
        raise ValueError("saved run package versions differ from the locked environment")
    if manifest.get("runtime", {}).get("lock_sha256") != file_hash(LOCK_PATH):
        raise ValueError("saved run references a different runtime lock")
    if manifest.get("raw_inputs", {}).get("intaset") != verify_intaset_input():
        raise ValueError("saved run InTaSet input receipt differs from current raw data")
    if manifest.get("raw_inputs", {}).get("nanodrone") != verify_nanodrone_inputs():
        raise ValueError("saved run Nano-Drone input receipt differs from current raw data")
    if manifest.get("script_sha256") != expected_script_hashes():
        raise ValueError("saved run script hashes differ from current benchmark scripts")

    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("saved run manifest has no output hashes")
    reject_symlink_tree(output_dir)
    actual_paths = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_paths != set(outputs):
        raise ValueError("saved run output file set differs from its hash manifest")
    for relative, expected_sha in outputs.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe saved run output path: {relative}")
        path = output_dir / relative_path
        cursor = output_dir
        for part in relative_path.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError(f"saved run output uses a symbolic link: {relative}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(output_dir):
            raise ValueError(f"saved run output escapes its directory: {relative}")
        if file_hash(resolved) != expected_sha:
            raise ValueError(f"saved run output hash mismatch: {relative}")
    return {
        "status": "SAVED_CONTROL_BENCHMARK_RUN_VERIFIED",
        "output_dir": str(output_dir),
        "output_files_verified": len(outputs),
        "raw_files_verified": 1 + len(manifest["raw_inputs"]["nanodrone"]["csv_files"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="20261102,20261103,20261104")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--verify-only", action="store_true", help="verify a completed run without executing models")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_saved_run(args.output_dir), ensure_ascii=False, indent=2))
        return
    if args.threads < 1:
        parser.error("--threads must be a positive integer")
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        parser.error("--seeds must contain at least one integer")

    runtime = verify_runtime()
    intaset_input = verify_intaset_input()
    nanodrone_input = verify_nanodrone_inputs()

    try:
        output_dir = resolve_run_directory(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is non-empty; choose a fresh path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir()
    intaset_output = output_dir / "intaset"
    nanodrone_output = output_dir / "nanodrone_external_v1"

    intaset_command = [
        sys.executable,
        str(ROOT / "scripts/run_intaset_aligned_budget_v1.py"),
        "--output-dir", str(intaset_output),
        "--seeds", ",".join(str(seed) for seed in seeds),
    ]
    run_child("intaset", intaset_command, logs_dir)

    nanodrone_command = [
        sys.executable,
        str(ROOT / "scripts/nanodrone_external_control_benchmark_v1.py"),
        "--author-root", str(NANODRONE_AUTHOR_ROOT),
        "--data-root", str(NANODRONE_DATA_ROOT),
        "--manifest", str(NANODRONE_MANIFEST),
        "--out-dir", str(nanodrone_output),
        "--threads", str(args.threads),
    ]
    run_child("nanodrone", nanodrone_command, logs_dir)

    master_summary = intaset_output / "intaset_aligned_master_summary.csv"
    with master_summary.open(encoding="utf-8-sig", newline="") as stream:
        master_rows = list(csv.DictReader(stream))
    if len(master_rows) != 148 or {row["condition"] for row in master_rows} != {"M1->M2", "M2->M1"}:
        raise ValueError("InTaSet replay summary does not contain the expected 148 bidirectional rows")
    if not (nanodrone_output / "nanodrone_metrics_by_run.csv").is_file():
        raise ValueError("Nano-Drone runner did not produce per-run scores")

    reject_symlink_tree(output_dir)
    output_hashes = {
        path.relative_to(output_dir).as_posix(): file_hash(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "status": "RAW_TO_RESULTS_REPRODUCTION_COMPLETE",
        "scope": "InTaSet M1<->M2 and Nano-Drone raw-to-result replay; outputs are conditional on recorded future inputs",
        "configuration": {
            "intaset_seeds": list(seeds),
            "state_observation_intervals_seconds": [0.5, 1.0],
            "target_calibration_budgets_seconds": [0, 1, 3, 5],
            "nanodrone_target_observation_step_seconds": 0.5,
            "nanodrone_scoring_horizons_steps": [1, 10, 50],
            "threads": args.threads,
        },
        "runtime": {
            "python": platform.python_version(),
            "locked_packages": runtime,
            "lock_sha256": file_hash(LOCK_PATH),
        },
        "raw_inputs": {"intaset": intaset_input, "nanodrone": nanodrone_input},
        "script_sha256": expected_script_hashes(),
        "outputs_sha256": output_hashes,
    }
    manifest_path = output_dir / "control_benchmark_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "output_dir": str(output_dir),
        "input_raw_files": 1 + len(nanodrone_input["csv_files"]),
        "output_files_hashed": len(output_hashes),
        "intaset_rows": len(master_rows),
        "nanodrone_train_rows": nanodrone_input["rows"]["train"],
        "nanodrone_test_rows": nanodrone_input["rows"]["test"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
