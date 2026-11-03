#!/usr/bin/env python3
"""Fetch and verify the public raw inputs for the control benchmark v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CONTROL_DOCS = ROOT / "docs/algorithmic-control-v1"
INTASET_MANIFEST = CONTROL_DOCS / "source_manifest.json"
NANODRONE_MANIFEST = CONTROL_DOCS / "nanodrone_data_manifest_20261103.json"
EXPECTED_INTASET_MANIFEST_SHA256 = "f52712fb927870b1492190d8f163ddddb5f68f9a772c746fc3bcb325e1f73b81"
EXPECTED_NANODRONE_MANIFEST_SHA256 = "3451d28b429e8b24f99c8d7918b272a439f13bfd09ae133f9d1bea41bce01caf"
EXPECTED_NANODRONE_COMMIT = "2d921b57d166fe2debe08a5d39bd07297c5abc39"
NANODRONE_REPOSITORY = "https://github.com/idsia-robotics/nanodrone-sysid-benchmark.git"
ZENODO_FILE_URL = "https://zenodo.org/records/17652851/files/InTaSet.zip?download=1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MD5_RE = re.compile(r"^[0-9a-f]{32}$")
LFS_OID_RE = re.compile(r"^oid sha256:([0-9a-f]{64})$", re.MULTILINE)
LFS_SIZE_RE = re.compile(r"^size ([0-9]+)$", re.MULTILINE)


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative_path(raw: str) -> Path:
    relative = Path(raw)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in raw
        or not relative.parts
    ):
        raise ValueError(f"unsafe relative data path: {raw}")
    return relative


def git_output(author_root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(author_root), *arguments],
        check=True,
        capture_output=True,
        text=not binary,
    )
    return result.stdout


def ensure_author_checkout(author_root: Path, *, fetch: bool) -> None:
    if author_root.exists():
        if not (author_root / ".git").exists():
            raise RuntimeError(
                f"author checkout exists but is not a Git repository: {author_root}"
            )
        actual = str(git_output(author_root, "rev-parse", "HEAD")).strip()
        if actual != EXPECTED_NANODRONE_COMMIT:
            raise RuntimeError(
                f"Nano-Drone checkout is {actual}, expected {EXPECTED_NANODRONE_COMMIT}; "
                "use a fresh --workspace-root or correct the checkout manually"
            )
        return

    if not fetch:
        raise FileNotFoundError(
            f"missing author checkout: {author_root}; rerun without --verify-only"
        )

    author_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="nanodrone-author-", dir=author_root.parent
    ) as temporary:
        candidate = Path(temporary) / "checkout"
        candidate.mkdir()
        env = os.environ.copy()
        # The CSV objects are downloaded and checked separately from Git LFS.
        # Keeping the pointer files in the author checkout lets the runner
        # verify them without requiring the Git LFS client.
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        subprocess.run(["git", "init", str(candidate)], check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "-C", str(candidate), "remote", "add", "origin", NANODRONE_REPOSITORY],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "git", "-C", str(candidate), "fetch", "--depth=1", "origin",
                EXPECTED_NANODRONE_COMMIT,
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        subprocess.run(
            ["git", "-C", str(candidate), "checkout", "--detach", "FETCH_HEAD"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        actual = str(git_output(candidate, "rev-parse", "HEAD")).strip()
        if actual != EXPECTED_NANODRONE_COMMIT:
            raise RuntimeError(f"fetched unexpected Nano-Drone commit: {actual}")
        os.replace(candidate, author_root)


def read_lfs_pointer(author_root: Path, relative: Path) -> tuple[bytes, str, int]:
    pointer = git_output(
        author_root,
        "show",
        f"{EXPECTED_NANODRONE_COMMIT}:{relative.as_posix()}",
        binary=True,
    )
    assert isinstance(pointer, bytes)
    text = pointer.decode("utf-8")
    oid_match = LFS_OID_RE.search(text)
    size_match = LFS_SIZE_RE.search(text)
    if oid_match is None or size_match is None:
        raise ValueError(f"pinned author commit has no valid Git LFS pointer: {relative}")
    local_pointer = author_root / relative
    if local_pointer.is_symlink() or not local_pointer.is_file():
        raise ValueError(f"author checkout is missing the expected LFS pointer: {relative}")
    if local_pointer.read_bytes() != pointer:
        raise ValueError(f"checked-out LFS pointer differs from pinned Git blob: {relative}")
    return pointer, oid_match.group(1), int(size_match.group(1))


def verify_file(path: Path, *, size_bytes: int, sha256: str, md5: str | None = None) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    if path.stat().st_size != size_bytes or file_hash(path) != sha256:
        return False
    return md5 is None or file_hash(path, "md5") == md5


def download_verified(
    url: str,
    destination: Path,
    *,
    size_bytes: int,
    sha256: str,
    md5: str | None = None,
) -> str:
    if verify_file(destination, size_bytes=size_bytes, sha256=sha256, md5=md5):
        return "already_verified"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".part",
                dir=destination.parent,
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                request = Request(
                    url,
                    headers={"User-Agent": "public-control-benchmark-v1/1.0"},
                )
                with urlopen(request, timeout=90) as response:
                    if response.status != 200:
                        raise RuntimeError(f"download returned HTTP {response.status}: {url}")
                    received = 0
                    for block in iter(lambda: response.read(1 << 20), b""):
                        received += len(block)
                        if received > size_bytes:
                            raise ValueError(f"download exceeds expected size: {url}")
                        output.write(block)
                output.flush()
                os.fsync(output.fileno())

            if not verify_file(
                temporary_path, size_bytes=size_bytes, sha256=sha256, md5=md5
            ):
                raise ValueError(
                    f"downloaded bytes do not match the pinned receipt for {destination}"
                )
            os.replace(temporary_path, destination)
            temporary_path = None
            return "downloaded"
        except (OSError, URLError, RuntimeError, ValueError) as error:
            last_error = error
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
                temporary_path = None
            if attempt < 2:
                time.sleep(2**attempt)

    raise RuntimeError(f"failed to fetch {destination}: {last_error}") from last_error


def load_and_check_manifests() -> tuple[dict[str, Any], dict[str, Any]]:
    if file_hash(INTASET_MANIFEST) != EXPECTED_INTASET_MANIFEST_SHA256:
        raise ValueError("reviewed InTaSet source manifest has changed")
    if file_hash(NANODRONE_MANIFEST) != EXPECTED_NANODRONE_MANIFEST_SHA256:
        raise ValueError("reviewed Nano-Drone data manifest has changed")
    intaset = json.loads(INTASET_MANIFEST.read_text(encoding="utf-8"))
    nanodrone = json.loads(NANODRONE_MANIFEST.read_text(encoding="utf-8"))
    if intaset.get("record_url") != "https://zenodo.org/records/17652851":
        raise ValueError("InTaSet manifest does not point to the reviewed Zenodo record")
    if intaset.get("file") != "InTaSet.zip":
        raise ValueError("InTaSet manifest names an unexpected source file")
    intaset_size = int(intaset["size_bytes"])
    intaset_sha = str(intaset["sha256_actual"])
    intaset_md5 = str(intaset["zenodo_md5_expected"])
    if intaset_size <= 0 or not SHA256_RE.fullmatch(intaset_sha) or not MD5_RE.fullmatch(intaset_md5):
        raise ValueError("InTaSet manifest contains an invalid size or digest")
    if nanodrone.get("source_repository") != NANODRONE_REPOSITORY.removesuffix(".git"):
        raise ValueError("Nano-Drone manifest does not name the reviewed author repository")
    if nanodrone.get("source_commit") != EXPECTED_NANODRONE_COMMIT:
        raise ValueError("Nano-Drone manifest does not pin the reviewed author commit")
    files = nanodrone.get("files")
    if not isinstance(files, list) or len(files) != 15:
        raise ValueError("Nano-Drone manifest must list exactly 15 public CSV objects")
    observed = nanodrone.get("observed")
    if not isinstance(observed, dict):
        raise ValueError("Nano-Drone manifest is missing observed-data totals")

    seen: set[str] = set()
    split_counts = {"train": 0, "test": 0}
    split_rows = {"train": 0, "test": 0}
    total_bytes = 0
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("Nano-Drone file entry must be a JSON object")
        relative = safe_relative_path(str(entry.get("path", "")))
        split = str(entry.get("split", ""))
        if split not in split_counts or relative.parts[:2] != ("data", split):
            raise ValueError(f"Nano-Drone file path does not match its split: {relative}")
        if relative.suffix.lower() != ".csv" or len(relative.parts) != 3:
            raise ValueError(f"unexpected Nano-Drone file path: {relative}")
        if relative.as_posix() in seen:
            raise ValueError(f"duplicate Nano-Drone file path: {relative}")
        seen.add(relative.as_posix())

        sha256 = str(entry.get("lfs_sha256", ""))
        size_bytes = int(entry.get("size_bytes", 0))
        rows = int(entry.get("rows", 0))
        if not SHA256_RE.fullmatch(sha256) or size_bytes <= 0 or rows <= 0:
            raise ValueError(f"invalid Nano-Drone receipt fields: {relative}")
        split_counts[split] += 1
        split_rows[split] += rows
        total_bytes += size_bytes

    if split_counts != {"train": 12, "test": 3}:
        raise ValueError(f"unexpected Nano-Drone split counts: {split_counts}")
    for split in ("train", "test"):
        if split_rows[split] != int(observed.get(f"{split}_rows", -1)):
            raise ValueError(f"Nano-Drone {split} row total differs from file entries")
    if total_bytes != int(observed.get("downloaded_bytes", -1)):
        raise ValueError("Nano-Drone byte total differs from file entries")
    return intaset, nanodrone


def run(workspace_root: Path, *, verify_only: bool) -> dict[str, Any]:
    workspace_root = workspace_root.expanduser().resolve()
    intaset_manifest, nanodrone_manifest = load_and_check_manifests()
    intaset_path = workspace_root / "docs/algorithmic-control-v1/source/InTaSet.zip"
    author_root = workspace_root / "data/research_control_v1/external/nanodrone"
    nanodrone_data_root = workspace_root / "data/research_control_v1/external/nanodrone_lfs"

    ensure_author_checkout(author_root, fetch=not verify_only)

    intaset_sha = str(intaset_manifest["sha256_actual"])
    intaset_size = int(intaset_manifest["size_bytes"])
    intaset_md5 = str(intaset_manifest["zenodo_md5_expected"])
    if not SHA256_RE.fullmatch(intaset_sha):
        raise ValueError("InTaSet manifest contains an invalid SHA-256")
    if verify_only:
        if not verify_file(
            intaset_path,
            size_bytes=intaset_size,
            sha256=intaset_sha,
            md5=intaset_md5,
        ):
            raise ValueError(f"InTaSet archive is missing or fails its receipt: {intaset_path}")
        intaset_action = "verified"
    else:
        intaset_action = download_verified(
            ZENODO_FILE_URL,
            intaset_path,
            size_bytes=intaset_size,
            sha256=intaset_sha,
            md5=intaset_md5,
        )

    nanodrone_actions = []
    for entry in nanodrone_manifest["files"]:
        relative = safe_relative_path(str(entry["path"]))
        sha256 = str(entry["lfs_sha256"])
        size_bytes = int(entry["size_bytes"])
        if not SHA256_RE.fullmatch(sha256):
            raise ValueError(f"invalid SHA-256 in Nano-Drone manifest: {relative}")
        _, pointer_sha, pointer_size = read_lfs_pointer(author_root, relative)
        if pointer_sha != sha256 or pointer_size != size_bytes:
            raise ValueError(f"manifest differs from pinned Git LFS pointer: {relative}")

        destination = nanodrone_data_root / relative
        if verify_only:
            if not verify_file(destination, size_bytes=size_bytes, sha256=sha256):
                raise ValueError(f"Nano-Drone object is missing or fails its receipt: {relative}")
            action = "verified"
        else:
            url_path = quote(relative.as_posix(), safe="/")
            url = (
                "https://media.githubusercontent.com/media/idsia-robotics/"
                f"nanodrone-sysid-benchmark/{EXPECTED_NANODRONE_COMMIT}/{url_path}"
            )
            action = download_verified(
                url,
                destination,
                size_bytes=size_bytes,
                sha256=sha256,
            )
        nanodrone_actions.append({"path": relative.as_posix(), "action": action})

    return {
        "status": "CONTROL_BENCHMARK_RAW_INPUTS_VERIFIED" if verify_only else "CONTROL_BENCHMARK_RAW_INPUTS_READY",
        "workspace_root": str(workspace_root),
        "intaset": {
            "path": str(intaset_path),
            "source_url": ZENODO_FILE_URL,
            "action": intaset_action,
            "size_bytes": intaset_size,
            "sha256": intaset_sha,
            "md5": intaset_md5,
        },
        "nanodrone_source_repository": NANODRONE_REPOSITORY,
        "nanodrone_author_commit": EXPECTED_NANODRONE_COMMIT,
        "nanodrone_csv_count": len(nanodrone_actions),
        "nanodrone_total_bytes": sum(int(entry["size_bytes"]) for entry in nanodrone_manifest["files"]),
        "nanodrone_actions": nanodrone_actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=ROOT,
        help="workspace root that will receive the raw inputs",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the inputs and pinned author checkout without downloading",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.workspace_root, verify_only=args.verify_only), indent=2))


if __name__ == "__main__":
    main()
