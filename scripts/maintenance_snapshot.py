#!/usr/bin/env python3
"""Index research files and copy compact result summaries without raw datasets."""

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_NAMES = {
    "summary.json", "audit_summary.json", "score_summary.json", "status.json",
    "macro_frontier.csv", "region_equal_frontier.csv", "method_summary.csv",
    "region_summary.csv", "summary_by_scenario_method.csv", "summary_by_scenario.csv",
    "forward_holdout_metrics.csv", "tail_frontier.csv", "pooled_frontier.csv",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "docs" / "evidence")
    if output.exists():
        raise FileExistsError(f"Snapshot already exists: {output}")
    output.mkdir(parents=True)
    catalog = ROOT / "docs" / "catalog"
    catalog.mkdir(exist_ok=True)

    scripts = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        match = re.search(r'(?s)"""(.*?)"""', text)
        scripts.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                        "description": " ".join(match.group(1).split())[:350] if match else "",
                        "sha256": digest(path)})
    write_csv(catalog / "scripts.csv", scripts, ["path", "bytes", "description", "sha256"])
    documents = []
    for path in sorted((ROOT / "docs").glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        title = next((x.lstrip("# ") for x in lines if x.startswith("#")), "")
        documents.append({"path": str(path.relative_to(ROOT)), "title": title,
                          "bytes": path.stat().st_size})
    write_csv(catalog / "documents.csv", documents, ["path", "title", "bytes"])

    inventory, copies = [], []
    results = ROOT / "results"
    for directory in sorted(results.iterdir()):
        if not directory.is_dir() or directory.is_symlink():
            continue
        files = [p for p in directory.rglob("*") if p.is_file() and not p.is_symlink()]
        inventory.append({"directory": str(directory.relative_to(ROOT)),
                          "files": len(files), "bytes": sum(p.stat().st_size for p in files)})
        for path in sorted(directory.iterdir()):
            if (not path.is_file() or path.is_symlink() or path.name not in SUMMARY_NAMES
                    or path.stat().st_size > 512 * 1024):
                continue
            relative = path.relative_to(results)
            target = output / "results" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if digest(path) != digest(target):
                raise RuntimeError(f"Snapshot mismatch: {path}")
            copies.append({"source": str(path.relative_to(ROOT)),
                           "snapshot": str(target.relative_to(ROOT)),
                           "bytes": target.stat().st_size, "sha256": digest(target)})
    write_csv(catalog / "local-artifacts.csv", inventory, ["directory", "files", "bytes"])
    manifest = {"scope": "compact summaries copied verbatim; inclusion does not validate an experiment",
                "raw_data_included": False, "model_checkpoints_included": False,
                "source_results_modified": False, "artifacts": copies,
                "artifact_count": len(copies), "artifact_bytes": sum(p["bytes"] for p in copies),
                "local_result_directories": len(inventory),
                "local_result_bytes": sum(p["bytes"] for p in inventory)}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
