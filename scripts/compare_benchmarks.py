#!/usr/bin/env python
"""Compare matched point forecasts with grid/country and time-block resampling."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transcif.data.dataset import load_bundle
from transcif.evaluation.comparison import paired_block_comparison


def compare(reference, candidate, manifest, grouping="target", reference_point="point_cif",
            candidate_point="point_cif", **options):
    reference, candidate, manifest = Path(reference), Path(candidate), Path(manifest)
    left, right = [json.loads((root / "benchmark.json").read_text()) for root in (reference, candidate)]
    if set(left["targets"]) != set(right["targets"]) or set(left["seeds"]) != set(right["seeds"]):
        raise ValueError("paired comparison requires identical target and training-seed sets")
    doc, grids = load_bundle(manifest)
    manifestsum = hashlib.sha256(manifest.read_bytes()).hexdigest()
    hashes, errors, common_origins = {}, [[], []], None
    for target in sorted(left["targets"]):
        by_method = [[], []]
        for seed in sorted(left["seeds"]):
            paths, summaries = [], []
            for root, benchmark in ((reference, left), (candidate, right)):
                matches = [r for r in benchmark["runs"] if (r["target"], r["seed"]) == (target, seed)]
                if len(matches) != 1:
                    raise ValueError("unique run per target/seed required")
                folder = root / matches[0]["run"]
                summary = json.loads((folder / "summary.json").read_text())
                if summary["manifest_sha256"] != manifestsum:
                    raise ValueError("manifest differs from experiment snapshot")
                for name, expected in summary["input_sha256"].items():
                    if name not in hashes:
                        hashes[name] = hashlib.sha256((manifest.parent / name).read_bytes()).hexdigest()
                    if hashes[name] != expected:
                        raise ValueError("input file differs from experiment snapshot: " + name)
                paths.append(folder / "forecasts.npz")
                summaries.append(summary)
            if summaries[0]["input_sha256"] != summaries[1]["input_sha256"]:
                raise ValueError("paired runs use different inputs or label vintages")
            forecasts = []
            for path, point_key in zip(paths, (reference_point, candidate_point)):
                with np.load(path) as values:
                    point = values[point_key] if point_key in values else (
                        values["cif"].mean(1) if point_key == "point_cif" else None)
                    if point is None:
                        raise ValueError("point forecast field absent: " + point_key)
                    origins = values["origins"].astype(str)
                    forecasts.append(point.copy())
                if common_origins is None:
                    common_origins = origins
                    stamps = pd.to_datetime(origins, utc=True)
                    gaps = np.diff(stamps.asi8)
                    if len(gaps) and (gaps.min() <= 0 or not (gaps == gaps[0]).all()):
                        raise ValueError("regular ordered forecast-window sampling required")
                if not np.array_equal(origins, common_origins):
                    raise ValueError("targets/seeds/methods must share the same ordered forecast origins")
            truth = np.stack([grids[target].labels(grids[target].example(origin, doc.get("horizon", 24)))["cif"]
                              for origin in common_origins])
            for i, point in enumerate(forecasts):
                if point.shape != truth.shape:
                    raise ValueError("forecast and truth shapes differ")
                by_method[i].append(point - truth)
        for i in range(2):
            errors[i].append(by_method[i])
    targets = sorted(left["targets"])
    groups = [grids[target].jurisdiction for target in targets] if grouping == "jurisdiction" else targets
    result = paired_block_comparison(*errors, groups=groups, **options)
    return {**result, "track": doc["track"], "grouping": grouping,
        "reference": str(reference), "candidate": str(candidate),
        "reference_point_field": reference_point, "candidate_point_field": candidate_point,
        "manifest_sha256": manifestsum, "input_fingerprints_verified": True,
        "window_spacing_hours": float((pd.Timestamp(str(common_origins[1])) - pd.Timestamp(str(common_origins[0]))).total_seconds() / 3600)
            if len(common_origins) > 1 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--grouping", choices=["target", "jurisdiction"], default="target")
    parser.add_argument("--reference-point", default="point_cif")
    parser.add_argument("--candidate-point", default="point_cif")
    parser.add_argument("--block-windows", type=int, default=7)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = vars(parser.parse_args())
    output = args.pop("output")
    if output.exists():
        parser.error("fresh comparison output required")
    result = compare(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["metrics"]))


if __name__ == "__main__":
    main()
