#!/usr/bin/env python
"""Verify native archived GFS semantics and extract explicitly supplied coordinates."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transcif.data.gfs_decode import decode_gfs_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True, help="JSON list of id/latitude/longitude dicts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("existing extracted evidence is never overwritten")
    report = decode_gfs_points(args.snapshot, json.loads(args.points.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Verified {len(report['fields'])} native fields at {len(report['points'])} points")


if __name__ == "__main__":
    main()
