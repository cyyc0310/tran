#!/usr/bin/env python
"""Collect a complete single-run GFS point-weather window with a resumable cache."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transcif.data.gfs_window import collect_gfs_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--first-lead", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--solar-negative-policy", choices=["error", "packing_resolution"], default="error")
    args = parser.parse_args()
    def progress(record):
        print(f"f{record['lead']:03d}: {'verified cache' if record['cached'] else 'downloaded'}", flush=True)
    report = collect_gfs_window(args.run, args.first_lead, args.horizon,
        json.loads(args.points.read_text()), args.cache, args.output,
        workers=args.workers, retries=args.retries, progress=progress,
        solar_negative_policy=args.solar_negative_policy)
    print(f"Complete {report['horizon']}h window; latest object write {report['latest_object_write_at']}")


if __name__ == "__main__":
    main()
