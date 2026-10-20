#!/usr/bin/env python
"""Save native single-run NOAA GFS weather fields and original object-time evidence."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transcif.data.gfs import collect_gfs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="timezone-aware initialization, e.g. 2023-01-01T00:00Z")
    parser.add_argument("--lead", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect_gfs(args.run, args.lead, args.output)
    for name, row in report["messages"].items():
        print(f"{name}: {row['level']}, {row['native_step']}, {row['end_byte']-row['start_byte']+1} bytes")
    print(f"S3 write time: {report['grib_object']['last_modified']}; public eligibility remains unverified")


if __name__ == "__main__":
    main()
