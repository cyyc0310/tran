#!/usr/bin/env python
"""Stage original AEMO DVD SCADA/registration snapshots without inventing CIF labels."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transcif.data.aemo import stage_mms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()
    def progress(row):
        print(f"{'/'.join(row['table'])}: {row['rows']} rows; {row['output']}", flush=True)
    stage_mms(args.inputs, args.output, args.chunk_size, progress=progress)


if __name__ == "__main__":
    main()
