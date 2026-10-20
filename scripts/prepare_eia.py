#!/usr/bin/env python
"""Stage native EIA-930 fuels/demand without manufacturing as-of input evidence."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transcif.data.eia930 import stage_balance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--regions", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = stage_balance(args.inputs, args.regions, args.output)
    for name, record in result["regions"].items():
        print(f"{name}: {record['rows']} rows, {record['negative_generation_rows']} rows with negative net output")


if __name__ == "__main__":
    main()
