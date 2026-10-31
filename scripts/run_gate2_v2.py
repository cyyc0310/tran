"""Run the independent V2 controlled inference suite in a fresh directory."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcif.research.benchmark_v2 import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["development", "stress"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.output, ROOT, args.suite)
    print(f"Finished {summary['trials']} trials; hash mismatches: {summary['frozen_hash_mismatches']}")


if __name__ == "__main__":
    main()
