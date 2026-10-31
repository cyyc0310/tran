"""Prepare a new-year proxy without computing forecast accuracy."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transcif.research.heldout_data import prepare_year


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, nargs="+", required=True)
    parser.add_argument("--historical", type=Path, default=ROOT / "results/research-prefix/data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()
    report = prepare_year(args.native, args.historical, args.output, args.year)
    print({region: {k: v for k, v in audit.items() if k in {"valid_days", "native_hours", "released_months"}}
           for region, audit in report["regions"].items()})


if __name__ == "__main__":
    main()
