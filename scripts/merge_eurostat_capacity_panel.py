"""Build the nine-country, three-year Eurostat capacity budget table."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COUNTRIES = ("NL", "ES", "BE", "AT", "SE", "CZ", "SK", "HU", "SI")
YEARS = (2020, 2021, 2022)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True,
                        help="five-country Eurostat extract directory")
    parser.add_argument("--expansion-dir", type=Path, required=True,
                        help="four-country Eurostat extract directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dirs = [args.base_dir.resolve(), args.expansion_dir.resolve()]
    pieces = [pd.read_csv(d / "capacity.csv") for d in dirs]
    panel = pd.concat(pieces, ignore_index=True)
    panel = panel.loc[panel.country.isin(COUNTRIES) & panel.stat_year.isin(YEARS)]
    panel = panel.sort_values(["country", "stat_year"]).reset_index(drop=True)
    if len(panel) != len(COUNTRIES) * len(YEARS):
        raise ValueError("capacity panel does not contain exactly nine countries × three years")
    if panel.duplicated(["country", "stat_year"]).any():
        raise ValueError("duplicate country/year capacity rows")
    if panel.thermal_capacity_share.isna().any():
        raise ValueError("thermal capacity share contains missing values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output, index=False)
    base_dir, expansion_dir = [p.relative_to(ROOT) for p in dirs]
    source_records = [
        {"path": str(base_dir / "capacity.csv"), "sha256": sha(dirs[0] / "capacity.csv"),
         "manifest": str(base_dir / "capacity.manifest.json"),
         "manifest_sha256": sha(dirs[0] / "capacity.manifest.json")},
        {"path": str(expansion_dir / "capacity.csv"), "sha256": sha(dirs[1] / "capacity.csv"),
         "manifest": str(expansion_dir / "capacity.manifest.json"),
         "manifest_sha256": sha(dirs[1] / "capacity.manifest.json")},
        {"path": str(expansion_dir / "acquisition_manifest.json"),
         "sha256": sha(dirs[1] / "acquisition_manifest.json")},
    ]
    manifest = {
        "schema": "transcif-eurostat-capacity-nine-country-r1",
        "rows": len(panel), "countries": sorted(panel.country.unique().tolist()),
        "stat_years": list(YEARS),
        "capacity_field": "thermal_capacity_share=CF/TOTAL",
        "sources": source_records,
        "vintage_note": "current revised Eurostat responses; not historical as-issued releases",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output, sha(args.output), len(panel), flush=True)


if __name__ == "__main__":
    main()
