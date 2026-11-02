"""Compare Chinese feature-range support before and after adding Serbia."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_energy_charts_loco as base  # noqa: E402
import run_energy_charts_serbia_expansion as expansion  # noqa: E402

CHINA_FEATURES = ROOT / "docs/evidence/china-province-public-inference-r2/public_features.csv"
COUNTRY_FEATURES = (
    "thermal", "nuclear", "hydro", "wind", "solar",
    "thermal_mean", "thermal_std", "thermal_change", "total_cv",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    manifest_path = base.DATA / "acquisition_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_frames = []
    input_paths = [manifest_path, base.DATA / "audit.json", expansion.RS_DATA,
                   expansion.RS_MANIFEST, CHINA_FEATURES]
    for country in (*base.COUNTRIES, "rs"):
        if country == "rs":
            entry = json.loads(expansion.RS_MANIFEST.read_text(encoding="utf-8"))
            raw, _ = expansion.load_serbia(entry["sha256"])
        else:
            raw_path = base.DATA / f"{country}_2022_2024.json"
            input_paths.append(raw_path)
            raw, _ = base.load_country(country, manifest)
        features = base.public_features(base.public_months(raw), raw.clock)
        features["year"] = raw.clock.dt.year.to_numpy()
        features["country"] = country.upper()
        source_year = features.dropna(subset=list(base.HISTORY)).loc[lambda frame: frame.year.eq(2022)]
        feature_frames.append(source_year)

    source = pd.concat(feature_frames, ignore_index=True)
    china = pd.read_csv(CHINA_FEATURES)
    rows = []
    country_members = {"five_country_source": [code.upper() for code in base.COUNTRIES],
                       "six_country_source": [code.upper() for code in (*base.COUNTRIES, "rs")]}
    province_rows = []
    for name, members in country_members.items():
        cohort = source.loc[source.country.isin(members)]
        bounds = {column: [float(cohort[column].min()), float(cohort[column].max())]
                  for column in COUNTRY_FEATURES}
        rows.extend({"source_cohort": name, "feature": column,
                     "source_min": limits[0], "source_max": limits[1]}
                    for column, limits in bounds.items())
        for province, block in china.groupby("province", sort=True):
            outside = {
                column: bool(((block[column] < bounds[column][0] - 1e-10)
                              | (block[column] > bounds[column][1] + 1e-10)).any())
                for column in COUNTRY_FEATURES
            }
            province_rows.append({"source_cohort": name, "province": province,
                                  **{f"outside_{key}": value for key, value in outside.items()},
                                  "outside_any_listed_coordinate": any(outside.values())})
    pd.DataFrame(rows).to_csv(output / "source_feature_ranges.csv", index=False)
    provinces = pd.DataFrame(province_rows)
    provinces.to_csv(output / "china_province_support.csv", index=False)
    counts = {}
    for name in country_members:
        block = provinces.loc[provinces.source_cohort.eq(name)]
        counts[name] = {
            "training_rows": int(len(source.loc[source.country.isin(country_members[name])])),
            "provinces_outside_any_of_nine_coordinates": int(block.outside_any_listed_coordinate.sum()),
            "out_by_feature": {
                column: int(block[f"outside_{column}"].sum()) for column in COUNTRY_FEATURES
            },
        }
    summary = {
        "experiment": "coordinate-wise Chinese public feature range diagnostic",
        "source_year": 2022,
        "china_inputs": "31-province NBS public inference feature snapshot, t−2=2026-09",
        "coordinates": list(COUNTRY_FEATURES),
        "metric": "province is outside if any forecast-hour coordinate is outside source min/max",
        "scope_limit": "univariate min/max only; not joint support, accuracy, or coverage",
        "cohorts": counts,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "input_sha256": {str(path.resolve()): sha(path) for path in sorted(set(input_paths))},
        "code_sha256": {str(path.resolve()): sha(path) for path in [Path(__file__).resolve(),
                            Path(base.__file__).resolve(), Path(expansion.__file__).resolve()]},
        "output_sha256": {path.name: sha(path) for path in output.iterdir() if path.is_file()},
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
