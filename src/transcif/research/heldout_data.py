"""Build an additional-year controlled proxy with unchanged historical inputs.

Data preparation may read native labels for schema/quality checks; no forecast
scores or CIF outcome summaries are computed here. Prediction policies must be
frozen separately before opening this year's validation scores.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from transcif.data.eia930 import normalize_balance
from .eia_proxy import REGIONS, _proxy_rows, _monthly_at_release


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_year(native_paths, historical, output, year=2025, regions=REGIONS):
    historical, output = Path(historical), Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh output directory for heldout data")
    if not native_paths or not regions or len(set(regions)) != len(regions):
        raise ValueError("Native paths and unique nonempty regions required")
    history_monthly = pd.read_csv(historical / "monthly.csv")
    historical_times = pd.to_datetime(history_monthly.period_start, utc=True)
    if (historical_times.dt.year >= year).any():
        raise ValueError("Historical inputs must precede the additional year")
    frames, schemas, native_sources = [], {}, []
    for snapshot, path in enumerate(map(Path, native_paths)):
        normalized = normalize_balance(pd.read_csv(path), regions)
        normalized["source_snapshot"] = snapshot
        schemas[str(snapshot)] = {"fuel_columns": [c for c in normalized if c.startswith("generation_")]}
        frames.append(normalized)
        native_sources.append({"file": str(path.resolve()), "sha256": sha(path),
                               "url": f"https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/{path.name}"})
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.loc[combined.valid_at.dt.year == year].copy()
    if combined.duplicated(["region", "valid_at"]).any():
        raise ValueError("Duplicate native hours across snapshots")
    observations, hourly_outputs, audits = [], {}, {}
    for region in regions:
        native = combined.loc[combined.region == region].copy()
        if native.empty:
            raise ValueError(f"No additional-year rows for {region}")
        native.index = pd.DatetimeIndex(native.valid_at)
        native = native.sort_index()
        hourly, quality_audit = _proxy_rows(native, region, schemas)
        monthly, eligibility = _monthly_at_release(native, region, schemas)
        history = pd.read_csv(historical / f"{region}_hourly.csv")
        if (pd.to_datetime(history.time, utc=True).dt.year >= year).any():
            raise ValueError("Historical hourly data must precede additional year")
        hourly_outputs[region] = pd.concat([history, hourly], ignore_index=True)
        observations.extend(monthly)
        audits[region] = {"valid_days": int(hourly.valid.sum() // 24),
                          "native_hours": len(hourly), "released_months": len(monthly),
                          "monthly_prefix_eligibility": eligibility,
                          "snapshot_quality": quality_audit["snapshots"]}
    all_monthly = pd.concat([history_monthly, pd.DataFrame(observations)], ignore_index=True)
    if all_monthly.duplicated(["region", "period_start"]).any():
        raise ValueError("Duplicate historical/heldout monthly support")
    output.mkdir(parents=True)
    for region, frame in hourly_outputs.items():
        frame.to_csv(output / f"{region}_hourly.csv", index=False)
    all_monthly.to_csv(output / "monthly.csv", index=False)
    report = {"year": year, "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
              "native_sources": native_sources, "native_schemas": schemas,
              "historical_input_hashes": {p.name: sha(p) for p in sorted(historical.glob("*.csv"))},
              "derived_hashes": {p.name: sha(p) for p in sorted(output.glob("*.csv"))},
              "regions": audits, "cif_scores_inspected": False,
              "protocol": "controlled_information_budget; month-end plus30d",
              "issued_public": False, "fixed_ef_proxy_not_full_emissions_truth": True,
              "notes": ["Original historical monthly rows kept; no hourly reaggregation of old months.",
                        "Monthly active fields/quality use only native valid_at before assumed release.",
                        "Evaluation quality uses full additional-year snapshots retrospectively.",
                        "Native revisions downloaded in2026; original historical vintages unavailable.",
                        "Missing boundary hours are not filled; incomplete UTC days excluded."]}
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
