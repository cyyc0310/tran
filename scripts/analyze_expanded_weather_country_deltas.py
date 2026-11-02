#!/usr/bin/env python3
"""Summarize paired R6b weather deltas by held-out target country."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/expanded-anchor-issued-r6b/configuration_metrics.csv"
FIELDS = ("mae", "shape_mae", "wis90", "coverage90", "width90")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(input_path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(input_path)
    selected = metrics.loc[
        metrics.method.isin(["paired_hgb_no_weather", "paired_hgb_weather"])
        & metrics.risk.eq("calibrated")
        & metrics.track.eq("paired_labels")
    ].copy()
    counts = selected.groupby(["country", "budget", "configuration", "method"]).size()
    if not counts.eq(1).all():
        raise ValueError("expected one row per configuration/method for selected R6b metrics")
    rows = []
    for (country, budget), group in selected.groupby(["country", "budget"]):
        wide = group.pivot(index="configuration", columns="method", values=list(FIELDS))
        if set(wide.index) != set(group.configuration.unique()) or wide.isna().any().any():
            raise ValueError(f"unpaired weather/no-weather rows for {country} {budget}")
        deltas = wide.xs("paired_hgb_weather", axis=1, level=1) - wide.xs("paired_hgb_no_weather", axis=1, level=1)
        for field in FIELDS:
            rows.append({"country": country, "budget": budget, "metric": field,
                         "no_weather_mean": float(wide.xs("paired_hgb_no_weather", axis=1, level=1)[field].mean()),
                         "weather_mean": float(wide.xs("paired_hgb_weather", axis=1, level=1)[field].mean()),
                         "weather_minus_no_weather": float(deltas[field].mean()),
                         "source_target_configurations": int(len(deltas)),
                         "source_target_delta_values": ";".join(f"{x:.9g}" for x in deltas[field].to_numpy())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_path, output = args.input.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    input_hash = sha(input_path)
    frame = summarize(input_path)
    output_csv = output / "country_paired_deltas.csv"
    frame.to_csv(output_csv, index=False)
    summary = {
        "status": "POST_HOC_COUNTRY_STRATIFIED_R6B_SCORE_DIAGNOSTIC",
        "target_labels_previously_viewed": True,
        "prediction_or_model_refit": False,
        "metric_source_sha256": input_hash,
        "analysis_script_sha256": sha(Path(__file__)),
        "output_sha256": sha(output_csv),
        "target_countries": sorted(frame.country.unique().tolist()),
        "source_target_configurations_per_country_budget": 3,
        "bootstrap_or_significance": False,
        "interpretation": "Descriptive target-country heterogeneity only; three held-out countries do not support a general cross-country uncertainty claim.",
    }
    (output / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(frame.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
