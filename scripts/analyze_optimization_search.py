"""Summarize frozen V3 outcomes by lawful input regimes, without model tuning."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transcif.research.search_analysis import paired_month_interval, input_regimes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Do not overwrite a prior diagnostic")
    args.output.mkdir(parents=True)
    scores = pd.read_csv(args.experiment / "day_metrics.csv")
    monthly = pd.read_csv(args.data / "monthly.csv")
    descriptors = []
    for (region, delay), frame in scores.groupby(["region", "delay"]):
        origins = sorted(frame.origin.unique())
        descriptors.append(input_regimes(monthly, origins, region, delay))
    descriptors = pd.concat(descriptors, ignore_index=True)
    # Canonical timestamp strings avoid differently formatted equal UTC times.
    scores.origin = pd.to_datetime(scores.origin, utc=True).astype(str)
    joined = scores.merge(descriptors, on=["region", "delay", "origin"], validate="many_to_one")
    joined.to_csv(args.output / "day_metrics_with_inputs.csv", index=False)
    intervals, regime_rows = [], []
    for delay, part in scores.groupby("delay"):
        for method in ("seasonal", "analog_shape_half", "source_policy"):
            intervals.append({"extra_delay": int(delay), "total_release_days": int(delay)+30,
                              "method": method, **paired_month_interval(part, method)})
    pd.DataFrame(intervals).to_csv(args.output / "synchronized_month_intervals.csv", index=False)
    metrics = ["crps", "raw_point_mae", "raw_shape_mae", "raw_level_mae", "coverage90", "width90"]
    for variable in ("age_regime", "coal_regime", "seasonal_history"):
        regional = joined.groupby(["delay", variable, "region", "method"], observed=True)[metrics].mean()
        macro = regional.groupby(level=["delay", variable, "method"]).mean().reset_index()
        for (delay, label), group in macro.groupby(["delay", variable]):
            baseline = group.loc[group.method.eq("latest")].iloc[0]
            population = joined.loc[joined.delay.eq(delay) & joined[variable].eq(label) & joined.method.eq("latest")]
            for row in group.to_dict("records"):
                regime_rows.append({"total_release_days": int(delay)+30, "variable": variable,
                                    "regime": label, "regions": int(population.region.nunique()),
                                    "region_days": len(population), "method": row["method"],
                                    **{key: row[key] for key in metrics},
                                    "delta_crps_vs_latest": row["crps"] - baseline.crps})
    pd.DataFrame(regime_rows).to_csv(args.output / "optimization_regimes.csv", index=False)
    bounds = []
    for folder in sorted(args.experiment.glob("*/predictions.npz")):
        with np.load(folder) as values:
            for name in values.files:
                if name.startswith("residual_") or name == "origins":
                    continue
                array = values[name]
                bounds.append({"fold": folder.parent.name, "method": name, "hours": array.size,
                               "raw_out_of_bounds": int(((array < 0) | (array > 1000)).sum())})
    pd.DataFrame(bounds).to_csv(args.output / "raw_point_bounds.csv", index=False)
    (args.output / "scope.json").write_text(json.dumps({
        "status": "post-score diagnostics, not a heldout-trained selector",
        "interval": "synchronized calendar months; equal regions with valid day counts",
        "regimes": "input-only age, coal share and same-month historical availability",
        "not_causal": "regime variables overlap geography/season/missingness; subgroup winners are not deployable policies",
        "new_hyperparameters_selected": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
