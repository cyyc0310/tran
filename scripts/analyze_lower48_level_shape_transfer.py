"""Post-hoc development diagnostic: anchor the level, transfer the hourly shape."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_lower48_actual_issue_dual_track import DEFAULT_OUT, source_fit
from transcif.research.cross_country import COARSE, FUELS, labels, load_inputs, metrics, monthly_inputs, predict_models


def main() -> None:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    scored_path = run / "scored_hourly.csv"
    if not scored_path.exists():
        raise FileNotFoundError("score the sealed baseline run first")

    models, _ = source_fit()
    source, _ = load_inputs(ROOT)
    gb = source["GB"]
    prepared = monthly_inputs(gb, lag=3).dropna(subset=list(COARSE)).copy()
    prepared["source_index"] = prepared.index
    cal = prepared.loc[prepared.year.eq(2023)].copy()
    cal_pred = predict_models(models, cal)
    cal_shape = cal_pred["coarse_recomposition"]
    cal_day = cal.clock.dt.strftime("%Y-%m-%d")
    cal_shape = cal_shape - pd.Series(cal_shape).groupby(cal_day.to_numpy()).transform("mean").to_numpy()
    cal_hybrid = cal_pred["anchor700"] + cal_shape
    cal_y = labels(gb.loc[cal.source_index])
    cal_residual = cal_y - cal_hybrid
    bias = float(cal_residual.mean())
    sigma = max(float(np.sqrt(np.mean((cal_residual - bias) ** 2))), 1.)

    frame = pd.read_csv(scored_path, parse_dates=["time", "clock"])
    rec = frame.coarse_recomposition_point.to_numpy(float)
    day = pd.to_datetime(frame.clock).dt.strftime("%Y-%m-%d")
    rec_shape = rec - pd.Series(rec).groupby(day.to_numpy()).transform("mean").to_numpy()
    frame["level_anchor_shape_point"] = frame.anchor700_point.to_numpy(float) + rec_shape
    frame["level_anchor_shape_bias"] = bias
    frame["level_anchor_shape_sigma"] = sigma
    end_points = ["official_generation", "official_consumption"]
    rows = []
    result = {"status": "post-hoc development diagnostic; target labels already inspected",
              "definition": "700-anchor level + zero-mean-within-local-day coarse-recomposition shape",
              "risk": "bias and sigma calibrated only on GB 2023 under the same lag-3 contract",
              "source_calibration": {"bias": bias, "sigma": sigma}, "tracks": {}}
    for endpoint in end_points:
        valid = np.isfinite(frame[endpoint].to_numpy(float))
        part = frame.loc[valid]
        result["tracks"][endpoint] = metrics(
            part, part[endpoint].to_numpy(float), part.level_anchor_shape_point.to_numpy(float), bias, sigma)
        for month, group in part.groupby("period"):
            m = metrics(group, group[endpoint].to_numpy(float), group.level_anchor_shape_point.to_numpy(float), bias, sigma)
            rows.append({"endpoint": endpoint, "period": month, **m})
    (run / "level_shape_diagnostic.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(rows).to_csv(run / "level_shape_monthly_scores.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
