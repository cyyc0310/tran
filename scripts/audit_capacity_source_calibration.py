"""Leave-one-BA-out audit of calibrating annual capacity shares to thermal EF."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_cross_country import ROOT
from transcif.research.cross_country import EF, FUELS, load_inputs


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--capacity-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    code_files = [Path(__file__), ROOT/"src/transcif/research/cross_country.py",
                  ROOT/"docs/CAPACITY_SOURCE_CALIBRATION_AUDIT_PROTOCOL_20261101.md"]
    code_hashes = {str(path.relative_to(ROOT)): digest(path) for path in code_files}
    (out/"startup_code_hashes.json").write_text(json.dumps(code_hashes, indent=2))
    raw, audit = load_inputs(ROOT)
    cap = json.loads(args.capacity_json.read_text())["regions"]
    rows = []
    for region, frame in raw.items():
        if region == "GB":
            continue
        total_capacity = float(cap[region]["coal_mw"]) + float(cap[region]["gas_mw"])
        coal_capacity_share = float(cap[region]["coal_mw"]) / total_capacity
        for year in (2022, 2023):
            fuel = frame.loc[frame.clock.dt.year.eq(year), list(FUELS)].to_numpy(float).sum(axis=0)
            thermal = fuel[:4]
            thermal_ef = float(thermal @ EF[:4] / thermal.sum())
            rows.append({"region": region, "year": year,
                         "coal_capacity_share": coal_capacity_share,
                         "coal_generation_share": float(fuel[0]/thermal.sum()),
                         "thermal_ef": thermal_ef})
    panel = pd.DataFrame(rows)
    results = []
    for heldout in sorted(panel.region.unique()):
        train = panel[(panel.region != heldout) & (panel.year == 2022)]
        target = panel[(panel.region == heldout) & (panel.year == 2023)].iloc[0]
        x = train.coal_capacity_share.to_numpy(float)
        y = train.thermal_ef.to_numpy(float)
        slope, intercept = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
        predictions = {
            "source_mean": float(y.mean()),
            "capacity_identity": float(400 + 600*target.coal_capacity_share),
            "source_calibrated_capacity": float(np.clip(intercept+slope*target.coal_capacity_share, 400, 1000)),
        }
        for method, prediction in predictions.items():
            results.append({"heldout": heldout, "method": method, "actual_2023_thermal_ef": target.thermal_ef,
                            "prediction": prediction, "absolute_error": abs(prediction-target.thermal_ef),
                            "squared_error": (prediction-target.thermal_ef)**2,
                            "train_regions": ",".join(train.region.astype(str)),
                            "fit_year": 2022, "score_year": 2023,
                            "slope": float(slope), "intercept": float(intercept)})
    panel.to_csv(out/"region_year_fuel_mix.csv", index=False)
    scored = pd.DataFrame(results)
    scored.to_csv(out/"loba_predictions.csv", index=False)
    summary = scored.groupby("method").agg(mae=("absolute_error", "mean"),
                                              rmse=("squared_error", lambda x: float(np.sqrt(x.mean())))).reset_index()
    summary.to_csv(out/"macro_metrics.csv", index=False)
    input_hashes = {str((ROOT/row["path"]).resolve()): digest(ROOT/row["path"]) for row in audit.values()}
    input_hashes[str(args.capacity_json.resolve())] = digest(args.capacity_json)
    (out/"input_hashes.json").write_text(json.dumps(input_hashes, indent=2))
    if any(digest(ROOT/path) != value for path, value in code_hashes.items()):
        raise RuntimeError("code or protocol changed during execution")
    verification = {
        "leave_region_out_predictions": len(scored),
        "prediction_rows": len(scored) == len(panel.region.unique())*3,
        "input_hash_mismatches": [path for path, value in input_hashes.items()
                                   if not Path(path).exists() or digest(Path(path)) != value],
        "code_hash_mismatches": [path for path, value in code_hashes.items()
                                 if digest(ROOT/path) != value],
    }
    (out/"post_run_verification.json").write_text(json.dumps(verification, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
