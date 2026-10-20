#!/usr/bin/env python
"""Diagnose a conditional capacity-only CIF envelope without reading labels."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transcif.data.dataset import FUELS, load_bundle
from transcif.models.dispatch import wind_cf
from transcif.physics.envelope import dispatch_cif_envelope


def diagnose(manifest, origin, target=None):
    manifest = Path(manifest)
    document, grids = load_bundle(manifest)
    target = target or document["target"]
    example = grids[target].example(origin, document["horizon"])
    values = example.values
    demand = np.asarray(values["demand"], dtype=float)
    capacity = np.broadcast_to(values["capacity"], (*demand.shape, len(FUELS))).astype(float).copy()
    # Fixed issued resource factors: these are assumptions, not actual
    # availability bounds, and do not include learned resource calibration.
    capacity[..., 4] *= wind_cf(torch.tensor(values["weather"][..., 1])).numpy()
    capacity[..., 5] *= np.clip(.85 * values["weather"][..., 2] / 1000, 0, 1)
    feasible = (demand > 0) & (demand <= capacity.sum(-1))
    low, high = np.full(demand.shape, np.nan), np.full(demand.shape, np.nan)
    if feasible.any():
        envelope = dispatch_cif_envelope(capacity[feasible], demand[feasible], values["ef"])
        low[feasible], high[feasible] = envelope.lower, envelope.upper
    widths = (high - low)[feasible]
    definition = next(row for row in document["regions"] if row["name"] == target)
    used_paths = [manifest, *(manifest.parent / definition[key] for key in ("releases", "forecasts"))]

    def nullable(values):
        return [[float(x) if np.isfinite(x) else None for x in row] for row in values]

    return {"track": document["track"], "region": target, "origin": str(example.origin),
        "provenance": example.provenance, "members": list(grids[target].members),
        "units": "gCO2/kWh", "axes": ["member", "horizon"],
        "interpretation": "conditional feasibility diagnostic; not a probability interval or target risk guarantee",
        "assumptions": ["single bus without storage, interchange, ramps or commitment",
            "domestic generation set equal to issued demand forecast; not valid generally for open grids",
            "wind/PV upper limits use fixed simplified issued-weather power factors",
            "all other primary capacity fully available; independent outputs between zero and upper limit",
            "fixed complete six-fuel direct CO2 EFs; no cost/merit-order or source-learned restrictions",
            "forecast-member envelope does not guarantee coverage of weather or demand outcomes"],
        "feasible": feasible.tolist(), "infeasible_member_hours": int((~feasible).sum()),
        "lower": nullable(low), "upper": nullable(high),
        "mean_conditional_width": float(widths.mean()) if widths.size else None,
        "input_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in used_paths}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--target")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("existing diagnostic is never overwritten")
    report = diagnose(args.manifest, args.origin, args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['region']}: conditional width={report['mean_conditional_width']}, "
          f"infeasible member-hours={report['infeasible_member_hours']}")


if __name__ == "__main__":
    main()
