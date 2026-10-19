#!/usr/bin/env python
"""Inference from public inputs only; no label file is opened."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transcif.data.dataset import FUELS, load_bundle
from transcif.models.dispatch import ProbabilisticDispatchNet
from transcif.training.fit import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--origin", required=True, help="timezone-aware hourly origin")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scenarios", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a fresh path")
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (checkpoint.get("schema_version") != 1 or checkpoint.get("fuels") != list(FUELS)
            or checkpoint.get("accounting") != "production_direct_co2"):
        raise ValueError("incompatible checkpoint")
    doc, grids = load_bundle(args.manifest)
    if checkpoint.get("cost_currency") != doc["cost_currency"]:
        raise ValueError("input and training cost_currency must match")
    example = grids[args.target].example(args.origin, int(doc.get("horizon", 24)))
    model = ProbabilisticDispatchNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    forecast = predict(model, [example], args.scenarios, args.seed)
    draws = forecast["cif"][0]
    output = {"region": args.target, "origin": args.origin,
        "accounting": checkpoint["accounting"], "training_track": checkpoint["training_track"],
        "input_track": doc["track"], "provenance": example.provenance,
        "mean_g_per_kwh": draws.mean(0).tolist(),
        "p05_g_per_kwh": np.quantile(draws, .05, axis=0).tolist(),
        "p95_g_per_kwh": np.quantile(draws, .95, axis=0).tolist(),
        "unserved_forecast_load_fraction": float(np.mean(
            forecast["unserved"] / forecast["demand"])), "scenarios": draws.tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
