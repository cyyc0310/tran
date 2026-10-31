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
from transcif.models.ensemble import (load_ensemble, predict_components, mixture,
                                     uncertainty_decomposition)
from transcif.evaluation.scheduling import mechanism_decisions
from transcif.inference import predict
from transcif.training.transfer import support_diagnostics


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
    if (checkpoint.get("schema_version") not in (1, 2) or checkpoint.get("fuels") != list(FUELS)
            or checkpoint.get("accounting") != "production_direct_co2"):
        raise ValueError("incompatible checkpoint")
    doc, grids = load_bundle(args.manifest)
    if checkpoint.get("cost_currency") != doc["cost_currency"]:
        raise ValueError("input and training cost_currency must match")
    example = grids[args.target].example(args.origin, int(doc.get("horizon", 24)))
    details = {}
    if checkpoint["schema_version"] == 2:
        models = load_ensemble(checkpoint)
        point_definition = "mean_of_mechanism_" + models[0].point_estimator
        transfer = checkpoint["transfer"]
        components = predict_components(models, [example], args.scenarios, args.seed,
                                        transfer["latent_scale"])
        forecast = mixture(components)
        support = support_diagnostics([example], transfer["support"])
        decisions = mechanism_decisions(components, support["supported"],
                                       **{k: v for k, v in transfer["decision_policy"].items()
                                          if k in ("alpha", "max_shortage")},
                                       duration=transfer["decision_policy"]["duration_hours"])
        accepted = bool(decisions["accepted"][0])
        details = {"mechanisms": len(models), "latent_scale": transfer["latent_scale"],
            "scenarios_per_mechanism": args.scenarios,
            "configuration_support": {"supported": bool(support["supported"][0]),
                "distance": float(support["distance"][0]),
                "distance_ratio": float(support["distance_ratio"][0]),
                "nearest_source_group": transfer["support"]["groups"][
                    int(support["nearest_source_group"][0])]},
            "uncertainty_g_per_kwh_squared": {key: value[0].tolist()
                for key, value in uncertainty_decomposition(components["cif"]).items()},
            "recommendation_accepted": accepted,
            "max_mechanism_unserved_fraction": float(decisions["max_mechanism_unserved_fraction"][0]),
            "schedule_proposals": {key: {"proposed_start_hour": int(decisions[key][0]),
                "recommended_start_hour": int(decisions[key][0]) if accepted else None}
                for key in ("mixture_mean", "mixture_cvar", "worst_mechanism_mean", "worst_mechanism_cvar")}}
    else:
        model = ProbabilisticDispatchNet(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["state_dict"])
        point_definition = model.point_estimator
        forecast = predict(model, [example], args.scenarios, args.seed)
    draws = forecast["cif"][0]
    output = {"region": args.target, "origin": args.origin,
        "accounting": checkpoint["accounting"], "training_track": checkpoint["training_track"],
        "input_track": doc["track"], "provenance": example.provenance, **details,
        "point_definition": point_definition,
        "point_g_per_kwh": forecast.get("point_cif", forecast["cif"].mean(1))[0].tolist(),
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
