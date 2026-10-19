#!/usr/bin/env python
"""Run fixed-protocol outer grid/seed sweeps; all fitting remains source-only."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transcif.benchmark import benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--scenarios", type=int, default=32)
    parser.add_argument("--mechanism-ensemble", action="store_true")
    parser.add_argument("--inner-unit", choices=["region", "jurisdiction"], default="region")
    parser.add_argument("--scale-candidates", nargs="+", type=float, default=[.5, 1., 2., 4.])
    parser.add_argument("--leave-jurisdiction-out", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.threads < 1 or args.scenarios < 2:
        parser.error("positive threads and scenarios>=2 required")
    torch.set_num_threads(args.threads)
    summary = benchmark(args.manifest, args.output, args.targets, args.seeds,
        steps=args.steps, scenarios=args.scenarios, mechanism_ensemble=args.mechanism_ensemble,
        inner_unit=args.inner_unit, scale_candidates=args.scale_candidates,
        leave_jurisdiction_out=args.leave_jurisdiction_out, device=args.device)
    for name, metrics in summary["target_macro"].items():
        print(f"{name}: target-macro MAE={metrics['mae']:.3f} CRPS={metrics['crps']:.3f}")
    print(args.output / "benchmark.json")


if __name__ == "__main__":
    main()
