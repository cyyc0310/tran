#!/usr/bin/env python
"""Run an explicit dataset, or a clearly identified synthetic integration demo."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transcif.data.synthetic import make_bundle
from transcif.experiment import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--demo", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenarios", type=int, default=64)
    parser.add_argument("--demo-days", type=int, default=90)
    parser.add_argument("--leave-jurisdiction-out", action="store_true")
    parser.add_argument("--mechanism-ensemble", action="store_true",
                        help="cross-fit independent source groups; scenarios are per mechanism")
    parser.add_argument("--inner-unit", choices=["region", "jurisdiction"], default="region")
    parser.add_argument("--scale-candidates", type=float, nargs="+", default=[.5, 1., 2., 4.])
    parser.add_argument("--no-target-scoring", action="store_true",
                        help="train and forecast without opening target labels")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1 or args.scenarios < 2:
        parser.error("positive threads and scenarios>=2 required")
    torch.set_num_threads(args.threads)
    manifest = args.manifest
    if args.demo:
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("demo output must be empty; existing runs are never overwritten")
        manifest = make_bundle(args.output / "synthetic_data", args.demo_days)
    summary = run(manifest, args.output, args.target, args.steps, args.seed,
                  args.scenarios, args.leave_jurisdiction_out, args.device,
                  score_target=not args.no_target_scoring, mechanism_ensemble=args.mechanism_ensemble,
                  inner_unit=args.inner_unit, scale_candidates=args.scale_candidates)
    print(f"track={summary['track']} target={summary['target']} sources={summary['sources']}")
    for name, metric in summary["metrics"].items():
        print(f"{name}: MAE={metric['mae']:.3f} CRPS={metric['crps']:.3f} "
              f"pointwise90={metric['pointwise_90_coverage']:.3f}")
    print(f"Saved {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
