"""Shared training controls for experiment and outer-benchmark commands."""


def add_training_arguments(parser):
    parser.add_argument("--objective", choices=["point", "crps"], default="point")
    parser.add_argument("--selection-metric", choices=["mae", "rmse", "crps"])
    parser.add_argument("--point-estimator", choices=["mean", "median", "physical"])
    parser.add_argument("--backbone", choices=["tcn", "context"], default="tcn")
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--training-scenarios", type=int, default=16)
    parser.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--sampler", choices=["uniform", "region_balanced"], default="uniform")
    parser.add_argument("--tree-baseline", action="store_true")


def training_options(args):
    return {"objective": args.objective, "selection_metric": args.selection_metric,
        "model_config": {"backbone": args.backbone, "point_estimator": args.point_estimator,
                         "hidden": args.hidden if args.hidden is not None else (64 if args.backbone == "context" else 32)},
        "lr": args.learning_rate, "batch_size": args.batch_size, "scenarios": args.training_scenarios,
        "schedule": args.schedule, "sampler": args.sampler}
