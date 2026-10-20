"""Shared training controls for experiment and outer-benchmark commands."""


def add_training_arguments(parser):
    parser.add_argument("--objective", choices=["point", "crps"], default="point")
    parser.add_argument("--selection-metric", choices=["mae", "rmse", "crps"])
    parser.add_argument("--point-estimator", choices=["mean", "median", "physical"])
    parser.add_argument("--backbone", choices=["tcn", "context", "shared"], default="tcn")
    parser.add_argument("--backbone-candidates", choices=["tcn", "context", "shared"], nargs="+")
    parser.add_argument("--selection-protocol", choices=["temporal", "source_group"], default="temporal")
    parser.add_argument("--selection-unit", choices=["region", "jurisdiction"], default="region")
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--learning-rate-candidates", nargs="+", type=float,
                        help="optional source-validation search; overrides --learning-rate, multiplies budget")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--training-scenarios", type=int, default=16)
    parser.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--sampler", choices=["uniform", "region_balanced"], default="uniform")
    parser.add_argument("--tree-baseline", action="store_true")


def training_options(args):
    return {"objective": args.objective, "selection_metric": args.selection_metric,
        "selection_protocol": args.selection_protocol, "selection_unit": args.selection_unit,
        "model_candidates": ([{"backbone": backbone, "hidden": args.hidden if args.hidden is not None
            else (64 if backbone == "context" else 32)} for backbone in args.backbone_candidates]
            if args.backbone_candidates is not None else None),
        "model_config": {"backbone": args.backbone, "point_estimator": args.point_estimator,
                         "hidden": args.hidden if args.hidden is not None else (64 if args.backbone == "context" else 32)},
        "lr": args.learning_rate, "lr_candidates": args.learning_rate_candidates,
        "batch_size": args.batch_size, "scenarios": args.training_scenarios,
        "schedule": args.schedule, "sampler": args.sampler}
