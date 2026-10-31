"""Post-target, source-fixed V9 mechanism ablations.

The plan is determined solely by frozen configurations and source choices:
for each source-primary scope, compare recompose minus additive for all six
forecasts at that scope's fixed round and spread factor. If that round is R2,
also compare conditional minus global residual risk for the four eligible
forecasts, using canonical R2 identities. Every pair keeps the same forecast,
round, spread factor, scored days, and V7 donor bank. Candidate-wide pairs are
descriptive; the source-primary-aligned pair is flagged, never reselected on
target outcomes. This script is not part of the V9 prediction freeze.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = ROOT / "scripts/analyze_coarse_dynamics_v9.py"


def module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return value


ANALYSIS = module("v9_ablation_base_analysis", ANALYZER_PATH)
RUNNER = ANALYSIS.RUNNER
V7 = ANALYSIS.V7
METRICS = ("crps", "empirical_coverage90", "empirical_width90",
           "empirical_interval_score90", "point_level_mae", "point_shape_mae")


def plan(roles: dict[str, Any], configs: list[dict[str, Any]],
         methods: list[str]) -> list[dict[str, Any]]:
    """Derive every matched comparison from source choices and frozen grid."""
    if set(roles) != {"1", "2", "joint"}:
        raise ValueError("Both source rounds and joint role required")
    lookup = {}
    for config in configs:
        key = int(config["round"]), config["forecast"], config["mode"]
        if key in lookup:
            raise ValueError("Aliased V9 forecast/mode configuration")
        lookup[key] = config
    if len(methods) != 6 or len(set(methods)) != 6:
        raise ValueError("Frozen six-forecast grid required")
    comparisons = []
    for scope in ("1", "2", "joint"):
        selected = roles[scope]["primary"]["configuration"]
        round_, factor = int(selected["round"]), float(selected["factor"])
        if factor not in RUNNER.V7.FACTORS:
            raise ValueError("Source-selected spread factor not in frozen grid")
        primary = next((c for c in configs if c["round"] == round_ and c["name"] == selected["name"]), None)
        if primary is None:
            raise ValueError("Source-selected primary configuration absent")
        if scope != "joint" and round_ != int(scope):
            raise ValueError("Round primary changed identity")
        for method in methods:
            ratio = lookup.get((round_, method, "recompose"))
            additive = lookup.get((round_, method, "additive"))
            if ratio is None or additive is None:
                raise ValueError("Matched additive/recompose grid incomplete")
            comparisons.append({"scope": scope, "mechanism": "mapping_recompose_minus_additive",
                                "forecast": method, "round": round_, "factor": factor,
                                "primary": {"round": round_, "name": ratio["name"], "factor": factor},
                                "reference": {"round": round_, "name": additive["name"], "factor": factor},
                                "source_primary_aligned": bool(method == primary["forecast"] and
                                                               primary["mode"] in ("recompose", "additive")),
                                "selection_scope": "source-fixed descriptive matched pair"})
        if round_ == 2:
            for method in methods:
                conditional = lookup.get((2, method, "residual_conditional"))
                global_ = lookup.get((2, method, "residual_global"))
                if (conditional is None) != (global_ is None):
                    raise ValueError("Conditional/global residual pair incomplete")
                if conditional is None:
                    continue
                comparisons.append({"scope": scope, "mechanism": "residual_conditional_minus_global",
                                    "forecast": method, "round": 2, "factor": factor,
                                    "primary": {"round": 2, "name": conditional["name"], "factor": factor},
                                    "reference": {"round": 2, "name": global_["name"], "factor": factor},
                                    "source_primary_aligned": bool(method == primary["forecast"] and
                                                                   primary["mode"] in ("residual_conditional", "residual_global")),
                                    "selection_scope": "source-fixed descriptive matched pair"})
            if sum(c["scope"] == scope and c["mechanism"] == "residual_conditional_minus_global"
                   for c in comparisons) != 4:
                raise ValueError("R2 requires four residual mechanism pairs")
    return comparisons


def compare(target_days: pd.DataFrame, comparisons: list[dict[str, Any]],
            *, n_bootstrap: int = 5000, seed: int = 17023) -> pd.DataFrame:
    available = [metric for metric in METRICS if metric in target_days]
    if not set(METRICS[:4]).issubset(available):
        raise ValueError("Target proper score/interval columns missing")
    tables = []
    for item in comparisons:
        paired = V7.paired_month_bootstrap(target_days, item["primary"], item["reference"],
                                            available, n_bootstrap=n_bootstrap, seed=seed)
        tables.append(paired.assign(scope=item["scope"], mechanism=item["mechanism"],
            forecast=item["forecast"], round=item["round"], factor=item["factor"],
            source_primary_aligned=item["source_primary_aligned"],
            selection_scope=item["selection_scope"],
            primary_configuration=json.dumps(item["primary"], sort_keys=True),
            reference_configuration=json.dumps(item["reference"], sort_keys=True)))
    return pd.concat(tables, ignore_index=True)


def analyze(root: Path, output: Path, *, n_bootstrap: int = 5000,
            seed: int = 17023) -> dict[str, Any]:
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new V9 ablation analysis directory")
    frozen = RUNNER.verify(root)
    RUNNER.verify_phase(root, "source")
    choices = RUNNER.read(root / "source_choices.json")
    RUNNER.validate_choices(root, choices)
    comparisons = plan(choices["selections"], frozen["configs"], frozen["coarse_methods"])
    # Open target scores only after the complete target phase has been verified.
    target = ANALYSIS.verified_scores(root, "target")
    result = compare(target["day"], comparisons, n_bootstrap=n_bootstrap, seed=seed)
    output.mkdir(parents=True)
    RUNNER.write(output / "source_fixed_ablation_plan.json", {"comparisons": comparisons,
        "source_choices_sha256": RUNNER.sha(root / "source_choices.json"),
        "target_scores_used_for_plan": False})
    result.to_csv(output / "matched_paired_month_bootstrap.csv", index=False)
    (output / "READ_ME.md").write_text(
        "# V9 source-fixed mechanism ablations\n\n"
        "Each scope uses its source-primary spread factor and round. Mapping pairs compare "
        "recompose against additive with the same coarse forecast. R2 risk pairs compare "
        "conditional against global crossfit residuals with the same coarse forecast. "
        "The source-primary-aligned pair is flagged; all grid pairs are descriptive, "
        "with no target selection. Differences are first method minus reference. "
        "Six-month paired bootstrap intervals describe the previously viewed 2026H1 "
        "regions and are not geographic generalization intervals.\n")
    manifest = [root / "prefit_freeze.json", root / "source_choices.json",
                root / "source_completed.json", root / "source_prediction_seal.json",
                root / "source_scores/score_seal.json", root / "target_completed.json",
                root / "target_prediction_seal.json", root / "target_scores/score_seal.json",
                root / "target_scores/day_metrics.csv", ANALYZER_PATH, Path(__file__).resolve()]
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "run": str(root),
                "target_selection_performed": False, "target_training_performed": False,
                "source_choices_sha256": RUNNER.sha(root / "source_choices.json"),
                "comparison_count": len(comparisons), "source_primary_aligned_count": sum(
                    bool(c["source_primary_aligned"]) for c in comparisons),
                "bootstrap": {"n_bootstrap": n_bootstrap, "seed": seed, "calendar_month_blocks": 6},
                "input_hashes": {str(path): RUNNER.sha(path) for path in manifest},
                "output_hashes": {path.name: RUNNER.sha(path) for path in sorted(output.iterdir()) if path.is_file()}}
    RUNNER.write(output / "analysis_metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    analyze(args.run, args.output, n_bootstrap=args.bootstrap)


if __name__ == "__main__":
    main()
