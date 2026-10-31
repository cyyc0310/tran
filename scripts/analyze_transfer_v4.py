"""Freeze source-only references, then analyze globally sealed V4 score tables.

Examples (the last --source directory supplies the final source selection)::

    python scripts/analyze_transfer_v4.py --source results/R1 results/R2 --output results/source-analysis
    python scripts/analyze_transfer_v4.py --source results/R1 results/R2 --external results/external \
        --baseline-selection results/source-analysis/source_only_baseline_selection.json \
        --output results/external-analysis

Only runner-produced score tables and seal metadata are read; raw target labels
and prediction archives are never opened. External comparisons use references
fixed before the external freeze. The bootstrap describes conditional temporal
uncertainty over the observed regions, not generalization to unseen regions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["variant", "method", "calibration"]
BASELINES = ("latest", "seasonal", "joint_half", "analog_half", "harmonic_ridge",
             "local_level", "pooled_ridge", "softmax_moe")
METRICS = ["crps", "mae", "raw_point_mae", "raw_level_mae", "raw_shape_mae",
           "empirical_coverage90", "empirical_width90", "empirical_interval_score90",
           "analytic_coverage90", "analytic_width90", "analytic_interval_score90",
           "empirical_lower_tail_miss", "empirical_upper_tail_miss", "empirical_path_coverage90"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_root(path):
    path = Path(path).resolve()
    return path.parent if path.name == "scores" else path


def verified_scores(path, *, external=False):
    """Check the score/global-seal bindings before opening any daily score table."""
    root = run_root(path)
    plan = json.loads((root / "evaluation_plan.json").read_text())
    expected_study, expected_year = ("holdout", 2025) if external else ("development", 2024)
    if plan.get("study") != expected_study or plan.get("year") != expected_year:
        raise ValueError(f"Expected {expected_study} {expected_year} scores: {root}")
    if external and len(set(plan.get("targets", []))) != 6:
        raise ValueError("External analysis requires the globally sealed six-region study.")
    if not external and set(plan.get("targets", [])) != {"CISO", "FPL", "ISNE"}:
        raise ValueError("Source analysis must use the three predeclared source calibration regions.")
    seal = json.loads((root / "all_predictions_sealed.json").read_text())
    if seal.get("plan_sha256") != sha(root / "evaluation_plan.json"):
        raise ValueError("Evaluation plan differs from the global prediction seal.")
    if external and (seal.get("target_labels_read") is not False
                     or seal.get("prefit_sha256") != sha(root / "prefit_freeze.json")):
        raise ValueError("External global seal lacks the prefit/no-target-label binding.")
    score_dir = root / "scores"
    verification = json.loads((score_dir / "postrun_verification.json").read_text())
    if not verification.get("all_frozen_hashes_match") or not verification.get("target_label_access_after_global_seal"):
        raise ValueError("Runner postrun verification failed.")
    if verification.get("global_seal_sha256") != sha(root / "all_predictions_sealed.json"):
        raise ValueError("Score verification is not bound to this global seal.")
    hashes = verification.get("score_hashes", {})
    required = {"macro_metrics.csv", "region_metrics.csv", "day_metrics.csv"}
    if external:
        required.add("fixed_primary_metrics.csv")
    if not required.issubset(hashes):
        raise ValueError("Runner verification lacks required score hashes.")
    for name, digest in hashes.items():
        if Path(name).name != name or sha(score_dir / name) != digest:
            raise ValueError(f"Score artifact hash mismatch: {name}")
    return root, score_dir, plan


def rank_candidates(macro, regional):
    """Same coverage gate, tie order and equal-region/equal-delay means as runner."""
    for frame, keys in ((macro, ["delay", *KEYS]), (regional, ["region", "delay", *KEYS])):
        if frame.empty or frame.duplicated(keys).any():
            raise ValueError("Empty or duplicate configuration score rows.")
    metrics = [column for column in METRICS if column in macro]
    if not {"crps", "empirical_coverage90", "empirical_width90", "empirical_interval_score90"}.issubset(metrics):
        raise ValueError("Missing source selection metrics.")
    if not np.isfinite(macro[metrics].to_numpy()).all() or not np.isfinite(regional[metrics].to_numpy()).all():
        raise ValueError("Score metrics must be finite.")
    # Every candidate must have exactly the same region x delay support.
    supports = regional.groupby(KEYS, sort=True).apply(
        lambda group: frozenset(zip(group.region, group.delay)), include_groups=False)
    if len(set(supports)) != 1:
        raise ValueError("Candidate grids have different region/delay support.")
    ranked = macro.groupby(KEYS, sort=True)[metrics].mean().reset_index()
    region_mean = regional.groupby(KEYS)[metrics].mean().reset_index()
    check = ranked.merge(region_mean, on=KEYS, suffixes=("_macro", "_region"), validate="one_to_one")
    for metric in metrics:
        if not np.allclose(check[metric + "_macro"], check[metric + "_region"], rtol=1e-9, atol=1e-9):
            raise ValueError("Macro table does not match equal-region/equal-delay averaging.")
    worst = regional.groupby(KEYS).empirical_coverage90.min().rename("worst_region_delay_coverage90")
    ranked = ranked.merge(worst, on=KEYS, validate="one_to_one")
    ranked["coverage_gap"] = np.abs(ranked.empirical_coverage90 - .9)
    ranked["gate_passed"] = ((ranked.coverage_gap <= .05 + 1e-12)
                             & (ranked.worst_region_delay_coverage90 >= .85 - 1e-12))
    ranked["gate_violation"] = (np.maximum(ranked.coverage_gap - .05, 0)
                                + np.maximum(.85 - ranked.worst_region_delay_coverage90, 0))
    ranked = ranked.sort_values(["gate_passed", "gate_violation", "crps", *KEYS],
                                ascending=[False, True, True, True, True, True], kind="stable")
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked.reset_index(drop=True)


def configuration(row):
    return {key: str(row[key]) for key in KEYS}


def source_baseline_selection(ranked):
    baseline = ranked.loc[ranked.method.isin(BASELINES)]
    if baseline.empty:
        raise ValueError("No predeclared baseline methods in source score grid.")
    passing = baseline.loc[baseline.gate_passed].sort_values(["crps", *KEYS], kind="stable")
    closest = baseline.sort_values(["gate_violation", "crps", *KEYS], kind="stable").iloc[0]
    unconstrained = baseline.sort_values(["crps", *KEYS], kind="stable").iloc[0]
    winner = None if passing.empty else passing.iloc[0]
    return {
        "status": "constraint_failed" if winner is None else "passed",
        "configuration": None if winner is None else configuration(winner),
        "reference_configuration": configuration(closest if winner is None else winner),
        "reference_role": "coverage_failed_reference" if winner is None else "source_selected_qualified_baseline",
        "best_by_violation": {"configuration": configuration(closest), "gate_violation": float(closest.gate_violation),
                              "worst_region_delay_coverage90": float(closest.worst_region_delay_coverage90)},
        "unconstrained_configuration": configuration(unconstrained),
        "unconstrained_role": "source_only_unconstrained_crps_baseline",
        "baseline_methods": list(BASELINES), "coverage_gap_limit": .05, "worst_region_delay_floor": .85,
        "selection_rule": "Coverage gate, then minimum source CRPS; failed gate never supplies a qualified winner",
    }


def subset(frame, config):
    selected = frame.loc[np.logical_and.reduce([frame[key].eq(value) for key, value in config.items()])]
    if selected.empty:
        raise ValueError(f"Fixed configuration absent from scores: {config}")
    return selected.copy()


def paired_month_bootstrap(days, primary, reference, *, n_bootstrap=5000, seed=17023):
    """Synchronous calendar-month blocks; daily sum/count then equal group mean.

    Pair daily rows strictly. Each replicate samples the same months jointly for
    every region, lag and metric, retaining contemporaneous cross-domain/lag
    dependence. Regions are fixed; twelve calendar blocks are a small sample.
    Percentile intervals are descriptive and make no multiplicity correction.
    """
    if not isinstance(n_bootstrap, int) or n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")
    metrics = [metric for metric in METRICS if metric in days]
    keys = ["region", "delay", "origin"]
    left, right = subset(days, primary), subset(days, reference)
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("Duplicate paired daily scores.")
    paired = left[keys + metrics].merge(right[keys + metrics], on=keys, how="outer",
                                        suffixes=("_primary", "_reference"), validate="one_to_one", indicator=True)
    if not paired._merge.eq("both").all():
        raise ValueError("Fixed methods have different scored-day support; refusing partial pairing.")
    timestamps = pd.to_datetime(paired.origin, utc=True)
    if not timestamps.dt.year.eq(2025).all():
        raise ValueError("External paired daily scores must be exclusively 2025.")
    paired["month"] = timestamps.dt.month
    months = np.sort(paired.month.unique())
    if len(months) != 12:
        raise ValueError("External bootstrap requires all twelve calendar months.")
    groups = list(paired.groupby(["region", "delay"], sort=True))
    values = np.empty((len(groups), 12, len(metrics), 2))
    counts = np.empty((len(groups), 12))
    for g, (_, group) in enumerate(groups):
        for m, month in enumerate(months):
            block = group.loc[group.month.eq(month)]
            counts[g, m] = len(block)
            if not len(block):
                raise ValueError("Every region/delay must have scored days in every month.")
            for side, name in enumerate(("primary", "reference")):
                values[g, m, :, side] = block[[metric + "_" + name for metric in metrics]].sum().to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite paired metric values.")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(12, size=(n_bootstrap, 12))
    multiplicity = np.stack([(sampled == month).sum(axis=1) for month in range(12)], axis=1)
    denominator = np.einsum("bm,gm->bg", multiplicity, counts)
    simulated = np.einsum("bm,gmks->bgks", multiplicity, values) / denominator[:, :, None, None]
    observed = values.sum(axis=1) / counts.sum(axis=1)[:, None, None]
    strata = [("overall", "all", np.arange(len(groups)))]
    for field, position in (("region", 0), ("delay", 1)):
        labels = sorted({key[position] for key, _ in groups})
        strata.extend((field, str(label), np.array([i for i, (key, _) in enumerate(groups) if key[position] == label]))
        for label in labels)
    rows = []
    for field, label, indices in strata:
        point = observed[indices].mean(axis=0)
        distribution = simulated[:, indices].mean(axis=1)
        delta = distribution[:, :, 0] - distribution[:, :, 1]
        lo, hi = np.quantile(delta, [.025, .975], axis=0)
        for j, metric in enumerate(metrics):
            rows.append({"stratum": field, "label": label, "metric": metric,
                         "primary_mean": float(point[j, 0]), "reference_mean": float(point[j, 1]),
                         "difference_primary_minus_reference": float(point[j, 0] - point[j, 1]),
                         "ci95_low": float(lo[j]), "ci95_high": float(hi[j]),
                         "n_paired_days": int(counts[indices].sum()), "n_region_delay_groups": len(indices),
                         "n_calendar_blocks": 12, "n_bootstrap": n_bootstrap, "seed": seed})
    return pd.DataFrame(rows)


def component_contrasts(macro):
    metrics = [metric for metric in METRICS if metric in macro]
    keys = ["delay", "variant", "method"]
    raw = macro.loc[macro.calibration.eq("source_residual"), keys + metrics]
    components = macro.loc[macro.calibration.str.startswith("component_transport")]
    if raw.empty or components.empty:
        return pd.DataFrame()
    merged = components.merge(raw, on=keys, suffixes=("_component", "_raw"), validate="many_to_one")
    result = merged[keys + ["calibration"]].copy()
    for metric in metrics:
        result[metric + "_delta_component_minus_raw"] = merged[metric + "_component"] - merged[metric + "_raw"]
    return result


def save_figure(fig, output, name):
    import matplotlib.pyplot as plt
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pareto_mask(values):
    """Minimize all coordinates; no choice or external selection is implied."""
    values = np.asarray(values, dtype=float)
    return np.asarray([not np.any(np.all(values <= row, axis=1) & np.any(values < row, axis=1))
                       for row in values])


def figures(ranked, output, prefix, primary=None, reference=None, contrasts=None):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output / ".cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    specs = [("empirical_width90", "empirical_coverage90", "90% interval width (gCO2/kWh)", "Hourly coverage"),
             ("empirical_interval_score90", "crps", "90% interval score (gCO2/kWh)", "CRPS (gCO2/kWh)"),
             ("empirical_width90", "crps", "90% interval width (gCO2/kWh)", "CRPS (gCO2/kWh)")]
    for ax, (x, y, xlabel, ylabel) in zip(axes, specs):
        for passed, color, marker, label in [(False, "#999999", "x", "Coverage gate failed"),
                                             (True, "#009E73", "o", "Coverage gate passed")]:
            part = ranked.loc[ranked.gate_passed.eq(passed)]
            ax.scatter(part[x], part[y], color=color, marker=marker, s=16, alpha=.65, label=label)
        for config, color, marker, label in [(primary, "#0072B2", "*", "Fixed primary"),
                                               (reference, "#D55E00", "s", "Source-fixed reference")]:
            if config is not None:
                part = subset(ranked, config)
                ax.scatter(part[x], part[y], color=color, marker=marker, s=90, label=label, zorder=5)
        if y == "empirical_coverage90":
            ax.axhline(.9, color="black", linestyle="--", linewidth=.8)
            objective = np.column_stack([ranked[x], np.abs(ranked[y] - .9)])
        else:
            objective = ranked[[x, y]].to_numpy()
        front = ranked.loc[pareto_mask(objective)].sort_values(x)
        ax.plot(front[x], front[y], color="#CC79A7", linewidth=.8,
                linestyle=":", label="Descriptive Pareto frontier")
        ax.set(xlabel=xlabel, ylabel=ylabel)
    axes[0].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle(f"{prefix}: descriptive trade-offs; choices fixed on source data", fontsize=10)
    save_figure(fig, output, prefix + "_tradeoffs")
    if contrasts is not None and not contrasts.empty:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
        for calibration, group in contrasts.groupby("calibration", sort=True):
            means = group.groupby("method").mean(numeric_only=True)
            for ax, metric, label in [(axes[0], "crps", "CRPS change (gCO2/kWh)"),
                                      (axes[1], "empirical_interval_score90", "Interval score change (gCO2/kWh)")]:
                ax.plot(means.index, means[metric + "_delta_component_minus_raw"], marker="o", label=calibration)
                ax.axhline(0, color="black", linewidth=.8)
                ax.tick_params(axis="x", rotation=70, labelsize=7)
                ax.set_ylabel(label)
        axes[0].legend(frameon=False, fontsize=7)
        fig.suptitle("Component transport minus raw residual; source development only", fontsize=10)
        save_figure(fig, output, prefix + "_component_vs_raw")


def paired_figure(comparisons, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    positions = comparisons[["stratum", "label"]].drop_duplicates().reset_index(drop=True)
    labels = ["Overall" if row.stratum == "overall" else f"{row.stratum}: {row.label}"
              for row in positions.itertuples()]
    roles = list(comparisons.reference_role.unique())
    for ax, metric, title in zip(axes, ["crps", "empirical_interval_score90"],
                                 ["CRPS difference (gCO2/kWh)", "Interval score difference (gCO2/kWh)"]):
        for r, role in enumerate(roles):
            part = comparisons.loc[comparisons.reference_role.eq(role) & comparisons.metric.eq(metric)]
            part = positions.merge(part, on=["stratum", "label"], validate="one_to_one")
            y = np.arange(len(part)) + (r - (len(roles) - 1) / 2) * .20
            color = ["#0072B2", "#D55E00"][r % 2]
            ax.hlines(y, part.ci95_low, part.ci95_high, color=color, linewidth=1.3)
            ax.scatter(part.difference_primary_minus_reference, y, color=color, s=22,
                       marker=["o", "s"][r % 2], label=role.replace("_", " "))
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linestyle="--", linewidth=.8)
        ax.set_xlabel(title)
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Fixed primary minus source-fixed references: synchronous month-block 95% CIs", fontsize=10)
    save_figure(fig, output, "external_fixed_paired_uncertainty")


def analyze_sources(sources, output):
    rankings, sources_metadata = [], []
    final = None
    for path in sources:
        root, scores, plan = verified_scores(path)
        macro, regional = pd.read_csv(scores / "macro_metrics.csv"), pd.read_csv(scores / "region_metrics.csv")
        ranked = rank_candidates(macro, regional)
        selected_path = scores / "selected_config.json"
        selected = json.loads(selected_path.read_text())
        if selected.get("status") != "passed":
            raise ValueError("Final source primary must have passed its predeclared gate.")
        best = ranked.loc[ranked.gate_passed].sort_values(["crps", *KEYS], kind="stable").iloc[0]
        if configuration(best) != selected.get("configuration"):
            raise ValueError("Fixed primary does not match source-only ranking rule.")
        for name, field in [("macro_metrics.csv", "macro_metrics_sha256"),
                            ("region_metrics.csv", "region_metrics_sha256")]:
            if selected.get(field) != sha(scores / name):
                raise ValueError("Source selection score binding differs.")
        ranked.insert(0, "source_round", root.name)
        rankings.append(ranked)
        metadata = {"root": str(root), "selection_regions": plan["targets"],
                    "artifacts": {str(scores / name): sha(scores / name) for name in
                                  ("macro_metrics.csv", "region_metrics.csv", "selected_config.json")}}
        sources_metadata.append(metadata)
        final = (root, scores, macro, regional, ranked, selected)
        contrasts = component_contrasts(macro)
        if not contrasts.empty:
            contrasts.to_csv(output / f"{root.name}_component_vs_raw.csv", index=False)
        figures(ranked, output, root.name, primary=selected["configuration"], contrasts=contrasts)
    combined = pd.concat(rankings, ignore_index=True)
    combined.to_csv(output / "source_candidate_ranking.csv", index=False)
    # This attribution is development evidence, not a fresh independent test.
    combined.groupby("source_round", sort=False).head(15).to_csv(output / "development_round_comparison.csv", index=False)
    root, scores, macro, regional, ranked, selected = final
    baseline = source_baseline_selection(ranked)
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "analysis_script_sha256": sha(__file__),
            "final_source_root": str(root), "source_rounds": sources_metadata,
            "primary_configuration": selected["configuration"], "primary_selection_sha256": sha(scores / "selected_config.json"),
            "selection_year": 2024, "target_scores_used": False, **baseline}


def validate_frozen_baseline(path, expected, external_root):
    path = Path(path).resolve()
    frozen = json.loads(path.read_text())
    sidecar = path.with_suffix(".sha256")
    if sidecar.read_text().split()[0] != sha(path):
        raise ValueError("Source-only baseline selection hash changed.")
    for field in ("source_rounds", "final_source_root", "primary_configuration", "primary_selection_sha256",
                  "status", "configuration", "reference_configuration", "unconstrained_configuration"):
        if frozen.get(field) != expected.get(field):
            raise ValueError(f"Baseline freeze differs from source-only recomputation: {field}")
    if frozen.get("target_scores_used") is not False or frozen.get("selection_year") != 2024:
        raise ValueError("Baseline freeze must use only source-2024 scores.")
    created = datetime.fromisoformat(frozen["created_utc"]).timestamp()
    freeze_time = min((external_root / name).stat().st_mtime for name in
                      ("prefit_freeze.json", "evaluation_plan.json", "all_predictions_sealed.json"))
    if created > freeze_time or path.stat().st_mtime > freeze_time:
        raise ValueError("Baseline selection was not fixed before the external freeze.")
    plan = json.loads((external_root / "evaluation_plan.json").read_text())
    selected = plan.get("selection", {})
    if (selected.get("source_only_baselines") != frozen
            or selected.get("source_only_baselines_sha256") != sha(path)):
        raise ValueError("External evaluation plan does not embed the exact pre-fixed baseline object/hash.")
    frozen_at = datetime.fromisoformat(selected["source_only_baselines_frozen_at_utc"]).timestamp()
    if not created <= frozen_at <= freeze_time:
        raise ValueError("Embedded baseline fixation time is outside the pre-external window.")
    # Verify the embedded selection also appears in a hash-bound prefit snapshot.
    manifest = json.loads((external_root / "prefit_freeze.json").read_text())
    bound = False
    for name, digest in manifest.get("snapshot_hashes", {}).items():
        if "selected_config" in Path(name).name and Path(name).suffix == ".json":
            snapshot = external_root / name
            if sha(snapshot) == digest and json.loads(snapshot.read_text()) == selected:
                bound = True
    if not bound:
        raise ValueError("External prefit snapshot does not bind the embedded source selection/baselines.")
    return frozen


def analyze_external(path, baseline_path, expected, output, n_bootstrap, seed):
    root = run_root(path)
    frozen = validate_frozen_baseline(baseline_path, expected, root)
    root, scores, plan = verified_scores(root, external=True)
    primary = frozen["primary_configuration"]
    if plan.get("selection", {}).get("configuration") != primary:
        raise ValueError("External primary differs from fixed source primary.")
    fixed = pd.read_csv(scores / "fixed_primary_metrics.csv")
    if len(subset(fixed, primary)) != len(fixed):
        raise ValueError("Fixed primary score table contains another configuration.")
    days = pd.read_csv(scores / "day_metrics.csv")
    macro, regional = pd.read_csv(scores / "macro_metrics.csv"), pd.read_csv(scores / "region_metrics.csv")
    ranked = rank_candidates(macro, regional)
    ranked.to_csv(output / "external_descriptive_candidates.csv", index=False)
    references = [(frozen["reference_role"], frozen["reference_configuration"]),
                  (frozen["unconstrained_role"], frozen["unconstrained_configuration"])]
    comparisons = []
    for role, reference in references:
        table = paired_month_bootstrap(days, primary, reference, n_bootstrap=n_bootstrap, seed=seed)
        table.insert(0, "reference_role", role)
        for key, value in reference.items():
            table["reference_" + key] = value
        comparisons.append(table)
    combined = pd.concat(comparisons, ignore_index=True)
    combined.to_csv(output / "external_fixed_paired_month_bootstrap.csv", index=False)
    fixed.to_csv(output / "external_fixed_primary_metrics.csv", index=False)
    figures(ranked, output, "external", primary=primary, reference=frozen["reference_configuration"])
    paired_figure(combined, output)
    write_json(output / "external_analysis_metadata.json", {
        "baseline_selection_path": str(Path(baseline_path).resolve()), "baseline_selection_sha256": sha(baseline_path),
        "global_prediction_seal_sha256": sha(root / "all_predictions_sealed.json"),
        "pairing": "exact region/delay/origin; all compared days retained",
        "bootstrap": "synchronous 12-month blocks; region-delay sum/count then equal-group macro",
        "uncertainty_scope": "conditional on observed six regions; 12 temporal blocks; unadjusted 95% percentile CIs",
        "reference_status": frozen["status"], "external_selection_performed": False,
        "n_bootstrap": n_bootstrap, "seed": seed,
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", nargs="+", type=Path, required=True,
                        help="Completed development runs in chronological order; final one determines fixed choices")
    parser.add_argument("--external", type=Path)
    parser.add_argument("--baseline-selection", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New immutable analysis directory")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17023)
    args = parser.parse_args(argv)
    if args.external is not None and args.baseline_selection is None:
        parser.error("--external requires a source-only --baseline-selection fixed before external freeze")
    args.output.mkdir(parents=True, exist_ok=False)
    selection = analyze_sources(args.source, args.output)
    if args.external is None:
        path = args.output / "source_only_baseline_selection.json"
        write_json(path, selection)
        path.with_suffix(".sha256").write_text(sha(path) + "  " + path.name + "\n")
        print(json.dumps({"baseline_selection": str(path.resolve()), "sha256": sha(path),
                          "status": selection["status"], "reference": selection["reference_configuration"]}))
    else:
        analyze_external(args.external, args.baseline_selection, selection, args.output, args.bootstrap, args.seed)
        print(json.dumps({"analysis": str(args.output.resolve()), "external_selection_performed": False}))


if __name__ == "__main__":
    main()
