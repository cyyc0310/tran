"""Analyze sealed V6 source choices and fixed 2026H1 external comparisons.

Reads verified score tables and audit metadata only. The runner's source-only
selected_config.json is copied byte-for-byte; external scores never select a
configuration. Six synchronous month blocks provide descriptive uncertainty
conditional on the six observed regions, not a global generalization interval.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


HELPER_PATH = Path(__file__).with_name("analyze_transfer_v4.py")
_SPEC = importlib.util.spec_from_file_location("conditional_v6_analysis_helpers", HELPER_PATH)
V4 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(V4)
KEYS = V4.KEYS
SOURCES = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
HOLDOUTS = ("AZPS", "DUK", "LGEE", "PACE", "PSCO", "SRP")
DELAYS = (0, 30, 90)
POINTS = ("support_gp_medium", "harmonic_ridge", "softmax_moe")
VARIANTS = ("conditional_v6", "anchored_v6")
ROLES = ("primary", "baseline", "unconstrained_baseline", "historical_v5", "cqr")
INTERVAL_METRICS = ["empirical_coverage90", "empirical_width90", "empirical_interval_score90",
                    "empirical_lower_tail_miss", "empirical_upper_tail_miss", "empirical_path_coverage90"]
SCORE_FILES = ("day_metrics.csv", "region_metrics.csv", "macro_metrics.csv", "day_counts.csv",
               "interval_day_metrics.csv", "interval_region_metrics.csv", "interval_macro_metrics.csv")


def read_json(path):
    return json.loads(Path(path).read_text())


def probability_calibrations(variant):
    if variant not in VARIANTS:
        raise ValueError("Declare a supported V6 variant explicitly.")
    ablation = "structured_nohindcast" if variant == "conditional_v6" else "noanchor"
    multipliers = (("x1", "x1p25", "x1p5", "x2") if variant == "conditional_v6" else
                   ("x1", "x1p1", "x1p2", "x1p25", "x1p5", "x2"))
    return tuple(f"{mode}_{suffix}"
        for mode in ("global", "loglinear", "extratrees", "structured", ablation)
        for suffix in multipliers) + (
            "historical_v5_component110", "historical_v5_risk0p5")


def _bound_file(root, name, expected):
    path = root / name
    if not isinstance(expected, str) or V4.sha(path) != expected:
        raise ValueError(f"Missing or changed V6 hash binding: {name}")


def verified_scores(path, *, external=False, variant="conditional_v6"):
    """Verify global pre-label seals, fitted bytes and postrun score bindings."""
    root = V4.run_root(path)
    plan = read_json(root / "evaluation_plan.json")
    expected = ("holdout", 2026, HOLDOUTS) if external else ("development", 2025, SOURCES)
    if (plan.get("study"), plan.get("year"), tuple(plan.get("targets", ()))) != expected:
        raise ValueError("V6 requires the predeclared study, year and region order.")
    config = read_json(root / "run_config.json")
    calibrations = probability_calibrations(variant)
    expected_config = {"study_version": 6, "variants": [variant], "methods": list(POINTS),
        "delays": list(DELAYS), "calibration_modes": list(calibrations),
        "interval_only_modes": ["cqr"], "source_year": 2025, "residual_fit_year": 2024,
        "selection_regions": list(SOURCES), "external_regions": list(HOLDOUTS),
        "external_year": 2026, "external_stop_exclusive": "2026-07-01T00:00:00+00:00",
        "proper_months": [1, 6], "bank_months": [7, 12]}
    multiplier_values = {"x1": 1., "x1p1": 1.1, "x1p2": 1.2, "x1p25": 1.25, "x1p5": 1.5, "x2": 2.}
    parameters = {calibration: {"mode": calibration.rsplit("_", 1)[0],
                               "multiplier": multiplier_values[calibration.rsplit("_", 1)[1]]}
                  for calibration in calibrations if not calibration.startswith("historical_v5_")}
    parameters.update({"historical_v5_component110": {"old_calibration": "seven_component110"},
                       "historical_v5_risk0p5": {"old_calibration": "adaptive_component110_beta0p5"}})
    expected_config["parameters"] = parameters
    if variant == "anchored_v6":
        expected_config["multipliers"] = multiplier_values
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError("V6 frozen grid and fit/selection/confirmation splits changed.")
    origins = pd.DatetimeIndex(pd.to_datetime(plan.get("origins", []), utc=True))
    expected_origins = pd.date_range("2026-01-01", "2026-06-30", freq="D", tz="UTC") if external else pd.date_range(
        "2025-01-01", "2025-12-31", freq="D", tz="UTC")
    if not origins.equals(expected_origins):
        raise ValueError("V6 evaluation origins must span the exact source year or confirmation half-year.")
    seal = read_json(root / "all_predictions_sealed.json")
    for name, field in (("evaluation_plan.json", "plan_sha256"),
                        ("prefit_freeze.json", "prefit_sha256"), ("fit_seal.json", "fit_seal_sha256")):
        _bound_file(root, name, seal.get(field))
    if seal.get("target_labels_read") is not False:
        raise ValueError("Global prediction seal lacks the no-target-label declaration.")
    entries = seal.get("entries", [])
    keys = [tuple(entry[key] for key in ("region", "delay", *KEYS)) for entry in entries]
    planned = {tuple(key) for key in plan.get("expected", [])}
    declared = {(region, delay, variant, method, calibration)
                for region in expected[2] for delay in DELAYS for method in POINTS
                for calibration in (*calibrations, "cqr")}
    if (not keys or len(set(keys)) != len(keys) or set(keys) != planned
            or planned != declared
            or {tuple(key) for key in seal.get("expected", [])} != planned
            or {entry["path"] for entry in entries} != set(seal.get("artifacts", {}))):
        raise ValueError("V6 global seal grid is incomplete or duplicated.")
    if {key[0] for key in keys} != set(expected[2]) or {key[1] for key in keys} != set(DELAYS):
        raise ValueError("V6 sealed region/delay support changed.")
    if {key[2] for key in keys} != {variant}:
        raise ValueError("Unexpected variant in the V6 seal.")
    commit = read_json(root / "evaluation_commit.json")
    _bound_file(root, "evaluation_plan.json", commit.get("plan_sha256"))
    _bound_file(root, "all_predictions_sealed.json", commit.get("global_seal_sha256"))
    fit = read_json(root / "fit_seal.json")
    _bound_file(root, "prefit_freeze.json", fit.get("prefit_sha256"))
    if not fit.get("artifacts"):
        raise ValueError("V6 fit seal has no fitted artifacts.")
    for name, digest in fit["artifacts"].items():
        _bound_file(root, name, digest)
    scores = root / "scores"
    verification = read_json(scores / "postrun_verification.json")
    if (verification.get("all_frozen_hashes_match") is not True
            or verification.get("target_label_access_after_global_seal") is not True):
        raise ValueError("V6 postrun verification failed.")
    _bound_file(root, "all_predictions_sealed.json", verification.get("global_seal_sha256"))
    hashes = verification.get("score_hashes", {})
    required = {*SCORE_FILES, "fixed_primary_metrics.csv" if external else "selected_config.json"}
    if not required.issubset(hashes):
        raise ValueError("V6 postrun verification omits required score tables.")
    for name, digest in hashes.items():
        if Path(name).name != name:
            raise ValueError("Score hashes must refer to score-directory basenames.")
        _bound_file(scores, name, digest)
    # Probabilistic and interval-only predictions have disjoint score tables,
    # whose union must cover exactly the same sealed region/configuration grid.
    score_keys = ["region", "delay", *KEYS]
    scored = pd.concat([pd.read_csv(scores / name, usecols=score_keys) for name in
                        ("region_metrics.csv", "interval_region_metrics.csv")], ignore_index=True)
    if scored.duplicated(score_keys).any() or set(scored.itertuples(index=False, name=None)) != planned:
        raise ValueError("V6 probability/interval score support differs from the global sealed grid.")
    metadata = {"root": str(root), "variant": variant, "year": expected[1], "regions": list(expected[2]),
                "global_prediction_seal_sha256": V4.sha(root / "all_predictions_sealed.json"),
                "fit_seal_sha256": V4.sha(root / "fit_seal.json"),
                "prefit_sha256": V4.sha(root / "prefit_freeze.json"),
                "postrun_verification_sha256": V4.sha(scores / "postrun_verification.json"),
                "score_hashes": hashes}
    return root, scores, plan, config, metadata


def rank_intervals(macro, regional):
    """CQR selection uses interval score and the same explicit coverage gate."""
    for frame, keys in ((macro, ["delay", *KEYS]), (regional, ["region", "delay", *KEYS])):
        if frame.empty or frame.duplicated(keys).any():
            raise ValueError("Empty or duplicate interval score rows.")
        if not set(INTERVAL_METRICS).issubset(frame) or not np.isfinite(frame[INTERVAL_METRICS]).all().all():
            raise ValueError("CQR requires finite interval metrics, with no invented CRPS.")
    supports = regional.groupby(KEYS, sort=True).apply(
        lambda group: frozenset(zip(group.region, group.delay)), include_groups=False)
    if len(set(supports)) != 1:
        raise ValueError("Interval candidates have different region/delay support.")
    ranked = macro.groupby(KEYS, sort=True)[INTERVAL_METRICS].mean().reset_index()
    means = regional.groupby(KEYS, sort=True)[INTERVAL_METRICS].mean().reset_index()
    merged = ranked.merge(means, on=KEYS, suffixes=("_macro", "_region"), validate="one_to_one")
    for metric in INTERVAL_METRICS:
        if not np.allclose(merged[metric + "_macro"], merged[metric + "_region"], atol=1e-9, rtol=1e-9):
            raise ValueError("CQR macro differs from equal-region/equal-delay means.")
    worst = regional.groupby(KEYS).empirical_coverage90.min().rename("worst_region_delay_coverage90")
    ranked = ranked.merge(worst, on=KEYS, validate="one_to_one")
    ranked["coverage_gap"] = np.abs(ranked.empirical_coverage90 - .9)
    ranked["gate_violation"] = (np.maximum(ranked.coverage_gap - .05, 0)
                                + np.maximum(.85 - ranked.worst_region_delay_coverage90, 0))
    ranked["gate_passed"] = ranked.gate_violation <= 1e-12
    return ranked.sort_values(["gate_passed", "gate_violation", "empirical_interval_score90", *KEYS],
                              ascending=[False, True, True, True, True, True], kind="stable").reset_index(drop=True)


def rank_probability(macro, regional):
    ranked = V4.rank_candidates(macro, regional)
    # Match runner._rank's summed-violation tolerance, including boundary ties.
    ranked["gate_passed"] = ranked.gate_violation <= 1e-12
    ranked = ranked.sort_values(["gate_passed", "gate_violation", "crps", *KEYS],
        ascending=[False, True, True, True, True, True], kind="stable").reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def _mode(calibration):
    for mode in ("structured_nohindcast", "structured", "extratrees", "loglinear", "global", "noanchor"):
        if calibration == mode or calibration.startswith(mode + "_") or calibration.startswith(mode + "-"):
            return mode
    return calibration


def validate_selection(source_root, scores, plan, selection, ranked, intervals):
    """Check the runner's frozen choices without modifying or replacing them."""
    if selection.get("source_year") != 2025 or tuple(selection.get("selection_regions", ())) != SOURCES:
        raise ValueError("V6 selection must use only the seven source regions in 2025.")
    for name in ("macro_metrics.csv", "region_metrics.csv", "interval_macro_metrics.csv", "interval_region_metrics.csv"):
        _bound_file(scores, name, selection.get(name.removesuffix(".csv") + "_sha256"))
    _bound_file(source_root, "fit_seal.json", selection.get("fit_seal_sha256"))
    _bound_file(source_root, "all_predictions_sealed.json", selection.get("prediction_global_seal_sha256"))
    if selection.get("fit_artifact_hashes") != read_json(source_root / "fit_seal.json")["artifacts"]:
        raise ValueError("Source selection fitted-artifact binding differs from the fit seal.")
    frozen = read_json(source_root / "prefit_freeze.json")
    package = Path(frozen["root"]) / "src/transcif"
    closure = {str(Path(path).relative_to(package)): digest
               for path, digest in frozen["code_hashes"].items() if Path(path).is_relative_to(package)}
    if selection.get("model_closure_hashes") != closure or selection.get("environment") != frozen["environment"]:
        raise ValueError("Source selection code/environment binding differs from the prefit freeze.")
    roles = selection.get("selections", {})
    if set(roles) != set(ROLES):
        raise ValueError("V6 selection must freeze exactly the five predeclared roles.")
    modes = ranked.calibration.map(_mode)
    families = {"primary": ranked.loc[modes.eq("structured")],
                "baseline": ranked.loc[modes.isin(("global", "loglinear", "extratrees"))],
                "unconstrained_baseline": ranked.loc[modes.isin(("global", "loglinear", "extratrees"))],
                "historical_v5": ranked.loc[ranked.calibration.isin(("historical_v5_component110", "historical_v5_risk0p5"))],
                "cqr": intervals}
    for role in ROLES:
        record, family = roles[role], families[role]
        if family.empty or record.get("status") not in ("passed", "failed", "unconstrained"):
            raise ValueError(f"Missing role family or explicit gate status: {role}")
        metric = "empirical_interval_score90" if role == "cqr" else "crps"
        if role in ("unconstrained_baseline", "historical_v5"):
            if record["status"] != "unconstrained":
                raise ValueError(f"Unconstrained source reference must retain its explicit status: {role}")
            chosen = family.sort_values([metric, *KEYS], kind="stable").iloc[0]
        else:
            passing = family.loc[family.gate_passed]
            expected_status = "failed" if passing.empty else "passed"
            if record["status"] != expected_status:
                raise ValueError(f"Runner did not preserve the source coverage-gate result: {role}")
            chosen = (family.sort_values(["gate_violation", metric, *KEYS], kind="stable").iloc[0]
                      if passing.empty else passing.sort_values([metric, *KEYS], kind="stable").iloc[0])
        if record.get("configuration") != V4.configuration(chosen):
            raise ValueError(f"Frozen source choice differs from its predeclared rule: {role}")
        metric_names = (metric, "empirical_coverage90", "empirical_width90", "empirical_interval_score90",
                        "worst_region_delay_coverage90", "gate_violation")
        if (set(record.get("metrics", {})) != set(metric_names)
                or not all(np.isclose(record["metrics"][name], chosen[name], atol=1e-10, rtol=1e-10)
                           for name in metric_names)):
            raise ValueError(f"Frozen source role metrics differ from verified source score tables: {role}")
    return roles


def paired_month_bootstrap(days, primary, reference, *, metrics=None, n_bootstrap=5000, seed=17023):
    """V4 sum/count algorithm with six synchronous 2026H1 calendar blocks."""
    if not isinstance(n_bootstrap, int) or n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")
    metrics = metrics or [metric for metric in V4.METRICS if metric in days]
    keys = ["region", "delay", "origin"]
    left, right = V4.subset(days, primary), V4.subset(days, reference)
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("Duplicate fixed-comparison daily rows.")
    paired = left[keys + metrics].merge(right[keys + metrics], on=keys, how="outer",
        suffixes=("_primary", "_reference"), validate="one_to_one", indicator=True)
    if not paired._merge.eq("both").all():
        raise ValueError("Fixed methods have different scored days; refusing partial pairing.")
    times = pd.to_datetime(paired.origin, utc=True)
    if not (times.ge("2026-01-01T00:00Z") & times.lt("2026-07-01T00:00Z")).all():
        raise ValueError("External daily rows must be exclusively 2026H1.")
    paired["month"] = times.dt.month
    if set(paired.month) != set(range(1, 7)):
        raise ValueError("External bootstrap requires all six calendar months.")
    groups = list(paired.groupby(["region", "delay"], sort=True))
    if {key for key, _ in groups} != {(region, delay) for region in HOLDOUTS for delay in DELAYS}:
        raise ValueError("Fixed comparisons require every predeclared region/delay.")
    values = np.empty((len(groups), 6, len(metrics), 2))
    counts = np.empty((len(groups), 6))
    for g, (_, group) in enumerate(groups):
        for m in range(6):
            block = group.loc[group.month.eq(m + 1)]
            counts[g, m] = len(block)
            if not len(block):
                raise ValueError("Every region/delay requires scored days in every month.")
            for side, name in enumerate(("primary", "reference")):
                values[g, m, :, side] = block[[metric + "_" + name for metric in metrics]].sum().to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite fixed-comparison metrics.")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(6, size=(n_bootstrap, 6))
    multiplicity = np.stack([(sampled == month).sum(axis=1) for month in range(6)], axis=1)
    denominator = np.einsum("bm,gm->bg", multiplicity, counts)
    simulated = np.einsum("bm,gmks->bgks", multiplicity, values) / denominator[:, :, None, None]
    observed = values.sum(axis=1) / counts.sum(axis=1)[:, None, None]
    strata = [("overall", "all", np.arange(len(groups)))]
    for field, position in (("region", 0), ("delay", 1)):
        for label in sorted({key[position] for key, _ in groups}):
            strata.append((field, str(label), np.array([i for i, (key, _) in enumerate(groups) if key[position] == label])))
    rows = []
    for field, label, indices in strata:
        point = observed[indices].mean(axis=0)
        distribution = simulated[:, indices].mean(axis=1)
        lo, hi = np.quantile(distribution[:, :, 0] - distribution[:, :, 1], [.025, .975], axis=0)
        for j, metric in enumerate(metrics):
            rows.append({"stratum": field, "label": label, "metric": metric,
                "primary_mean": float(point[j, 0]), "reference_mean": float(point[j, 1]),
                "difference_primary_minus_reference": float(point[j, 0] - point[j, 1]),
                "ci95_low": float(lo[j]), "ci95_high": float(hi[j]),
                "n_paired_days": int(counts[indices].sum()), "n_region_delay_groups": len(indices),
                "n_calendar_blocks": 6, "n_bootstrap": n_bootstrap, "seed": seed})
    return pd.DataFrame(rows)


def audit_links(root):
    """Link saved proper/bank/selection/target feature audits without raw labels."""
    paths = set(root.rglob("*.json")) | set(root.rglob("*.csv"))
    return [{"path": str(path), "sha256": V4.sha(path)} for path in sorted(paths)
            if "predictions" not in path.parts
            and any("audit" in part.lower() or "feature" in part.lower()
                    for part in path.relative_to(root).parts)]


def region_figure(regional, roles, output, prefix):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for role in ("primary", "baseline", "unconstrained_baseline", "historical_v5"):
        record = roles[role]
        selected = V4.subset(regional, record["configuration"]).groupby("region", sort=True).mean(numeric_only=True)
        label = role + (" (gate failed)" if record["status"] == "failed" else "")
        for ax, metric, ylabel in zip(axes, ["empirical_coverage90", "empirical_width90", "crps"],
            ["Hourly coverage", "90% interval width (gCO2/kWh)", "CRPS (gCO2/kWh)"]):
            ax.plot(selected.index, selected[metric], marker="o", linewidth=1, label=label)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=35)
    axes[0].axhline(.9, color="black", linestyle="--", linewidth=.8)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(prefix + ": source-fixed configurations by region", fontsize=10)
    V4.save_figure(fig, output, prefix + "_by_region")


def interval_figure(probability_ranked, interval_ranked, roles, output, prefix):
    """Show CQR as intervals only, without assigning it a CRPS coordinate."""
    import matplotlib.pyplot as plt
    primary = V4.subset(probability_ranked, roles["primary"]["configuration"])
    cqr = V4.subset(interval_ranked, roles["cqr"]["configuration"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for ax, metric, ylabel in zip(axes, ["empirical_coverage90", "empirical_interval_score90"],
        ["Hourly coverage", "90% interval score (gCO2/kWh)"]):
        ax.scatter(interval_ranked.empirical_width90, interval_ranked[metric],
                   color="#999999", s=20, alpha=.6, label="All CQR candidates (descriptive)")
        for table, color, marker, label in ((primary, "#0072B2", "*", "Source-fixed primary"),
                                           (cqr, "#D55E00", "s", "Source-fixed CQR")):
            ax.scatter(table.empirical_width90, table[metric], color=color, marker=marker, s=90, label=label)
        ax.set(xlabel="90% interval width (gCO2/kWh)", ylabel=ylabel)
    axes[0].axhline(.9, color="black", linestyle="--", linewidth=.8)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(prefix + ": interval-only CQR comparison; configurations fixed on source", fontsize=10)
    V4.save_figure(fig, output, prefix + "_cqr_interval_tradeoffs")


def paired_figure(comparisons, output):
    import matplotlib.pyplot as plt
    positions = comparisons[["stratum", "label"]].drop_duplicates().reset_index(drop=True)
    labels = ["Overall" if row.stratum == "overall" else f"{row.stratum}: {row.label}"
              for row in positions.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    roles = list(comparisons.reference_role.unique())
    for ax, metric, title in zip(axes, ["crps", "empirical_interval_score90"],
        ["CRPS difference (gCO2/kWh)", "Interval score difference (gCO2/kWh)"]):
        for i, role in enumerate(roles):
            part = positions.merge(comparisons.loc[comparisons.reference_role.eq(role)
                & comparisons.metric.eq(metric)], on=["stratum", "label"], validate="one_to_one")
            y = np.arange(len(part)) + (i - (len(roles) - 1) / 2) * .20
            color = ("#0072B2", "#D55E00", "#009E73")[i]
            ax.hlines(y, part.ci95_low, part.ci95_high, color=color, linewidth=1.3)
            ax.scatter(part.difference_primary_minus_reference, y, color=color, marker=("o", "s", "^")[i],
                       s=22, label=role.replace("_", " "))
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linestyle="--", linewidth=.8)
        ax.set_xlabel(title)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Source-fixed primary minus references: descriptive six-month block 95% intervals", fontsize=10)
    V4.save_figure(fig, output, "external_fixed_paired_uncertainty")


def source_analysis(path, output, *, variant="conditional_v6"):
    root, scores, plan, config, metadata = verified_scores(path, variant=variant)
    macro, regional = pd.read_csv(scores / "macro_metrics.csv"), pd.read_csv(scores / "region_metrics.csv")
    ranked = rank_probability(macro, regional)
    interval_ranked = rank_intervals(pd.read_csv(scores / "interval_macro_metrics.csv"),
                                     pd.read_csv(scores / "interval_region_metrics.csv"))
    selected_path = scores / "selected_config.json"
    selection = read_json(selected_path)
    roles = validate_selection(root, scores, plan, selection, ranked, interval_ranked)
    ranked.to_csv(output / "source_candidate_ranking.csv", index=False)
    interval_ranked.to_csv(output / "source_interval_candidate_ranking.csv", index=False)
    regional.to_csv(output / "source_candidates_by_region.csv", index=False)
    V4.figures(ranked, output, "source_v6", primary=roles["primary"]["configuration"],
               reference=roles["baseline"]["configuration"])
    region_figure(regional, roles, output, "source_v6")
    interval_figure(ranked, interval_ranked, roles, output, "source_v6")
    shutil.copyfile(selected_path, output / "selection.json")
    metadata.update(selection_path=str(selected_path), selection_sha256=V4.sha(selected_path),
                    roles=roles, audit_links=audit_links(root))
    return root, selection, metadata


def validate_external_selection(root, config, plan, selection, source_root):
    if config.get("source_selection") != selection or plan.get("selection") != selection:
        raise ValueError("External run does not embed the exact source-frozen selection.")
    provenance = config.get("source_provenance", {})
    if (provenance.get("fitted_bytes_identical") is not True
            or provenance.get("source_fit_seal_sha256") != V4.sha(source_root / "fit_seal.json")):
        raise ValueError("External fitted bytes are not bound to the source-frozen fit.")
    source_fit = read_json(source_root / "fit_seal.json")["artifacts"]
    external_fit = read_json(root / "fit_seal.json")["artifacts"]
    if provenance.get("source_fit_artifact_hashes") != source_fit:
        raise ValueError("External provenance does not retain the exact source fit-artifact manifest.")
    for name, digest in source_fit.items():
        # The external run configuration embeds the source choice/provenance;
        # its fitted model and calibration artifact bytes remain unchanged.
        if name == "run_config.json":
            continue
        if external_fit.get(name) != digest:
            raise ValueError("External model/calibrator bytes differ from source selection.")


def external_analysis(path, source_root, selection, output, n_bootstrap, seed, *, variant="conditional_v6"):
    root, scores, plan, config, metadata = verified_scores(path, external=True, variant=variant)
    validate_external_selection(root, config, plan, selection, source_root)
    roles = selection["selections"]
    primary = roles["primary"]["configuration"]
    macro, regional = pd.read_csv(scores / "macro_metrics.csv"), pd.read_csv(scores / "region_metrics.csv")
    ranked = rank_probability(macro, regional).rename(columns={"rank": "descriptive_order"})
    ranked["external_selection_performed"] = False
    ranked.to_csv(output / "external_descriptive_candidates.csv", index=False)
    regional.to_csv(output / "external_candidates_by_region.csv", index=False)
    interval_macro = pd.read_csv(scores / "interval_macro_metrics.csv")
    interval_regional = pd.read_csv(scores / "interval_region_metrics.csv")
    interval_ranked = rank_intervals(interval_macro, interval_regional)
    interval_ranked.to_csv(output / "external_descriptive_interval_candidates.csv", index=False)
    interval_regional.to_csv(output / "external_interval_candidates_by_region.csv", index=False)
    needed = [roles[role]["configuration"] for role in ROLES if role != "cqr"]
    chunks = []
    for chunk in pd.read_csv(scores / "day_metrics.csv", chunksize=50000):
        mask = np.zeros(len(chunk), dtype=bool)
        for configuration in needed:
            mask |= np.logical_and.reduce([chunk[key].eq(value) for key, value in configuration.items()])
        chunks.append(chunk.loc[mask])
    days = pd.concat(chunks, ignore_index=True)
    comparisons = []
    for role in ("baseline", "unconstrained_baseline", "historical_v5"):
        table = paired_month_bootstrap(days, primary, roles[role]["configuration"], n_bootstrap=n_bootstrap, seed=seed)
        table.insert(0, "reference_role", role)
        table["reference_source_status"] = roles[role]["status"]
        table["primary_source_status"] = roles["primary"]["status"]
        for key, value in roles[role]["configuration"].items():
            table["reference_" + key] = value
        comparisons.append(table)
    combined = pd.concat(comparisons, ignore_index=True)
    combined.to_csv(output / "external_fixed_paired_month_bootstrap.csv", index=False)
    interval_days = pd.read_csv(scores / "interval_day_metrics.csv")
    interval_keys = ["region", "delay", *KEYS, "origin", *INTERVAL_METRICS]
    interval_days = pd.concat([V4.subset(days, primary)[interval_keys],
                              V4.subset(interval_days, roles["cqr"]["configuration"])[interval_keys]], ignore_index=True)
    cqr = paired_month_bootstrap(interval_days, primary, roles["cqr"]["configuration"],
                                 metrics=INTERVAL_METRICS, n_bootstrap=n_bootstrap, seed=seed)
    cqr["reference_role"] = "cqr"
    cqr["reference_source_status"] = roles["cqr"]["status"]
    cqr.to_csv(output / "external_fixed_cqr_interval_comparison.csv", index=False)
    V4.subset(macro, primary).to_csv(output / "external_fixed_primary_metrics.csv", index=False)
    V4.subset(interval_macro, roles["cqr"]["configuration"]).to_csv(output / "external_fixed_cqr_metrics.csv", index=False)
    V4.figures(ranked, output, "external_v6", primary=primary, reference=roles["baseline"]["configuration"])
    paired_figure(combined, output)
    region_figure(regional, roles, output, "external_v6")
    interval_figure(ranked, interval_ranked, roles, output, "external_v6")
    metadata.update(audit_links=audit_links(root), external_selection_performed=False,
                    source_selection_sha256=V4.sha(source_root / "scores/selected_config.json"),
                    fixed_role_status={role: roles[role]["status"] for role in ROLES})
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--external", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default="conditional_v6",
                        help="Predeclared source/external variant; never inferred from score performance")
    parser.add_argument("--output", type=Path, required=True, help="New immutable analysis directory")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17023)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output / ".matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(args.output / ".cache"))
    source_root, selection, source_metadata = source_analysis(args.source, args.output, variant=args.variant)
    external_metadata = (external_analysis(args.external, source_root, selection, args.output, args.bootstrap, args.seed, variant=args.variant)
                         if args.external is not None else None)
    V4.write_json(args.output / "analysis_metadata.json", {
        "study_version": 6, "variant": args.variant, "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script_sha256": V4.sha(__file__), "analysis_helper_sha256": V4.sha(HELPER_PATH),
        "source": source_metadata, "external": external_metadata,
        "selection_sha256": V4.sha(args.output / "selection.json"),
        "external_selection_performed": False,
        "bootstrap": "synchronous six calendar-month blocks; region-delay sum/count then equal-group mean",
        "uncertainty_scope": "descriptive conditional on the fixed six observed regions and 2026H1; six blocks; unadjusted percentile intervals; no global generalization guarantee",
        "cqr_scope": "interval scores only; no CRPS or probabilistic distribution is attributed to CQR",
        "n_bootstrap": args.bootstrap, "seed": args.seed})
    print(json.dumps({"analysis": str(args.output.resolve()), "selection_sha256": V4.sha(args.output / "selection.json"),
                      "primary_status": selection["selections"]["primary"]["status"], "external_selection_performed": False}))


if __name__ == "__main__":
    main()
