"""Freeze cross-round source choices before scoring either V6 external run.

The freeze phase never reads external labels or score tables. Analyze requires
that immutable joint freeze and reports source-fixed comparisons only. Temporal
bootstrap intervals describe six observed 2026H1 month blocks, not unseen grids.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
HELPER = HERE.with_name("analyze_conditional_transfer_v6.py")
SPEC = importlib.util.spec_from_file_location("coarse_risk_joint_helpers", HELPER)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
V4 = A.V4
VARIANTS = ("conditional_v6", "anchored_v6")
IMPLEMENTATIONS = (HERE, HELPER, A.HELPER_PATH)
PROTOCOL = HERE.parents[1] / "docs/COARSE_RISK_V6_JOINT_CONFIRMATION.md"


def source_bundle(path, variant):
    root, scores, plan, config, metadata = A.verified_scores(path, variant=variant)
    selected = A.read_json(scores / "selected_config.json")
    ranked = A.rank_probability(pd.read_csv(scores / "macro_metrics.csv"), pd.read_csv(scores / "region_metrics.csv"))
    intervals = A.rank_intervals(pd.read_csv(scores / "interval_macro_metrics.csv"),
                                 pd.read_csv(scores / "interval_region_metrics.csv"))
    A.validate_selection(root, scores, plan, selected, ranked, intervals)
    manifest = A.read_json(root / "prefit_freeze.json")
    return {"root": root, "variant": variant, "selection": selected, "ranked": ranked,
            "interval_ranked": intervals, "metadata": metadata,
            "selection_sha256": V4.sha(scores / "selected_config.json"),
            "frozen_code_protocol_hashes": {path: digest for path, digest in manifest["code_hashes"].items()
                                            if Path(path).suffix in (".py", ".md")}}


def _role_record(bundle, record):
    return {"source_run": str(bundle["root"]), "variant": bundle["variant"],
            "configuration": deepcopy(record["configuration"]), "status": record["status"],
            "source_metrics": deepcopy(record["metrics"])}


def joint_roles(bundles):
    """Use source-only gates, scores and stable configuration tie ordering."""
    result = {}
    for role in A.ROLES:
        candidates = [_role_record(bundle, bundle["selection"]["selections"][role]) for bundle in bundles]
        metric = "empirical_interval_score90" if role == "cqr" else "crps"
        if role in ("primary", "baseline", "cqr"):
            passing = [record for record in candidates if record["status"] == "passed"]
            choices = passing or candidates
            order = lambda record: (([] if passing else [record["source_metrics"]["gate_violation"]])
                                    + [record["source_metrics"][metric]]
                                    + [record["configuration"][key] for key in A.KEYS])
        else:
            choices = candidates
            order = lambda record: [record["source_metrics"][metric],
                                    *[record["configuration"][key] for key in A.KEYS]]
        result[role] = deepcopy(min(choices, key=order))
    second = bundles[1]
    primary = result["source2_primary"] = _role_record(second, second["selection"]["selections"]["primary"])
    calibration = primary["configuration"]["calibration"]
    if not calibration.startswith("structured_x"):
        raise ValueError("Second-round primary must be a structured multiplier configuration.")
    for role, mode in (("source2_noanchor", "noanchor"), ("source2_simple_anchor", "global")):
        config = {**primary["configuration"], "calibration": mode + "_" + calibration.removeprefix("structured_")}
        row = V4.subset(second["ranked"], config)
        if len(row) != 1:
            raise ValueError(f"Exactly one source2 same-point/same-multiplier {mode} comparison is required.")
        row = row.iloc[0]
        metrics = {metric: float(row[metric]) for metric in ("crps", *A.INTERVAL_METRICS,
                    "worst_region_delay_coverage90", "gate_violation") if metric in row}
        result[role] = {"source_run": str(second["root"]), "variant": second["variant"],
            "configuration": config, "status": "passed" if row.gate_passed else "failed", "source_metrics": metrics,
            "selection_rule": "Exact source2-primary point method and multiplier; no comparison-score selection"}
    return result


def _artifact(root, name, digest):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Artifact path must remain within its sealed run.")
    A._bound_file(root, name, digest)


def sealed_external(path, bundle, *, require_unscored):
    """Check external metadata/archives without reading labels or scores."""
    root = V4.run_root(path)
    if require_unscored and (root / "scores").exists():
        raise ValueError("Joint freeze requires both external scores directories to be absent.")
    variant = bundle["variant"]
    config, plan = A.read_json(root / "run_config.json"), A.read_json(root / "evaluation_plan.json")
    source_config = A.read_json(bundle["root"] / "run_config.json")
    if config != {**source_config, "source_selection": bundle["selection"],
                  "source_provenance": config.get("source_provenance")}:
        raise ValueError("External grid/bridge configuration differs from its exact source-frozen configuration.")
    if config.get("source_provenance", {}).get("source_run") != str(bundle["root"]):
        raise ValueError("External provenance points to another source run.")
    if (config.get("variants") != [variant] or config.get("calibration_modes") != list(A.probability_calibrations(variant))
            or config.get("methods") != list(A.POINTS) or config.get("delays") != list(A.DELAYS)):
        raise ValueError("External variant or complete configuration grid changed.")
    if (plan.get("study"), plan.get("year"), tuple(plan.get("targets", ()))) != ("holdout", 2026, A.HOLDOUTS):
        raise ValueError("Joint confirmation requires exactly the six new regions in 2026H1.")
    origins = pd.DatetimeIndex(pd.to_datetime(plan.get("origins", []), utc=True))
    if not origins.equals(pd.date_range("2026-01-01", "2026-06-30", tz="UTC")):
        raise ValueError("Joint confirmation requires all 181 predeclared daily origins.")
    seal = A.read_json(root / "all_predictions_sealed.json")
    if seal.get("target_labels_read") is not False:
        raise ValueError("External global seal must declare no prior target-label access.")
    for name, field in (("evaluation_plan.json", "plan_sha256"), ("prefit_freeze.json", "prefit_sha256"),
                        ("fit_seal.json", "fit_seal_sha256")):
        _artifact(root, name, seal.get(field))
    expected = {(region, delay, variant, method, calibration)
        for region in A.HOLDOUTS for delay in A.DELAYS for method in A.POINTS
        for calibration in (*A.probability_calibrations(variant), "cqr")}
    entries = seal.get("entries", [])
    keys = [tuple(entry[key] for key in ("region", "delay", *A.KEYS)) for entry in entries]
    if (len(keys) != len(expected) or set(keys) != expected
            or len({entry["path"] for entry in entries}) != len(entries)
            or {tuple(key) for key in seal.get("expected", [])} != expected
            or {tuple(key) for key in plan.get("expected", [])} != expected
            or {entry["path"] for entry in entries} != set(seal.get("artifacts", {}))):
        raise ValueError("External seal lacks the complete unique new-region prediction grid.")
    for entry in entries:
        kind = "interval_only" if entry["calibration"] == "cqr" else "probabilistic"
        if entry.get("kind") != kind:
            raise ValueError("CQR/probabilistic archive kind differs from the declared calibration.")
    for name, digest in seal["artifacts"].items():
        _artifact(root, name, digest)
    fit = A.read_json(root / "fit_seal.json")
    _artifact(root, "prefit_freeze.json", fit.get("prefit_sha256"))
    for name, digest in fit.get("artifacts", {}).items():
        _artifact(root, name, digest)
    _artifact(root, "run_config.json", fit.get("artifacts", {}).get("run_config.json"))
    if seal.get("fit_artifacts") != fit["artifacts"]:
        raise ValueError("Global prediction seal differs from its fitted-artifact manifest.")
    commit = A.read_json(root / "evaluation_commit.json")
    _artifact(root, "evaluation_plan.json", commit.get("plan_sha256"))
    _artifact(root, "all_predictions_sealed.json", commit.get("global_seal_sha256"))
    A.validate_external_selection(root, config, plan, bundle["selection"], bundle["root"])
    manifest = A.read_json(root / "prefit_freeze.json")
    label_data = Path(plan.get("label_data", "")).resolve()
    if not plan.get("label_data") or str(label_data) != manifest.get("external_data"):
        raise ValueError("External label_data must be its separately frozen confirmation dataset.")
    confirmation_hashes = {}
    for name in ("monthly.csv", *(f"{region}_hourly.csv" for region in A.HOLDOUTS)):
        digest = manifest.get("data_hashes", {}).get(str(label_data / name))
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("External prefit lacks a valid confirmation label/coarse-data hash: " + name)
        confirmation_hashes[name] = digest
    embedded_selection = False
    for name, digest in manifest.get("snapshot_hashes", {}).items():
        _artifact(root, name, digest)
        if Path(name).name == "selected_config.json" and A.read_json(root / name) == bundle["selection"]:
            if digest != bundle["selection_sha256"]:
                raise ValueError("Embedded source selection bytes differ from their original source hash.")
            embedded_selection = True
    if not embedded_selection:
        raise ValueError("External prefit snapshot must bind the exact source selection before prediction.")
    if require_unscored and (root / "scores").exists():
        raise ValueError("External scoring started during the joint-freeze verification.")
    return {"root": str(root), "variant": variant, "prediction_count": len(expected),
        "global_prediction_seal_sha256": V4.sha(root / "all_predictions_sealed.json"),
        "prefit_sha256": V4.sha(root / "prefit_freeze.json"), "fit_seal_sha256": V4.sha(root / "fit_seal.json"),
        "evaluation_plan_sha256": V4.sha(root / "evaluation_plan.json"),
        "confirmation_label_data_root": str(label_data), "confirmation_data_hashes": confirmation_hashes,
        "source_selection_sha256": bundle["selection_sha256"],
        "scores_absent_verified_at_freeze": bool(require_unscored), "target_labels_read_at_global_seal": False,
        "frozen_code_protocol_hashes": {path: digest for path, digest in manifest["code_hashes"].items()
                                       if Path(path).suffix in (".py", ".md")}}


def freeze_joint(source_paths, external_paths, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Joint freeze requires a new output directory.")
    implementation_hashes = {str(path): V4.sha(path) for path in IMPLEMENTATIONS}
    protocol_hash = V4.sha(PROTOCOL)
    dependency_hashes = {**implementation_hashes, str(PROTOCOL): protocol_hash}
    bundles = [source_bundle(path, variant) for path, variant in zip(source_paths, VARIANTS)]
    external = [sealed_external(path, bundle, require_unscored=True) for path, bundle in zip(external_paths, bundles)]
    if external[0]["confirmation_data_hashes"] != external[1]["confirmation_data_hashes"]:
        raise ValueError("The two external runs must use identical confirmation hourly and monthly data hashes.")
    fixed = joint_roles(bundles)
    if any((Path(record["root"]) / "scores").exists() for record in external):
        raise ValueError("External scoring began before joint choice fixation.")
    output.mkdir(parents=True)
    (output / "snapshot").mkdir()
    for path in (*IMPLEMENTATIONS, PROTOCOL):
        shutil.copyfile(path, output / "snapshot" / path.name)
        if (V4.sha(path) != dependency_hashes[str(path)]
                or V4.sha(output / "snapshot" / path.name) != dependency_hashes[str(path)]):
            raise ValueError("Joint implementation/protocol changed while fixation was in progress.")
    sources = []
    for i, bundle in enumerate(bundles, 1):
        folder = output / f"source-r{i}"
        folder.mkdir()
        shutil.copyfile(bundle["root"] / "scores/selected_config.json", folder / "selection.json")
        if V4.sha(folder / "selection.json") != bundle["selection_sha256"]:
            raise ValueError("Source selection changed while the joint freeze was being written.")
        bundle["ranked"].to_csv(folder / "candidate_ranking.csv", index=False)
        bundle["interval_ranked"].to_csv(folder / "interval_candidate_ranking.csv", index=False)
        sources.append({"root": str(bundle["root"]), "variant": bundle["variant"],
            "selection_sha256": bundle["selection_sha256"], "metadata": bundle["metadata"],
            "selections": bundle["selection"]["selections"],
            "frozen_code_protocol_hashes": bundle["frozen_code_protocol_hashes"]})
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "study": "coarse_risk_v6_joint",
        "source_year": 2025, "confirmation_year": 2026, "confirmation_stop_exclusive": "2026-07-01T00:00:00+00:00",
        "sources": sources, "external_runs": external, "fixed_roles": fixed,
        "confirmation_data_hashes": external[0]["confirmation_data_hashes"],
        "implementation_hashes": implementation_hashes,
        "protocol_path": str(PROTOCOL), "protocol_sha256": protocol_hash,
        "snapshot_hashes": {str(path.relative_to(output)): V4.sha(path) for path in sorted(output.rglob("*")) if path.is_file()},
        "selection_rule": "Source-only passed coverage gate then minimum CRPS/IS across rounds; if all fail minimum violation then score; unconstrained references minimum source CRPS; stable variant/method/calibration ties",
        "target_scores_read": False, "external_selection_performed": False,
        "both_external_score_directories_absent": True,
        "bootstrap_replicates": 5000, "bootstrap_seed": 17023,
        "uncertainty_scope": "Descriptive conditional on six observed regions and six synchronous 2026H1 month blocks; no global CI"}
    if any((Path(record["root"]) / "scores").exists() for record in external):
        raise ValueError("External scoring started before joint-freeze publication.")
    record["created_utc"] = datetime.now(timezone.utc).isoformat()
    V4.write_json(output / "joint_freeze.json", record)
    (output / "joint_freeze.sha256").write_text(V4.sha(output / "joint_freeze.json") + "  joint_freeze.json\n")
    return record


def verify_joint_freeze(output):
    output = Path(output).resolve()
    path = output / "joint_freeze.json"
    if (output / "joint_freeze.sha256").read_text().split()[0] != V4.sha(path):
        raise ValueError("Immutable joint freeze hash changed.")
    frozen = A.read_json(path)
    if (frozen.get("target_scores_read") is not False or frozen.get("external_selection_performed") is not False
            or frozen.get("both_external_score_directories_absent") is not True):
        raise ValueError("Joint selection lacks its no-target-score freeze declarations.")
    for name, digest in frozen["implementation_hashes"].items():
        if V4.sha(name) != digest:
            raise ValueError("Joint analysis implementation changed after fixation.")
    if V4.sha(frozen["protocol_path"]) != frozen["protocol_sha256"]:
        raise ValueError("Joint confirmation protocol changed after fixation.")
    for name, digest in frozen["snapshot_hashes"].items():
        _artifact(output, name, digest)
    return frozen


def _fixed_daily(runs, config, *, interval=False):
    scores = runs[config["variant"]] / "scores"
    file = "interval_day_metrics.csv" if interval else "day_metrics.csv"
    parts = []
    for chunk in pd.read_csv(scores / file, chunksize=50000):
        mask = np.logical_and.reduce([chunk[key].eq(value) for key, value in config.items()])
        parts.append(chunk.loc[mask])
    selected = pd.concat(parts, ignore_index=True)
    if selected.empty:
        raise ValueError("Source-fixed configuration has no external scored days.")
    return selected


def analyze_joint(output):
    output = Path(output).resolve()
    frozen = verify_joint_freeze(output)
    analysis = output / "analysis"
    if analysis.exists():
        raise FileExistsError("Joint analysis is immutable; its analysis directory already exists.")
    bundles = [source_bundle(Path(record["root"]), variant) for record, variant in zip(frozen["sources"], VARIANTS)]
    if joint_roles(bundles) != frozen["fixed_roles"]:
        raise ValueError("Joint roles differ from immutable source-only fixation.")
    runs, metadata = {}, []
    for bundle, saved in zip(bundles, frozen["external_runs"]):
        if bundle["selection_sha256"] != saved["source_selection_sha256"]:
            raise ValueError("Source selection changed after joint freeze.")
        root, _, _, _, verification = A.verified_scores(saved["root"], external=True, variant=bundle["variant"])
        actual = sealed_external(root, bundle, require_unscored=False)
        for key in ("global_prediction_seal_sha256", "prefit_sha256", "fit_seal_sha256", "evaluation_plan_sha256"):
            if actual[key] != saved[key]:
                raise ValueError("External seal/fitted bytes changed after joint fixation: " + key)
        if (actual["confirmation_data_hashes"] != saved["confirmation_data_hashes"]
                or actual["confirmation_data_hashes"] != frozen["confirmation_data_hashes"]):
            raise ValueError("External confirmation data hashes differ from the identical-data joint freeze.")
        if saved.get("scores_absent_verified_at_freeze") is not True:
            raise ValueError("Both external score directories must have been absent at fixation.")
        created = datetime.fromisoformat(frozen["created_utc"])
        postrun_time = datetime.fromtimestamp((root / "scores/postrun_verification.json").stat().st_mtime, timezone.utc)
        if created > postrun_time:
            raise ValueError("Joint choices were fixed after external postrun publication.")
        runs[bundle["variant"]] = root
        metadata.append(verification)
    analysis.mkdir()
    os.environ.setdefault("MPLCONFIGDIR", str(analysis / ".matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(analysis / ".cache"))
    import matplotlib
    matplotlib.use("Agg")
    roles = frozen["fixed_roles"]
    frames = {role: _fixed_daily(runs, record["configuration"], interval=role == "cqr") for role, record in roles.items()}
    comparisons = []
    for role in ("baseline", "unconstrained_baseline", "historical_v5"):
        configs = (roles["primary"]["configuration"], roles[role]["configuration"])
        days = frames["primary"] if configs[0] == configs[1] else pd.concat([frames["primary"], frames[role]], ignore_index=True)
        table = A.paired_month_bootstrap(days, *configs)
        table["reference_role"] = role
        table["primary_source_status"], table["reference_source_status"] = roles["primary"]["status"], roles[role]["status"]
        comparisons.append(table)
    combined = pd.concat(comparisons, ignore_index=True)
    combined.to_csv(analysis / "joint_fixed_paired_month_bootstrap.csv", index=False)
    for role, name in (("source2_noanchor", "noanchor"), ("source2_simple_anchor", "simple_anchor")):
        ablation = A.paired_month_bootstrap(pd.concat([frames["source2_primary"], frames[role]], ignore_index=True),
            roles["source2_primary"]["configuration"], roles[role]["configuration"])
        ablation["reference_role"] = role
        ablation["primary_source_status"], ablation["reference_source_status"] = roles["source2_primary"]["status"], roles[role]["status"]
        ablation.to_csv(analysis / f"source2_primary_minus_exact_{name}.csv", index=False)
    columns = ["region", "delay", *A.KEYS, "origin", *A.INTERVAL_METRICS]
    interval_days = pd.concat([frames["primary"][columns], frames["cqr"][columns]], ignore_index=True)
    cqr = A.paired_month_bootstrap(interval_days, roles["primary"]["configuration"], roles["cqr"]["configuration"], metrics=A.INTERVAL_METRICS)
    cqr.to_csv(analysis / "joint_fixed_cqr_interval_comparison.csv", index=False)
    role_tables = []
    for role, record in roles.items():
        metrics = A.INTERVAL_METRICS if role == "cqr" else [metric for metric in V4.METRICS if metric in frames[role]]
        regional = frames[role].groupby(["region", "delay"], sort=True)[metrics].mean().reset_index()
        regional["role"], regional["source_status"] = role, record["status"]
        for key, value in record["configuration"].items():
            regional[key] = value
        role_tables.append(regional)
    by_region = pd.concat(role_tables, ignore_index=True)
    by_region.to_csv(analysis / "fixed_roles_region_delay_metrics.csv", index=False)
    by_region.groupby(["role", "source_status", *A.KEYS], sort=True).mean(numeric_only=True).drop(columns="delay").to_csv(
        analysis / "fixed_roles_macro_metrics.csv")
    for variant, root in runs.items():
        for filename in ("macro_metrics.csv", "region_metrics.csv", "interval_macro_metrics.csv", "interval_region_metrics.csv"):
            shutil.copyfile(root / "scores" / filename, analysis / (variant + "_descriptive_" + filename))
    A.paired_figure(combined, analysis)
    plot_roles(by_region, analysis)
    individual_reports = []
    for bundle in bundles:
        folder = analysis / "individual_rounds" / bundle["variant"]
        folder.mkdir(parents=True)
        shutil.copyfile(bundle["root"] / "scores/selected_config.json", folder / "selection.json")
        independent = A.external_analysis(runs[bundle["variant"]], bundle["root"], bundle["selection"], folder,
                                         5000, 17023, variant=bundle["variant"])
        V4.write_json(folder / "analysis_metadata.json", {"source": bundle["metadata"], "external": independent,
            "source_selection_sha256": bundle["selection_sha256"], "external_selection_performed": False})
        individual_reports.append(str(folder))
    V4.write_json(analysis / "analysis_metadata.json", {"joint_freeze_sha256": V4.sha(output / "joint_freeze.json"),
        "joint_protocol_sha256": frozen["protocol_sha256"],
        "confirmation_data_hashes": frozen["confirmation_data_hashes"],
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_fixed_roles": roles,
        "external_verifications": metadata, "external_selection_performed": False,
        "n_bootstrap": 5000, "seed": 17023, "n_calendar_blocks": 6,
        "uncertainty_scope": frozen["uncertainty_scope"],
        "cqr_scope": "Interval scores/coverage/width only; no CRPS attributed to CQR",
        "individual_round_reports_preserved": individual_reports,
        "audit_links": {variant: A.audit_links(root) for variant, root in runs.items()}})
    return frozen


def plot_roles(regional, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    means = regional.groupby("role", sort=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for role, row in means.iterrows():
        axes[0].scatter(row.empirical_width90, row.empirical_coverage90, label=role)
        if np.isfinite(row.get("crps", np.nan)):
            axes[1].scatter(row.empirical_interval_score90, row.crps, label=role)
    axes[0].axhline(.9, color="black", linestyle="--", linewidth=.8)
    axes[0].set(xlabel="90% interval width", ylabel="Hourly coverage")
    axes[1].set(xlabel="90% interval score", ylabel="CRPS")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Joint source-fixed roles; CQR appears only on interval axes", fontsize=10)
    V4.save_figure(fig, output, "joint_fixed_role_tradeoffs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "analyze"), required=True)
    for name in ("source-r1", "source-r2", "external-r1", "external-r2"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New joint freeze root, reused only for analyze/ subdirectory")
    args = parser.parse_args(argv)
    paths = (args.source_r1, args.source_r2, args.external_r1, args.external_r2)
    if args.phase == "freeze":
        if any(path is None for path in paths):
            parser.error("freeze requires both --source-r* and both --external-r* paths")
        frozen = freeze_joint(paths[:2], paths[2:], args.output)
    else:
        if any(path is not None for path in paths):
            parser.error("analyze uses only paths bound in --output/joint_freeze.json")
        frozen = analyze_joint(args.output)
    print(json.dumps({"phase": args.phase, "joint_root": str(args.output.resolve()),
                      "primary": frozen["fixed_roles"]["primary"], "external_selection_performed": False}))


if __name__ == "__main__":
    main()
