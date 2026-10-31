"""Audit and describe source-fixed V7 development retests.

No target ranking or fitting is performed.  The six-month bootstrap is a
conditional descriptive summary of already viewed 2026H1 regions, not a new
confirmation or a population/generalization confidence interval.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_fuel_transport_v7.py"
_spec = importlib.util.spec_from_file_location("fuel_v7_analysis_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUNNER)
IDENTITY = ["round", "name", "mode", "factor"]
CELL = ["region", "delay", *IDENTITY, "semester"]
DAY = ["region", "delay", *IDENTITY, "origin"]
INTERVAL = ["empirical_coverage90", "empirical_width90", "empirical_interval_score90",
            "empirical_lower_tail_miss", "empirical_upper_tail_miss", "empirical_path_coverage90"]
METRICS = ["crps", *INTERVAL, "mae", "raw_point_mae", "raw_level_mae", "raw_shape_mae",
           "point_level_mae", "point_shape_mae"]
SCORE_FILES = {"day_metrics.csv", "region_metrics.csv", "macro_metrics.csv", "day_counts.csv"}
ROLE_COLORS = {"primary": "#0072B2", "baseline": "#D55E00", "simple_anchor": "#009E73",
               "unconstrained_baseline": "#CC79A7", "cqr": "#7A5195",
               "exact_direct": "#E69F00", "exact_independent": "#56B4E9", "exact_pooled": "#555555"}
ROLE_LABELS = {"primary": "Fuel joint", "baseline": "Source best baseline", "simple_anchor": "Fresh anchor",
               "unconstrained_baseline": "Lowest source CRPS", "cqr": "CQR interval",
               "exact_direct": "Direct analog", "exact_independent": "Independent fuels", "exact_pooled": "Pooled fuels"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def member(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Seal member escapes run directory")
    return path


def check_hashes(root, hashes, expected=None):
    if expected is not None and set(hashes) != set(expected):
        raise ValueError("Incomplete sealed file manifest")
    for name, digest in hashes.items():
        if sha(member(root, name)) != digest:
            raise ValueError("Changed sealed bytes: " + name)


def subset(frame, config):
    mask = np.ones(len(frame), dtype=bool)
    for key in ("round", "name", "factor"):
        mask &= frame[key].eq(config[key]).to_numpy()
    result = frame.loc[mask].copy()
    if result.empty:
        raise ValueError("Fixed configuration has no rows: " + str(config))
    if "mode" in result and result["mode"].nunique() != 1:
        raise ValueError("Configuration identity aliases multiple modes")
    return result


def assert_equal_table(actual, expected, keys, values):
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError("Duplicate aggregate identity")
    a = actual.set_index(keys).sort_index()
    b = expected.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("Aggregate grid differs from daily scores")
    for column in values:
        if column not in a or column not in b or not np.allclose(a[column], b[column], rtol=1e-9, atol=1e-9, equal_nan=True):
            raise ValueError("Aggregate score differs: " + column)


def verified_scores(root, stage):
    """Verify CSV bytes, complete identities, and equal-region/delay aggregation."""
    folder = root / (stage + "_scores")
    seal = read_json(folder / "score_seal.json")
    check_hashes(folder, seal, SCORE_FILES)
    region = pd.read_csv(folder / "region_metrics.csv")
    macro = pd.read_csv(folder / "macro_metrics.csv")
    counts = pd.read_csv(folder / "day_counts.csv")
    header = pd.read_csv(folder / "day_metrics.csv", nrows=0)
    metrics = [m for m in METRICS if m in header]
    if not set(["crps", *INTERVAL]).issubset(metrics):
        raise ValueError("Missing proper/interval score columns")
    days = pd.read_csv(folder / "day_metrics.csv", usecols=[*DAY, "semester", *metrics])
    if days.duplicated(DAY).any() or days.empty:
        raise ValueError("Duplicate or empty daily scores")
    time = pd.to_datetime(days.origin, utc=True)
    year = 2025 if stage == "source" else 2026
    end = f"{year + 1}-01-01" if stage == "source" else "2026-07-01"
    if not (time.ge(f"{year}-01-01") & time.lt(end) & time.eq(time.dt.normalize())).all():
        raise ValueError("Wrong scoring year or non-midnight origin")
    if not days.semester.eq(np.where(time.dt.month.le(6), "H1", "H2")).all():
        raise ValueError("Semester differs from origin")
    regions = RUNNER.SOURCES if stage == "source" else RUNNER.TARGETS
    semesters = ("H1", "H2") if stage == "source" else ("H1",)
    expected = {(r, d, c["round"], c["name"], c["mode"], f, s)
                for r in regions for d in RUNNER.DELAYS for c in RUNNER.configs()
                for f in ((1.,) if c["mode"] == "cqr" else RUNNER.FACTORS) for s in semesters}
    if set(map(tuple, region[CELL].to_numpy())) != expected or len(region) != len(expected):
        raise ValueError("Incomplete score configuration/region/delay/semester grid")
    cqr = days["mode"].eq("cqr")
    if days.loc[cqr, "crps"].notna().any():
        raise ValueError("CQR must not have CRPS")
    if not np.isfinite(days.loc[~cqr, "crps"]).all() or not np.isfinite(days[INTERVAL]).all().all():
        raise ValueError("Nonfinite proper/interval scores")
    # Each method/factor must be evaluated on exactly the same retained days.
    support = days.groupby(["region", "delay", *IDENTITY], sort=False).origin.agg(frozenset)
    for _, group in support.groupby(level=[0, 1]):
        if len(set(group)) != 1:
            raise ValueError("Candidates have different retained day sets")
    grouped = days.groupby(CELL, sort=True)
    recomputed = grouped[metrics].mean().reset_index()
    recomputed["days"] = grouped.size().to_numpy()
    assert_equal_table(region, recomputed, CELL, [*metrics, "days"])
    macro_keys = [*IDENTITY, "semester"]
    assert_equal_table(macro, region.groupby(macro_keys)[metrics].mean().reset_index(), macro_keys, metrics)
    count_keys = ["region", "round", "name", "delay", "factor"]
    assert_equal_table(counts, days.groupby(count_keys).size().reset_index(name="days"), count_keys, ["days"])
    return {"day": days, "region": region, "macro": macro, "metrics": metrics}


def verified_prediction_seal(root, stage):
    seal = read_json(root / (stage + "_prediction_seal.json"))
    configs = {(c["round"], c["name"]): c for c in RUNNER.configs()}
    regions = RUNNER.SOURCES if stage == "source" else RUNNER.TARGETS
    expected = {(r, d, *key) for r in regions for d in RUNNER.DELAYS for key in configs}
    seen = set()
    for entry in seal["entries"]:
        key = (entry["region"], entry["delay"], entry["round"], entry["name"])
        if key in seen or key not in expected:
            raise ValueError("Duplicate/unexpected prediction entry")
        seen.add(key)
        config = configs[key[2:]]
        if any(entry.get(k) != v for k, v in config.items()) or entry["interval_only"] != (config["mode"] == "cqr"):
            raise ValueError("Prediction configuration metadata changed")
        if sha(member(root, entry["path"])) != entry["sha256"]:
            raise ValueError("Prediction bytes changed")
        support = entry["support"]
        if config["mode"] not in ("fresh_anchor", "extratrees", "cqr"):
            for metric in ("unsupported_target_mass", "mean_effective_donors", "min_effective_donors", "zero_generation_fallback_fraction"):
                if metric not in support or not np.isfinite(support[metric]) or support[metric] < 0:
                    raise ValueError("Invalid target donor support diagnostics")
            if support["unsupported_target_mass"] > 1 or support["zero_generation_fallback_fraction"] > 1 or not support.get("ess_scope"):
                raise ValueError("Invalid support probability/scope")
    if seen != expected:
        raise ValueError("Incomplete prediction grid")
    if stage == "source":
        fits = {f"lodo-{r}-{d}.pkl" for r in RUNNER.SOURCES for d in RUNNER.DELAYS} | {f"full-{d}.pkl" for d in RUNNER.DELAYS}
        check_hashes(root / "fits", seal["fits"], fits)
    else:
        if seal.get("all_grid_complete") is not True or seal.get("target_labels_read_in_this_prediction_phase") is not False:
            raise ValueError("Target predictions were not sealed before label scoring")
        if seal.get("target_already_viewed_in_V6") is not True:
            raise ValueError("V7 target development scope missing")
    return seal


def validate_selection(root, choice, regional):
    """Call the actual frozen producer's source-only selection implementation."""
    expected_files = SCORE_FILES | {"ranking.csv", "score_seal.json"}
    check_hashes(root / "source_scores", choice["source_score_hashes"], expected_files)
    if choice.get("target_model_scores_read") is not False:
        raise ValueError("Source selection must precede target scores")
    if sha(root / "source_prediction_seal.json") != choice["source_seal_sha256"]:
        raise ValueError("Source prediction seal hash differs")
    actual, ranking = RUNNER.select(regional)
    if actual != choice["selections"]:
        raise ValueError("Source-fixed choices differ from runner.select")
    saved = pd.read_csv(root / "source_scores/ranking.csv")
    columns = list(ranking.select_dtypes("number"))
    assert_equal_table(saved, ranking, IDENTITY, [c for c in columns if c not in IDENTITY])
    return actual


def verify_run(root, source_only=False):
    frozen = read_json(root / "prefit_freeze.json")
    for path, digest in {**frozen["code"], **frozen["inputs"]}.items():
        if sha(path) != digest:
            raise ValueError("Frozen input/code changed: " + path)
    if frozen["code"].get(str(RUNNER_PATH)) != sha(RUNNER_PATH):
        raise ValueError("Selection implementation absent from frozen code")
    if frozen["configs"] != RUNNER.configs() or frozen["factors"] != list(RUNNER.FACTORS) or frozen["delays"] != list(RUNNER.DELAYS):
        raise ValueError("V7 frozen grid changed")
    source_seal = verified_prediction_seal(root, "source")
    source = verified_scores(root, "source")
    choice = read_json(root / "source_choices.json")
    roles = validate_selection(root, choice, source["region"])
    result = dict(source=source, roles=roles, source_seal=source_seal, frozen=frozen)
    if source_only:
        return result
    completed = read_json(root / "completed.json")
    choice_hash = sha(root / "source_choices.json")
    if completed.get("source_choices_sha256") != choice_hash or completed.get("frozen_hashes_verified") is not True:
        raise ValueError("Completion/source choice binding differs")
    if completed.get("target_seal_sha256") != sha(root / "target_prediction_seal.json"):
        raise ValueError("Completion/target prediction seal binding differs")
    if completed.get("target_score_seal_sha256") != sha(root / "target_scores/score_seal.json"):
        raise ValueError("Completion/target score seal binding differs")
    target_seal = verified_prediction_seal(root, "target")
    if target_seal.get("source_choices_sha256") != choice_hash:
        raise ValueError("Target prediction seal/source choice binding differs")
    if pd.Timestamp(choice["created_utc"]) > pd.Timestamp(target_seal["created_utc"]) or pd.Timestamp(target_seal["created_utc"]) > pd.Timestamp(completed["created_utc"]):
        raise ValueError("Source selection/prediction/completion order differs")
    result.update(target=verified_scores(root, "target"), target_seal=target_seal)
    return result


def fixed_tables(scores, roles):
    regional, macros, delays = [], [], []
    for scope, records in roles.items():
        for role, record in records.items():
            group = subset(scores["region"], record["configuration"])
            context = {"scope": scope, "role": role, "source_status": record["status"]}
            regional.append(group.assign(**context))
            for semester, half in group.groupby("semester"):
                metrics = scores["metrics"]
                base = {**context, **record["configuration"], "mode": half["mode"].iloc[0], "semester": semester}
                macros.append({**base, **half[metrics].mean().to_dict(), "worst_coverage": half.empirical_coverage90.min(),
                               "n_region_delay_groups": len(half), "n_scored_days": int(half.days.sum())})
                for delay, delayed in half.groupby("delay"):
                    delays.append({**base, "delay": delay, **delayed[metrics].mean().to_dict(),
                                   "worst_coverage": delayed.empirical_coverage90.min()})
    return pd.DataFrame(macros), pd.concat(regional, ignore_index=True), pd.DataFrame(delays)


def paired_month_bootstrap(days, primary, reference, metrics, n_bootstrap=5000, seed=17023):
    """Equal-region/delay means, synchronous sum/count calendar-month resampling."""
    keys = ["region", "delay", "origin"]
    left, right = subset(days, primary), subset(days, reference)
    left["origin"] = pd.to_datetime(left.origin, utc=True)
    right["origin"] = pd.to_datetime(right.origin, utc=True)
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("Duplicate paired daily identity")
    paired = left[keys + metrics].merge(right[keys + metrics], on=keys, how="outer", validate="one_to_one",
                                         suffixes=("_primary", "_reference"), indicator=True)
    if not paired._merge.eq("both").all():
        raise ValueError("Fixed comparisons have different scored days")
    time = pd.to_datetime(paired.origin, utc=True)
    if not (time.ge("2026-01-01") & time.lt("2026-07-01")).all():
        raise ValueError("Bootstrap only describes 2026H1")
    paired["month"] = time.dt.month
    groups = list(paired.groupby(["region", "delay"], sort=True))
    if {key for key, _ in groups} != {(r, d) for r in RUNNER.TARGETS for d in RUNNER.DELAYS}:
        raise ValueError("Paired bootstrap requires all six regions/three delays")
    values = np.zeros((len(groups), 6, len(metrics), 2))
    counts = np.zeros((len(groups), 6))
    for g, (_, group) in enumerate(groups):
        for m in range(6):
            block = group[group.month.eq(m + 1)]
            counts[g, m] = len(block)
            if not len(block):
                raise ValueError("Each region/delay requires all six month blocks")
            for side, suffix in enumerate(("primary", "reference")):
                matrix = block[[m + "_" + suffix for m in metrics]].to_numpy()
                if not np.isfinite(matrix).all():
                    raise ValueError("Undefined metric in paired comparison; CQR has no CRPS")
                values[g, m, :, side] = matrix.sum(0)
    if n_bootstrap < 100:
        raise ValueError("At least 100 bootstrap replicates required")
    sampled = np.random.default_rng(seed).integers(6, size=(n_bootstrap, 6))
    multiplicity = np.stack([(sampled == m).sum(1) for m in range(6)], axis=1)
    denominator = np.einsum("bm,gm->bg", multiplicity, counts)
    simulated = np.einsum("bm,gmks->bgks", multiplicity, values) / denominator[:, :, None, None]
    observed = values.sum(1) / counts.sum(1)[:, None, None]
    strata = [("overall", "all", np.arange(len(groups)))]
    for field, position in (("region", 0), ("delay", 1)):
        for label in sorted({key[position] for key, _ in groups}):
            strata.append((field, str(label), np.array([i for i, (key, _) in enumerate(groups) if key[position] == label])))
    rows = []
    for field, label, indices in strata:
        point = observed[indices].mean(0)
        delta = simulated[:, indices].mean(1)
        lo, hi = np.quantile(delta[:, :, 0] - delta[:, :, 1], [.025, .975], axis=0)
        for i, metric in enumerate(metrics):
            rows.append(dict(stratum=field, label=label, metric=metric, primary_mean=point[i, 0], reference_mean=point[i, 1],
                difference_primary_minus_reference=point[i, 0] - point[i, 1], ci95_low=lo[i], ci95_high=hi[i],
                n_paired_days=int(counts[indices].sum()), n_region_delay_groups=len(indices), n_calendar_blocks=6,
                n_bootstrap=n_bootstrap, seed=seed))
    return pd.DataFrame(rows)


def planned_comparisons(roles):
    records = []
    for scope, selected in roles.items():
        references = (["baseline", "simple_anchor", "unconstrained_baseline", "cqr"] if scope == "joint"
                      else ["baseline", "simple_anchor", "cqr", "exact_direct", "exact_independent", "exact_pooled"])
        for reference in references:
            records.append(dict(scope=scope, comparison="primary_minus_" + reference, exploratory=False,
                                primary=selected["primary"]["configuration"], reference=selected[reference]["configuration"],
                                metrics=INTERVAL if reference == "cqr" else ["crps", *INTERVAL]))
    # Counterpart chosen by exact source-selected R2 primary identity, never target outcomes.
    config = roles["2"]["primary"]["configuration"]
    name = config["name"]
    if name.startswith(("adaptive_mean-", "adaptive_mixture-")):
        mean = {**config, "name": name.replace("adaptive_mixture-", "adaptive_mean-")}
        mixture = {**config, "name": name.replace("adaptive_mean-", "adaptive_mixture-")}
        records.append(dict(scope="2", comparison="matched_mean_minus_mixture", exploratory=True,
                            primary=mean, reference=mixture, metrics=["crps", *INTERVAL]))
    return records


def support_tables(seal, roles):
    all_rows = pd.DataFrame([{**{k: entry[k] for k in ("region", "delay", "round", "name", "mode")}, **entry["support"]}
                             for entry in seal["entries"]])
    fixed = []
    for scope, records in roles.items():
        for role, record in records.items():
            config = record["configuration"]
            selected = all_rows[all_rows["round"].eq(config["round"]) & all_rows.name.eq(config["name"])]
            fixed.append(selected.assign(scope=scope, role=role, factor=config["factor"]))
    return all_rows, pd.concat(fixed, ignore_index=True)


def figures(macro, regional, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for scope in ("1", "2", "joint"):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), constrained_layout=True)
        display = ("primary", "baseline", "simple_anchor", "cqr") if scope == "joint" else (
            "primary", "simple_anchor", "exact_direct", "exact_independent", "exact_pooled", "cqr")
        table = macro[macro.scope.eq(scope)]
        region = regional[regional.scope.eq(scope)]
        for role in display:
            row = table[table.role.eq(role)].iloc[0]
            color = ROLE_COLORS[role]
            label = ROLE_LABELS[role] + (" [source gate failed]" if row.source_status == "failed" else "")
            axes[0].scatter(row.empirical_width90, row.empirical_coverage90, color=color, label=label, s=55)
            if np.isfinite(row.crps):
                axes[1].scatter(row.crps, row.empirical_interval_score90, color=color, s=55)
            values = region[region.role.eq(role)].groupby("region").empirical_coverage90.mean().reindex(RUNNER.TARGETS)
            axes[2].plot(values.index, values, color=color, marker="o", linewidth=1, label=label)
        axes[0].set(xlabel="90% interval width (gCO2/kWh)", ylabel="Hourly coverage")
        axes[1].set(xlabel="CRPS (gCO2/kWh)", ylabel="90% interval score (gCO2/kWh)")
        axes[2].set(ylabel="Coverage, mean across 3 delays")
        axes[2].tick_params(axis="x", rotation=35)
        for ax in (axes[0], axes[2]):
            ax.axhline(.9, color="black", linestyle="--", linewidth=.8)
        axes[0].legend(fontsize=7, loc="best")
        fig.suptitle(f"V7 {'joint' if scope == 'joint' else 'round ' + scope}: source-fixed roles, viewed target retest", fontsize=11)
        fig.supxlabel("Display roles fixed by source selection/identity; CQR has no CRPS. Other fixed roles remain in tables.", fontsize=9)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{scope}_fixed_role_summary.{extension}", dpi=170)
        plt.close(fig)


def historical_background(root, target_days, choices, output, v6_joint):
    """Same-day background only: older point-fit regions/years/budgets differ."""
    frozen_path = v6_joint / "joint_freeze.json"
    frozen = read_json(frozen_path)
    sidecar = v6_joint / "joint_freeze.sha256"
    if sha(frozen_path) != sidecar.read_text().split()[0]:
        raise ValueError("Historical V6 joint freeze hash differs")
    current = read_json(root / "prefit_freeze.json")
    for region in RUNNER.TARGETS:
        name = region + "_hourly.csv"
        hashes = [digest for path, digest in current["inputs"].items() if Path(path).name == name]
        if hashes != [frozen["confirmation_data_hashes"][name]]:
            raise ValueError("Historical/current target hourly data differ")
    comparisons, tables, input_hashes = [], [], {str(frozen_path): sha(frozen_path), str(sidecar): sha(sidecar)}
    primary = choices["joint"]["primary"]["configuration"]
    new = subset(target_days, primary)
    new = new.assign(round=7, name="v7_joint_primary", mode="v7_joint_primary", factor=1.)
    for role in ("primary", "source2_simple_anchor"):
        record = frozen["fixed_roles"][role]
        config = record["configuration"]
        external = next(r for r in frozen["external_runs"] if r["variant"] == config["variant"])
        run = Path(external["root"])
        # Postrun seals in V6 bind every original score byte, including daily CSV.
        postrun_path = run / "scores/postrun_verification.json"
        postrun = read_json(postrun_path)
        day_path = run / "scores/day_metrics.csv"
        global_hash = sha(run / "all_predictions_sealed.json")
        if (postrun.get("all_frozen_hashes_match") is not True or postrun.get("target_label_access_after_global_seal") is not True
                or postrun.get("global_seal_sha256") != global_hash or external["global_prediction_seal_sha256"] != global_hash):
            raise ValueError("Historical V6 prediction/postrun binding differs")
        check_hashes(run / "scores", postrun["score_hashes"])
        if postrun["score_hashes"].get("day_metrics.csv") != sha(day_path):
            raise ValueError("Historical V6 daily score seal missing/different")
        old = pd.read_csv(day_path, usecols=lambda c: c in ["region", "delay", "origin", "variant", "method", "calibration", "crps", *INTERVAL])
        for key, value in config.items():
            old = old[old[key].eq(value)]
        old = old.assign(round=6, name="v6_" + role, mode="v6_" + role, factor=1.)
        merged = pd.concat([new, old], ignore_index=True)
        reference = dict(round=6, name="v6_" + role, factor=1.)
        result = paired_month_bootstrap(merged, dict(round=7, name="v7_joint_primary", factor=1.), reference, ["crps", *INTERVAL])
        comparisons.append(result.assign(comparison="v7_primary_minus_v6_" + role, interpretation="Historical background; different training budgets; no algorithm attribution"))
        for name, group in (("v7_joint_primary", new), ("v6_" + role, old)):
            region = group.groupby(["region", "delay"])[["crps", *INTERVAL]].mean().reset_index()
            tables.append(region.assign(historical_comparison=role, method_label=name))
        input_hashes.update({str(day_path): sha(day_path), str(postrun_path): sha(postrun_path)})
    pd.concat(comparisons).to_csv(output / "historical_v6_same_day_background.csv", index=False)
    pd.concat(tables).to_csv(output / "historical_v6_region_delay_background.csv", index=False)
    return input_hashes


def analyze(root, output, source_only=False, v6_joint=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new analysis output directory")
    verified = verify_run(root, source_only=source_only)
    output.mkdir(parents=True)
    source_tables = fixed_tables(verified["source"], verified["roles"])
    for name, table in zip(("macro", "region_delay", "delay"), source_tables):
        table.to_csv(output / f"source_fixed_roles_{name}.csv", index=False)
    region_keys = ["scope", "role", "source_status", *IDENTITY, "semester", "region"]
    source_tables[1].groupby(region_keys)[verified["source"]["metrics"]].mean().reset_index().to_csv(
        output / "source_fixed_roles_region.csv", index=False)
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), run=str(root), source_only=source_only,
        source_choices_sha256=sha(root / "source_choices.json"), source_prediction_seal_sha256=sha(root / "source_prediction_seal.json"),
        prefit_freeze_sha256=sha(root / "prefit_freeze.json"), selections=verified["roles"],
        selection_recomputed_with="Frozen runner.select from source H1 regional CSV; H2 stress never selects",
        target_selection_performed=False, target_training_performed=False,
        scope="Previously viewed 2026H1 regions: development retest, NOT new confirmation",
        bootstrap=dict(n_bootstrap=5000, seed=17023, calendar_month_blocks=6,
                       interpretation="Conditional descriptive six-month blocks; not global/generalization CI"),
        historical_context="V6 uses different source point-fit/training budgets; no algorithm attribution",
        code_hashes={str(Path(__file__).resolve()): sha(__file__), str(RUNNER_PATH): sha(RUNNER_PATH)})
    if not source_only:
        target = verified["target"]
        tables = fixed_tables(target, verified["roles"])
        for name, table in zip(("macro", "region_delay", "delay"), tables):
            table.to_csv(output / f"target_fixed_roles_{name}.csv", index=False)
        tables[1].groupby(region_keys)[target["metrics"]].mean().reset_index().to_csv(
            output / "target_fixed_roles_region.csv", index=False)
        comparisons = planned_comparisons(verified["roles"])
        results = []
        for record in comparisons:
            result = paired_month_bootstrap(target["day"], record["primary"], record["reference"], record["metrics"])
            results.append(result.assign(scope=record["scope"], comparison=record["comparison"], exploratory=record["exploratory"],
                primary_configuration=json.dumps(record["primary"], sort_keys=True), reference_configuration=json.dumps(record["reference"], sort_keys=True)))
        pd.concat(results, ignore_index=True).to_csv(output / "fixed_paired_month_bootstrap.csv", index=False)
        for name, table in zip(("all_grid", "fixed_roles"), support_tables(verified["target_seal"], verified["roles"])):
            table.to_csv(output / f"target_support_{name}.csv", index=False)
        # All-candidate tables are descriptive and retain producer ordering; no target ranking.
        target["macro"].to_csv(output / "target_all_candidates_descriptive_macro.csv", index=False)
        target["region"].to_csv(output / "target_all_candidates_descriptive_region_delay.csv", index=False)
        figures(tables[0], tables[1], output)
        metadata.update(completed_sha256=sha(root / "completed.json"), target_prediction_seal_sha256=sha(root / "target_prediction_seal.json"),
                        target_score_seal_sha256=sha(root / "target_scores/score_seal.json"), comparisons=comparisons)
        if v6_joint is not None:
            metadata["historical_input_hashes"] = historical_background(root, target["day"], verified["roles"], output, Path(v6_joint).resolve())
    (output / "READ_ME.md").write_text(
        "# V7 source-fixed development analysis\n\n"
        "Selections are reconstructed with the frozen runner from 2025H1 LODO scores. "
        "2025H2 is a source stress test and never selects configurations. Failed coverage "
        "roles retain their failed status and diagnostic configuration.\n\n"
        "Target results concern previously viewed 2026H1 regions; they are a development "
        "retest, not a new independent confirmation. All target comparisons retain source "
        "configuration identities. Candidate tables are descriptive without target ranking.\n\n"
        "Bootstrap differences are primary minus reference, using 5,000 synchronous resamples "
        "of only six calendar-month blocks (seed 17023), equal region/delay weights, "
        "and identical scored days. The intervals are conditional descriptive summaries, "
        "not global or generalization confidence intervals. CQR has interval metrics only.\n\n"
        "Each round's direct, independent-fuel, and pooled controls use its exact source-fixed "
        "primary forecast, bandwidth, temperature, and spread factor. Round 2 mean versus "
        "mixture uses its source-selected primary's matched counterpart and is exploratory. "
        "ESS is a donor-distribution support diagnostic; mixture ESS is an expert-weighted "
        "component summary, not exact mixture ESS.\n\n"
        "Historical V6 comparisons, when requested, use the identical target days and "
        "verified hourly-data bytes. Their training budgets differ and do not support "
        "algorithm attribution.\n")
    metadata["output_hashes"] = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "analysis_metadata.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/fuel-transport-v7")
    parser.add_argument("--output", type=Path, default=ROOT / "results/fuel-transport-v7-analysis")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--v6-joint", type=Path, default=ROOT / "results/coarse-risk-v6-joint-confirmation")
    parser.add_argument("--skip-v6-history", action="store_true", help="Explicitly omit optional historical-budget background")
    args = parser.parse_args()
    analyze(args.run, args.output, args.source_only, None if args.skip_v6_history else args.v6_joint)


if __name__ == "__main__":
    main()
