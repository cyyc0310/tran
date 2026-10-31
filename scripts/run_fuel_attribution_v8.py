"""Immutable V8 mapping ablation followed by explicitly illegal mix oracles."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


A7 = module("v8_prior_audit", ROOT / "scripts/analyze_fuel_transport_v7.py")
V7 = A7.RUNNER
from transcif.research.eia_proxy import FUELS
PATHS = module("v8_matched_paths", ROOT / "scripts/fuel_attribution_v8.py")
PROTOCOL = ROOT / "docs/FUEL_ATTRIBUTION_V8_PROTOCOL.md"
MODES = ("direct_analog", "additive_anchor", "recompose_joint")
ORACLE_MODES = ("oracle_additive_anchor", "oracle_recompose_joint")
sha, write = V7.sha, V7.write


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def identities(upstream):
    choices = read(upstream / "source_choices.json")["selections"]
    grid = V7.configs()
    return [next(c for c in grid if c["round"] == r and
                 c["name"] == choices[str(r)]["primary"]["configuration"]["name"])
            for r in (1, 2)]


def config(base, mode):
    return {**base, "mode": mode, "name": base["name"].replace("recompose_joint", mode)}


def grid(frozen, stage):
    return [config(c, mode) for c in frozen["identities"]
            for mode in (ORACLE_MODES if stage == "oracle" else MODES)]


def freeze(output, upstream):
    if output.exists():
        raise FileExistsError("Use a new V8 output directory")
    # Verify the full prior evidence before reusing trusted local fit pickles.
    prior = A7.verify_run(upstream)
    old = prior["frozen"]
    dependencies = {**old["code"], **old["inputs"]}
    for name in ("prefit_freeze.json", "source_choices.json", "source_prediction_seal.json",
                 "target_prediction_seal.json", "completed.json"):
        path = upstream / name
        dependencies[str(path)] = sha(path)
    for path in (upstream / "fits").glob("*.pkl"):
        dependencies[str(path)] = sha(path)
    for folder in ("source_scores", "target_scores"):
        for path in (upstream / folder).iterdir():
            if path.is_file():
                dependencies[str(path)] = sha(path)
    code = [Path(__file__).resolve(), Path(PATHS.__file__).resolve(), PROTOCOL,
            Path(A7.__file__).resolve()]
    record = {"created_utc": now(), "upstream": str(upstream), "source": old["source"],
              "target": old["target"], "dependencies": dependencies, "inputs": old["inputs"],
              "code": {str(p): sha(p) for p in code}, "identities": identities(upstream),
              "factors": list(V7.FACTORS), "delays": list(V7.DELAYS),
              "samples": V7.N_SAMPLES, "seed": V7.SEED, "environment": V7.environment(),
              "scope": "Previously viewed targets; mapping attribution, NOT confirmation"}
    output.mkdir(); (output / "predictions").mkdir()
    write(output / "prefit_freeze.json", record)
    return record


def verify(output):
    frozen = read(output / "prefit_freeze.json")
    for path, digest in {**frozen["dependencies"], **frozen["code"]}.items():
        if sha(path) != digest:
            raise ValueError("Frozen bytes changed: " + path)
    if (frozen["factors"] != list(V7.FACTORS) or frozen["delays"] != list(V7.DELAYS)
            or frozen["samples"] != V7.N_SAMPLES or frozen["seed"] != V7.SEED
            or frozen["environment"] != V7.environment()
            or frozen["identities"] != identities(Path(frozen["upstream"]))):
        raise ValueError("Frozen environment/grid/identity changed")
    return frozen


def selection(table):
    if not set(table["mode"]).issubset(MODES):
        raise ValueError("Only lawful mappings may enter source selection")
    groups = table[table.semester.eq("H1")].groupby(["round", "name", "mode", "factor"])
    ranking = groups.mean(numeric_only=True).drop(columns=["delay", "days"]).reset_index()
    ranking["worst_coverage"] = groups.empirical_coverage90.min().to_numpy()
    ranking["violation"] = (np.maximum(.9 - ranking.empirical_coverage90, 0)
        + np.maximum(ranking.empirical_coverage90 - .95, 0)
        + np.maximum(.85 - ranking.worst_coverage, 0))

    def item(row):
        metrics = ("crps", "empirical_coverage90", "empirical_width90",
                   "empirical_interval_score90", "worst_coverage", "violation")
        return {"configuration": {"round": int(row["round"]), "name": row["name"], "factor": float(row.factor)},
                "status": "passed" if row.violation <= 1e-12 else "failed",
                "metrics": {k: float(row[k]) for k in metrics}}

    def choose(rows):
        passed = rows[rows.violation.le(1e-12)]
        keys = ([] if len(passed) else ["violation"]) + ["crps", "round", "name", "factor"]
        return item((passed if len(passed) else rows).sort_values(keys).iloc[0])

    result = {}
    for r in (1, 2):
        part = ranking[ranking["round"].eq(r)]
        records = {role: choose(part[part["mode"].eq(mode)]) for role, mode in
                   (("primary", "recompose_joint"), ("baseline", "additive_anchor"))}
        primary = records["primary"]["configuration"]
        for role, mode in (("exact_additive", "additive_anchor"), ("exact_direct", "direct_analog")):
            match = part[part["mode"].eq(mode) & part.factor.eq(primary["factor"])]
            if len(match) != 1:
                raise ValueError("Exact mapping control is not unique")
            records[role] = item(match.iloc[0])
        result[str(r)] = records
    result["joint"] = {role: choose(ranking[ranking["mode"].eq(mode)]) for role, mode in
                       (("primary", "recompose_joint"), ("baseline", "additive_anchor"))}
    joint_round = str(result["joint"]["primary"]["configuration"]["round"])
    for role in ("exact_additive", "exact_direct"):
        result["joint"][role] = result[joint_round][role]
    return result, ranking


def validate_choices(output, choice):
    if choice.get("oracle_used") is not False or choice.get("target_scores_used") is not False:
        raise ValueError("Selection scope is not source-only lawful")
    if set(choice["source_score_hashes"]) != A7.SCORE_FILES | {"ranking.csv", "score_seal.json"}:
        raise ValueError("Incomplete source score manifest")
    for name, digest in choice["source_score_hashes"].items():
        if sha(output / "source_scores" / name) != digest:
            raise ValueError("Source scores changed")
    actual, _ = selection(pd.read_csv(output / "source_scores/region_metrics.csv"))
    if actual != choice["selections"]:
        raise ValueError("Source choice does not reproduce")
    if sha(output / "source_prediction_seal.json") != choice["source_seal_sha256"]:
        raise ValueError("Source prediction seal changed")


def check_entries(output, frozen, stage, seal):
    regions = V7.SOURCES if stage == "source" else V7.TARGETS
    configs = {(c["round"], c["name"]): c for c in grid(frozen, stage)}
    expected = {(r, d, *c) for r in regions for d in V7.DELAYS for c in configs}
    seen = set()
    for entry in seal["entries"]:
        key = entry["region"], entry["delay"], entry["round"], entry["name"]
        path = (output / entry["path"]).resolve()
        if (key in seen or key not in expected or not path.is_relative_to(output.resolve())
                or any(entry.get(k) != v for k, v in configs[key[2:]].items())
                or entry["interval_only"] is not False or sha(path) != entry["sha256"]):
            raise ValueError("Incomplete, changed or aliased prediction grid")
        seen.add(key)
    if seen != expected or seal.get("complete_grid") is not True:
        raise ValueError("Incomplete prediction grid")


def verify_phase(output, stage):
    if stage not in ("source", "legal", "oracle"):
        raise ValueError("Unknown stage")
    frozen = verify(output)
    done = read(output / f"{stage}_completed.json")
    if done["prefit_freeze_sha256"] != sha(output / "prefit_freeze.json"):
        raise ValueError("Frozen protocol completion binding changed")
    sealpath = output / f"{stage}_prediction_seal.json"
    if sha(sealpath) != done["prediction_seal_sha256"]:
        raise ValueError("Prediction completion binding changed")
    seal = read(sealpath)
    if seal.get("hourly_labels_read_in_this_prediction_phase") is not False or seal.get("illegal_future_mix") != (stage == "oracle"):
        raise ValueError("Prediction phase information contract changed")
    check_entries(output, frozen, stage, seal)
    folder = output / f"{stage}_scores"
    if sha(folder / "score_seal.json") != done["score_seal_sha256"]:
        raise ValueError("Score completion binding changed")
    hashes = read(folder / "score_seal.json")
    if set(hashes) != A7.SCORE_FILES:
        raise ValueError("Incomplete score file manifest")
    for name, digest in hashes.items():
        if sha(folder / name) != digest:
            raise ValueError("Score bytes changed")
    choice = read(output / "source_choices.json")
    validate_choices(output, choice)
    if done["source_choices_sha256"] != sha(output / "source_choices.json"):
        raise ValueError("Source choice completion binding changed")
    if pd.Timestamp(seal["created_utc"]) > pd.Timestamp(done["created_utc"]):
        raise ValueError("Completion precedes prediction seal")
    if stage != "source":
        if seal.get("source_choices_sha256") != sha(output / "source_choices.json"):
            raise ValueError("Prediction source-choice binding changed")
        if pd.Timestamp(choice["created_utc"]) > pd.Timestamp(seal["created_utc"]):
            raise ValueError("Target predictions precede source choice")
    if stage == "oracle":
        if seal.get("illegal_future_mix") is not True or done.get("eligible_for_selection") is not False:
            raise ValueError("Oracle misuse guard missing")
        if seal["legal_completed_sha256"] != sha(output / "legal_completed.json"):
            raise ValueError("Oracle detached from legal completion")
    return seal


def load_fit(frozen, region, delay, stage):
    name = f"lodo-{region}-{delay}.pkl" if stage == "source" else f"full-{delay}.pkl"
    path = Path(frozen["upstream"]) / "fits" / name
    if sha(path) != frozen["dependencies"][str(path)]:
        raise ValueError("Fitted model changed")
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    expected = set(V7.SOURCES) - ({region} if stage == "source" else set())
    if set(bundle["training_regions"]) != expected or set(bundle["bank"].source_regions_) != expected:
        raise ValueError("Wrong leave-one-domain-out fitted bank")
    return bundle


def audit_summary(audit):
    return {k: ({"shape": list(v.shape), "sha256": hashlib.sha256(v.tobytes()).hexdigest()}
                if isinstance(v, np.ndarray) else v) for k, v in audit.items()}


def save(output, stage, region, delay, days, base, result, modes):
    entries = []
    for mode in modes:
        c = config(base, mode)
        path = output / "predictions" / f"{stage}-{region}-{delay}-r{c['round']}-{c['name']}.npz"
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, origins=pd.DatetimeIndex(days).as_unit("ns").asi8, **result[mode])
        entries.append({**c, "region": region, "delay": delay, "path": str(path.relative_to(output)),
                        "sha256": sha(path), "interval_only": False, "audit": audit_summary(result["audit"])})
    return entries


def parity(path, expected, origins):
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["origins"], pd.DatetimeIndex(origins).as_unit("ns").asi8):
            raise ValueError("Prediction origins changed")
        differences = {}
        for key in ("point", "ensemble"):
            if archive[key].shape != expected[key].shape or not np.allclose(archive[key], expected[key], atol=1e-10, rtol=1e-12):
                raise ValueError("Matched prediction parity failed: " + str(path))
            differences[key] = float(np.max(np.abs(archive[key] - expected[key])))
    return differences


def oracle_mix(monthly, region, origins):
    """Illegal retrospective month composition; never called by lawful phases."""
    rows = monthly[monthly.region.eq(region)].copy()
    rows["period_start"] = pd.to_datetime(rows.period_start, utc=True)
    if rows.period_start.duplicated().any():
        raise ValueError("Duplicate region-month oracle rows")
    rows = rows.set_index("period_start")
    values = []
    for day in pd.DatetimeIndex(origins):
        month = pd.Timestamp(year=day.year, month=day.month, day=1, tz="UTC")
        if month not in rows.index:
            raise ValueError("Missing oracle month; do not drop or impute: " + str(month))
        record = rows.loc[month]
        expected_end = month + pd.offsets.MonthBegin(1)
        if pd.to_datetime(record.period_end, utc=True) != expected_end:
            raise ValueError("Oracle month support is not one calendar month")
        value = record[["fuelshare_" + f for f in FUELS]].to_numpy(dtype=float)
        if not np.isfinite(value).all() or np.any(value < 0) or not np.isclose(value.sum(), 1., atol=1e-8):
            raise ValueError("Invalid actual composition")
        values.append(value)
    return np.stack(values)


def finish(output, stage, seal):
    write(output / f"{stage}_completed.json", {"created_utc": now(),
        "prefit_freeze_sha256": sha(output / "prefit_freeze.json"),
        "prediction_seal_sha256": sha(output / f"{stage}_prediction_seal.json"),
        "score_seal_sha256": sha(output / f"{stage}_scores/score_seal.json"),
        "source_choices_sha256": sha(output / "source_choices.json"),
        "eligible_for_selection": stage == "source", "scope": "Seen-region development"})
    verify_phase(output, stage)


def run_phase(output, upstream, stage):
    frozen = freeze(output, upstream) if stage == "source" else verify(output)
    if (output / f"{stage}_prediction_seal.json").exists() or (output / f"{stage}_scores").exists():
        raise FileExistsError("Stage is immutable")
    if stage != "source":
        verify_phase(output, "source")
    if stage == "oracle":
        legal = verify_phase(output, "legal")
        write(output / "oracle_opened.json", {"created_utc": now(), "illegal_future_mix": True,
            "legal_completed_sha256": sha(output / "legal_completed.json")})
    source, target = Path(frozen["source"]), Path(frozen["target"])
    monthly = pd.concat([pd.read_csv(source / "monthly.csv"), pd.read_csv(target / "monthly.csv")], ignore_index=True)
    year = 2025 if stage == "source" else 2026
    end = "2026-01-01" if stage == "source" else "2026-07-01"
    days = pd.date_range(f"{year}-01-01", end, inclusive="left", tz="UTC")
    regions = V7.SOURCES if stage == "source" else V7.TARGETS
    prior_stage = "source" if stage == "source" else "target"
    previous = read(Path(frozen["upstream"]) / f"{prior_stage}_prediction_seal.json")
    old_entries = {(e["region"], e["delay"], e["round"], e["name"]): e for e in previous["entries"]}
    legal_entries = {(e["region"], e["delay"], e["round"], e["mode"]): e for e in legal["entries"]} if stage == "oracle" else {}
    entries, matches = [], []
    for delay in V7.DELAYS:
        for region in regions:
            bundle = load_fit(frozen, region, delay, stage)
            coarse = V7.make_features(monthly, region, days, delay)
            actual = oracle_mix(monthly, region, days) if stage == "oracle" else None
            for base in frozen["identities"]:
                result = PATHS.generate_matched(bundle["bank"], coarse, days, base, oracle_mix=actual)
                if stage == "oracle":
                    for mode in MODES:
                        e = legal_entries[region, delay, base["round"], mode]
                        if result["audit"]["draw_sha256"] != e["audit"]["draw_sha256"]:
                            raise ValueError("Oracle changed lawful routing")
                        parity(output / e["path"], result[mode], days)
                else:
                    for mode in ("direct_analog", "recompose_joint"):
                        c = config(base, mode)
                        e = old_entries[region, delay, base["round"], c["name"]]
                        path = Path(frozen["upstream"]) / e["path"]
                        if sha(path) != e["sha256"]:
                            raise ValueError("Prior prediction archive changed")
                        matches.append({"region": region, "delay": delay, "round": base["round"],
                                        "mode": mode, **parity(path, result[mode], days)})
                entries.extend(save(output, stage, region, delay, days, base, result,
                                    ORACLE_MODES if stage == "oracle" else MODES))
            print("Predicted", stage, region, delay, flush=True)
    verify(output)
    seal = {"created_utc": now(), "entries": entries, "complete_grid": True,
            "v7_parity": matches, "illegal_future_mix": stage == "oracle",
            "hourly_labels_read_in_this_prediction_phase": False}
    if stage != "source":
        seal["source_choices_sha256"] = sha(output / "source_choices.json")
    if stage == "oracle":
        seal["legal_completed_sha256"] = sha(output / "legal_completed.json")
    check_entries(output, frozen, stage, seal)
    write(output / f"{stage}_prediction_seal.json", seal)
    V7.score_entries(output, entries, source if stage == "source" else target, year, output / f"{stage}_scores")
    if stage == "source":
        choices, ranking = selection(pd.read_csv(output / "source_scores/region_metrics.csv"))
        ranking.to_csv(output / "source_scores/ranking.csv", index=False)
        write(output / "source_choices.json", {"created_utc": now(), "selections": choices,
            "oracle_used": False, "target_scores_used": False,
            "source_score_hashes": {p.name: sha(p) for p in (output / "source_scores").iterdir()},
            "source_seal_sha256": sha(output / "source_prediction_seal.json")})
    finish(output, stage, seal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("source", "legal", "oracle"))
    parser.add_argument("--upstream", type=Path, default=ROOT / "results/fuel-transport-v7")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_phase(args.output.resolve(), args.upstream.resolve(), args.phase)


if __name__ == "__main__":
    main()
