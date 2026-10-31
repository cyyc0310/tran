"""Two source-LODO development rounds; six previously viewed target regions.

Source choices precede saving every target prediction and reading target labels.
This extra seal does NOT turn the already diagnosed targets into a new holdout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import importlib.metadata
import json
import pickle
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transcif.research.coarse_forecast_v7 import released_forecasts
from transcif.research.fuel_transport_v7 import FuelTrajectoryBank
from transcif.research.baselines_v7 import DirectBaselines
from transcif.research.dynamic_v2 import read_year, daily_profiles
from transcif.research.structural_v3 import fit_model

spec = importlib.util.spec_from_file_location("v7_metric_helpers", ROOT / "scripts/run_transfer_v4.py")
V4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(V4)
SOURCES = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
TARGETS = ("AZPS", "DUK", "LGEE", "PACE", "PSCO", "SRP")
DELAYS = (0, 30, 90)
FACTORS = (.75, 1., 1.25, 1.5)
N_SAMPLES, SEED = 128, 17023
PROTOCOL = ROOT / "docs/FUEL_TRANSPORT_V7_PROTOCOL.md"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def configs():
    records = []
    modes = ("recompose_joint", "direct_analog", "recompose_independent", "pooled_recompose")
    for bw in (.1, .3, 1.):
        for mode in modes:
            records.append(dict(round=1, name=f"latest-{mode}-b{bw}", mode=mode, forecast="latest", bandwidth=bw, temperature=.03))
    for temp in (.03, .1):
        for forecast in ("adaptive_mean", "adaptive_mixture"):
            for mode in modes:
                records.append(dict(round=2, name=f"{forecast}-t{temp}-{mode}-b0.3", mode=mode,
                                    forecast=forecast, bandwidth=.3, temperature=temp))
    for round_ in (1, 2):
        for mode in ("fresh_anchor", "extratrees", "cqr"):
            records.append(dict(round=round_, name=mode, mode=mode))
    return records


def environment():
    return {"python": sys.version, "executable": sys.executable,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")}}


def freeze(output, source, target):
    if output.exists(): raise FileExistsError("Use a new V7 output directory")
    output.mkdir(parents=True); (output / "fits").mkdir(); (output / "predictions").mkdir()
    code = [Path(__file__).resolve(), PROTOCOL, ROOT / "scripts/run_transfer_v4.py", *sorted((ROOT / "src/transcif").rglob("*.py"))]
    files = [source / "monthly.csv", target / "monthly.csv",
             *[source / f"{r}_hourly.csv" for r in SOURCES], *[target / f"{r}_hourly.csv" for r in TARGETS]]
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "code": {str(p): sha(p) for p in code},
        "inputs": {str(p): sha(p) for p in files}, "source": str(source), "target": str(target),
        "configs": configs(), "delays": DELAYS, "factors": FACTORS, "seed": SEED, "samples": N_SAMPLES,
        "source_choice_period": "2025H1 LODO", "source_stress_period": "2025H2 LODO",
        "target_scope": "Previously viewed 2026H1 regions; development retest, NOT independent confirmation",
        "environment": environment()}
    write(output / "prefit_freeze.json", record)
    return record


def verify(output):
    record = json.loads((output / "prefit_freeze.json").read_text())
    for p, digest in {**record["code"], **record["inputs"]}.items():
        if sha(p) != digest: raise ValueError("Frozen input/code changed: " + p)
    if record["configs"] != configs() or record["factors"] != list(FACTORS):
        raise ValueError("Configuration grid changed")
    if record.get("delays") != list(DELAYS) or record.get("seed") != SEED or record.get("samples") != N_SAMPLES:
        raise ValueError("Frozen sampling/delay grid changed")
    if record.get("environment") != environment(): raise ValueError("Frozen numerical environment changed")
    return record


def load_training(source, target):
    monthly = pd.concat([pd.read_csv(source / "monthly.csv"), pd.read_csv(target / "monthly.csv")], ignore_index=True)
    proper, cal = {}, {}
    for region in SOURCES:
        frame = read_year(source / f"{region}_hourly.csv", 2024)
        proper[region] = frame[frame.time.dt.month.le(6)].copy()
        cal[region] = frame[frame.time.dt.month.gt(6)].copy()
    return monthly, proper, cal


def make_features(monthly, region, days, delay):
    return released_forecasts(monthly, region, days, delay)


def collect(frames, monthly, delay):
    result = {}
    for region, frame in frames.items():
        days, truth = daily_profiles(frame)
        result[region] = dict(days=days, truth=truth, coarse=make_features(monthly, region, days, delay))
    return result


def flatten(records, regions, field):
    if field == "features": return np.concatenate([records[r]["coarse"]["features"] for r in regions])
    return np.concatenate([records[r][field] for r in regions])


def fit_bundle(proper, cal, monthly, delay, excluded, proper_records, cal_records):
    regions = tuple(r for r in SOURCES if r != excluded)
    frames = {r: proper[r] for r in regions}
    bank = FuelTrajectoryBank().fit(frames)
    structural = fit_model(frames, monthly, cutoff="2024-07-01T00:00:00Z", delay_days=delay)
    rr = lambda records: np.concatenate([np.repeat(r, len(records[r]["days"])) for r in regions])
    baseline = DirectBaselines().fit(flatten(proper_records, regions, "features"), flatten(proper_records, regions, "truth"), rr(proper_records),
        flatten(cal_records, regions, "features"), flatten(cal_records, regions, "truth"), rr(cal_records))
    residuals, labels = [], []
    for region in regions:
        record = cal_records[region]
        point = structural.predict(record["days"], region, monthly)["analog_shape_half"] + record["coarse"]["anchor"][:, None]
        residuals.append(record["truth"] - point); labels.extend([region] * len(point))
    return {"bank": bank, "structural": structural, "baseline": baseline, "residuals": np.concatenate(residuals),
            "residual_regions": np.asarray(labels), "training_regions": regions, "excluded": excluded, "delay": delay}


def day_rng(origin, offset=0):
    value = pd.Timestamp(origin).value // (86400 * 10**9)
    return np.random.default_rng(np.random.SeedSequence([SEED, int(value), offset]))


def support_summary(result):
    return {"unsupported_target_mass": float(np.mean(result["unsupported_target_mass"])),
            "mean_effective_donors": float(np.mean(result["effective_donors"])),
            "min_effective_donors": float(np.min(result["effective_donors"])),
            "zero_generation_fallback_fraction": float(np.mean(result["zero_generation_fallback_fraction"])),
            "ess_scope": "Conditional donor distribution before finite sampling"}


def raw_predictions(bundle, monthly, region, days, coarse):
    """Generate all grid members from common lawful inputs, no outcome argument."""
    bank = bundle["bank"]
    direct = bundle["baseline"].predict(coarse["features"], days, n_scenarios=N_SAMPLES, seed=SEED)
    center = bundle["structural"].predict(days, region, monthly)["analog_shape_half"] + coarse["anchor"][:, None]
    labels = bundle["residual_regions"]
    weights = np.array([1 / (len(bundle["training_regions"]) * np.count_nonzero(labels == r)) for r in labels])
    ensembles = np.stack([center[i] + bundle["residuals"][day_rng(day).choice(len(labels), N_SAMPLES, p=weights)]
                          for i, day in enumerate(days)])
    direct["fresh_anchor"] = dict(point=center, ensemble=np.clip(ensembles, 0, 1000))
    cache = {}
    for config in configs():
        name, mode = config["name"], config["mode"]
        if mode in direct:
            yield config, direct[mode], {}
            continue
        key = (config["forecast"], config["temperature"], mode, config["bandwidth"])
        if key not in cache:
            options = dict(mode=mode, bandwidth=config["bandwidth"], n_scenarios=N_SAMPLES, seed=SEED)
            if config["forecast"] == "latest":
                result = bank.predict(coarse["experts"][:, 0], days, **options)
                support = support_summary(result)
            else:
                probability = coarse["weights"][str(config["temperature"])]; mix = coarse["experts"]
                if config["forecast"] == "adaptive_mean":
                    result = bank.predict(np.einsum("ne,nef->nf", probability, mix), days, **options)
                    support = support_summary(result)
                else:
                    outputs = [bank.predict(mix[:, e], days, **options) for e in range(3)]
                    stacked = np.stack([x["ensemble"] for x in outputs], axis=1)
                    paths = []
                    for i, day in enumerate(days):
                        indices = day_rng(day, 997).choice(3, N_SAMPLES, p=probability[i])
                        paths.append(stacked[i, indices, np.arange(N_SAMPLES)])
                    ensemble = np.asarray(paths)
                    result = {"point": ensemble.mean(1), "ensemble": ensemble}
                    support = {"unsupported_target_mass": float(np.mean(sum(probability[:, e] * outputs[e]["unsupported_target_mass"] for e in range(3)))),
                        "mean_effective_donors": float(np.mean(sum(probability[:, e] * outputs[e]["effective_donors"] for e in range(3)))),
                        "min_effective_donors": float(min(np.min(x["effective_donors"]) for x in outputs)),
                        "zero_generation_fallback_fraction": float(np.mean(sum(probability[:, e] * outputs[e]["zero_generation_fallback_fraction"] for e in range(3)))),
                        "ess_scope": "Expert-weighted component ESS; NOT exact mixture ESS; min across all components"}
            cache[key] = {"point": result["point"], "ensemble": result["ensemble"]}, support
        prediction, support = cache[key]
        yield config, prediction, support


def save_predictions(output, stage, region, delay, days, predictions):
    entries = []
    for config, result, support in predictions:
        name = f"{stage}-{region}-{delay}-r{config['round']}-{config['name']}.npz"
        path = output / "predictions" / name
        if path.exists(): raise FileExistsError(path)
        np.savez_compressed(path, origins=pd.DatetimeIndex(days).as_unit("ns").asi8, **result)
        entries.append(dict(region=region, delay=delay, **config, path=str(path.relative_to(output)),
                            sha256=sha(path), support=support, interval_only=config["mode"] == "cqr"))
    return entries


def score_entries(output, entries, data, year, folder):
    # Reject a changed member of the grid before parsing any hourly outcomes.
    for entry in entries:
        if sha(output / entry["path"]) != entry["sha256"]:
            raise ValueError("Prediction bytes changed")
    folder.mkdir()
    all_rows = []; counts = []
    with (folder / "day_metrics.csv").open("x", newline="") as handle:
        writer = None
        for region in sorted({e["region"] for e in entries}):
            days, truths = daily_profiles(read_year(data / f"{region}_hourly.csv", year))
            lookup = {d.value: y for d, y in zip(days, truths)}
            for entry in [e for e in entries if e["region"] == region]:
                path = output / entry["path"]
                if sha(path) != entry["sha256"]: raise ValueError("Prediction bytes changed")
                with np.load(path, allow_pickle=False) as archive: a = {k: archive[k] for k in archive.files}
                for factor in ((1.,) if entry["interval_only"] else FACTORS):
                    metrics = []
                    for i, timestamp in enumerate(a["origins"]):
                        if int(timestamp) not in lookup: continue
                        y = lookup[int(timestamp)]
                        if entry["interval_only"]:
                            values = {"empirical_" + k: v for k, v in V4.interval_metrics(a["lower"][i], a["upper"][i], y).items()}
                        else:
                            mean = a["ensemble"][i].mean(0)
                            paths = np.clip(mean + factor * (a["ensemble"][i] - mean), 0, 1000)
                            # Same inverse-CDF empirical interval convention as V6.
                            lower, upper = np.quantile(paths, [.05, .95], axis=0, method="inverted_cdf")
                            values = V4.score_day(a["point"][i], paths, lower, upper, y)
                            values["point_level_mae"] = float(abs(a["point"][i].mean() - y.mean()))
                            values["point_shape_mae"] = float(np.mean(abs((a["point"][i]-a["point"][i].mean())-(y-y.mean()))))
                        row = dict(region=region, delay=entry["delay"], round=entry["round"], name=entry["name"], mode=entry["mode"],
                            factor=factor, origin=str(pd.Timestamp(int(timestamp), tz="UTC")), semester="H1" if pd.Timestamp(int(timestamp)).month <= 6 else "H2")
                        # Fixed union schema prevents CQR creating pseudo distribution metrics.
                        keys = [*V4.score_day(np.zeros(24), np.zeros((2,24)), np.zeros(24), np.zeros(24), np.zeros(24)),
                                "point_level_mae", "point_shape_mae"] if writer is None else None
                        if writer is None:
                            writer = csv.DictWriter(handle, fieldnames=[*row, *keys]); writer.writeheader()
                        writer.writerow({**row, **values}); metrics.append({**row, **values})
                    frame = pd.DataFrame(metrics)
                    if frame.empty: raise ValueError("No scored days")
                    for semester, group in frame.groupby("semester"):
                        numeric = [c for c in group.select_dtypes("number") if c not in ("round", "delay", "factor")]
                        all_rows.append({k: frame.iloc[0][k] for k in ("region", "delay", "round", "name", "mode", "factor")} | {
                            "semester": semester, "days": len(group), **group[numeric].mean().to_dict()})
                    counts.append(dict(region=region, round=entry["round"], name=entry["name"], delay=entry["delay"], factor=factor, days=len(frame)))
            print("Scored", year, region, flush=True)
    table = pd.DataFrame(all_rows); table.to_csv(folder / "region_metrics.csv", index=False)
    pd.DataFrame(counts).to_csv(folder / "day_counts.csv", index=False)
    numeric = [c for c in table.select_dtypes("number") if c not in ("round", "delay", "factor", "days")]
    table.groupby(["round", "name", "mode", "factor", "semester"])[numeric].mean().reset_index().to_csv(folder / "macro_metrics.csv", index=False)
    write(folder / "score_seal.json", {p.name: sha(p) for p in folder.iterdir() if p.is_file()})
    return table


def select(table):
    table = table[table.semester == "H1"]
    grouped = table.groupby(["round", "name", "mode", "factor"])
    ranking = grouped.mean(numeric_only=True).drop(columns=["delay", "days"]).reset_index()
    ranking["worst_coverage"] = grouped.empirical_coverage90.min().to_numpy()
    ranking["violation"] = np.maximum(.9-ranking.empirical_coverage90,0)+np.maximum(ranking.empirical_coverage90-.95,0)+np.maximum(.85-ranking.worst_coverage,0)
    selected = {}

    def record(candidate, status):
        metric_names = ["crps", "empirical_coverage90", "empirical_width90",
                        "empirical_interval_score90", "worst_coverage", "violation"]
        return {"configuration": {"round": int(candidate['round']), "name": candidate['name'], "factor": float(candidate.factor)},
            "status": status, "metrics": {k: float(candidate[k]) for k in metric_names if pd.notna(candidate.get(k, np.nan))}}

    for round_ in (1, 2):
        selected[str(round_)] = {}
        for role, mask in [("primary", ranking["mode"].eq("recompose_joint")),
                           ("baseline", ranking["mode"].isin(["direct_analog", "fresh_anchor", "extratrees"])),
                           ("simple_anchor", ranking["mode"].eq("fresh_anchor")),
                           ("unconstrained_baseline", ranking["mode"].isin(["direct_analog", "fresh_anchor", "extratrees"])),
                           ("cqr", ranking["mode"].eq("cqr"))]:
            rows = ranking[(ranking["round"] == round_) & mask].copy()
            metric = "empirical_interval_score90" if role == "cqr" else "crps"
            if role == "unconstrained_baseline":
                selected[str(round_)][role] = record(rows.sort_values([metric,"name","factor"]).iloc[0], "unconstrained")
                continue
            passed = rows[rows.violation <= 1e-12]
            candidate = (passed if len(passed) else rows).sort_values((["violation"] if not len(passed) else []) + [metric,"name","factor"]).iloc[0]
            selected[str(round_)][role] = record(candidate, "passed" if len(passed) else "failed")
        primary = selected[str(round_)]["primary"]["configuration"]
        for role, mode in [("exact_direct", "direct_analog"), ("exact_independent", "recompose_independent"), ("exact_pooled", "pooled_recompose")]:
            rows = ranking[(ranking['round'] == round_) & ranking.name.eq(primary['name'].replace("recompose_joint", mode)) & ranking.factor.eq(primary['factor'])]
            if len(rows) != 1: raise ValueError("Missing exact primary mechanism control: " + role)
            row=rows.iloc[0];selected[str(round_)][role]=record(row,"passed" if row.violation<=1e-12 else "failed")
    selected['joint'] = {}
    for role in ('primary','baseline','simple_anchor','unconstrained_baseline','cqr'):
        choices=[selected[str(r)][role] for r in (1,2)]
        metric='empirical_interval_score90' if role=='cqr' else 'crps'
        passed=[r for r in choices if r['status']=='passed']
        available=passed or choices
        key=lambda r: (([] if passed or role=='unconstrained_baseline' else [r['metrics']['violation']]) +
            [r['metrics'][metric],r['configuration']['round'],r['configuration']['name'],r['configuration']['factor']])
        selected['joint'][role]=min(available,key=key)
    return selected, ranking


def validate_choices(output, choice):
    """Reconstruct source-only decisions from committed metrics before reuse."""
    for name,digest in choice['source_score_hashes'].items():
        if sha(output/'source_scores'/name)!=digest:raise ValueError('Source scores changed')
    actual,_=select(pd.read_csv(output/'source_scores/region_metrics.csv'))
    if actual!=choice['selections']:raise ValueError('Source-derived choice changed')
    if sha(output/'source_prediction_seal.json')!=choice['source_seal_sha256']:raise ValueError('Source seal changed')


def source_phase(output, source, target):
    freeze(output, source, target)
    monthly, proper, cal = load_training(source, target)
    entries = []
    origins = pd.date_range("2025-01-01", "2026-01-01", inclusive="left", tz="UTC")
    for delay in DELAYS:
        a, b = collect(proper, monthly, delay), collect(cal, monthly, delay)
        for region in SOURCES:
            bundle = fit_bundle(proper, cal, monthly, delay, region, a, b)
            with (output / "fits" / f"lodo-{region}-{delay}.pkl").open("xb") as f: pickle.dump(bundle, f)
            coarse = make_features(monthly, region, origins, delay)
            entries.extend(save_predictions(output, "source", region, delay, origins, raw_predictions(bundle, monthly, region, origins, coarse)))
            print("Source predicted", delay, region, flush=True)
        full = fit_bundle(proper, cal, monthly, delay, None, a, b)
        with (output / "fits" / f"full-{delay}.pkl").open("xb") as f: pickle.dump(full, f)
    verify(output)
    if len(entries) != len(SOURCES)*len(DELAYS)*len(configs()): raise ValueError("Incomplete source prediction grid")
    write(output / "source_prediction_seal.json", {"entries": entries, "fits": {p.name: sha(p) for p in (output/"fits").glob("*.pkl")}})
    score_entries(output, entries, source, 2025, output / "source_scores")
    table = pd.read_csv(output / "source_scores/region_metrics.csv")
    selections, ranking = select(table); ranking.to_csv(output / "source_scores/ranking.csv", index=False)
    write(output / "source_choices.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "selections": selections,
        "source_score_hashes": {p.name: sha(p) for p in (output/"source_scores").iterdir()},
        "source_seal_sha256": sha(output/"source_prediction_seal.json"), "target_model_scores_read": False})
    verify(output)


def target_phase(output):
    record = verify(output); source, target = Path(record["source"]), Path(record["target"])
    if (output/"target_prediction_seal.json").exists() or (output/"target_scores").exists(): raise FileExistsError("Target phase is immutable")
    choice = json.loads((output/"source_choices.json").read_text()); choice_hash = sha(output/"source_choices.json")
    validate_choices(output,choice)
    seal = json.loads((output/"source_prediction_seal.json").read_text())
    if sha(output/"source_prediction_seal.json") != choice["source_seal_sha256"]: raise ValueError("Source seal changed")
    monthly = pd.concat([pd.read_csv(source/"monthly.csv"), pd.read_csv(target/"monthly.csv")], ignore_index=True)
    origins = pd.date_range("2026-01-01", "2026-07-01", inclusive="left", tz="UTC"); entries=[]
    for delay in DELAYS:
        path = output/"fits"/f"full-{delay}.pkl"
        if sha(path)!=seal["fits"][path.name]: raise ValueError("Fitted artifact changed")
        with path.open("rb") as f: bundle=pickle.load(f)
        for region in TARGETS:
            coarse=make_features(monthly,region,origins,delay)
            entries.extend(save_predictions(output,"target",region,delay,origins,raw_predictions(bundle,monthly,region,origins,coarse)))
            print("Target predicted",delay,region,flush=True)
    verify(output)
    if sha(output/"source_choices.json")!=choice_hash:raise ValueError("Source choices changed")
    write(output/"target_prediction_seal.json",{"created_utc":datetime.now(timezone.utc).isoformat(),"entries":entries,
        "source_choices_sha256":choice_hash,"all_grid_complete":len(entries)==len(TARGETS)*len(DELAYS)*len(configs()),
        "target_labels_read_in_this_prediction_phase":False,"target_already_viewed_in_V6":True})
    if len(entries)!=len(TARGETS)*len(DELAYS)*len(configs()):raise ValueError("Incomplete prediction grid")
    score_entries(output,entries,target,2026,output/"target_scores")
    verify(output)
    write(output/"completed.json",{"created_utc":datetime.now(timezone.utc).isoformat(),"source_choices_sha256":choice_hash,
        "target_seal_sha256":sha(output/"target_prediction_seal.json"),
        "target_score_seal_sha256":sha(output/"target_scores/score_seal.json"),"frozen_hashes_verified":True,
        "scope":"Development retest on already seen targets; no target-hourly training"})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--phase",choices=("source","target"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source",type=Path,default=Path("results/heldout-2026h1-data"))
    parser.add_argument("--target",type=Path,default=Path("results/external-v6-data"))
    args=parser.parse_args();output=args.output.resolve()
    if args.phase=="source":source_phase(output,args.source.resolve(),args.target.resolve())
    else:target_phase(output)


if __name__=="__main__":main()
