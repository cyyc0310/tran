"""Frozen source-2025 development and new-region 2026H1 confirmation.

Source point bytes stay unchanged. Conditional residual models fit 2024H1,
paired banks/CQR offsets use 2024H2. Every target prediction is globally sealed
before any confirmation hourly number is parsed.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V5 = _module("conditional_v6_v5", ROOT / "scripts/run_risk_transfer_v5.py")
V4 = _module("conditional_v6_v4", ROOT / "scripts/run_transfer_v4.py")
SOURCES = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
HOLDOUTS = ("AZPS", "DUK", "LGEE", "PACE", "PSCO", "SRP")
POINTS = ("support_gp_medium", "harmonic_ridge", "softmax_moe")
DELAYS = (0, 30, 90)
VARIANT = "conditional_v6"
PROTOCOL = "docs/CONDITIONAL_TRANSFER_V6_PROTOCOL.md"
MULTIPLIERS = {"x1": 1., "x1p25": 1.25, "x1p5": 1.5, "x2": 2.}
MODES = ("global", "loglinear", "extratrees", "structured", "structured_nohindcast")
CONFIGS = {f"{mode}_{suffix}": {"mode": mode, "multiplier": value}
           for mode in MODES for suffix, value in MULTIPLIERS.items()}
CONFIGS.update({"historical_v5_component110": {"old_calibration": "seven_component110"},
                "historical_v5_risk0p5": {"old_calibration": "adaptive_component110_beta0p5"}})
V4.PROTOCOL = PROTOCOL


def grid_parameters():
    return {"study_version": 6, "variants": [VARIANT], "methods": list(POINTS),
        "delays": list(DELAYS), "calibration_modes": list(CONFIGS), "parameters": CONFIGS,
        "interval_only_modes": ["cqr"], "residual_fit_year": 2024,
        "proper_months": [1, 6], "bank_months": [7, 12], "source_year": 2025,
        "selection_regions": list(SOURCES), "external_regions": list(HOLDOUTS),
        "external_year": 2026, "external_stop_exclusive": "2026-07-01T00:00:00+00:00",
        "n_samples": 128, "ensemble_seed": 17023, "projection_bounds": [0., float(V4.EF.max())]}


def validate_grid(config):
    if any(config.get(key) != value for key, value in grid_parameters().items()):
        raise ValueError("Current V6 grid differs from the frozen source grid")
    if [r.get("delay") for r in config.get("records", [])] != list(DELAYS):
        raise ValueError("V6 must contain the complete fixed delay grid")
    for r in config["records"]:
        if any(r.get(key) != f"{folder}/{prefix}-{r['delay']}.pkl" for key, folder, prefix in
               (("path", "fit", "conditional_v6"), ("base_path", "base", "round2"),
                ("historical_path", "historical", "risk_v5"))):
            raise ValueError("Frozen V6 fit paths differ")


def verify_external(manifest):
    directory = Path(manifest["external_data"] or "")
    if not manifest.get("external_data") or str(directory / "audit.json") not in manifest["data_hashes"]:
        raise ValueError("Need a separately audited confirmation dataset")
    audit = json.loads((directory / "audit.json").read_text())
    if tuple(audit.get("external_regions", ())) != HOLDOUTS or audit.get("final_external_year") != 2026:
        raise ValueError("External regions/year differ from V6 protocol")
    for name in ("cif_scores_inspected", "cif_summaries_computed", "target_hourly_used_for_fit_or_calibration"):
        if audit.get(name) is not False:
            raise ValueError("External audit is not clean: " + name)
    if not audit.get("protocol_sha256_before") or audit["protocol_sha256_before"] != audit.get("protocol_sha256_after"):
        raise ValueError("Preparation protocol changed")
    required = {"monthly.csv", *(f"{r}_hourly.csv" for r in HOLDOUTS)}
    if not required <= set(audit.get("derived_hashes", {})):
        raise ValueError("External audit omits required files")
    for name, digest in audit["derived_hashes"].items():
        if manifest["data_hashes"].get(str(directory / name)) != digest:
            raise ValueError("External derived bytes differ from audit")
    return audit


def _write_fit(output, config):
    validate_grid(config)
    V4.write_json(output / "run_config.json", config)
    artifacts = {r[key]: V4.sha(output / r[key]) for r in config["records"]
                 for key in ("path", "base_path", "historical_path")}
    artifacts["run_config.json"] = V4.sha(output / "run_config.json")
    V4.verify_frozen_inputs(output)
    V4.write_json(output / "fit_seal.json", {"prefit_sha256": V4.sha(output / "prefit_freeze.json"), "artifacts": artifacts})


def _rank(macro, regional, *, metric="crps"):
    keys = ["variant", "method", "calibration"]
    numeric = [c for c in macro.select_dtypes(include="number") if c != "delay"]
    table = macro.groupby(keys)[numeric].mean().reset_index()
    minimum = regional.groupby(keys).empirical_coverage90.min().rename("worst_region_delay_coverage90")
    table = table.merge(minimum, on=keys)
    table["coverage_gap"] = abs(table.empirical_coverage90-.9)
    table["gate_violation"] = np.maximum(table.coverage_gap-.05, 0)+np.maximum(.85-table.worst_region_delay_coverage90, 0)
    table["gate_passed"] = table.gate_violation <= 1e-12
    return table


def _choose(table, mask=None, *, metric="crps", unconstrained=False):
    candidates = table if mask is None else table.loc[mask]
    if candidates.empty:
        raise ValueError("Missing predeclared selection family")
    passing = candidates.loc[candidates.gate_passed]
    if unconstrained:
        choices, status = candidates, "unconstrained"
    else:
        choices, status = (passing, "passed") if len(passing) else (candidates, "failed")
    order = [metric, "variant", "method", "calibration"] if status != "failed" else ["gate_violation", metric, "variant", "method", "calibration"]
    row = choices.sort_values(order, kind="stable").iloc[0]
    return {"status": status, "configuration": {k: str(row[k]) for k in ("variant", "method", "calibration")},
        "metrics": {k: float(row[k]) for k in (metric, "empirical_coverage90", "empirical_width90", "empirical_interval_score90", "worst_region_delay_coverage90", "gate_violation")}}


def select_roles(macro, regional, interval_macro, interval_regional):
    ranked = _rank(macro, regional)
    primary = ranked.calibration.str.startswith("structured_x")
    baseline = ranked.calibration.str.startswith(("global_", "loglinear_", "extratrees_"))
    historical = ranked.calibration.str.startswith("historical_v5_")
    return {"primary": _choose(ranked, primary), "baseline": _choose(ranked, baseline),
        "unconstrained_baseline": _choose(ranked, baseline, unconstrained=True),
        "historical_v5": _choose(ranked, historical, unconstrained=True),
        "cqr": _choose(_rank(interval_macro, interval_regional), metric="empirical_interval_score90")}


def load_selection(path):
    path = Path(path).resolve()
    source = path.parent.parent
    selected = json.loads(path.read_text())
    if selected.get("source_year") != 2025 or tuple(selected.get("selection_regions", ())) != SOURCES:
        raise ValueError("Selection must be source-only 2025")
    frozen = V4.verify_frozen_inputs(source)
    fit = V4._verify_artifacts(source, "fit_seal.json")
    for name in ("macro_metrics", "region_metrics", "interval_macro_metrics", "interval_region_metrics"):
        if selected.get(name+"_sha256") != V4.sha(path.parent / (name+".csv")):
            raise ValueError("Source selection score bytes changed")
    if selected.get("fit_seal_sha256") != V4.sha(source / "fit_seal.json") or selected.get("fit_artifact_hashes") != fit["artifacts"]:
        raise ValueError("Source selection fitted bytes changed")
    if selected.get("model_closure_hashes") != V4.model_closure_hashes(frozen) or selected.get("environment") != frozen["environment"]:
        raise ValueError("Source selection environment/code closure changed")
    if selected.get("prediction_global_seal_sha256") != V4.sha(source / "all_predictions_sealed.json"):
        raise ValueError("Source prediction seal changed")
    frames = [pd.read_csv(path.parent / (name+".csv")) for name in
              ("macro_metrics", "region_metrics", "interval_macro_metrics", "interval_region_metrics")]
    if selected["selections"] != select_roles(*frames):
        raise ValueError("Selection roles differ from the frozen source rule")
    config = json.loads((source / "run_config.json").read_text())
    validate_grid(config)
    return source, config, selected


def fit_phase(data, baseline, historical, output, *, external_data=None, selection=None):
    from transcif.research.conditional_transfer_v6 import ConditionalTransport, information_features
    from transcif.research.quantile_baseline_v6 import QuantileBaseline
    data, baseline, historical, output = [Path(p).resolve() for p in (data, baseline, historical, output)]
    _, original_seal, originals, dependencies = V5.verify_base(baseline, ROOT)
    historical_seal = V4._verify_artifacts(historical, "fit_seal.json")
    historical_config = json.loads((historical / "run_config.json").read_text())
    V5.validate_grid(historical_config)
    for r in historical_config["records"]:
        if historical_seal["artifacts"][r["base_path"]] != original_seal["artifacts"][f"fit/round2-{r['delay']}.pkl"]:
            raise ValueError("Historical V5 point model provenance differs")
    extra = [ROOT / "scripts/run_conditional_transfer_v6.py", ROOT / "scripts/run_risk_transfer_v5.py",
             baseline / "fit_seal.json", baseline / "prefit_freeze.json", baseline / "run_config.json",
             historical / "fit_seal.json", historical / "run_config.json", historical / "prefit_freeze.json"]
    extra.extend(baseline / r["path"] for r in originals)
    extra.extend(historical / r["path"] for r in historical_config["records"])
    source, source_config, selected = None, None, None
    if external_data is not None:
        if selection is None:
            raise ValueError("Confirmation requires a frozen source selection")
        source, source_config, selected = load_selection(selection)
        extra.extend([Path(selection).resolve(), *(source / name for name in
            ("prefit_freeze.json", "fit_seal.json", "run_config.json", "all_predictions_sealed.json", "evaluation_plan.json"))])
        extra.extend(source / name for name in selected["fit_artifact_hashes"])
    manifest = V4.freeze_inputs(ROOT, data, output, extra_files=extra, external_data=external_data)
    if external_data is not None:
        verify_external(manifest)
        old = json.loads((source / "prefit_freeze.json").read_text())
        if old["environment"] != manifest["environment"] or V4.model_closure_hashes(old) != V4.model_closure_hashes(manifest):
            raise ValueError("Source environment or model code changed")
        for relative in ("scripts/run_conditional_transfer_v6.py", "scripts/run_risk_transfer_v5.py"):
            if old["code_hashes"].get(str(ROOT / relative)) != V4.sha(ROOT / relative):
                raise ValueError("Source-selected runner changed")
        for name, digest in selected["fit_artifact_hashes"].items():
            if name == "run_config.json":
                continue
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, destination)
            if V4.sha(destination) != digest:
                raise ValueError("Copied source fitted bytes differ")
        config = {**source_config, "source_selection": selected,
            "source_provenance": {"source_run": str(source), "source_fit_seal_sha256": V4.sha(source / "fit_seal.json"),
                                  "source_fit_artifact_hashes": selected["fit_artifact_hashes"], "fitted_bytes_identical": True}}
        _write_fit(output, config)
        return config
    monthly = V4.monthly_inputs(manifest)
    origins = V4._origins(2024)
    truths = {r: V4.daily_profiles(V4.read_year(data / f"{r}_hourly.csv", 2024)) for r in SOURCES}
    records, audits = [], {}
    for original in originals:
        delay = original["delay"]
        paths = {"base_path": f"base/round2-{delay}.pkl", "historical_path": f"historical/risk_v5-{delay}.pkl",
                 "path": f"fit/conditional_v6-{delay}.pkl"}
        for key, source_file in (("base_path", baseline / original["path"]),
                                 ("historical_path", historical / f"fit/risk_v5-{delay}.pkl")):
            destination = output / paths[key]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, destination)
            if V4.sha(destination) != V4.sha(source_file):
                raise ValueError("Original fitted artifact copy differs")
        with (output / paths["base_path"]).open("rb") as handle:
            model = pickle.load(handle)["model"]
        parts = {method: [] for method in POINTS}
        for region in SOURCES:
            predictions = V5.base_points(model, origins, region, monthly)
            x, names = information_features(origins, region, monthly, delay)
            days, truth = truths[region]
            ix = origins.get_indexer(days)
            if not len(ix) or np.any(ix < 0):
                raise ValueError("Missing source fitting days")
            for method in POINTS:
                p = predictions[method]
                parts[method].append({"point": p["point"][ix], "truth": truth, "features": x[ix],
                    "regions": np.repeat(region, len(ix)), "days": days.to_numpy(),
                    "level_scales": p["level_scale"][ix], "shape_scales": p["shape_scale"][ix]})
        calibrators = {}
        for method in POINTS:
            values = {name: np.concatenate([part[name] for part in parts[method]]) for name in parts[method][0]}
            for mode in MODES:
                calibrators[method, mode] = ConditionalTransport(SOURCES, mode=mode, fit_year=2024).fit(**values)
                audits[f"{delay}/{method}/{mode}"] = calibrators[method, mode].audit_
            calibrators[method, "cqr"] = QuantileBaseline(SOURCES, fit_year=2024).fit(**values)
            audits[f"{delay}/{method}/cqr"] = calibrators[method, "cqr"].audit_
        destination = output / paths["path"]
        destination.parent.mkdir(exist_ok=True)
        with destination.open("xb") as handle:
            pickle.dump({"calibrators": calibrators}, handle)
        records.append({"variant": VARIANT, "delay": delay, **paths})
        print(f"Fitted delay={delay}: 15 conditional models, 3 CQR; source 2024 only", flush=True)
    V4.write_json(output / "calibration_audit.json", audits)
    config = {**grid_parameters(), "records": records, "source_selection": None,
        "base_fit_seal_sha256": V4.sha(baseline / "fit_seal.json"), "base_code_dependency_hashes": dependencies,
        "historical_fit_seal_sha256": V4.sha(historical / "fit_seal.json"), "feature_names": list(names)}
    _write_fit(output, config)
    return config


def _save_prediction(folder, origins, p, ensemble, lower, upper, **metadata):
    if ensemble is not None:
        ensemble, lower, upper, boundary = V5.project_distribution(ensemble, lower, upper)
    else:
        bounds = np.stack((lower, upper), axis=1)
        boundary = {"preprojection_interval_endpoints_below_zero": np.mean(bounds < 0, axis=(1,2)),
                    "preprojection_interval_endpoints_above_max": np.mean(bounds > V4.EF.max(), axis=(1,2))}
        lower, upper = np.clip(lower, 0, V4.EF.max()), np.clip(upper, 0, V4.EF.max())
    arrays = {"origins": origins.as_unit("ns").asi8, "point": p["point"],
              "analytic_lower": lower, "analytic_upper": upper, **boundary, **metadata}
    if ensemble is not None:
        arrays["ensemble"] = np.asarray(ensemble, dtype="float32")
    if any(not np.isfinite(v).all() for v in arrays.values()) or lower.shape != (len(origins),24) or np.any(lower > upper):
        raise ValueError("Invalid predicted arrays")
    if ensemble is not None and ensemble.shape != (len(origins),128,24):
        raise ValueError("Invalid ensemble shape")
    with folder.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return {key: float(value.mean()) for key, value in boundary.items()}


def predict_seal_phase(output, *, study="development"):
    from transcif.research.conditional_transfer_v6 import information_features, information_feature_audit
    output = Path(output).resolve()
    manifest = V4.verify_frozen_inputs(output)
    V4._verify_artifacts(output, "fit_seal.json")
    config = json.loads((output / "run_config.json").read_text())
    validate_grid(config)
    if study == "development":
        targets, year, origins, selected = SOURCES, 2025, V4._origins(2025), None
    elif study == "holdout":
        verify_external(manifest)
        targets, year, origins = HOLDOUTS, 2026, V4._origins(2026, "2026-06-30")
        selected = config.get("source_selection")
        if not selected or not config.get("source_provenance"):
            raise ValueError("Confirmation needs exact source-selected artifacts")
    else:
        raise ValueError("Unknown study")
    monthly = V4.monthly_inputs(manifest)
    modes = [*CONFIGS, "cqr"]
    expected = [(r,d,VARIANT,m,c) for r in targets for d in DELAYS for m in POINTS for c in modes]
    V4.write_json(output / "evaluation_plan.json", {"study": study, "year": year, "targets": list(targets),
        "origins": [str(day) for day in origins], "selection": selected,
        "label_data": manifest["external_data"] if study == "holdout" else manifest["data"],
        "expected": [list(key) for key in expected]})
    folder = output / "predictions"
    folder.mkdir()
    entries, feature_audits = [], {}
    for record in config["records"]:
        V4.verify_frozen_inputs(output)
        V4._verify_artifacts(output, "fit_seal.json")
        with (output / record["base_path"]).open("rb") as handle:
            model = pickle.load(handle)["model"]
        with (output / record["path"]).open("rb") as handle:
            calibrators = pickle.load(handle)["calibrators"]
        with (output / record["historical_path"]).open("rb") as handle:
            old = pickle.load(handle)["calibrators"]
        for region in targets:
            predictions = V5.base_points(model, origins, region, monthly)
            x, names = information_features(origins, region, monthly, record["delay"])
            feature_audits[f"{record['delay']}/{region}"] = information_feature_audit(x)
            for method in POINTS:
                p = predictions[method]
                base_outputs = {mode: calibrators[method,mode].predict(p["point"],x,
                    level_scales=p["level_scale"],shape_scales=p["shape_scale"],n_samples=128,seed=17023) for mode in MODES}
                for name, parameters in CONFIGS.items():
                    metadata = {}
                    if "mode" in parameters:
                        result = base_outputs[parameters["mode"]]
                        center = result["corrected_point"]
                        factor = parameters["multiplier"]
                        ensemble = center[:,None]+factor*(result["ensemble"]-center[:,None])
                        lower, upper = (center+factor*(result[key]-center) for key in ("lower","upper"))
                        metadata = {key: result[key] for key in ("corrected_point","level_scale","shape_scale","bias")}
                    else:
                        old_name = parameters["old_calibration"]
                        transformed = V5.transform(origins,region,monthly,p,record["delay"],old_name)
                        options = {"ages": V4.age_bins(transformed["age"]), "level_scale": transformed["level_scale"], "shape_scale": transformed["shape_scale"]}
                        c = old[method,old_name]
                        lower, upper = c.intervals(transformed["point"],transformed["scale"],mode="component_transport_110",**options)
                        ensemble = c.ensemble(transformed["point"],transformed["scale"],mode="component_transport_110",**options,n_samples=128,seed=17023)
                    path = folder / f"{region}-{record['delay']}-{method}-{name}.npz"
                    boundary = _save_prediction(path,origins,p,ensemble,lower,upper,**metadata)
                    entries.append({"region":region,"delay":record["delay"],"variant":VARIANT,"method":method,
                                    "calibration":name,"path":str(path.relative_to(output)),"kind":"probabilistic","preprojection_boundary_mass":boundary})
                lower,upper = calibrators[method,"cqr"].intervals(p["point"],x,level_scales=p["level_scale"],shape_scales=p["shape_scale"])
                path=folder/f"{region}-{record['delay']}-{method}-cqr.npz"
                boundary=_save_prediction(path,origins,p,None,lower,upper)
                entries.append({"region":region,"delay":record["delay"],"variant":VARIANT,"method":method,
                                "calibration":"cqr","path":str(path.relative_to(output)),"kind":"interval_only","preprojection_boundary_mass":boundary})
            print(f"Sealed candidate files ready: delay={record['delay']} region={region}",flush=True)
    V4.write_json(output/"information_feature_audit.json",feature_audits)
    pd.DataFrame([{**{key:e[key] for key in ("region","delay","variant","method","calibration")},**e["preprojection_boundary_mass"]} for e in entries]).to_csv(output/"preprojection_boundary_audit.csv",index=False)
    V4.seal_predictions(output,expected,entries)
    V4.write_json(output/"evaluation_commit.json",{"plan_sha256":V4.sha(output/"evaluation_plan.json"),"global_seal_sha256":V4.sha(output/"all_predictions_sealed.json")})
    return entries


def score_phase(output):
    output=Path(output).resolve()
    if (output/"scores").exists():
        raise FileExistsError("Scores are immutable")
    access=V4.LabelAccess(output)
    commit=json.loads((output/"evaluation_commit.json").read_text())
    if commit.get("plan_sha256")!=V4.sha(output/"evaluation_plan.json") or commit.get("global_seal_sha256")!=access.global_digest:
        raise ValueError("Score commit differs from global seal")
    plan=json.loads((output/"evaluation_plan.json").read_text())
    frozen=V4.verify_frozen_inputs(output)
    folder=output/"scores";folder.mkdir()
    summaries={False:[],True:[]};counts=[]
    with (folder/"day_metrics.csv").open("x",newline="") as probfile, (folder/"interval_day_metrics.csv").open("x",newline="") as intfile:
        writers={False:None,True:None};handles={False:probfile,True:intfile}
        for region in plan["targets"]:
            days,truths=V4.daily_profiles(access.read(Path(plan["label_data"])/f"{region}_hourly.csv",plan["year"]))
            truth_by_day={d.value:y for d,y in zip(days,truths)}
            for entry in access.seal["entries"]:
                if entry["region"]!=region: continue
                interval=entry["kind"]=="interval_only"
                metrics=[]
                with np.load(output/entry["path"],allow_pickle=False) as stored:
                    names=("origins","point","analytic_lower","analytic_upper")+(() if interval else ("ensemble",))
                    archive={name:stored[name] for name in names}
                    for i,origin in enumerate(archive["origins"]):
                        truth=truth_by_day.get(int(origin))
                        if truth is None: continue
                        if interval:
                            metric={"empirical_"+k:v for k,v in V4.interval_metrics(archive["analytic_lower"][i],archive["analytic_upper"][i],truth).items()}
                        else:
                            metric=V4.score_day(archive["point"][i],archive["ensemble"][i],archive["analytic_lower"][i],archive["analytic_upper"][i],truth)
                        row={k:entry[k] for k in ("region","delay","variant","method","calibration")}
                        row.update(origin=str(pd.Timestamp(int(origin),tz="UTC")),**metric)
                        if writers[interval] is None:
                            writers[interval]=csv.DictWriter(handles[interval],fieldnames=list(row));writers[interval].writeheader()
                        writers[interval].writerow(row);metrics.append(metric)
                if not metrics: raise ValueError("No matching outcome days")
                group={k:entry[k] for k in ("region","delay","variant","method","calibration")}
                summaries[interval].append({**group,**pd.DataFrame(metrics).mean().to_dict()})
                counts.append({**group,"days":len(metrics),"kind":entry["kind"]})
            print(f"Scored after global seal: {region}",flush=True)
    frames={}
    for interval in (False,True):
        prefix="interval_" if interval else ""
        regional=pd.DataFrame(summaries[interval]);regional.to_csv(folder/f"{prefix}region_metrics.csv",index=False)
        numeric=[c for c in regional.select_dtypes(include="number") if c!="delay"]
        macro=regional.groupby(["delay","variant","method","calibration"])[numeric].mean().reset_index()
        macro.to_csv(folder/f"{prefix}macro_metrics.csv",index=False)
        frames[prefix]=(macro,regional)
    pd.DataFrame(counts).to_csv(folder/"day_counts.csv",index=False)
    if plan["study"]=="development":
        # Re-read the committed tables: selection and later verification must
        # use exactly the same floating-point CSV representation.
        committed=[pd.read_csv(folder/(name+".csv")) for name in
                   ("macro_metrics","region_metrics","interval_macro_metrics","interval_region_metrics")]
        selected={"source_year":2025,"selection_regions":list(SOURCES),"selections":select_roles(*committed),
            "model_closure_hashes":V4.model_closure_hashes(frozen),"environment":frozen["environment"],
            "fit_seal_sha256":V4.sha(output/"fit_seal.json"),"fit_artifact_hashes":V4._verify_artifacts(output,"fit_seal.json")["artifacts"],
            "prediction_global_seal_sha256":access.global_digest}
        selected.update({name+"_sha256":V4.sha(folder/(name+".csv")) for name in
                        ("macro_metrics","region_metrics","interval_macro_metrics","interval_region_metrics")})
        V4.write_json(folder/"selected_config.json",selected)
    else:
        fixed=plan["selection"]["selections"]["primary"]["configuration"]
        macro=frames[""][0]
        macro.loc[np.logical_and.reduce([macro[k].eq(v) for k,v in fixed.items()])].to_csv(folder/"fixed_primary_metrics.csv",index=False)
    V4.verify_frozen_inputs(output);V4._verify_artifacts(output,"all_predictions_sealed.json")
    V4.write_json(folder/"postrun_verification.json",{"all_frozen_hashes_match":True,"global_seal_sha256":access.global_digest,
        "target_label_access_after_global_seal":True,"score_hashes":{p.name:V4.sha(p) for p in sorted(folder.iterdir()) if p.is_file()}})
    return frames[""][0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("fit","predict_seal","score","all"),default="all")
    parser.add_argument("--study",choices=("development","holdout"),default="development")
    parser.add_argument("--data",type=Path,default=Path("results/heldout-2026h1-data"))
    parser.add_argument("--baseline",type=Path,default=Path("results/transfer-v4-round2-source2024"))
    parser.add_argument("--historical",type=Path,default=Path("results/risk-transfer-v5-source2024"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--external-data",type=Path)
    parser.add_argument("--selection",type=Path)
    args=parser.parse_args()
    if args.study=="holdout" and (args.external_data is None or args.selection is None):
        parser.error("Confirmation needs --external-data and --selection")
    if args.study=="development" and (args.external_data is not None or args.selection is not None):
        parser.error("Development cannot consume confirmation inputs or an external selection")
    if args.phase in ("fit","all"):
        fit_phase(args.data,args.baseline,args.historical,args.output,external_data=args.external_data,selection=args.selection)
    if args.phase in ("predict_seal","all"): predict_seal_phase(args.output,study=args.study)
    if args.phase in ("score","all"): print(score_phase(args.output).to_string(index=False))


if __name__=="__main__": main()
