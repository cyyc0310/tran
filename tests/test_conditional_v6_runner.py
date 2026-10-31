import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SPEC=importlib.util.spec_from_file_location("conditional_v6_test_runner",Path(__file__).parents[1]/"scripts/run_conditional_transfer_v6.py")
runner=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def grid():
    return {**runner.grid_parameters(), "records":[{"delay":d,"path":f"fit/conditional_v6-{d}.pkl",
        "base_path":f"base/round2-{d}.pkl","historical_path":f"historical/risk_v5-{d}.pkl"} for d in runner.DELAYS]}


def frames():
    rows=[];interval=[]
    for r in runner.SOURCES:
        for d in runner.DELAYS:
            for m in runner.POINTS:
                for c in [*runner.CONFIGS,"cqr"]:
                    row=dict(region=r,delay=d,variant=runner.VARIANT,method=m,calibration=c,
                        empirical_coverage90=.9024646969078041,empirical_width90=100.12345678901234,
                        empirical_interval_score90=105.45678901234567)
                    if c=="cqr": interval.append(row)
                    else: rows.append({**row,"crps":30.123456789012345})
    regional,iregional=pd.DataFrame(rows),pd.DataFrame(interval)
    keys=["delay","variant","method","calibration"]
    return regional.groupby(keys).mean(numeric_only=True).reset_index(),regional,iregional.groupby(keys).mean(numeric_only=True).reset_index(),iregional


def test_grid_rejects_changed_multiplier_mapping(monkeypatch):
    config=grid();runner.validate_grid(config)
    changed=copy.deepcopy(runner.CONFIGS)
    changed["structured_x1"]["multiplier"]=9.
    monkeypatch.setattr(runner,"CONFIGS",changed)
    with pytest.raises(ValueError,match="grid"):
        runner.validate_grid(config)


def test_failed_coverage_keeps_fixed_diagnostic_not_qualified():
    data=frames()
    for frame in data: frame["empirical_coverage90"]=.7
    chosen=runner.select_roles(*data)
    assert chosen["primary"]["status"]==chosen["baseline"]["status"]==chosen["cqr"]["status"]=="failed"
    assert chosen["unconstrained_baseline"]["status"]=="unconstrained"
    assert chosen["historical_v5"]["status"]=="unconstrained"
    assert chosen["primary"]["configuration"]["calibration"].startswith("structured_x")
    assert not chosen["baseline"]["configuration"]["calibration"].startswith("structured")


def test_source_selection_roundtrip_and_tamper(tmp_path):
    source=tmp_path/"source";scores=source/"scores";scores.mkdir(parents=True)
    manifest={"root":str(tmp_path),"code_hashes":{},"data_hashes":{},"snapshot_hashes":{},"environment":{"fixture":True}}
    runner.V4.write_json(source/"prefit_freeze.json",manifest)
    runner.V4.write_json(source/"run_config.json",grid())
    artifacts={"run_config.json":runner.V4.sha(source/"run_config.json")}
    for record in grid()["records"]:
        for key in ("path","base_path","historical_path"):
            path=source/record[key];path.parent.mkdir(exist_ok=True);path.write_bytes(b"frozen fit")
            artifacts[record[key]]=runner.V4.sha(path)
    runner.V4.write_json(source/"fit_seal.json",{"prefit_sha256":runner.V4.sha(source/"prefit_freeze.json"),"artifacts":artifacts})
    runner.V4.write_json(source/"all_predictions_sealed.json",{"fixture":True})
    names=("macro_metrics","region_metrics","interval_macro_metrics","interval_region_metrics")
    for name,frame in zip(names,frames()): frame.to_csv(scores/(name+".csv"),index=False)
    committed=[pd.read_csv(scores/(name+".csv")) for name in names]
    selected={"source_year":2025,"selection_regions":list(runner.SOURCES),"selections":runner.select_roles(*committed),
        "model_closure_hashes":{},"environment":manifest["environment"],"fit_artifact_hashes":artifacts,
        "fit_seal_sha256":runner.V4.sha(source/"fit_seal.json"),"prediction_global_seal_sha256":runner.V4.sha(source/"all_predictions_sealed.json")}
    selected.update({name+"_sha256":runner.V4.sha(scores/(name+".csv")) for name in names})
    runner.V4.write_json(scores/"selected_config.json",selected)
    assert runner.load_selection(scores/"selected_config.json")[2]==selected
    selected["selections"]["primary"]["configuration"]["calibration"]="global_x1"
    (scores/"selected_config.json").write_text(json.dumps(selected))
    with pytest.raises(ValueError,match="roles"):
        runner.load_selection(scores/"selected_config.json")


def test_score_cannot_create_outputs_without_global_seal(tmp_path):
    source=tmp_path/"unsealed";source.mkdir()
    with pytest.raises(FileNotFoundError): runner.score_phase(source)
    assert not (source/"scores").exists()


def test_cqr_archive_has_no_fake_ensemble(tmp_path):
    p={"point":np.full((1,24),500.)}
    origins=pd.DatetimeIndex(["2026-01-01"],tz="UTC")
    path=tmp_path/"cqr.npz"
    runner._save_prediction(path,origins,p,None,np.full((1,24),-10.),np.full((1,24),1200.))
    with np.load(path) as stored:
        assert "ensemble" not in stored
        assert np.all(stored["analytic_lower"]==0.) and np.all(stored["analytic_upper"]==1000.)
