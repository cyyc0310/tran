import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from coarse_bridge_v6 import AnchorTransport, MODES, coarse_anchor
from transcif.research.conditional_transfer_v6 import ConditionalTransport, FEATURE_NAMES, information_features
from transcif.research.eia_proxy import FUELS

SPEC = importlib.util.spec_from_file_location("test_anchored_v6_runner", ROOT / "scripts/run_anchored_transfer_v6.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def cohort():
    rng = np.random.default_rng(17027)
    days = pd.DatetimeIndex([f"2024-{month:02d}-{day:02d}" for region in range(2)
                             for month in range(1, 13) for day in range(1, 11)], tz="UTC")
    regions = np.repeat(["A", "B"], 120)
    x = np.abs(rng.normal(size=(len(days), len(FEATURE_NAMES)))) * 10
    x[:, FEATURE_NAMES.index("seasonal_hindcast_valid_months")] = 12
    x[:, FEATURE_NAMES.index("seasonal_hindcast_bias")] = np.linspace(-30, 70, len(days))
    point = np.full((len(days), 24), 200.)
    shape = np.arange(24.) - 11.5
    remainder = rng.normal(size=(len(days), 1)) * 2 + rng.normal(size=(len(days), 1)) * shape
    truth = point + coarse_anchor(x)[:, None] + remainder
    return point, truth, x, regions, days


def test_anchor_priority_counts_and_unsupported_zero():
    x = np.zeros((4, len(FEATURE_NAMES)))
    for method, bias in (("seasonal", 10), ("harmonic", 20), ("latest", 30)):
        x[:, FEATURE_NAMES.index(f"{method}_hindcast_bias")] = bias
    x[0, FEATURE_NAMES.index("seasonal_hindcast_valid_months")] = 2
    x[:2, FEATURE_NAMES.index("harmonic_hindcast_valid_months")] = 3
    x[:3, FEATURE_NAMES.index("hindcast_months")] = 4
    np.testing.assert_array_equal(coarse_anchor(x), [10, 20, 30, 0])


@pytest.mark.parametrize("mode", MODES[:-1])
def test_anchor_is_subtracted_for_fit_added_for_prediction_and_preserves_shape(mode):
    point, truth, x, regions, days = cohort()
    original_point = point.copy()
    anchor = coarse_anchor(x)
    expected = ConditionalTransport(["A", "B"], mode).fit(
        point, truth - anchor[:, None], x, regions, days=days, level_scales=10., shape_scales=3.)
    model = AnchorTransport(["A", "B"], mode).fit(
        point, truth, x, regions, days=days, level_scales=10., shape_scales=3.)
    base = expected.predict(point[:3], x[:3], level_scales=10., shape_scales=3., seed=19)
    result = model.predict(point[:3], x[:3], level_scales=10., shape_scales=3., seed=19)
    for key in ("corrected_point", "lower", "upper"):
        np.testing.assert_array_equal(result[key], base[key] + anchor[:3, None])
    np.testing.assert_array_equal(result["ensemble"], base["ensemble"] + anchor[:3, None, None])
    np.testing.assert_array_equal(result["bias"], base["bias"] + anchor[:3])
    for key in ("level_scale", "shape_scale"):
        np.testing.assert_array_equal(result[key], base[key])
    centered = lambda paths: paths - paths.mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(centered(result["ensemble"]), centered(base["ensemble"]), atol=1e-12)
    np.testing.assert_array_equal(point, original_point)
    assert model.audit_["raw_point_unchanged"]


def test_noanchor_is_the_original_structured_model_exactly():
    point, truth, x, regions, days = cohort()
    original = ConditionalTransport(["A", "B"], "structured").fit(point, truth, x, regions, days=days)
    ablation = AnchorTransport(["A", "B"], "noanchor").fit(point, truth, x, regions, days=days)
    expected = original.predict(point[:5], x[:5])
    actual = ablation.predict(point[:5], x[:5])
    assert expected.keys() == actual.keys()
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])


def test_anchor_uses_only_the_lawful_extractor_and_future_poison_cannot_change_prediction():
    starts = pd.date_range("2022-01-01", periods=36, freq="MS", tz="UTC")
    frame = pd.DataFrame({"region": "T", "period_start": starts,
                          "period_end": starts + pd.offsets.MonthBegin(1),
                          "available_at": starts + pd.offsets.MonthBegin(1) + pd.Timedelta(days=30)})
    for fuel in FUELS:
        frame[f"fuelshare_{fuel}"] = float(fuel == "gas")
    future = frame.iloc[[-1]].copy()
    future["available_at"] = pd.Timestamp("2028-01-01", tz="UTC")
    future["fuelshare_gas"] = np.nan
    x, _ = information_features(["2024-06-01"], "T", frame)
    poisoned, _ = information_features(["2024-06-01"], "T", pd.concat([frame, future]))
    np.testing.assert_array_equal(x, poisoned)
    np.testing.assert_array_equal(coarse_anchor(x), coarse_anchor(poisoned))
    point, truth, features, regions, days = cohort()
    model = AnchorTransport(["A", "B"]).fit(point, truth, features, regions, days=days)
    actual, other = model.predict(point[:1], x), model.predict(point[:1], poisoned)
    for key in actual:
        np.testing.assert_array_equal(actual[key], other[key])


def test_bridge_pickle_loads_in_fresh_python_process(tmp_path):
    point, truth, x, regions, days = cohort()
    model = AnchorTransport(["A", "B"], "global").fit(point, truth, x, regions, days=days)
    artifact, inputs, outputs = (tmp_path / name for name in ("model.pkl", "input.npz", "output.npz"))
    artifact.write_bytes(pickle.dumps(model))
    np.savez(inputs, point=point[:2], features=x[:2])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "scripts"), str(ROOT / "src")])}
    code = ("import pickle,sys,numpy as np; "
            "model=pickle.load(open(sys.argv[1],'rb')); x=np.load(sys.argv[2]); "
            "np.savez(sys.argv[3],**model.predict(x['point'],x['features']))")
    subprocess.run([sys.executable, "-c", code, str(artifact), str(inputs), str(outputs)],
                   env=env, check=True, capture_output=True, text=True, timeout=60)
    expected = model.predict(point[:2], x[:2])
    with np.load(outputs) as actual:
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])


def valid_grid():
    return {**runner.grid_parameters(), "records": [
        {"delay": delay, "path": f"fit/conditional_v6-{delay}.pkl",
         "base_path": f"base/round2-{delay}.pkl", "historical_path": f"historical/risk_v5-{delay}.pkl"}
        for delay in runner.DELAYS]}


def test_all_five_modes_share_the_frozen_six_multiplier_grid():
    expected = {"x1": 1., "x1p1": 1.1, "x1p2": 1.2, "x1p25": 1.25, "x1p5": 1.5, "x2": 2.}
    assert runner.MULTIPLIERS == runner.R6.MULTIPLIERS == expected
    assert runner.grid_parameters()["multipliers"] == expected
    assert runner.CONFIGS is runner.R6.CONFIGS
    for mode in MODES:
        for suffix, multiplier in expected.items():
            assert runner.CONFIGS[f"{mode}_{suffix}"] == {"mode": mode, "multiplier": multiplier}
    assert len(runner.CONFIGS) == 5 * 6 + 2
    changed = json.loads(json.dumps(valid_grid()))
    changed["multipliers"]["x1p1"] = 1.11
    with pytest.raises(ValueError, match="grid"):
        runner.validate_grid(changed)


def fake_bridge_root(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "docs").mkdir()
    for relative in runner.BRIDGE_FILES:
        (root / relative).write_text("frozen bridge: " + relative)
    (root / "scripts/run_transfer_v4.py").write_text("frozen V4")
    (root / runner.PROTOCOL).write_text("frozen anchored protocol")
    monkeypatch.setattr(runner, "ROOT", root)
    return root


def test_wrapper_rejects_same_name_policy_or_script_changes(tmp_path, monkeypatch):
    root = fake_bridge_root(tmp_path, monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    config = valid_grid()
    manifest = {"code_hashes": {str(root / relative): runner.V4.sha(root / relative)
                                for relative in runner.BRIDGE_FILES}}
    (source / "prefit_freeze.json").write_text(json.dumps(manifest))
    runner.verify_bridge_source(source, config)
    changed = json.loads(json.dumps(config))
    changed["bridge_policy"]["mode_map"]["noanchor"] = "global"
    with pytest.raises(ValueError, match="bridge_policy"):
        runner.verify_bridge_source(source, changed)
    for relative in runner.BRIDGE_FILES:
        original = (root / relative).read_text()
        (root / relative).write_text(original + " edited")
        with pytest.raises(ValueError, match="bridge script"):
            runner.verify_bridge_source(source, config)
        (root / relative).write_text(original)


def test_delegated_fit_freezes_both_scripts_and_restores_core_even_on_error(tmp_path, monkeypatch):
    import transcif.research.conditional_transfer_v6 as core
    root = fake_bridge_root(tmp_path, monkeypatch)
    data = tmp_path / "data"
    data.mkdir()
    (data / "monthly.csv").write_text("region,value\nR,1\n")
    output = tmp_path / "output"
    original_class, original_freeze = core.ConditionalTransport, runner.V4.freeze_inputs

    def delegated(*args, **kwargs):
        assert core.ConditionalTransport is AnchorTransport
        manifest = runner.V4.freeze_inputs(root, data, output)
        assert all(str(root / relative) in manifest["code_hashes"] for relative in runner.BRIDGE_FILES)
        raise RuntimeError("delegated fit stopped")

    monkeypatch.setattr(runner, "_ORIGINAL_FIT", delegated)
    with pytest.raises(RuntimeError, match="delegated fit stopped"):
        runner.fit_phase(data, "baseline", "historical", output)
    assert core.ConditionalTransport is original_class
    assert runner.V4.freeze_inputs is original_freeze
