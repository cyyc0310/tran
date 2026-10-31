import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transcif.research.fuel_transport_v7 import FuelTrajectoryBank, GENERATION_COLUMNS

spec = importlib.util.spec_from_file_location("risk_v9_test", Path(__file__).parents[1] / "scripts/risk_paths_v9.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def bank():
    frames = {}
    for region, gas in [("A", 30.), ("B", 100.), ("C", 300.)]:
        t = pd.date_range("2024-01-01", periods=96, freq="h", tz="UTC")
        f = pd.DataFrame({"time": t, "valid": True})
        for col in GENERATION_COLUMNS:
            f[col] = 0.
        wave = np.sin(np.arange(96) * 2 * np.pi / 24)
        f["generation_gas"] = gas * (1 + .3 * wave)
        f["generation_wind"] = 100 * (1 - .4 * wave)
        frames[region] = f
    return FuelTrajectoryBank().fit(frames)


def forecasts(n=2):
    m = np.array([.25, .35, .05, .05, .05, .15, .05, .05])
    draws = np.tile(m, (n, 16, 1))
    draws[:, ::2, 0] += .05
    draws[:, ::2, 1] -= .05
    return {method: dict(mix=np.tile(m, (n, 1)), draws=draws.copy(), audit={}) for method in R.METHODS}


def test_complete_paired_grid_bounds_mean_and_batch_invariance():
    b = bank()
    days = pd.DatetimeIndex(["2025-01-03", "2025-02-08"], tz="UTC")
    fs = forecasts()
    first = R.predict_paths(b, fs, days)
    reverse = {k: {**v, "mix": v["mix"][::-1], "draws": v["draws"][::-1]} for k, v in fs.items()}
    second = R.predict_paths(b, reverse, days[::-1].as_unit("us"))
    assert len(R.configs()) == 32
    assert len(first) == 20
    assert {c["name"] for c in R.configs()} == set(first)
    for key, out in first.items():
        assert out["ensemble"].shape == (2, 16, 24)
        assert np.isfinite(out["ensemble"]).all()
        assert 0 <= out["ensemble"].min() <= out["ensemble"].max() <= 1000
        np.testing.assert_array_equal(out["point"], out["ensemble"].mean(1))
        np.testing.assert_array_equal(out["ensemble"], second[key]["ensemble"][::-1])


def test_inner_source_region_never_enters_crossfit_center(monkeypatch):
    b = bank(); R._RESIDUAL_CACHE.clear()
    original = b._weights
    masks = []
    def observe(mix, origin, mode, bandwidth, eligible):
        excluded = np.unique(b.regions_[~eligible])
        assert len(excluded) == 1
        assert np.all(b.regions_[eligible] != excluded[0])
        masks.append(str(excluded[0]))
        return original(mix, origin, mode, bandwidth, eligible)
    monkeypatch.setattr(b, "_weights", observe)
    residual = R.crossfit_residuals(b)
    assert set(masks) == {"A", "B", "C"}
    assert not residual.flags.writeable
    assert np.isfinite(residual).all()


def test_residual_cache_changes_when_training_labels_change():
    b = bank(); one = R.crossfit_residuals(b).copy()
    b.raw_cif_ = b.raw_cif_ + 10
    two = R.crossfit_residuals(b)
    np.testing.assert_allclose(two - one, 10)


def test_prediction_rejects_invalid_mix_dates_and_missing_method():
    b = bank(); fs = forecasts()
    with pytest.raises(ValueError, match="cutoff"):
        R.predict_paths(b, fs, ["2024-01-01", "2024-01-02"])
    bad = forecasts(); bad.pop("latest")
    with pytest.raises(ValueError, match="grid"):
        R.predict_paths(b, bad, ["2025-01-01", "2025-01-02"])
    fs["seasonal"]["draws"][0, 0, 0] = -1
    with pytest.raises(ValueError, match="simplex"):
        R.predict_paths(b, fs, ["2025-01-01", "2025-01-02"])


def test_identical_coarse_forecasts_share_mapping_draws():
    out = R.predict_paths(bank(), forecasts(), ["2025-01-01", "2025-01-02"])
    for mode in ("additive", "recompose"):
        np.testing.assert_array_equal(out[f"latest--{mode}"]["ensemble"], out[f"release_bridge--{mode}"]["ensemble"])
