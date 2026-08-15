"""Tests for transcif.data.windows and loaders."""

import numpy as np
import pytest

from transcif.data.windows import build_windows
from transcif.data.loaders import load_region_data
from transcif.config import SEQ_LEN, HORIZON


def test_build_windows_shapes():
    n = 1000
    rs = np.random.rand(n).astype(np.float32)
    cif = (np.random.rand(n) * 400).astype(np.float32)
    x, y_rs, y_cif = build_windows(rs, cif)
    assert x.shape[1] == SEQ_LEN
    assert y_rs.shape[1] == HORIZON
    assert y_cif.shape[1] == HORIZON
    assert x.shape[0] == y_rs.shape[0] == y_cif.shape[0]
    assert x.shape[0] > 0


def test_build_windows_empty_on_short_series():
    rs = np.random.rand(10).astype(np.float32)
    cif = np.random.rand(10).astype(np.float32)
    x, y_rs, y_cif = build_windows(rs, cif)
    assert x.shape[0] == 0


def test_load_region_data_keys():
    # Construct a tiny fake CSV in a temp dir to exercise the loader.
    import pandas as pd
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        df = pd.DataFrame({
            "hour": pd.date_range("2023-01-01", periods=500, freq="h"),
            "renew_share": np.random.rand(500),
            "cif_real_gco2_per_kwh": np.random.rand(500) * 400,
        })
        f = Path(td) / "US_CISO_2023_hourly.csv"
        df.to_csv(f, index=False)
        cfg = {"file": "US_CISO_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 342.8}
        data = load_region_data("US_CISO", {"US_CISO": cfg}, data_dir=Path(td))
        assert "rs" in data and "cif" in data
        assert data["ef_nr"] == 342.8
        # mean_rs must be derived from the training split only (500 * 0.8 = 400
        # rows) so the zero-shot config vector does not leak test-period data.
        train_split = int(len(df) * 0.8)
        assert abs(data["mean_rs"] - df["renew_share"].iloc[:train_split].mean()) < 1e-5
        # Sanity: the full-series mean should NOT match once the leak is fixed.
        assert abs(data["mean_rs"] - df["renew_share"].mean()) > 1e-6


def test_discover_uk_regions_efnr_train_split_only(monkeypatch, tmp_path):
    """UK ef_nr estimation must use the TRAINING split only (leak fix).

    Companion to the mean_rs regression above: ef_nr enters the zero-shot
    config vector and the config-distance weighting, so a full-series
    estimate would leak test-period CIF exactly like mean_rs did pre-fix.

    Discriminating design (median is robust, so the rows are laid out to
    make full-series vs train-split medians differ):
      - train (first 960 h): bimodal ratios 600 x480 / 1000 x480 -> median 800
      - test  (last  240 h): ratio 600 -> full-series median would be 600
    """
    import pandas as pd
    from transcif.data.loaders import discover_uk_regions
    import transcif.data.loaders as loaders

    n_hours = 1200
    split = int(n_hours * 0.8)  # 960
    rng = np.random.default_rng(0)
    rs = (0.3 + 0.4 * rng.random(n_hours)).astype(np.float32)  # in (0.05, 0.95)

    ratios = np.empty(n_hours, dtype=np.float64)
    ratios[:480] = 600.0            # train half A
    ratios[480:split] = 1000.0      # train half B
    ratios[split:] = 600.0          # test portion
    # ef_r = 0 -> cif = (1 - rs) * ratio makes cif / (1 - rs) == ratio exactly
    cif = ((1.0 - rs) * ratios).astype(np.float32)

    df = pd.DataFrame({
        "hour": pd.date_range("2023-01-01", periods=n_hours, freq="h"),
        "renew_share": rs,
        "cif_real_gco2_per_kwh": cif,
    })
    df.to_csv(tmp_path / "UK_TEST_2023_hourly.csv", index=False)

    # Isolate the module-level UK_REGIONS dict so the test doesn't pollute
    # the shared config state for other tests.
    monkeypatch.setattr(loaders, "UK_REGIONS", {})

    discovered = discover_uk_regions(data_dir=tmp_path)

    assert "UK_TEST" in discovered, "region should pass the discovery filters"
    ef_nr = discovered["UK_TEST"]["ef_nr"]
    assert ef_nr == pytest.approx(800.0, abs=0.01), (
        f"ef_nr {ef_nr} != train-split median 800; a full-series estimate "
        f"would collapse to 600 (test-period leak)"
    )
