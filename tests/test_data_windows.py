"""Tests for transcif.data.windows and loaders."""

import numpy as np

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
