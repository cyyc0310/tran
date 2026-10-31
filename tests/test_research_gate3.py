import json

import numpy as np
import pandas as pd
import pytest

from transcif.research.eia_proxy import FUELS
from transcif.research.gate3 import _load_source_frames, score_fold
from transcif.research.references import features


def test_reference_information_is_frozen_at_origin():
    rows = []
    for released, value in [("2023-12-01T00:00:00Z", .1), ("2024-01-01T12:00:00Z", .8)]:
        row = {"region": "CISO", "available_at": released, "period_end": "2023-11-01T00:00:00Z"}
        row.update({f"fuelshare_{f}": value if f == "coal" else (1 - value) / 7 for f in FUELS})
        rows.append(row)
    monthly = pd.DataFrame(rows)
    times = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    x, mix, valid = features(times, "CISO", monthly)
    assert valid.all()
    assert np.all(mix[:, 0] == .1)
    monthly.loc[1, "fuelshare_coal"] = 999
    changed, _, _ = features(times, "CISO", monthly)
    np.testing.assert_array_equal(x, changed)


def test_source_reader_excludes_target_and_future(tmp_path, monkeypatch):
    from transcif.research import gate3
    monkeypatch.setattr(gate3, "REGIONS", ("CISO", "ERCO"))
    pd.DataFrame({"time": ["2023-12-31T23:00:00Z", "2024-01-01T00:00:00Z"], "value": [1, 999]}).to_csv(tmp_path / "ERCO_hourly.csv", index=False)
    # There is deliberately no target file. Prediction-side reading must work.
    frames = _load_source_frames(tmp_path, "CISO")
    assert set(frames) == {"ERCO"}
    assert frames["ERCO"].value.tolist() == [1]


def test_score_rejects_wrong_region_or_factor_before_loading_labels(tmp_path):
    (tmp_path / "prediction_seal.json").write_text(json.dumps({"target": "CISO", "other_ef": 400}))
    with pytest.raises(ValueError, match="region/EF"):
        score_fold(tmp_path / "nonexistent-data", tmp_path, "ERCO", 400)
    with pytest.raises(ValueError, match="region/EF"):
        score_fold(tmp_path / "nonexistent-data", tmp_path, "CISO", 0)
