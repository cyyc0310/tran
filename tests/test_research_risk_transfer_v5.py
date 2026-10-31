import numpy as np
import pandas as pd
import pytest

from transcif.research.risk_transfer_v5 import coarse_risk, risk_scales
from transcif.research.eia_proxy import FUELS


def monthly():
    starts = pd.date_range("2022-01-01", periods=15, freq="MS", tz="UTC")
    frame = pd.DataFrame({"region": "R", "period_start": starts, "period_end": starts+pd.offsets.MonthBegin(1),
                          "available_at": starts+pd.offsets.MonthBegin(1)+pd.Timedelta(days=30), "coverage": 1.})
    for fuel in FUELS:
        frame[f"fuelshare_{fuel}"] = float(fuel == "gas")
    frame["fuelshare_coal"] = np.arange(15)/30
    frame["fuelshare_gas"] = 1-frame.fuelshare_coal
    return frame


def test_future_poison_has_no_effect_and_release_delay_changes_risk():
    frame = monthly()
    origins = pd.DatetimeIndex(["2023-03-15"], tz="UTC")
    result = coarse_risk(origins, "R", frame)
    poison = frame.iloc[[-1]].copy()
    poison.available_at = "2027-01-01"
    poison.fuelshare_coal = np.nan
    altered = coarse_risk(origins.as_unit("us"), "R", pd.concat([frame, poison]))
    for a, b in zip(result, altered):
        np.testing.assert_allclose(a, b)
    assert coarse_risk(origins, "R", frame, 90)[2][0] < result[2][0]


def test_risk_changes_only_level_scale_and_preserves_point_and_shape():
    origins = pd.DatetimeIndex(["2023-03-15"], tz="UTC")
    point = np.arange(24.)[None, :]
    pred = {"point": point, "level_scale": np.array([20.]), "shape_scale": np.full((1, 24), 10.),
            "scale": np.full((1, 24), np.sqrt(500)), "age": np.array([43.])}
    changed = risk_scales(origins, "R", monthly(), pred, beta=1.)
    assert changed["level_scale"][0] > pred["level_scale"][0]
    np.testing.assert_array_equal(changed["shape_scale"], pred["shape_scale"])
    np.testing.assert_array_equal(changed["point"], point)
    np.testing.assert_array_equal(pred["level_scale"], [20.])
    with pytest.raises(ValueError, match="beta"):
        risk_scales(origins, "R", monthly(), pred, beta=-1)


def test_empty_history_keeps_prior_scale_and_reports_missing_support():
    risk, age, count = coarse_risk(pd.DatetimeIndex(["2020-01-01"], tz="UTC"), "R", monthly())
    np.testing.assert_array_equal(risk, [0.])
    np.testing.assert_array_equal(count, [0])
    assert age[0] == 365.
