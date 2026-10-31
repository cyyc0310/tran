"""Behavioral contracts for source-only V10 information baselines."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/information_models_v10.py"
SPEC = importlib.util.spec_from_file_location("information_models_v10_under_test", PATH)
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)
InfoModel = MODEL.InfoModel


def source_fixture():
    fit_regions = np.array(["BPAT"] * 10 + ["CISO"] * 20)
    xfit = np.column_stack([
        np.r_[np.arange(10, dtype=float), np.arange(20, dtype=float) + 100],
        np.full(30, np.nan),
        np.r_[np.arange(10, dtype=float), np.arange(20, dtype=float)],
    ])
    xfit[2, 2] = np.nan
    anchors_fit = np.full(30, 300.0)
    yfit = anchors_fit[:, None] + np.arange(24)[None, :] + np.arange(30)[:, None] * .1
    cal_regions = np.array(["BPAT", "CISO", "CISO", "CISO"])
    xcal = np.array([[1e6, 999, np.nan], [1e6, 999, 2],
                     [1e6, 999, 3], [1e6, 999, 4]], dtype=float)
    anchors_cal = np.full(4, 300.0)
    ycal = anchors_cal[:, None] + np.arange(24)[None, :] + 10
    return xfit, yfit, fit_regions, anchors_fit, xcal, ycal, cal_regions, anchors_cal


@pytest.mark.parametrize("kind", ["ridge", "extra_trees"])
def test_source_only_imputation_weighting_and_public_prediction(kind):
    data = source_fixture()
    model = InfoModel(kind).fit(*data, exclude_region="ERCO")
    assert model.medians_[0] == pytest.approx(104.5)
    assert model.medians_[1] == 0
    assert model.all_missing_columns_.tolist() == [False, True, False]
    # Source BPAT/CISO carry equal total mass despite 10 versus 20 days.
    assert model.scaler_.mean_[0] == pytest.approx((4.5 + 109.5) / 2)
    assert model.scaler_.mean_[4] == pytest.approx(1.0)  # all-missing column indicator
    assert model.audit_["proper_regions"] == {"BPAT": 10, "CISO": 20}
    assert model.audit_["calibration_regions"] == {"BPAT": 1, "CISO": 3}
    assert model.audit_["excluded_evaluation_region"] == "ERCO"
    assert model.audit_["proper_fit_only_transform"] is True
    assert "truth" not in inspect.signature(InfoModel.predict).parameters

    origin = pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC")
    prediction = model.predict(data[4][:2], data[7][:2], origin)
    assert prediction["point"].shape == (2, 24)
    assert prediction["ensemble"].shape == (2, 128, 24)
    assert np.isfinite(prediction["ensemble"]).all()
    assert np.all((prediction["ensemble"] >= 0) & (prediction["ensemble"] <= 1000))
    assert np.all((prediction["point"] >= 0) & (prediction["point"] <= 1000))


def test_full_day_equal_region_calibration_and_origin_determinism():
    fit_regions = np.array(["BPAT"] * 20 + ["CISO"] * 20)
    xfit = np.zeros((40, 2))
    anchors_fit = np.full(40, 200.0)
    yfit = np.full((40, 24), 200.0)
    cal_regions = np.array(["BPAT"] + ["CISO"] * 9)
    xcal = np.zeros((10, 2))
    anchors_cal = np.full(10, 200.0)
    first = np.arange(24, dtype=float) - 12
    second = np.arange(24, dtype=float) + 40
    ycal = 200 + np.vstack([first, *([second] * 9)])
    model = InfoModel("ridge").fit(xfit, yfit, fit_regions, anchors_fit,
                                    xcal, ycal, cal_regions, anchors_cal)
    days = pd.date_range("2026-01-01", periods=3, tz="UTC")
    x = np.zeros((3, 2))
    anchors = np.full(3, 200.0)
    together = model.predict(x, anchors, days)
    separate = [model.predict(x[i:i + 1], anchors[i:i + 1], days[i:i + 1]) for i in range(3)]
    for i, one in enumerate(separate):
        np.testing.assert_array_equal(together["point"][i:i + 1], one["point"])
        np.testing.assert_array_equal(together["ensemble"][i:i + 1], one["ensemble"])
    draws = together["ensemble"][0] - together["point"][0]
    assert all(np.array_equal(draw, first) or np.array_equal(draw, second) for draw in draws)
    # The nine CISO calibration days must not give CISO nine times the mass.
    first_count = sum(np.array_equal(draw, first) for draw in draws)
    assert 35 <= first_count <= 93


def test_outer_region_and_target_labels_are_rejected_from_both_splits():
    data = list(source_fixture())
    with pytest.raises(ValueError, match="Outer evaluation region"):
        InfoModel("ridge").fit(*data, exclude_region="BPAT")
    data[6] = np.array(["ERCO", "CISO", "CISO", "CISO"])
    with pytest.raises(ValueError, match="Outer evaluation region"):
        InfoModel("ridge").fit(*data, exclude_region="ERCO")
    data[6] = np.array(["AZPS", "CISO", "CISO", "CISO"])
    with pytest.raises(ValueError, match="seven declared source"):
        InfoModel("ridge").fit(*data)


def test_invalid_feature_and_prediction_contracts_fail_closed():
    data = list(source_fixture())
    data[0][0, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        InfoModel("ridge").fit(*data)
    model = InfoModel("ridge").fit(*source_fixture())
    with pytest.raises(ValueError, match="feature width"):
        model.predict(np.zeros((1, 2)), np.array([300.]), ["2026-01-01"])
    with pytest.raises(ValueError, match="n_samples"):
        model.predict(np.zeros((1, 3)), np.array([300.]), ["2026-01-01"], n_samples=1)
    with pytest.raises(ValueError, match="kind"):
        InfoModel("unsupported")
